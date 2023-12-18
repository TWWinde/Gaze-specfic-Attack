import argparse
import os
import yacs.config
import sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import torch
import torchvision.utils
from gaze_estimation import (GazeEstimationMethod, create_dataloader,
                                    create_logger, create_loss, create_model,
                                    create_optimizer, create_scheduler,
                                    create_tensorboard_writer, get_default_config)
from gaze_estimation.utils import (AverageMeter, compute_angle_error,
                                          create_train_output_dir, load_config,
                                          save_config, set_seeds, setup_cudnn)
from gaze_estimation.datasets import create_dataset
import yaml
sys.path.append('./GMI-Attack/')
sys.path.append('./Celeba/')
'''
    This file is used to train gaze estimator. The preparation of target model.
'''

parser = argparse.ArgumentParser(description='train gaze estimator.')
parser.add_argument('--pretraining', default=True, help=" pre train the model without the target person but the others.")
parser.add_argument('--dataset', choices=['mpiifacegaze', 'nvgaze', 'mpiiface'], default='mpiifacegaze', help='Specify the dataset')
args = parser.parse_args()


def train(epoch, model, optimizer, scheduler, loss_function, train_loader,
          config, tensorboard_writer, logger):

    '''
    trainer of gaze estimator
    '''

    logger.info(f'Train {epoch}')
    model.train()
    device = torch.device("cuda:0")

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    if config.mode == GazeEstimationMethod.GazeCapture.name or config.mode == GazeEstimationMethod.NVGaze.name:
        for step, (images, gazes) in enumerate(train_loader):
            if config.tensorboard.train_images and step == 0:
                image = torchvision.utils.make_grid(images,
                                                    normalize=True,
                                                    scale_each=True)
                tensorboard_writer.add_image('Train/Image', image, epoch)

            images = images.to(device)
            gazes = gazes.to(device)
            optimizer.zero_grad()
            rz = transforms.Resize((224, 224))
            outputs = model(rz(images))
            loss = loss_function(outputs, gazes)
            loss.backward()
            optimizer.step()
            angle_error = compute_angle_error(outputs, gazes).mean()
            num = images.size(0)
            loss_meter.update(loss.item(), num)
            angle_error_meter.update(angle_error.item(), num)

            if step % config.train.log_period == 0:
                logger.info(f'Epoch {epoch} '
                            f'Step {step}/{len(train_loader)} '
                            f'lr {scheduler.get_last_lr()[0]:.6f} '
                            f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            f'angle error {angle_error_meter.val:.2f} '
                            f'({angle_error_meter.avg:.2f})')
    else:
        for step, (images, poses, gazes) in enumerate(train_loader):
            if config.tensorboard.train_images and step == 0:
                image = torchvision.utils.make_grid(images,
                                                    normalize=True,
                                                    scale_each=True)
                tensorboard_writer.add_image('Train/Image', image, epoch)

            images = images.to(device)
            poses = poses.to(device)
            gazes = gazes.to(device)
            optimizer.zero_grad()
            if config.mode == GazeEstimationMethod.MPIIGaze.name:
                outputs = model(images, poses)
            elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
                rz = transforms.Resize((224, 224))
                outputs = model(rz(images))
            else:
                raise ValueError
            loss = loss_function(outputs, gazes)
            loss.backward()
            optimizer.step()
            angle_error = compute_angle_error(outputs, gazes).mean()

            num = images.size(0)
            loss_meter.update(loss.item(), num)
            angle_error_meter.update(angle_error.item(), num)

            if step % config.train.log_period == 0:
                logger.info(f'Epoch {epoch} '
                            f'Step {step}/{len(train_loader)} '
                            f'lr {scheduler.get_last_lr()[0]:.6f} '
                            f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            f'angle error {angle_error_meter.val:.2f} '
                            f'({angle_error_meter.avg:.2f})')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')

    tensorboard_writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
    tensorboard_writer.add_scalar('Train/lr',
                                  scheduler.get_last_lr()[0], epoch)
    tensorboard_writer.add_scalar('Train/AngleError', angle_error_meter.avg,
                                  epoch)
    tensorboard_writer.add_scalar('Train/Time', elapsed, epoch)


def validate(epoch, model, loss_function, val_loader, config,
             tensorboard_writer, logger):
    logger.info(f'Val {epoch}')
    model.eval()
    device = torch.device("cuda:0")

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    with torch.no_grad():
        for step, (images, poses, gazes) in enumerate(val_loader):
            if config.tensorboard.val_images and epoch == 0 and step == 0:
                image = torchvision.utils.make_grid(images,
                                                    normalize=True,
                                                    scale_each=True)
                tensorboard_writer.add_image('Val/Image', image, epoch)

            images = images.to(device)
            poses = poses.to(device)
            gazes = gazes.to(device)

            if config.mode == GazeEstimationMethod.MPIIGaze.name:
                outputs = model(images, poses)
            elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
                rz = transforms.Resize((224, 224))
                outputs = model(rz(images))
            else:
                raise ValueError
            loss = loss_function(outputs, gazes)

            angle_error = compute_angle_error(outputs, gazes).mean()

            num = images.size(0)
            loss_meter.update(loss.item(), num)
            angle_error_meter.update(angle_error.item(), num)

    logger.info(f'Epoch {epoch} '
                f'loss {loss_meter.avg:.4f} '
                f'angle error {angle_error_meter.avg:.2f}')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')

    if epoch > 0:
        tensorboard_writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/AngleError', angle_error_meter.avg,
                                      epoch)
    tensorboard_writer.add_scalar('Val/Time', elapsed, epoch)

    if config.tensorboard.model_params:
        for name, param in model.named_parameters():
            tensorboard_writer.add_histogram(name, param, epoch)

#  dataset for training and updating


def get_dataloader(person, num):
    if args.dataset == 'mpiifacegaze':
        config = get_default_config()
        config.merge_from_file('./configs/mpiifacegaze/experiments.yaml')
    elif args.dataset == 'nvgaze':
        config = get_default_config()
        config.merge_from_file('./configs/NVGaze.yaml')
    elif args.dataset == 'mpiigaze':
        config = get_default_config()
        config.merge_from_file('./configs/mpiigaze/resnet_preact_train.yaml')
    config = get_default_config()
    config.merge_from_file('./configs/mpiifacegaze/experiments.yaml')
    config.freeze()
    train_dataset, val_dataset = create_dataset(config, True, person, False, False, num)
    # image, gaze
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.val_dataloader.num_workers,
        pin_memory=config.train.train_dataloader.pin_memory,
        drop_last=config.train.train_dataloader.drop_last,
    )
    update_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.val_dataloader.num_workers,
        pin_memory=config.train.val_dataloader.pin_memory,
        drop_last=False,
    )

    return train_loader, update_loader


def main(i, j):
    if args.dataset == 'mpiifacegaze':
        with open('./configs/mpiifacegaze/experiments.yaml', 'r') as f:
            config_dict = yaml.safe_load(f)
    elif args.dataset == 'nvgaze':
        with open('configs/mpiigaze1/resnet_preact_train.yaml', 'r') as f:
            config_dict = yaml.safe_load(f)
    elif args.dataset == 'mpiigaze':
        with open('./configs/NVGaze.yaml', 'r') as f:
            config_dict = yaml.safe_load(f)
    config = yacs.config.CfgNode(config_dict)
    set_seeds(config.train.seed)
    if args.pretraining:
        person = [k for k in range(15) if k != i]
    else:
        person = [i]
    print(person)
    num = [10, 50, 100, 500, 1000]  # the number of samples which are  taken from each participant
    output_dir = create_train_output_dir(config)
    save_config(config, output_dir)
    logger = create_logger(name=__name__, output_dir=output_dir, filename='log.txt')
    logger.info(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloader(person, num[j])
    model = create_model(config)
    loss_function = create_loss(config)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    tensorboard_writer = create_tensorboard_writer(config, output_dir)

    for epoch in range(1, config.scheduler.epochs + 1):
        train(epoch, model, optimizer, scheduler, loss_function, train_loader,
              config, tensorboard_writer, logger)
        scheduler.step()

        if (epoch % config.train.checkpoint_period == 0
                or epoch == config.scheduler.epochs):
            os.makedirs(f'./result/{args.dataset}', exist_ok=True)
            torch.save(model, os.path.join(f'./result/{args.dataset}', f"gaze_estimator_{i}_{num[j]}.pth"))


if __name__ == '__main__':
    for i in range(15):
        for j in range(5):
            main(i, j)


