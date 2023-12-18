import os
import torch.nn as nn
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
sys.path.append('./GMI-Attack/GMI-Attack/')
sys.path.append('./GMI-Attack/GMI-Attack/Gaze-specific_Attack/')

'''
This file is used to train gaze estimation models with different amount of samples with/without pretraining.
'''


def train(epoch, model, optimizer, scheduler, loss_function, train_loader,
          config,  logger):
    logger.info(f'Train {epoch}')

    model.train()

    # device = torch.device(config.device)
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


def validate(epoch, model, loss_function, val_loader, config, logger):
    logger.info(f'Val {epoch}')

    model.eval()

    # device = torch.device(config.device)
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


def main(i, j):
    a = [10, 50, 100, 500, 1000]
    num = a[j]
    b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    #del b[i]
    print(torch.cuda.is_available())
    config = get_default_config()
    config.merge_from_file('configs/NVGaze.yaml')
    config.freeze()
    train_dataset, val_dataset = create_dataset(config, True, [b[i]], False, False, False, num)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.val_dataloader.num_workers,
        pin_memory=config.train.train_dataloader.pin_memory,
        drop_last=config.train.train_dataloader.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.val_dataloader.num_workers,
        pin_memory=config.train.val_dataloader.pin_memory,
        drop_last=False,
    )

    set_seeds(config.train.seed)

    output_dir = create_train_output_dir(config)
    save_config(config, output_dir)
    logger = create_logger(name=__name__, output_dir=output_dir, filename='log.txt')
    logger.info(config)
    os.makedirs('./target_models/with_pretraining/',exist_ok=True)
    target_path = f"./target_models/with_pretraining/target_model_pretrain_{i}.pth"
    model = torch.load(target_path)
    model = nn.DataParallel(model).cuda()
    loss_function = create_loss(config)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)

    for epoch in range(1, config.scheduler.epochs + 1):
        train(epoch, model, optimizer, scheduler, loss_function, train_loader, config, logger)
        scheduler.step()

        if epoch % config.train.val_period == 0:
            validate(epoch, model, loss_function, val_loader, config, logger)

        if (epoch % config.train.checkpoint_period == 0
                or epoch == config.scheduler.epochs):
            torch.save(model, os.path.join('./target_models', 'with_pretraining', f"target_model_pretrain_{i}_{num}.pth"))


if __name__ == '__main__':
    for i in range(14):
        for j in range(5):
            main(i, j)



