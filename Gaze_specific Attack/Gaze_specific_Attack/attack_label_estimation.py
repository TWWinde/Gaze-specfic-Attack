import torch
import torchvision.transforms as transforms
import numpy as np
import yaml
import yacs.config
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from scipy.stats import entropy
from tqdm import trange
from sklearn.neighbors import KernelDensity
from gaze_estimation import get_default_config, create_dataloader, create_logger, create_model, create_loss, \
    create_optimizer, create_scheduler, GazeEstimationMethod
from gaze_estimation.datasets import create_dataset
from gaze_estimation.utils import create_train_output_dir, set_seeds, save_config, compute_angle_error, \
    AverageMeter
import matplotlib.pyplot as plt

'''
This file implements label estimation attack using MLP. 

'''


def dataset_for_train_update():
    config = get_default_config()
    config.merge_from_file('./experiments.yaml')
    config.freeze()
    train_dataset, val_dataset = create_dataset(config, True, [i for i in range(1,15)], False, False)
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
        shuffle=True,
        num_workers=config.train.val_dataloader.num_workers,
        pin_memory=config.train.val_dataloader.pin_memory,
        drop_last=False,
    )

    return train_loader, update_loader


def dataset_for_test():  # MPIIfacegaze dataset
    config = get_default_config()
    config.merge_from_file('./configs/mpiifacegaze/resnet_simple_14_train.yaml')
    config.freeze()
    train_dataset, val_dataset = create_dataset(config, True, [0], False, False)
    testloader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
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

    return testloader


def update_test_model(updateloader, testloader, optimizer, model, loss_function, outputoriginal, epochsNum=1):

    ######## update
    model.train()
    for epoch in range(epochsNum):  # loop over the dataset multiple times
        running_loss = 0.0
        labelsTotal = []
        for step, (images, poses, gazes) in enumerate(updateloader):
            images, gazes = torch.tensor(images), torch.tensor(gazes)
            images, gazes = images.to(device), gazes.to(device)
            optimizer.zero_grad()
            rz = transforms.Resize((224, 224))
            outputs = model(rz(images))

            loss = loss_function(outputs, gazes)
            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            labelsTotal.append(gazes.cpu().detach().numpy())
            if len(labelsTotal) == 60:
                labelsTotal = np.vstack(labelsTotal)
                break

            running_loss += loss.item()
    labelsTotal = np.array(labelsTotal)
    ####### test
    model.eval()
    updatedOutputs = []
    with torch.no_grad():
        for step, (images, poses, gazes) in enumerate(testloader):
            images, gazes = torch.tensor(images), torch.tensor(gazes)
            images, gazes = images.to(device), gazes.to(device)
            rz = transforms.Resize((224, 224))
            outputs = model(rz(images))
            updatedOutputs.append(torch.exp(outputs).cpu().detach().numpy())
            if len(updatedOutputs) == 60:
                updatedOutputs = np.vstack(updatedOutputs)
                break

    outputDiff = np.array(outputoriginal) - np.array(updatedOutputs)

    return model, labelsTotal, outputDiff


def train(epoch, model, optimizer, scheduler, loss_function, train_loader, config, logger):
    logger.info(f'Train {epoch}')
    model.train()
    device = torch.device("cuda:0")

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    for step, (images, gazes) in enumerate(train_loader):

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

    torch.save(model, os.path.join('/projects/tang/GMI-Attack/GMI-Attack/Celeba/result', "Gaze_estimator_for_inference_attack.pth"))

    return model


def train_model(train_loader, model, optimizer, scheduler, loss_function,config, logger ):

    for epoch in range(1, config.scheduler.epochs + 1):
        model = train(epoch, model, optimizer, scheduler, loss_function, train_loader,
                      config, logger)

    return model


def testModel(testloader, model):
    model.eval()

    outputoriginal = []
    with torch.no_grad():
        for step, (images, poses, gazes) in enumerate(testloader):
            images, gazes = torch.tensor(images), torch.tensor(gazes)
            images, gazes = images.to(device), gazes.to(device)
            rz = transforms.Resize((224, 224))
            outputs = model(rz(images))

            outputoriginal.append(torch.exp(outputs).cpu().detach().numpy())  # exp as activation function
            if len(outputoriginal) == 60:
                combined_array = np.vstack(outputoriginal)
                break


    return combined_array


def kernel_density_estimation(data1,data2):

    kde1 = KernelDensity(bandwidth=0.1, kernel='gaussian')
    kde1.fit(data1)

    kde2 = KernelDensity(bandwidth=0.1, kernel='gaussian')
    kde2.fit(data2)

    pdf1 = np.exp(kde1.score_samples(data1))
    pdf2 = np.exp(kde2.score_samples(data2))
    return pdf1, pdf2


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, hist1, hist2):

        kl_divergence = -torch.sum(hist1 * (torch.log(hist1) - torch.log(hist2)))
        return torch.tensor(kl_divergence, dtype=torch.float).to('cuda')


def torch_histogram(data, bins=10, range=None, density=False):

    data = (data - data.min()) / (data.max() - data.min())

    hist = torch.histc(data, bins=bins, min=0, max=1)

    if density:
        hist = hist / hist.sum()

    return hist


kl=[]
def train_model_adversary(gaze_pre_model ,device, labelsTotal, outputDiff, step):

    gaze_pre_model.to(device)
    optimizer = optim.Adam(gaze_pre_model.parameters(), lr=2e-4)

    inputs, groundtruth = torch.from_numpy(outputDiff).to(device), torch.from_numpy(labelsTotal).to(device)

    optimizer.zero_grad()
    outputs = gaze_pre_model(inputs)

    loss_fn = KLDivergenceLoss()

    pdf1, pdf2 = kernel_density_estimation(outputs.cpu().detach().numpy(), groundtruth.cpu().detach().numpy())
    kl_divergence1 = entropy(pdf1,pdf2)

    kl_loss = nn.KLDivLoss()(outputs, groundtruth)
    kl_divergence = torch.from_numpy(np.asarray(kl_divergence1)).to(device)
    zero = torch.zeros(1920, 2).to(device)
    mse_loss = nn.MSELoss()(outputs, groundtruth)
    loss = mse_loss #- 0.3*nn.MSELoss()(outputs, zero) #mse_loss
    loss.backward()
    optimizer.step()
    data2 = outputs.cpu().detach().numpy()
    data1 = groundtruth.cpu().detach().numpy()
    plot(data1, data2, kl_divergence1, step)

    print(f" Steps: {step: d} Loss: {loss.item():.4f} kl_divergence1: {kl_divergence1.item():.4f}, end='\r'")

    return gaze_pre_model


class Gazepred(nn.Module):
    def __init__(self, attackInput=2, gaze_size=2):
        super(Gazepred, self).__init__()
# The encoder is considered all layers except the last one, which is considered to be the decoder in this attack.
        self.preAttack = nn.Sequential(
            nn.Linear(attackInput, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, gaze_size),
            )

    def forward(self, inputAttack):
        x =self.preAttack(inputAttack)
        x = torch.tanh(x)
        return x


def plot(data1, data2, kl_divergence, step):

    plt.figure(figsize=(10, 8))

    plt.scatter(data1[:, 0], data1[:, 1], color='blue', label='groundtruth')

    plt.scatter(data2[:, 0], data2[:, 1], color='red', label='prediction')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'Scatter Plot of groundtruth and prediction KL_divergence: {kl_divergence:.4f}')
    plt.text(8, 6, f'KL Divergence: {kl_divergence:.4f}', fontsize=20, color='black')

    plt.legend()

    plt.savefig(f'./result/scatters_img/scatter_plot_14_parson_{step}.png')

    plt.close()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    use_cuda = torch.cuda.is_available()
    with open('./configs/gazecapture.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)

    config = yacs.config.CfgNode(config_dict)
    set_seeds(config.train.seed)

    output_dir = create_train_output_dir(config)
    save_config(config, output_dir)
    logger = create_logger(name=__name__,
                           output_dir=output_dir,
                           filename='log.txt')
    logger.info(config)
    model = create_model(config)
    loss_function = create_loss(config)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    train_loader, update_loader = dataset_for_train_update()
    test_loader = dataset_for_test()  # MPIIFaceGaze
    target_path = os.path.join('./result/label_inference gaze_estimator_original_14.pth')
    model = torch.load(target_path)
    model = nn.DataParallel(model).cuda()
    if all(param.requires_grad for param in model.parameters()):
        print("Model has been successfully loaded with checkpoint")
    else:
        print('Nope')
    # train_model(train_loader, model, optimizer, scheduler, loss_function,config, logger)
    outputoriginal = testModel(test_loader, model)
    print(outputoriginal.shape)
    Epoch = 500
    gaze_pre_model = Gazepred()
    for i in trange(Epoch):
        model, labelsTotal, outputDiff = update_test_model(update_loader, test_loader, optimizer, model,
                                                           loss_function, outputoriginal, epochsNum=1)

        gaze_pre_model = train_model_adversary(gaze_pre_model, device, labelsTotal, outputDiff, i)

    torch.save(gaze_pre_model, os.path.join(config.train.output_dir, "gaze_estimator_for_label_attack.pth"))
    pass
