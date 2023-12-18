import torch, os, time, random, generator, discri, classify, utils
import numpy as np
import torch.nn as nn
import torchvision.utils as tvls
import torchvision.transforms as transforms
from gaze_estimation.config import get_default_config
from gaze_estimation.datasets import create_dataset
from gaze_estimation.utils import (AverageMeter, compute_angle_error)
from generator import InversionNet

'''
This file is used to do attack experiments for different models which are trained on different amount of samples.

'''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs('./result/attack_MPIIGaze/', exist_ok=True)


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def inversion(G, D, T, E, gazes, ids, ground_truth, blurred, lr=2e-1, momentum=0.9, lamda=1,
              iter_times=3000,
              clip_range=1):
    gazes = gazes.to(device)
    blurred = blurred.to(device)
    ids = ids.long().to(device)
    ground_truth = ground_truth.to(device)
    criterion = nn.L1Loss(reduction='mean').to(device)  ####
    criterion1 = nn.CrossEntropyLoss().to(device)
    bs = gazes.shape[0]
    c = 0

    ids = torch.nn.functional.one_hot(ids, num_classes=14)
    ids = ids.to(torch.float32)
    ids = ids.to(device)

    G.eval()
    D.eval()
    T.eval()
    E.eval()

    rz = transforms.Resize((224, 224))
    total_loss = 999999
    for random_seed in range(8):
        tf = time.time()

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        z = torch.randn(bs, 100).to(device).float()
        z.requires_grad = True
        v = torch.zeros(bs, 100).to(device).float()

        for i in range(iter_times):
            fake = G((blurred, z))
            label = D(fake)
            fake_in = rz(fake)
            out = T(fake_in)
            identity = E(fake)
            if z.grad is not None:
                z.grad.data.zero_()

            #mse = torch.mean((ground_truth - fake.cpu()) ** 2)
            #max_pixel_value = 1.0
            #psnr = 10 * torch.log10((max_pixel_value ** 2) / mse)

            Prior_Loss = -label.mean()  # delete -
            Gaze_Loss = criterion(out, gazes)
            Iden_Loss = criterion1(identity[1], ids)
            Total_Loss = 10 * Prior_Loss + lamda * Gaze_Loss + lamda * Iden_Loss
            # Total_Loss = Prior_Loss
            c += 1

            print(
                f"Steps: {c: d} Loss: {Total_Loss.item():.4f} Prior_Loss: {Prior_Loss.item():.4f} "
                f"Angle Error: {compute_angle_error(out, gazes).mean():.4f} Identity_Loss: {Iden_Loss.item():.4f}", end='\r')
            # print(f"Loss: {Total_Loss.item():.2f} Prior_Loss: {Prior_Loss.item():.2f} Gaze_Loss: {Gaze_Loss.item():.2f} Iden_Loss: {Iden_Loss.item():2f}", end='\r')

            Total_Loss.backward()

            v_prev = v.clone()
            gradient = z.grad.data
            v = momentum * v - lr * gradient
            z = z + (- momentum * v_prev + (1 + momentum) * v)
            z = torch.clamp(z.detach(), -clip_range, clip_range).float()
            z.requires_grad = True

            if (i + 1) % 300 == 0:
                if total_loss > Total_Loss.item():
                    total_loss = Total_Loss.item()
                    fake_img = G((blurred, z.detach()))
                    imgs = []
                    for j in range(1):  # depends on the number of attack targets
                        img = fake_img[j]
                        blurr_img = blurred[j]
                        img = torch.concat(
                            (img[0].unsqueeze(0).cpu(), blurr_img[0].unsqueeze(0).cpu(),
                             ground_truth[j][0].unsqueeze(0)))
                        img = img.unsqueeze(1)
                        imgs.append(img)

                    img = torch.concat(imgs)
                    tvls.save_image(img, f'./result/attack_MPIIGaze/img_{c}.png', nrow=9)


def main():
    a = [10, 50, 100, 500, 1000]
    #num = a[k]
    target_path = f"./target_models/with_pretraining/target_model_pretrain_{1}_{1000}.pth"
    g_path = "/projects/tang/GMI-Attack/GMI-Attack/Celeba/result/models_gan_lpw/lpw_G_auxiliary.tar"
    d_path = "/projects/tang/GMI-Attack/GMI-Attack/Celeba/result/models_gan_lpw/lpw_D_auxiliary.tar"
    e_path = "/projects/tang/GMI-Attack/GMI-Attack/Celeba/result/Identity_classifier_NVGaze.tar"
    os.makedirs(f'/projects/tang/GMI-Attack/GMI-Attack/Celeba/result/model_pretrain_{1}_{1000}/', exist_ok=True)

    log_path = "./attack_logs"
    os.makedirs(log_path, exist_ok=True)
    log_file = f"attack_log_NVGaze_with_pretrained_p{1}_{1000}.txt"
    utils.Tee(os.path.join(log_path, log_file), 'w')

    E = torch.load(e_path)
    freeze(E)
    E = nn.DataParallel(E).to(device)

    T = torch.load(target_path)
    T = nn.DataParallel(T).cuda()

    G = InversionNet()
    G = nn.DataParallel(G).cuda()
    ckp_G = torch.load(g_path)['state_dict']
    utils.load_my_state_dict(G, ckp_G)

    D = discri.DGWGAN(3, 32)
    D = nn.DataParallel(D).cuda()
    ckp_D = torch.load(d_path)['state_dict']
    utils.load_my_state_dict(D, ckp_D)

    config = get_default_config()
    config.merge_from_file('./configs/NVGaze.yaml')
    config.freeze()

    train_dataset, val_dataset = create_dataset(config, True, [5], True, True, False, 100)

    images = []
    blurred = []
    gazes = []
    ids = []
    for d in train_dataset:
        images.append(d[0])
        blurred.append(d[1])
        gazes.append(d[2])
        ids.append(d[3])

    images = torch.cat([image.unsqueeze(0) for image in images])
    blurred = torch.cat([image.unsqueeze(0) for image in blurred])
    gazes = torch.cat([gaze.unsqueeze(0) for gaze in gazes])
    ids = torch.from_numpy(np.array(ids))

    os.makedirs(f'./result/NVGaze_auxiliary/model_pretrain_{1}_{1000}', exist_ok= True)
    tvls.save_image(images, f'./result/NVGaze_auxiliary/model_pretrain_{1}_{1000}/target_img.png')

    inversion(G, D, T, E, gazes, ids, images, blurred)

    del gazes, ids, images, blurred


if __name__ == '__main__':
        main()


