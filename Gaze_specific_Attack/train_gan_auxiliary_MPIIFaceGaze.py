import utils
from utils import *
from torch.autograd import grad
import torchvision.transforms as transforms
from discri import DGWGAN
from generator import Generator, InversionNet
from gaze_estimation.utils import (AverageMeter, compute_angle_error,
                                   create_train_output_dir, load_config,
                                   save_config, set_seeds, setup_cudnn)
from gaze_estimation.config import get_default_config
from gaze_estimation.datasets import create_dataset
from torch.utils.data import DataLoader
from losses import noise_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = get_default_config()
config.merge_from_file('configs/mpiifacegaze/resnet_simple_14_train.yaml')
config.freeze()
train_dataset, val_dataset = create_dataset(config, True, [10, 11, 12, 13, 14], False, True)
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


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).to(device)
    z = x + alpha * (y - x)
    z = z.to(device)
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).to(device), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


save_img_dir = "result/imgs_celeba_gan_face_auxiliary"
save_model_dir = "result/models_celeba_gan_face_auxiliary"
os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_img_dir, exist_ok=True)



log_path = "./attack_face_auxiliary_logs"
os.makedirs(log_path, exist_ok=True)
log_file = "GAN_face_auxiliary.txt"
utils.Tee(os.path.join(log_path, log_file), 'w')

if __name__ == "__main__":

    dataset_name = "celeba"
    file = "./" + dataset_name + ".json"
    args = load_params(json_file=file)

    file_path = args['dataset']['train_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    z_dim = args[model_name]['z_dim']
    epochs = args[model_name]['epochs']
    n_critic = args[model_name]['n_critic']
    rz = transforms.Resize((224, 224))
    print("---------------------Training [%s]------------------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name])

    target_path = "/projects/tang/GMI-Attack/GMI-Attack/Celeba/experiments/mpiifacegaze/resnet_simple_14/exp00" \
                  "/facegaze_estimator.pth"

    set_seeds(config.train.seed)

    T = torch.load(target_path)
    Net = InversionNet()  # encoder of generator
    DG = DGWGAN(3, 32)  # decoder of generator

    T = nn.DataParallel(T).to(device)
    Net = torch.nn.DataParallel(Net).to(device)
    DG = torch.nn.DataParallel(DG).to(device)

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(Net.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0

    for epoch in range(epochs):
        start = time.time()
        for i, (images, blurred, poses, gazes) in enumerate(train_loader):
            step += 1
            # imgs = images.to(device)
            imgs = images.to(device)
            blurred = blurred.to(device)
            bs = imgs.size(0)

            freeze(Net)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).to(device)
            f_imgs = Net((blurred, z))

            r_logit = DG(imgs)
            f_logit = DG(f_imgs)

            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = gradient_penalty(imgs.data, f_imgs.data)
            dg_loss = - wd + gp * 10.0

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            if step % n_critic == 0:
                # train G
                freeze(DG)
                unfreeze(Net)
                z1 = torch.randn(bs, z_dim).to(device)
                output1 = Net((blurred, z1))

                z2 = torch.randn(bs, z_dim).to(device)
                output2 = Net((blurred, z2))

                logit_dg = DG(output1)
                diff_loss = noise_loss(T, rz(output1), rz(output2))  # L1 loss

                # calculate g_loss
                g_loss = - logit_dg.mean() - diff_loss * 0.5
                print(f'{i}/{len(train_loader)} dg_loss: {dg_loss} g_loss: {g_loss}', end='\r')

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start
        print("Epoch:%d \t Time:%.2f" % (epoch, interval))
        if (epoch + 1) % 1 == 0:

            save_size = 8
            z = torch.randn(save_size, z_dim).to(device)
            index = torch.randint(0, train_dataset.__len__() - 1, (save_size,))
            actual = torch.zeros(save_size, 3, 64, 64)
            blurred_img = torch.zeros(save_size, 3, 64, 64)
            images = torch.zeros(save_size * 3, 3, 64, 64)
            i = 0
            for x in index:
                img, blur_img, tmp, tmp2 = train_dataset.__getitem__(x)
                actual[i] = img
                blurred_img[i] = blur_img
                i += 1
            fake_images = Net((blurred_img, z))
            for j in range(save_size):
                images[j] = actual[j]
                images[1 * save_size + j] = blurred_img[j]
                images[2 * save_size + j] = fake_images[j]
            save_tensor_images(images.detach(), os.path.join(save_img_dir, "result_auxiliary_{}.png".format(epoch)),
                               nrow=save_size)

        torch.save({'state_dict': Net.state_dict()}, os.path.join(save_model_dir, "MPIIFaceGaze_G_auxiliary.tar"))
        torch.save({'state_dict': DG.state_dict()}, os.path.join(save_model_dir, "MPIIFaceGaze_D_auxiliary.tar"))
