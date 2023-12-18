import utils
from utils import *
from torch.autograd import grad
from discri import DGWGAN
from generator import Generator
from gaze_estimation.config import get_default_config 
from gaze_estimation.datasets import create_dataset
from torch.utils.data import DataLoader

'''
train the attack model(GAN) using data from LPW Dataset in order to attack data from NVGaze Dataset.
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = get_default_config()
config.merge_from_file('./configs/lpw.yaml')
config.freeze()
train_dataset, val_dataset = create_dataset(config, True, [i for i in range(15)], False, True, True)
# public does not matter here
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
    g = grad(o, z, grad_outputs = torch.ones(o.size()).to(device), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

save_img_dir = "result/lpw_without_aux"
save_model_dir= "result/lpw_without_aux"
os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_img_dir, exist_ok=True)

dataset_name = "celeba"

log_path = "./attack_logs"
os.makedirs(log_path, exist_ok=True)
log_file = "GAN_lpw_no_aux.txt"
utils.Tee(os.path.join(log_path, log_file), 'w')

if __name__ == "__main__":
    
    file = "./" + dataset_name + ".json"
    args = load_params(json_file=file)

    file_path = args['dataset']['train_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    z_dim = args[model_name]['z_dim']
    epochs = args[model_name]['epochs']
    n_critic = args[model_name]['n_critic']

    print("---------------------Training [%s]------------------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name])

    # dataset, dataloader = init_gaze_face_data('./data/MPIIFaceGaze_normalized', [1,4,5,7,13])

    G = Generator(z_dim, 32)   # z_dim =100
    DG = DGWGAN(3, 32)  # in_dim=3, dim=64
    
    G = torch.nn.DataParallel(G).to(device)
    DG = torch.nn.DataParallel(DG).to(device)

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0
    for epoch in range(epochs):
        start = time.time()
        for i, (images,labels,files) in enumerate(train_loader):
            step += 1
            imgs = images.to(device)
            bs = imgs.size(0)
            
            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).to(device)
            f_imgs = G(z)

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
                unfreeze(G)
                z = torch.randn(bs, z_dim).to(device)
                f_imgs = G(z)
                logit_dg = DG(f_imgs)
                # calculate g_loss
                g_loss = - logit_dg.mean()
                print(f'{i}/{len(train_loader)} dg_loss: {dg_loss} g_loss: {g_loss}', end='\r')
                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start
        print("Epoch:%d \t Time:%.2f" % (epoch, interval))
        if (epoch+1) % 1 == 0:
            z = torch.randn(32, z_dim).to(device)
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "result_image1_{}.png".format(epoch)), nrow = 8)
        
        torch.save({'state_dict':G.state_dict()}, os.path.join(save_model_dir, "lpw_no_aux_G.tar"))
        torch.save({'state_dict':DG.state_dict()}, os.path.join(save_model_dir, "lpw_no_aux_D.tar"))

