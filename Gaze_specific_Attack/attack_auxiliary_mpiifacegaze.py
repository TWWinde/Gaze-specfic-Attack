import torch, os, time, random, generator, discri, classify, utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
import torchvision.transforms as transforms
from gaze_estimation.config import get_default_config
from gaze_estimation.datasets import create_dataset
from gaze_estimation.utils import (AverageMeter, compute_angle_error)
from generator import InversionNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_path = "./attack_logs"
os.makedirs(log_path, exist_ok=True)
log_file = "attack_gazecapture_auxiliary.txt"
utils.Tee(os.path.join(log_path, log_file), 'w')
os.makedirs('./result/gazecapture_auxiliary/', exist_ok=True)


def inversion(G, D, T, gazes, ids, ground_truth, blurred, lr=1e-1, momentum=0.9, lamda=100, iter_times=15000, clip_range=1):
	gazes = gazes.to(device)
	blurred = blurred.to(device)
	ids = ids.long().to(device)
	criterion = nn.L1Loss(reduction='mean').to(device)
	bs = gazes.shape[0]
	c=0
	
	iden = torch.zeros(10)
	for i in range(10):
		iden[i] = i
	iden = iden.view(-1).long().to(device)

	G.eval()
	D.eval()
	T.eval()

	rz = transforms.Resize((224,224))

	for random_seed in range(5):
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

			if z.grad is not None:
				z.grad.data.zero_()

			Prior_Loss =  label.mean()  # delete -
			Gaze_Loss = criterion(out, gazes)
			Total_Loss = Prior_Loss +  lamda * Gaze_Loss
			# Total_Loss = Prior_Loss +  lamda*Iden_Loss
			c +=1

			print(f"Steps: {c: d} Loss: {Total_Loss.item():.2f} Prior_Loss: {Prior_Loss.item():.2f} Angle Error: {compute_angle_error(out, gazes).mean():.2f}", end='\r')

			Total_Loss.backward()
			
			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -clip_range, clip_range).float()
			z.requires_grad = True

			if (i+1) % 300 == 0:
				fake_img = G((blurred, z.detach()))
				imgs = []
				for j in range(10):
					img = fake_img[j]
					blurr_img = blurred[j]
					img = torch.concat((img[0].unsqueeze(0).cpu(),blurr_img[0].unsqueeze(0).cpu(), ground_truth[j][0].unsqueeze(0)))
					img = img.unsqueeze(1)
					imgs.append(img)

				img = torch.concat(imgs)
				tvls.save_image(img, f'./result/gazecapture_auxiliary/img_{c}.png', nrow=9)

if __name__ == '__main__':
	target_path = "./experiments/mpiifacegaze/resnet_simple_14/exp00" \
				  "/facegaze_estimator.pth"
	g_path = "./result/models_celeba_gan_face_auxiliary/gazecapture_G_auxiliary.tar"
	d_path = "./result/models_celeba_gan_face_auxiliary/gazecapture_D_auxiliary.tar"
	
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
	config.merge_from_file('configs/mpiifacegaze/resnet_simple_14_train.yaml')
	config.freeze()

	train_dataset, val_dataset = create_dataset(config, False, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], True, True)

	images = []
	blurred = []
	gazes = []
	ids = []
	for d in train_dataset:
		images.append(d[0])
		blurred.append(d[1])
		gazes.append(d[3])
		ids.append(d[4])

	images = torch.cat([image.unsqueeze(0) for image in images])
	blurred = torch.cat([image.unsqueeze(0) for image in blurred])
	gazes = torch.cat([gaze.unsqueeze(0) for gaze in gazes])
	ids = torch.from_numpy(np.array(ids))

	os.makedirs('./result/', exist_ok=True)
	tvls.save_image(images, f'./result/target_img.png')
	inversion(G, D, T, gazes, ids, images, blurred)