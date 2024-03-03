# DualView: Gaze-specific Attack
This repo contains code for the  DualView: Gaze-specific Attack part of our work ["PrivatEyes: Appearance-based Gaze Estimation Using Federated Secure
Multi-Party Computation"](https://arxiv.org/abs/2402.18970), which has been accepted by [ETRA](https://etra.acm.org/index.html)



## Overview
Gaze estimation models are susceptible to a large number of vulnerabilities and attacks, such as
reconstruction and inference attacks. Nonetheless, such attacks have never been studied in the gaze community. In
this work, we present a new gaze-specific attack (DualView) to evaluate the amount of information leakage from gaze
estimation models and/or their training process. We further apply them to our FL training (PrivatEyes).

Attack goal: DualView aims to infer the user’s appearance (i.e. view1: how the user looks like) and the respective gaze
distribution (i.e. view2: where the user is looking) with an end-to-end attack model. The key motivation for studying this
attack is to empirically quantify the amount of information deducible from the training process.

![img.png](/img.png)


## Setup
First, clone this repository:
```
git clone [https://git.hcics.simtech.uni-stuttgart.de/tang/GMI-Attack.git](https://github.com/TWWinde/Gaze-specfic-Attack.git)
cd /GMI-Attack/Gaze_specific_Attack
```

The code is tested for Python 3.8 and the packages listed in [requirements.txt](oasis.yml).
The basic requirements are PyTorch and Torchvision.
The easiest way to get going is to setup a conda environment via
```
conda create -n myenv python=3.8 
conda activate myenv
conda install -c anaconda cudatoolkit=10.1
conda install -c conda-forge cudatoolkit-dev=10.1
conda install -c nvidia cudnn=7.6.5
pip install -r requirements.txt
```

## Dataset
We evaluate our method on the 
[MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild), 
[MPIIFaceGaze](https://www.perceptualui.org/research/datasets/MPIIFaceGaze/), 
[GazeCapture](https://gazecapture.csail.mit.edu), 
[NVGaze](https://sites.google.com/nvidia.com/nvgaze), 
[LPW](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/labelled-pupils-in-the-wild-lpw), 
and [Celeba](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets. They are all public datasets, you can download them in corresponding links. 
Our attack models(threat model) are trained on GazeCapture, Celeba and LPW datasets, then we implement them to attack data from MPIIGaze, MPIIFaceGaze and NVGaze datasets.

## Input pipeline
1. Download datasets into `/data` folder. 
2. Transfer datasets into `.h5` files using the scripts in /dataset_preprocess. For example,
run the script `python /dataset_preprocess/preprocess_MPIIFaceGaze.py` to generate the needed MPIIFaceGaze.h5 file.

       cd /dataset_preprocess
   
       python preprocess_MPIIFaceGaze.py
    the generated .h5 files will be located in `/result` folder
3. The dataset class files are located in `/gaze_estimation/datasets`.
4. Using `create_dataset` function to get train and test datasets.
## Usage
### 1. Train target model(Gaze estimator)
 Execute `python train_gaze_estimator.py` to train target model, specify the needed dataset at the beginning. Here we take a very simple gaze estimation model as an example. You can also use your own model and put the checkpoint into `/target_models`

### 2. Train attack model(GAN)
####
1) GAN with auxiliary(blurred private image)

   The inputs of the model are random noise and auxiliary information(blured images). execute `train_gan_auxiliary_xxx.py`: Train the attack models including generator and discriminator networks with public data corresponding to the first step Publice knowledge distillation.
using GazeCapture, LPW, Celeba datasets.

####
 2) GAN without auxiliary 

    The input of the model is only random noise. Run `train_gan_without_auxiliary_xxx.py`: Train the attack models including generator and discriminator networks with public data corresponding to the first step Publice knowledge distillation.
using GazeCapture, LPW, Celeba datasets.
### 3. Train evaluation classifier
 Execute `train_vgg_identity_classifer_xxx.py`, which is a classification model, in oder to evaluate the attack results

### 4. Implement attack 
 Execute `attack_auxiliary_xxx` or `attack_without_auxiliary_xxx` to implement attacks w/o auxiliary. The corresponding results will be saved into `/result` file

## Citation
If you use this work please cite
```
@misc{elfares2024privateyes,
      title={PrivatEyes: Appearance-based Gaze Estimation Using Federated Secure Multi-Party Computation}, 
      author={Mayar Elfares and Pascal Reisert and Zhiming Hu and Wenwu Tang and Ralf Küsters and Andreas Bulling},
      year={2024},
      eprint={2402.18970},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}  
```

## Contact
Please feel free to open an issue or contact us personally, if you have any questions or need any help.

Mayar.Elfares@vis.uni-stuttgart.de

twwinde@gmail.com





