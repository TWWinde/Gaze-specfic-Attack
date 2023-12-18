import json
import os
import pathlib
import sys
from typing import Callable, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transform
from torchvision.transforms import functional


def logError(msg, critical=False):
    print(msg)
    if critical:
        sys.exit(1)


def readJson(filename):
    if not os.path.isfile(filename):
        logError('Warning: No such file %s!' % filename)
        return None

    with open(filename) as f:
        try:
            data = json.load(f)
        except:
            data = None

    if data is None:
        logError('Warning: Could not read file %s!' % filename)
        return None

    return data
class OnePersonDataset(Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path,
                 transform: Callable, face:bool = False, auxiliary:bool=False, len:int=3000):

        self.person_id_str = person_id_str
        self.dataset_path = dataset_path
        self.person_path = os.path.join(self.dataset_path, self.person_id_str)
        self.transform = transform
        self.face = face
        self.auxiliary = auxiliary
        self.len = len
        self.jason_path = os.path.join('/datasets/external/gazecapture', self.person_id_str, 'dotInfo.json')
        self.dotInfo = readJson(self.jason_path)
        self.image_paths = sorted(os.listdir(self.person_path))


    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            # im = Image.new("RGB", self.imSize, "white")
        return im

    def __getitem__(self,index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        image = self.loadImage(os.path.join(self.person_path, self.image_paths[index]))
        image = np.array(image)
        # print(self.image_paths[index])
        if self.image_paths[index] != '' and self.image_paths[index] != '00000.jpg':
            number = int(self.image_paths[index].split('.')[0].lstrip('0'))
        else:
            number = 0
        #number = int(self.image_paths[index].lstrip('0').split('.')[0])
        gaze = np.array([self.dotInfo['XCam'][number], self.dotInfo['YCam'][number]], np.float32)
        image = self.transform(image)
        blurred = transform.functional.gaussian_blur(image, 13)
        gaze = torch.from_numpy(gaze)
        if self.face:
            if self.auxiliary:
                return image, blurred, gaze, int(self.person_id_str[1:])
            else:
                return image, gaze, int(self.person_id_str[1:])
        else:
            if self.auxiliary:
                return image, blurred, gaze
            else:
                return image, gaze

    def __len__(self) -> int:
        # 3000
        return len(self.image_paths)
