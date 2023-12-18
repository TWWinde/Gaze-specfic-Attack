import pathlib
from typing import Callable, Tuple
import h5py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transform
from torchvision.transforms import functional
'''
Dataset Class for each person of this dataset. There are in total 15 persons. We can control the person we use
for training in create_dataset function.
'''


class OnePersonDataset(Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path,
                 transform: Callable, face:bool = True, auxiliary:bool = False, len:int=3000):
        self.person_id_str = person_id_str
        self.dataset_path = dataset_path
        self.transform = transform
        self.face = face
        self.auxiliary = auxiliary
        self.len = len

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        blur = transform.GaussianBlur(9)
        with h5py.File(self.dataset_path, 'r') as f:
            image = f.get(f'{self.person_id_str}/image/{index:04}')[()]
            pose = f.get(f'{self.person_id_str}/pose/{index:04}')[()]
            gaze = f.get(f'{self.person_id_str}/gaze/{index:04}')[()]
        image = self.transform(image)
        blurred = transform.functional.gaussian_blur(image, 13)
        pose = torch.from_numpy(pose)
        gaze = torch.from_numpy(gaze)
        if self.face:
            if self.auxiliary:
                return image, blurred, pose, gaze, int(self.person_id_str[1:])
            else:
                return image, pose, gaze, int(self.person_id_str[1:])
        else:
            if self.auxiliary:
                return image, blurred, pose, gaze
            else:
                return image, pose, gaze

    def __len__(self) -> int:
        return self.len
