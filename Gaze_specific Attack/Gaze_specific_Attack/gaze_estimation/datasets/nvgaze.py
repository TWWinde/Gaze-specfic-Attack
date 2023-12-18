import pathlib
from typing import Callable, Tuple
import h5py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transform
from torchvision.transforms import functional


class OnePersonDataset(Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path,
                 transform: Callable, face:bool = True, auxiliary:bool = False, len:int=3000):
        self.person_id_str = person_id_str
        self.dataset_path = dataset_path
        self.transform = transform
        self.face = face
        self.auxiliary = auxiliary
        self.len = len if self.get_len() > len else self.get_len()

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        blur = transform.GaussianBlur(9)
        with h5py.File(self.dataset_path, 'r') as f:

            image = f.get(f'{self.person_id_str}/image/{index:06}')[()]
            gaze = f.get(f'{self.person_id_str}/gaze/{index:06}')[()]
        image = self.transform(image)
        blurred = transform.functional.gaussian_blur(image, 13)
        gaze = torch.from_numpy(gaze)

        if self.face:
            if self.auxiliary:
                return image, blurred, gaze,  int(self.person_id_str[1:])
            else:
                return image, gaze, (int(self.person_id_str[1:])-1)
        else:
            if self.auxiliary:
                return image, blurred, gaze
            else:
                return image, gaze

    def get_len(self):
        with h5py.File(self.dataset_path, 'r') as f:
            image_dataset = f[f'{self.person_id_str}/image']
            dataset = list(image_dataset.keys())
            dataset_length = len(dataset)

        return dataset_length

    def __len__(self) -> int:
        return self.len
