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
        self.len = len

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.dataset_path, 'r') as f:
            image = f.get(f'{self.person_id_str}/image/{index:04}')[()]
            pupil_loc = f.get(f'{self.person_id_str}/pupil_loc/{index:04}')[()]
        image = self.transform(image)
        blurred = transform.functional.gaussian_blur(image, 13)
        pupil_loc = torch.from_numpy(pupil_loc)

        if self.face:
            if self.auxiliary:
                return image, blurred, pupil_loc,  int(self.person_id_str[1:])
            else:
                return image, pupil_loc, int(self.person_id_str[1:])
        else:
            if self.auxiliary:
                return image, blurred, pupil_loc
            else:
                return image, pupil_loc

    def __len__(self) -> int:
        return self.len