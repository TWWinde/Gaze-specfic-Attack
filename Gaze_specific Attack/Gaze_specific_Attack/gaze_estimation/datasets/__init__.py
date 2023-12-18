import os
import pathlib
from typing import List, Union

import numpy as np
import torch
import yacs.config
from torch.utils.data import Dataset

from ..transforms import create_transform
from ..types import GazeEstimationMethod

gazecapture_path = './data/gazecapture'
name_list = os.listdir(gazecapture_path)
name_list = np.array(name_list, object)
name_list = name_list[[os.path.isdir(os.path.join(gazecapture_path, r)) for r in name_list]]
name_list.sort()


def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool = True, public: List = list(range(15)), face: bool = False,
                   auxiliary: bool = False, train_gan: bool = False, length: int = 3000) -> Union[List[Dataset], Dataset]:
    if config.mode == GazeEstimationMethod.MPIIGaze.name:
        from .mpiigaze import OnePersonDataset
    elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        from .mpiifacegaze import OnePersonDataset
    elif config.mode == GazeEstimationMethod.GazeCapture.name:
        from .gazecapture import OnePersonDataset
    elif config.mode == GazeEstimationMethod.LPW.name:
        from .lpw import OnePersonDataset
    elif config.mode == GazeEstimationMethod.NVGaze.name:
        from .nvgaze import OnePersonDataset
    else:
        raise ValueError

    dataset_dir = pathlib.Path(config.dataset.dataset_dir)
    assert dataset_dir.exists()
    assert config.train.test_id in range(-1, 15)
    # assert config.test.test_id in range(15)
    person_ids = [f'p{index:02}' for index in public]
    if config.mode == GazeEstimationMethod.GazeCapture.name:
        if train_gan:
            person_ids = name_list

        else:
            person_ids = name_list[10:]
    if config.mode == GazeEstimationMethod.LPW.name:
        if train_gan:
            person_ids = [f'p{index:02}' for index in range(1,23)]
        else:
            person_ids = name_list[10:]

    transform = create_transform(config)

    if is_train:
        if config.train.test_id == -1:
            if config.mode == GazeEstimationMethod.MPIIFaceGaze.name or config.mode == GazeEstimationMethod.NVGaze.name:
                train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(person_id, dataset_dir, transform, face, auxiliary, length)
                    for person_id in person_ids
                ])

            else:
                train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(person_id, dataset_dir, transform, face, auxiliary)
                    for person_id in person_ids
                ])
            # assert len(train_dataset) == 45000
        else:
            test_person_id = person_ids[config.train.test_id]
            train_dataset = torch.utils.data.ConcatDataset([
                OnePersonDataset(person_id, dataset_dir, transform)
                for person_id in person_ids if person_id != test_person_id
            ])

        val_ratio = config.train.val_ratio
        assert val_ratio < 1
        val_num = int(len(train_dataset) * val_ratio)
        train_num = len(train_dataset) - val_num
        lengths = [train_num, val_num]
        return torch.utils.data.dataset.random_split(train_dataset, lengths)
    else:
        train_dataset = torch.utils.data.ConcatDataset([
            OnePersonDataset(person_id, dataset_dir, transform, face, auxiliary, 1)
            for person_id in person_ids])
        OnePersonDataset('p01', dataset_dir, transform, face, auxiliary, 1)
        return torch.utils.data.dataset.random_split(train_dataset, [len(person_ids), 0])

        for p in person_ids:
            d = OnePersonDataset(p, dataset_dir, transform, True, 10)
            images.append(d[0])
            poses.append(d[1])
            gazes.append(d[2])
            ids.append(int(p[1:]))

        images = torch.cat([image.unsqueeze(0) for image in images])
        gazes = torch.cat([gaze.unsqueeze(0) for gaze in gazes])
        poses = torch.cat([pose.unsqueeze(0) for pose in poses])

        return images, poses, gazes, ids
