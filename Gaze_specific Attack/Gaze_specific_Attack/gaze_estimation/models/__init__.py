import importlib

import torch
import yacs.config


def create_model(config: yacs.config.CfgNode) -> torch.nn.Module:
    if config.mode == 'GazeCapture':
        dataset_name = 'mpiifacegaze'
    elif config.mode == 'NVGaze':
        dataset_name = 'mpiifacegaze'
    else:
        dataset_name = config.mode.lower()
    module = importlib.import_module(
        f'gaze_estimation.models.{dataset_name}.{config.model.name}')
    model = module.Model(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model
