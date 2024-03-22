import torch
import torch_geometric as ptg
import numpy as np
import json
import os
import os.path as osp
import time
import argparse
import wandb
import copy
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
import hydra
import pickle

from stdgmrf.models.dgmrf import *
import constants_dgmrf as constants
from stdgmrf import utils
from stdgmrf.datasets.dummy_dataset import DummyDataset
from callbacks import *


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="conf", config_name="config")
def analyse_stdgmrf(config: DictConfig):
    assert 'wandb_run' in config

    # load config of pre-trained model
    model_path, full_config = utils.get_wandb_model(config['wandb_run'], return_config=True)

    # replace single quotes by double quotes
    wandb_config = eval(full_config['config'].replace("'", '"'))

    # update wandb config with overrides
    overrides = HydraConfig.get().overrides.task
    print(overrides)

    parser = OverridesParser.create()
    parsed_overrides = parser.parse_overrides(overrides=overrides)

    for override in parsed_overrides:
        key = override.key_or_group
        value = override.value()
        wandb_config[key] = value

    config = wandb_config
    config = OmegaConf.create(config)
    print(config)

    utils.seed_all(config['seed'])

    if not config['device'] == "cpu" and torch.cuda.is_available():
        print('Use GPU')
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = 'cuda'
    else:
        print('Use CPU.')
        device = 'cpu'

    # initialize model based on given config
    model = setup_model(config, device)

    # load pre-trained model
    model.load_state_dict(torch.load(model_path))

    model.eval()

    dir = osp.join(config['output_dir'], 'example_transformations', config['wandb_run'])
    os.makedirs(dir, exist_ok=True)

    # load posterior estimate
    results = utils.get_wandb_results(config['wandb_run'])
    post_mean = results['post_mean'].reshape(1, model.T, model.num_nodes).to(device)
    post_samples = results['post_samples'].reshape(-1, model.T, model.num_nodes).to(device)
    data = results['data'].reshape(-1, model.T, model.num_nodes).to(device)

    # apply temporal layers
    h_mean = model.dgmrf.apply_temporal(post_mean)
    h_samples = model.dgmrf.apply_temporal(post_samples)
    h_data = model.dgmrf.apply_temporal(data)

    # apply spatial layers
    z_mean = model.dgmrf.apply_spatial(h_mean)
    z_samples = model.dgmrf.apply_spatial(h_samples)
    z_data = model.dgmrf.apply_spatial(h_data)

    torch.save(post_mean, osp.join(dir, 'x_mean.pt'))
    torch.save(h_mean, osp.join(dir, 'h_mean.pt'))
    torch.save(z_mean, osp.join(dir, 'z_mean.pt'))

    torch.save(post_samples, osp.join(dir, 'x_samples.pt'))
    torch.save(h_samples, osp.join(dir, 'h_samples.pt'))
    torch.save(z_samples, osp.join(dir, 'z_samples.pt'))

    torch.save(data, osp.join(dir, 'data.pt'))
    torch.save(h_data, osp.join(dir, 'h_data.pt'))
    torch.save(z_data, osp.join(dir, 'z_data.pt'))


if __name__ == "__main__":
    analyse_stdgmrf()
