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
import hydra
import pickle

# import visualization as vis
from structuredKS.models.dgmrf import *
import constants_dgmrf as constants
# import utils_dgmrf as utils
from structuredKS import utils
from structuredKS.datasets.dummy_dataset import DummyDataset
from callbacks import *


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="conf", config_name="config")
def analyse_stdgmrf(config: DictConfig):
    assert 'wandb_run' in config

    model_path, full_config = utils.get_wandb_model(config['wandb_run'], return_config=True)

    wandb_config = eval(full_config['config'].replace("'", '"'))
    wandb_config['data_dir'] = config.data_dir
    wandb_config['output_dir'] = config.output_dir
    wandb_config['wandb_run'] = config.wandb_run

    config = wandb_config
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
