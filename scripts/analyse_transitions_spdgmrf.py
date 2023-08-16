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

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_model(run_path):
    api = wandb.Api()
    artifact = api.artifact(run_path, type='models')
    model_path = osp.join(artifact.download(), 'model.pt')

    return model_path

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="conf", config_name="config")
def analyse_dgmrf(config: DictConfig):

    print(f'hydra working dir: {os.getcwd()}')
    print(config)
    seed_all(config['seed'])

    if not config['device'] == "cpu" and torch.cuda.is_available():
        print('Use GPU')
        # Make all tensors created go to GPU
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        # For reproducability on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = 'cuda'
    else:
        print('Use CPU.')
        device = 'cpu'


    print('load data')

    dataset_dict = utils.load_dataset(config["dataset"], config["data_dir"], device=device)
    spatial_graph = dataset_dict["spatial_graph"]
    temporal_graph = dataset_dict["temporal_graph"]

    if config.get('use_features', False) or config.get('use_features_dynamics', False):
        features = dataset_dict["covariates"].to(torch.float32)
        features = features - features.mean(0)
        print('features std min', features.std(0).min())
        features = features / features.std(0)
        features = features[:, [0, 3, 4]]
        print(f'use {features.size(1)} features')
        print(features.min(), features.max())
    else:
        features = None

    data = dataset_dict["data"].to(torch.float32)
    masks = dataset_dict["masks"] # shape [T, num_nodes]
    joint_mask = masks.reshape(-1)

    N = masks.numel()
    T = masks.size(0)

    print(f'initial guess = {data.mean()}')
    initial_guess = torch.ones(N) * data.mean()

    model = SpatiotemporalInference(config, initial_guess, data, joint_mask,
                                    spatial_graph.to_dict(), temporal_graph.to_dict(),
                                    T=T, gt=dataset_dict.get('gt', None),
                                    data_mean=dataset_dict.get('data_mean', 0),
                                    data_std=dataset_dict.get('data_std', 1),
                                    features=features,
                                    true_post_mean=dataset_dict.get("true_posterior_mean", None),
                                    true_post_std=dataset_dict.get("true_posterior_std", None))


    if 'wandb_run' in config:
        # load pre-trained model
        model_path = get_model(config.get('wandb_run'))
        model.load_state_dict(torch.load(model_path))

        F = model.dgmrf.get_transition_matrix()
        print(F.size())

        dir = osp.join(config['output_dir'], 'transition_matrices', config['wandb_run'])
        os.makedirs(dir, exist_ok=True)
        torch.save(F, osp.join(dir, 'F.pt'))

        torch.save(spatial_graph.pos, osp.join(dir, 'pos.pt'))
        torch.save(dataset_dict['true_transition_matrix'], osp.join(dir, 'true_F.pt'))

    else:
        print('Please specify which wandb_run to load')

if __name__ == "__main__":
    analyse_dgmrf()
