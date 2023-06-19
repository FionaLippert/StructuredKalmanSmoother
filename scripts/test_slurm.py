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


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def run_dgmrf():

    print(f'hydra working dir: {os.getcwd()}')

    print('setup wandb')

    # run wandb offline and sync later
    os.environ["WANDB_MODE"] = "offline"

    # Init wandb
    wandb_name = f"test"
    run = wandb.init(project='test_slurm', name=wandb_name)

if __name__ == "__main__":
    run_dgmrf()
