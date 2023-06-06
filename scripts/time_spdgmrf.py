import torch
import torch_geometric as ptg
import numpy as np
import json
import os
import time
import argparse
import wandb
import copy
from timeit import default_timer as timer
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
import utils_dgmrf as utils
from structuredKS.datasets.dummy_dataset import DummyDataset
from callbacks import *

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="conf", config_name="config")
def time_dgmrf(config: DictConfig):

    print(f'hydra working dir: {os.getcwd()}')

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

    print('setup wandb')


    # Init wandb
    wandb_name = f"{config['transition_type']}-" \
                 f"indepT={config['independent_time']}-layers={config['n_layers']}-bias={config['use_bias']}-" \
                 f"{config['n_transitions']}trans-{config['dataset']}-{time.strftime('%H-%M')}"
    run = wandb.init(project=config['experiment'], config=config, name=wandb_name)

    print('load data')

    dataset_dict = utils.load_dataset(config["dataset"], config["data_dir"], device=device)
    spatial_graph = dataset_dict["spatial_graph"]
    temporal_graph = dataset_dict["temporal_graph"]

    data = dataset_dict["data"]
    masks = dataset_dict["masks"] # shape [T, num_nodes]
    joint_mask = masks.reshape(-1)

    M = data.numel()
    N = masks.numel()
    T = masks.size(0)

    print(f'initial guess = {data.mean()}')
    initial_guess = torch.ones(N) * data.mean()

    model = SpatiotemporalInference(config, initial_guess, data, joint_mask,
                                    spatial_graph.to_dict(), temporal_graph.to_dict(), T=T, gt=dataset_dict.get('gt', None),
                                    data_mean=dataset_dict.get('data_mean', 0), data_std=dataset_dict.get('data_std', 1))


    for param_name, param_value in model.dgmrf.state_dict().items():
        print("{}: {}".format(param_name, param_value))


    # dataloaders contain data masks defining which observations to use for training, validation, testing
    ds_train = DummyDataset(dataset_dict['train_idx'], config["val_interval"])
    ds_val = DummyDataset(dataset_dict['val_idx'], 1)
    ds_test = DummyDataset(dataset_dict['test_idx'], 1)
    dl_train = DataLoader(ds_train, batch_size=1, shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)

    wandb_logger = WandbLogger(log_model='all')

    wandb.run.summary["num_nodes"] = model.num_nodes

    trainer = pl.Trainer(
        max_epochs=int(config["n_iterations"] / config["val_interval"]),
        log_every_n_steps=1,
        logger=wandb_logger,
        deterministic=True,
        accelerator='gpu',
        devices=1,
    )

    # time variational training
    start = timer()
    trainer.fit(model, dl_train, dl_val)
    end = timer()
    train_time = (end - start) / config["n_iterations"]

    # time inference
    start = timer()
    trainer.test(model, dl_test)
    end = timer()

    if config.get('use_KS', False):
        wandb.run.summary["KS_inference_time"] = (end - start)
        wandb.run.summary["KS_train_iter_time"] = train_time
    else:
        wandb.run.summary["DGMRF_inference_time"] = (end - start)
        wandb.run.summary["DGMRF_train_iter_time"] = train_time


if __name__ == "__main__":
    time_dgmrf()