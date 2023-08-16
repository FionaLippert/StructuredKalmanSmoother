import torch
import torch_geometric as ptg
import numpy as np
import json
import os
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
from structuredKS.models.baselines import *
import constants_dgmrf as constants
import utils_dgmrf as utils
from structuredKS.datasets.dummy_dataset import DummyDataset
from callbacks import *

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="conf", config_name="config")
def run_baselines(config: DictConfig):

    print(f'hydra working dir: {os.getcwd()}')

    seed_all(config['seed'])

    print('setup wandb')

    # Init wandb
    wandb_name = f"ARIMA-{config['dataset']}-{time.strftime('%H-%M')}"
    run = wandb.init(project=config['experiment'], config=config, name=wandb_name)

    print('load data')

    dataset_dict = utils.load_dataset(config["dataset"], config["data_dir"], device='cpu')

    data = dataset_dict["data"]
    masks = dataset_dict["masks"] # shape [T, num_nodes]
    joint_mask = masks.reshape(-1)

    T = masks.size(0)

    model = NodeARIMA(config, data, joint_mask, dataset_dict['train_masks'].reshape(-1),
                      T=T, gt=dataset_dict.get('gt', None),
                      true_post_mean=dataset_dict.get("true_posterior_mean", None),
                      true_post_std=dataset_dict.get("true_posterior_std", None)
                      )


    # dataloaders contain data masks defining which observations to use for training, validation, testing
    ds_val = DummyDataset(dataset_dict['val_masks'].reshape(-1), 1)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)
    ds_test = DummyDataset(dataset_dict['test_masks'].reshape(-1), 1)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)

    wandb_logger = WandbLogger(log_model='all')

    trainer = pl.Trainer(
        log_every_n_steps=1,
        logger=wandb_logger,
        deterministic=True,

    )

    if config.get('final', False):
        # final model fitting and prediction
        model.final_prediction = True
        trainer.test(model, dl_test)
        results = trainer.predict(model, dl_test, return_predictions=True)

    else:
        # hyperparamter tuning or other exploration on validation set
        trainer.final_prediction = False
        trainer.test(model, dl_val)
        results = trainer.predict(model, dl_val, return_predictions=True)

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    result_path = os.path.join(ckpt_dir, config['experiment'], run.id, 'results')
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    artifact = wandb.Artifact(f'results-{run.id}', type='results')
    artifact.add_dir(result_path)
    run.log_artifact(artifact)

    print(f'val nodes = {dataset_dict["val_masks"].sum(0).nonzero().flatten()}')
    print(f'test nodes = {dataset_dict["test_masks"].sum(0).nonzero().flatten()}')


if __name__ == "__main__":
    run_baselines()
