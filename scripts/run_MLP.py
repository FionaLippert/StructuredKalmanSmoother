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
from stdgmrf.models.baselines import *
import constants_dgmrf as constants
import utils_dgmrf as utils
from stdgmrf.datasets.dummy_dataset import DummyDataset
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


@hydra.main(config_path="conf", config_name="config_MLP")
def run_MLP(config: DictConfig):

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
    wandb_name = f"MLP-{config['dataset']}-{time.strftime('%H-%M')}"
    run = wandb.init(project=config['experiment'], config=config, name=wandb_name)

    print('load data')

    dataset_dict = utils.load_dataset(config["dataset"], config["data_dir"], device=device)

    data = dataset_dict["data"].to(torch.float32)
    masks = dataset_dict["masks"] # shape [T, num_nodes]
    features = dataset_dict["covariates"].to(torch.float32) # shape [T * num_nodes, num_features]
    features = features - features.mean(0)
    features = features / features.std(0)
    print(f'use {features.size(1)} features')
    joint_mask = masks.reshape(-1)

    if not config.get('final', False):
        # exclude all test data for training and validation runs
        trainval_mask = torch.logical_not(dataset_dict["test_masks"].reshape(-1))
        data = data[trainval_mask[joint_mask]]
        joint_mask = torch.logical_and(joint_mask, trainval_mask)


    targets = torch.zeros(joint_mask.size())
    targets[joint_mask] = data

    model = MLP(config, targets, features)

    earlystopping_callback = EarlyStopping(monitor="val_mse", mode="min",
                                           patience=config["early_stopping_patience"])

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        log_every_n_steps=1,
        logger=wandb_logger,
        deterministic=True,
        max_epochs=int(config["n_iterations"] / config["val_interval"]),
        accelerator='gpu',
        callbacks=[earlystopping_callback]
    )

    if 'wandb_run' in config:
        # load pre-trained model
        model_path = get_model(config.get('wandb_run'))
        model.load_state_dict(torch.load(model_path))
    else:
        # train model
        ds_train = DummyDataset(dataset_dict['train_masks'].reshape(-1), config["val_interval"])
        dl_train = DataLoader(ds_train, batch_size=1, shuffle=False)
        ds_val = DummyDataset(dataset_dict['val_masks'].reshape(-1), 1)
        dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)
        trainer.fit(model, dl_train, dl_val)

    if config.get('final', False):
        ds_test = DummyDataset(dataset_dict['test_masks'].reshape(-1), 1)
        dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)
        trainer.test(model, dl_test)
        results = trainer.predict(model, dl_test, return_predictions=True)
    else:
        ds_val = DummyDataset(dataset_dict['val_masks'].reshape(-1), 1)
        dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)
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

    # save model
    model_path = os.path.join(ckpt_dir, config['experiment'], run.id, 'models')
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))

    # save as artifact for version control
    artifact = wandb.Artifact(f'model-{run.id}', type='models')
    artifact.add_dir(model_path)
    run.log_artifact(artifact)

    print(f'val nodes = {dataset_dict["val_masks"].sum(0).nonzero().flatten()}')


if __name__ == "__main__":
    run_MLP()
