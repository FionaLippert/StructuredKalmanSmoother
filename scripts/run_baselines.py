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

    # if not config['device'] == "cpu" and torch.cuda.is_available():
    #     print('Use GPU')
    #     # Make all tensors created go to GPU
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
    #
    #     # For reproducability on GPU
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     device = 'cuda'
    # else:
    #     print('Use CPU.')
    #     device = 'cpu'

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

    model = NodeARIMA(config, data, joint_mask, T=T, gt=dataset_dict.get('gt', None))


    # dataloaders contain data masks defining which observations to use for training, validation, testing
    ds_train = DummyDataset(dataset_dict['train_idx'], config["val_interval"])
    ds_val = DummyDataset(dataset_dict['val_idx'], 1)
    ds_test = DummyDataset(dataset_dict['test_idx'], 1)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)

    wandb_logger = WandbLogger(log_model='all')
    # log_predictions_callback = LogPredictionsCallback(wandb_logger, t=1)

    # if ("true_posterior_mean" in dataset_dict) and ("gt" in dataset_dict):
    #
    #     true_mean = dataset_dict["true_posterior_mean"].squeeze()
    #     true_std = dataset_dict["true_posterior_std"].squeeze()
    #     test_mask = torch.logical_not(joint_mask)
    #
    #     gt = dataset_dict["gt"]
    #     residuals = (gt - true_mean)
    #     print(residuals.max(), residuals.min())
    #     print(residuals[test_mask].max(), residuals[test_mask].min())
    #
    #     wandb.run.summary["test_mae_optimal"] = residuals[test_mask].abs().mean().item()
    #     wandb.run.summary["test_rmse_optimal"] = torch.pow(residuals[test_mask], 2).mean().sqrt().item()
    #     wandb.run.summary["test_mape_optimal"] = (residuals[test_mask] / gt[test_mask]).abs().mean().item()
    #
    #     inference_callback = LatticeInferenceCallback(wandb_logger, config, dataset_dict['grid_size'],
    #                                                   true_mean, true_std, residuals)
    # else:
    #     tidx = 0
    #
    #     val_nodes = dataset_dict['val_nodes'].cpu()
    #     train_nodes = dataset_dict['train_nodes'].cpu()
    #     test_nodes = dataset_dict['test_nodes'].cpu()
    #
    #     # zoom in to crossing
    #     lat_max, lat_min, lon_max, lon_min = (37.330741, 37.315718, -121.883005, -121.903327)
    #     # lat_max, lat_min, lon_max, lon_min = (37.345741, 37.300718, -121.833005, -121.953327)
    #     node_mask = (temporal_graph.lat < lat_max) * (temporal_graph.lat > lat_min) * \
    #                 (temporal_graph.lon < lon_max) * (temporal_graph.lon > lon_min)
    #     subset = node_mask.nonzero().squeeze()
    #     mark_subset = torch.tensor([idx for idx, i in enumerate(subset) if (i in dataset_dict['train_nodes']
    #                                                                      or i in dataset_dict['val_nodes'])])
    #     print(f'subset size = {len(subset)}')
    #     print(f'test nodes = {test_nodes}')
    #     print(f'train nodes = {train_nodes}')
    #
    #     print(temporal_graph)
    #
    #     inference_callback = GraphInferenceCallback(wandb_logger, config, temporal_graph, tidx,
    #                                                 val_nodes, train_nodes, test_nodes, subset=subset, mark_subset=mark_subset)

    # earlystopping_callback = EarlyStopping(monitor="val_rec_loss", mode="min", patience=config["early_stopping_patience"])


    trainer = pl.Trainer(
        log_every_n_steps=1,
        logger=wandb_logger,
        deterministic=True
    )

    trainer.test(model, dl_test)
    results = trainer.predict(model, dl_test, return_predictions=True)

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    result_path = os.path.join(ckpt_dir, config['experiment'], run.id, 'results')
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    artifact = wandb.Artifact(f'results-{run.id}', type='results')
    artifact.add_dir(result_path)
    run.log_artifact(artifact)


if __name__ == "__main__":
    run_baselines()
