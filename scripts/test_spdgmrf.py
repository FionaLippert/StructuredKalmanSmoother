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
def run_dgmrf(config: DictConfig):

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
    dl_test = DataLoader(ds_val, batch_size=1, shuffle=False)

    wandb_logger = WandbLogger(log_model='all')
    # log_predictions_callback = LogPredictionsCallback(wandb_logger, t=1)

    if ("true_posterior_mean" in dataset_dict) and ("gt" in dataset_dict):

        true_mean = dataset_dict["true_posterior_mean"].squeeze()
        true_std = dataset_dict["true_posterior_std"].squeeze()
        test_mask = torch.logical_not(joint_mask)

        gt = dataset_dict["gt"]
        residuals = (gt - true_mean)
        print(residuals.max(), residuals.min())
        print(residuals[test_mask].max(), residuals[test_mask].min())

        wandb.run.summary["test_mae_optimal"] = residuals[test_mask].abs().mean().item()
        wandb.run.summary["test_rmse_optimal"] = torch.pow(residuals[test_mask], 2).mean().sqrt().item()
        wandb.run.summary["test_mape_optimal"] = (residuals[test_mask] / gt[test_mask]).abs().mean().item()

        inference_callback = LatticeInferenceCallback(wandb_logger, config, dataset_dict['grid_size'],
                                                      true_mean, true_std, residuals)
    else:
        tidx = 0

        val_nodes = dataset_dict['val_nodes'].cpu()
        # ridx = torch.randperm(len(val_nodes))[:4]
        # val_nodes = val_nodes[5:10].cpu()
        train_nodes = dataset_dict['train_nodes'].cpu()
        print(train_nodes)
        test_nodes = dataset_dict['test_nodes'].cpu()
        # ridx = torch.randperm(len(train_nodes))[:4]
        # train_nodes = train_nodes[5:10].cpu()
        # val_nodes = [892, 893, 785, 784]
        # train_nodes = [891, 960, 789, 770]

        # zoom in to crossing
        lat_max, lat_min, lon_max, lon_min = (37.330741, 37.315718, -121.883005, -121.903327)
        # lat_max, lat_min, lon_max, lon_min = (37.345741, 37.300718, -121.833005, -121.953327)
        node_mask = (temporal_graph.lat < lat_max) * (temporal_graph.lat > lat_min) * \
                    (temporal_graph.lon < lon_max) * (temporal_graph.lon > lon_min)
        subset = node_mask.nonzero().squeeze()
        mark_subset = torch.tensor([idx for idx, i in enumerate(subset) if (i in dataset_dict['train_nodes']
                                                                         or i in dataset_dict['val_nodes'])])
        print(f'subset size = {len(subset)}')
        print(f'test nodes = {test_nodes}')
        print(f'train nodes = {train_nodes}')

        print(temporal_graph)

        inference_callback = GraphInferenceCallback(wandb_logger, config, temporal_graph, tidx,
                                                    val_nodes, train_nodes, test_nodes, subset=subset, mark_subset=mark_subset)

    earlystopping_callback = EarlyStopping(monitor="val_rec_loss", mode="min", patience=config["early_stopping_patience"])


    trainer = pl.Trainer(
        max_epochs=int(config["n_iterations"] / config["val_interval"]),
        log_every_n_steps=1,
        logger=wandb_logger,
        deterministic=True,
        accelerator='gpu',
        devices=1,
        callbacks=[inference_callback, earlystopping_callback],
    )

    def print_params(model, config, header=None):
        if header:
            print(header)

        print("Aggregation weights:")
        for layer_i, layer in enumerate(model.layers):
            print("Layer {}".format(layer_i))
            if hasattr(layer, "activation_weight"):
                print("non-linear weight: {:.4}".format(layer.activation_weight.item()))
            else:
                print("self: {:.4}, neighbor: {:.4}".format(
                    layer.self_weight[0].item(), layer.neighbor_weight[0].item()))

            if hasattr(layer, "degree_power"):
                print("degree power: {:.4}".format(layer.degree_power[0].item()))

    trainer.fit(model, dl_train, dl_val)
    trainer.test(model, dl_test)
    results = trainer.predict(model, dl_test, return_predictions=True)

    for param_name, param_value in model.dgmrf.state_dict().items():
        print("{}: {}".format(param_name, param_value))
    if hasattr(model.dgmrf, 'dgmrf'):
        print_params(model.dgmrf.dgmrf, config)
    else:
        print_params(model.dgmrf, config)
    print(f'noise var = {model.noise_var}')



    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # save results
    result_path = os.path.join(ckpt_dir, config['experiment'], run.id, 'results')
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # save as artifact for version control
    artifact = wandb.Artifact(f'results-{run.id}', type='results')
    artifact.add_dir(result_path)
    run.log_artifact(artifact)


if __name__ == "__main__":
    run_dgmrf()