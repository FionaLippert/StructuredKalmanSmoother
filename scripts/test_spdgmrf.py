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

def get_model(run_path):
    api = wandb.Api()
    artifact = api.artifact(run_path, type='models')
    model_path = osp.join(artifact.download(), 'model.pt')

    return model_path

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="conf", config_name="config")
def run_dgmrf(config: DictConfig):

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

    if config.get('use_features', False) or config.get('use_features_dynamics', False):
        features = dataset_dict["covariates"].to(torch.float32)
        features = features - features.mean(0)
        print('features std min', features.std(0).min())
        features = features / features.std(0)
        print(f'use {features.size(1)} features')
        print(features.min(), features.max())
    else:
        features = None

    data = dataset_dict["data"].to(torch.float32)
    masks = dataset_dict["masks"] # shape [T, num_nodes]
    joint_mask = masks.reshape(-1)

    val_nodes = dataset_dict["val_masks"].sum(0).nonzero().squeeze()
    # val_nodes = val_nodes[~torch.isin(val_nodes, test_nodes)]
    val_nodes = val_nodes[torch.randperm(val_nodes.numel())[:5]]

    if not config.get('final', False):
        # exclude all test data for training and validation runs
        trainval_mask = torch.logical_not(dataset_dict["test_masks"].reshape(-1))
        data = data[trainval_mask[joint_mask]]
        joint_mask = torch.logical_and(joint_mask, trainval_mask)

        # don't use test set yet
        test_nodes = val_nodes
    else:
        test_nodes = dataset_dict["test_masks"].sum(0).nonzero().squeeze()
        test_nodes = test_nodes[torch.randperm(test_nodes.numel())[:5]]

    M = data.numel()
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


    for param_name, param_value in model.dgmrf.state_dict().items():
        print("{}: {}".format(param_name, param_value))


    wandb_logger = WandbLogger() #log_model='all')
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
        tidx = T // 2

        # val_nodes = dataset_dict['val_nodes'].cpu()
        # ridx = torch.randperm(len(val_nodes))[:4]
        # val_nodes = val_nodes[5:10].cpu()
        # train_nodes = dataset_dict['train_nodes'].cpu()
        # print(train_nodes)
        # test_nodes = dataset_dict['test_nodes'].cpu()
        # ridx = torch.randperm(len(train_nodes))[:4]
        # train_nodes = train_nodes[5:10].cpu()
        # val_nodes = [892, 893, 785, 784]
        # train_nodes = [891, 960, 789, 770]

        # val_nodes = dataset_dict["val_masks"].sum(0).nonzero().squeeze()
        # train_nodes = dataset_dict["train_masks"].sum(0).nonzero().squeeze().cpu()
        # test_nodes = dataset_dict["test_masks"].sum(0).nonzero().squeeze()

        # zoom in to crossing
        # lat_max, lat_min, lon_max, lon_min = (37.330741, 37.315718, -121.883005, -121.903327)
        # # lat_max, lat_min, lon_max, lon_min = (37.345741, 37.300718, -121.833005, -121.953327)
        # node_mask = (temporal_graph.lat < lat_max) * (temporal_graph.lat > lat_min) * \
        #             (temporal_graph.lon < lon_max) * (temporal_graph.lon > lon_min)
        # subset = node_mask.nonzero().squeeze()
        # mark_subset = torch.tensor([idx for idx, i in enumerate(subset) if (i in dataset_dict['train_nodes']
        #                                                                  or i in dataset_dict['val_nodes'])])
        # print(f'subset size = {len(subset)}')
        # print(f'test nodes = {test_nodes}')
        # print(f'train nodes = {train_nodes}')

        # subset = torch.cat([val_nodes[torch.randperm(val_nodes.numel())[:5]],
        #                     test_nodes[torch.randperm(test_nodes.numel())[:5]]], dim=0)
        # mark_subset = torch.tensor([idx for idx, i in enumerate(subset) if i not in test_nodes])

        # val_nodes = val_nodes[~torch.isin(val_nodes, test_nodes)]
        # val_nodes = val_nodes[torch.randperm(val_nodes.numel())[:5]]
        # test_nodes = test_nodes[torch.randperm(test_nodes.numel())[:5]]

        inference_callback = GraphInferenceCallback(wandb_logger, config, temporal_graph, tidx,
                                        test_nodes.cpu(), val_nodes.cpu())#, subset=subset, mark_subset=mark_subset)

    earlystopping_callback = EarlyStopping(monitor="val_rec_loss", mode="min", patience=config["early_stopping_patience"])


    if config.get('use_KS', False):
        callbacks = []
    else:
        callbacks = [inference_callback, earlystopping_callback]

    trainer = pl.Trainer(
        max_epochs=int(config["n_iterations"] / config["val_interval"]),
        log_every_n_steps=1,
        logger=wandb_logger,
        deterministic=True,
        accelerator='gpu',
        devices=1,
        callbacks=callbacks,
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
        if config.get('save_prediction', False):
            results = trainer.predict(model, dl_test, return_predictions=True)
    else:
        ds_val = DummyDataset(dataset_dict['val_masks'].reshape(-1), 1)
        dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)
        trainer.test(model, dl_val)
        if config.get('save_prediction', False):
            results = trainer.predict(model, dl_val, return_predictions=True)

    for param_name, param_value in model.dgmrf.state_dict().items():
        print("{}: {}".format(param_name, param_value))
    # if hasattr(model.dgmrf, 'dgmrf'):
    #     print_params(model.dgmrf.dgmrf, config)
    # else:
    #     print_params(model.dgmrf, config)
    print(f'noise var = {model.noise_var}')


    ckpt_dir = osp.join(config['output_dir'], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # save results
    if config.get('save_prediction', False):
        result_path = osp.join(ckpt_dir, config['experiment'], run.id, 'results')
        os.makedirs(result_path, exist_ok=True)
        with open(osp.join(result_path, 'results.pickle'), 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        # save as artifact for version control
        artifact = wandb.Artifact(f'results-{run.id}', type='results')
        artifact.add_dir(result_path)
        run.log_artifact(artifact)

    # save model
    model_path = osp.join(ckpt_dir, config['experiment'], run.id, 'models')
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))

    # save as artifact for version control
    artifact = wandb.Artifact(f'model-{run.id}', type='models')
    artifact.add_dir(model_path)
    run.log_artifact(artifact)


if __name__ == "__main__":
    run_dgmrf()
