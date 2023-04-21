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

print('import KS package stuff')

# import visualization as vis
from structuredKS.models.dgmrf import *
import constants_dgmrf as constants
import utils_dgmrf as utils
from structuredKS.datasets.dummy_dataset import DummyDataset
from callbacks import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

print('define config')

config = {"seed": 0,
          # "dataset": "spatiotemporal_20x20_obs=0.7_T=20_diff=0.01_adv=constant_ntrans=4_0",
          # "dataset": "spatiotemporal_20x20_obs=0.7_T=20_diff=0.0_adv=zero_ntrans=1_0",
          "dataset": "pems_start=0_end=215",
          "noise_std": 0.001,
          "n_layers": 1,
          "n_transitions": 1,
          "diff_K": 1,
          "non_linear": False,
          "fix_gamma": False,
          "log_det_method": "eigvals",
          "use_bias": True,
          "n_training_samples": 10,
          "n_post_samples": 100,
          "vi_layers": 1,
          "features": False,
          "optimizer": "adam",
          "lr": 0.1,
          "val_interval": 100,
          "n_iterations": 10000,
          "use_dynamics": True,
          "independent_time": False,
          "use_hierarchy": False,
          "transition_type": "diffusion",
          "inference_rtol": 1e-7,
          "device": "gpu",
          "early_stopping_patience": 100}

# experiment = 'constant_advection_07_dgmrf_varying_layers'
experiment = 'pems100-200_dgmrf_varying_layers'

print('set seed')

seed_all(config['seed'])

print('setup cuda')

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
wandb_name = f"{config['dataset']}-{config['transition_type']}-diffK={config['diff_K']}-" \
             f"indepT={config['independent_time']}-layers={config['n_layers']}-bias={config['use_bias']}-{time.strftime('%H-%M')}"
# wandb.init(project=constants.WANDB_PROJECT, config=config, name=wandb_name)
wandb.init(project=experiment, config=config, name=wandb_name)

print('load data')

dataset_dict = utils.load_dataset(config["dataset"], device=device)
# graphs = dataset_dict["spatiotemporal_graphs"]
spatial_graph = dataset_dict["spatial_graph"]
temporal_graph = dataset_dict["temporal_graph"]

data = dataset_dict["data"]
masks = dataset_dict["masks"] # shape [T, num_nodes]
joint_mask = masks.reshape(-1)

# M = graphs["data"].num_nodes
# N = graphs["latent"].num_nodes
# T = len(graphs)
M = data.numel()
N = masks.numel()
T = masks.size(0)

print(f'initial guess = {data.mean()}')
initial_guess = torch.ones(N) * data.mean()
# initial_guess = torch.ones(N).reshape(T, -1) * data.mean()
# model = SpatiotemporalInference(graphs, initial_guess, config)
# spatial_graph = ptg.data.Data(**graphs.get_example(0)["latent", "spatial", "latent"],
#                               pos=graphs.get_example(0)["latent"].pos)
# temporal_graph = ptg.data.Data(**graphs.get_example(1)["latent", "temporal", "latent"])


# if not config['independent_time'] and not config['use_dynamics']:
#     edges0 = graphs.get_example(0)["latent", "spatial", "latent"].edge_index.size(1)
#     edges1 = graphs.get_example(1)["latent", "spatial", "latent"].edge_index.size(1)
#     print(edges0, edges1)
#     spatial_graph = graphs["latent", "spatial", "latent"]
#     spatial_graph.pos = graphs["latent"].pos
#     temporal_graph = None
#     print(graphs["latent"].mask.sum(), graphs["data"].x.size())
# else:
#     spatial_graph = graphs.get_example(0)["latent", "spatial", "latent"]
#     spatial_graph.pos = graphs.get_example(0)["latent"].pos
#     temporal_graph = graphs.get_example(1)["latent", "temporal", "latent"]

model = SpatiotemporalInference(config, initial_guess, data, joint_mask,
                                spatial_graph.to_dict(), temporal_graph.to_dict(), T=T, gt=dataset_dict.get('gt', None),
                                data_mean=dataset_dict.get('data_mean', 0), data_std=dataset_dict.get('data_std', 1))
# samples = model.vi_dist.sample()
# print(samples.size())
# log_det = model.vi_dist.log_det()
# print(log_det)
# post_mean, post_std = model.vi_dist.posterior_estimate()
# print(post_mean.size(), post_std.size())
#
# ll = model.dgmrf(samples)
# print(ll.size())
# log_det = model.dgmrf.log_det()
# print(log_det)
#
# assert 0

# dataloaders contain data masks defining which observations to use for training, validation, testing
ds_train = DummyDataset(dataset_dict['train_idx'], config["val_interval"])
ds_val = DummyDataset(dataset_dict['val_idx'], 1)
# ds_train = DummyDataset(torch.arange(M), config["val_interval"])
# ds_val = DummyDataset(torch.arange(M), 1)
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
    tidx = 3
    # G_t = graphs.get_example(tidx)
    # G_t = ptg.data.Data(edge_index=G_t["latent", "spatial", "latent"].edge_index,
    #                     num_nodes=G_t["latent"].num_nodes,
    #                     mask=G_t["latent"].mask,
    #                     pos=G_t["latent"].pos)

    val_nodes = dataset_dict['val_nodes']
    ridx = torch.randperm(len(val_nodes))[:4]
    val_nodes = val_nodes[ridx].cpu()
    train_nodes = dataset_dict['train_nodes']
    ridx = torch.randperm(len(train_nodes))[:4]
    train_nodes = train_nodes[ridx].cpu()
    # val_nodes = [892, 893, 785, 784]
    # train_nodes = [891, 960, 789, 770]

    inference_callback = GraphInferenceCallback(wandb_logger, config, temporal_graph, tidx, val_nodes, train_nodes)

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

    print("Raw parameters:")
    for param_name, param_value in model.state_dict().items():
        print("{}: {}".format(param_name, param_value))

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
# print_params(model.dgmrf, config)
for param_name, param_value in model.dgmrf.state_dict().items():
    print("{}: {}".format(param_name, param_value))
trainer.test(model, dl_test)



# if hasattr(model.dgmrf, 'dynamics'):
#     if hasattr(model.dgmrf.dynamics.transition_model, 'diff_coeff'):
#         print(f'diff estimate = {model.dgmrf.dynamics.transition_model.diff_coeff}')
#     if hasattr(model.dgmrf.dynamics.transition_model, 'velocity'):
#         print(f'velocity estimate = {model.dgmrf.dynamics.transition_model.velocity}')


# dgmrf = ParallelDGMRF(graphs, config)
# opt_params = tuple(dgmrf.parameters())
#
# # Train using VI
# vi_dist = ParallelVI(graphs, initial_guess, config)
# opt_params += tuple(vi_dist.parameters())
#
# opt = utils.get_optimizer(config["optimizer"])(opt_params, lr=config["lr"])
# total_loss = torch.zeros(1)

# print('start iterations')
#
# for iteration_i in range(config["n_iterations"]):
#     opt.zero_grad()
#
#     print(f'iteration {iteration_i}')
#
#     # sample from q(x)
#     print('sample from q(x)')
#     samples = vi_dist.sample() # shape [T, n_samples, num_nodes]
#
#     print('compute g(x)')
#     # compute log-likelihood of samples given prior p(x)
#     Gx = dgmrf(samples) # shape (T, n_samples, n_nodes)
#     prior_ll = (-0.5 * torch.sum(torch.pow(Gx, 2)) + dgmrf.log_det()) / N
#
#
#     print('compute p(y given x)')
#     # compute data log-likelihood given samples
#     obs_model = ObservationModel()
#     print(f'sample size = {samples.size()}', T, N)
#     samples = samples.transpose(0, 1) # shape [n_samples, T, num_nodes]
#     samples = samples.reshape(config["n_training_samples"], -1) # shape [n_samples, T * num_nodes]
#     y_hat = obs_model(samples, graphs["latent", "observation", "data"])
#     print(f'y_hat size = {y_hat.size()}')
#
#     y = graphs["data"].x
#     noise_std = graphs.get_example(0)['data'].noise_std
#     log_noise_std = torch.log(noise_std)
#     data_ll = -0.5 * torch.sum(torch.pow((y - y_hat), 2)) / (M * noise_std**2) - log_noise_std
#
#     vi_entropy = 0.5 * vi_dist.log_det() / N
#
#     print('compute loss')
#     loss = - (prior_ll + vi_entropy + data_ll)
#
#
#     # Train
#     loss.backward()
#     opt.step()
#
#     total_loss += loss.detach()



# test dgmrf
# x = graphs["latent"].x
# Gx = dgmrf(graphs)
# prior_ll = (-0.5 * torch.sum(torch.pow(Gx, 2)) + dgmrf.log_det(len(graphs))) / N
#
# vi_entropy = 0
#
# obs_model = ObservationModel()
# y_hat = obs_model(x, graphs["latent", "observation", "data"])
# y = graphs["data"].x
#
# noise_std = graphs.get_example(0)['data'].noise_std
# log_noise_std = torch.log(noise_std)
# data_ll = -0.5 * torch.sum(torch.pow((y - y_hat), 2)) / (M * noise_std**2) - log_noise_std
#
# elbo = prior_ll + vi_entropy + data_ll
# print(prior_ll, data_ll)