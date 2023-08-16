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
from structuredKS.models.KS import AdvectionDiffusionTransition
import constants_dgmrf as constants
import structuredKS.utils as utils_stdgmrf
import utils_dgmrf as utils
from structuredKS.datasets.dummy_dataset import DummyDataset
from callbacks import *

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def diffusion_coeffs(adj, d):
    # assume dt=1, cell_size=1
    # otherwise: return adj * d * dt / cell_size
    return adj * d

def advection_coeffs(edge_velocities):
    # assume dt=1, face_length=1, cell_size=1
    return -0.5 * edge_velocities

def transition_matrix(coeffs):
    # add self-edges
    # coeffs has shape [num_nodes, num_nodes] or [batch_size, num_nodes, num_nodes]
    diag = torch.diag_embed(1 - coeffs.sum(-1))
    F = coeffs + diag

    return F

def transition_matrix_exponential(coeffs, k_max=1):
    # coeffs has shape [num_nodes, num_nodes] or [batch_size, num_nodes, num_nodes]

    F = torch.eye(coeffs.size(-2), dtype=coeffs.dtype)
    if coeffs.dim() == 3:
        F = F.unsqueeze(0).repeat(coeffs.size(0), 1, 1)
    A = coeffs - torch.diag_embed(coeffs.sum(-1))
    term_k = F
    z = 1
    for i in range(k_max):
        k = i+1
        z /= k
        term_k = z * A @ term_k
        F += term_k

    return F



@hydra.main(config_path="conf", config_name="config")
def run_baselines(config: DictConfig):

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
    wandb_name = f"KS-{config['dataset']}-{time.strftime('%H-%M')}"
    run = wandb.init(project=config['experiment'], config=config, name=wandb_name)

    print('load data')

    dataset_dict = utils.load_dataset(config["dataset"], config["data_dir"], device=device)

    data = dataset_dict["data"].to(torch.float32)
    masks = dataset_dict["masks"] # shape [T, num_nodes]
    joint_mask = masks.reshape(-1)

    T = masks.size(0)
    num_nodes = masks.size(1)
    print(f'data = {data}')


    n_transitions = 4
    v = torch.tensor([-0.3, 0.3])
    velocities = v.unsqueeze(1).repeat(1, num_nodes)
    diffusion = 0.01

    G_temp = dataset_dict['temporal_graph']
    G_spatial = dataset_dict['spatial_graph']

    normals = G_temp.edge_attr
    print(normals)
    normals = torch.stack([utils_stdgmrf.get_normal(G_temp.pos[u], G_temp.pos[v], max=30 - 1)
                           for u, v in G_temp.edge_index.T], dim=0)
    print(normals)

    G_temp.edge_attr = normals

    adj = ptg.utils.to_dense_adj(G_spatial.edge_index, max_num_nodes=num_nodes).squeeze(0)

    edge_velocities = (normals * v.unsqueeze(0)).sum(-1)  # [num_edges]
    edge_velocities = ptg.utils.to_dense_adj(G_temp.edge_index, edge_attr=edge_velocities,
                                             max_num_nodes=num_nodes).squeeze(0)  # [num_nodes, num_nodes]

    F_coeffs = torch.zeros(num_nodes, num_nodes)
    F_coeffs += diffusion_coeffs(adj, diffusion)
    F_coeffs += advection_coeffs(edge_velocities)

    F = transition_matrix_exponential(F_coeffs.to(torch.float64), k_max=2)
    F = torch.linalg.matrix_power(F, n_transitions)


    def transition_model(ensemble, transpose=False):
        # simple identity transition function
        # ensemble has shape [state_dim, ensemble_size]
        alpha = ensemble[-1, :] # get alpha parameter
        ensemble = torch.cat([alpha.unsqueeze(0) * ensemble[:-1, :], ensemble[-1, :].unsqueeze(0)], dim=0)
        return ensemble

    def true_transition(ensemble, transpose=False):
        # F = dataset_dict['true_transition_matrix'].to(torch.float64)

        if transpose:
            ensemble = F.transpose(0, 1) @ ensemble.to(torch.float64)
        else:
            ensemble = F @ ensemble.to(torch.float64)


        return ensemble

    advdiff = AdvectionDiffusionTransition(config, G_temp)

    def adv_diff_transition(ensemble, transpose=False):
        v = ensemble[-2:, :]
        #diffusion = torch.nn.functional.relu(ensemble[-3, :])
        diffusion = torch.clamp(torch.pow(ensemble[-3, :], 2), min=0, max=1)
        print(f'average v = {v.mean(-1)}, abs max = {v.abs().max(-1)}')
        print(f'average diffusion = {diffusion.mean()}, abs max = {diffusion.abs().max()}')

        ensemble = ensemble.to(torch.float64)


        for i in range(ensemble.size(-1)):
            # treat each ensemble member separately

            # edge_velocities = (normals * v[:, i].unsqueeze(0)).sum(-1)  # [num_edges]
            # edge_velocities = ptg.utils.to_dense_adj(G_temp.edge_index, edge_attr=edge_velocities,
            #                                          max_num_nodes=num_nodes).squeeze(0)  # [num_nodes, num_nodes]
            #
            # F_coeffs = torch.zeros(num_nodes, num_nodes)
            # F_coeffs += diffusion_coeffs(adj, diffusion[i])
            # F_coeffs += advection_coeffs(edge_velocities)
            #
            # F = transition_matrix_exponential(F_coeffs.to(torch.float64), k_max=1)
            # F = torch.linalg.matrix_power(F, n_transitions)
            #
            # # F = dataset_dict['true_transition_matrix'].to(torch.float64)
            #
            # if transpose:
            #     ensemble[:-3, i] = (F.transpose(0, 1) @ ensemble[:-3, i].unsqueeze(-1)).squeeze(-1)
            # else:
            #     ensemble[:-3, i] = (F @ ensemble[:-3, i].unsqueeze(-1)).squeeze(-1)

            for t in range(4):
                ensemble[:-3, i] = advdiff.forward(ensemble[:-3, i], velocity=v[:, i],
                                                   diffusion=diffusion[i], transpose=transpose)

        return ensemble

    #
    # model = EnKS(config, data, joint_mask, dataset_dict['train_masks'].reshape(-1), true_transition,
    #              config.get('ensemble_size', 100),
    #               T=T, gt=dataset_dict.get('gt', None),
    #               true_post_mean=dataset_dict.get("true_posterior_mean", None),
    #               true_post_std=dataset_dict.get("true_posterior_std", None)
    #               )

    velocity = 2 * torch.rand(2, ) - 1
    diff_param = torch.rand(1, ) * 0.5

    print(f'initial velocity estimate = {velocity}')
    print(f'initial diffusion estimate = {torch.pow(diff_param, 2)}')

    # velocity = torch.tensor([-0.3, 0.3])
    # diff_param = torch.tensor([0.1])

    model = EnKS(config, data, joint_mask, dataset_dict['train_masks'].reshape(-1), adv_diff_transition,
                 config.get('ensemble_size', 100),
                 initial_params=torch.cat([diff_param, velocity]),
                 initial_std_params=torch.tensor([0.01, 0.01, 0.01]),
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
        # deterministic=True
    )

    if config.get('final', False):
        # final model fitting and prediction
        model.final_prediction = True
        trainer.test(model, dl_test) # run EM and evaluation on held out test set
        if config.get('save_prediction', False):
            results = trainer.predict(model, dl_test, return_predictions=True)

    else:
        # hyperparamter tuning or other exploration on validation set
        trainer.final_prediction = False
        trainer.test(model, dl_val) # run EM and evaluation on held out validation set
        if config.get('save_prediction', False):
            results = trainer.predict(model, dl_val, return_predictions=True)

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    if config.get('save_prediction', False):
        result_path = os.path.join(ckpt_dir, config['experiment'], run.id, 'results')
        os.makedirs(result_path, exist_ok=True)

        with open(os.path.join(result_path, 'results.pickle'), 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        artifact = wandb.Artifact(f'results-{run.id}', type='results')
        artifact.add_dir(result_path)
        run.log_artifact(artifact)


if __name__ == "__main__":
    run_baselines()
