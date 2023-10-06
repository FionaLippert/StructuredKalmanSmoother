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

def get_model(run_path, return_config=False):
    api = wandb.Api()

    user, project, run_id = run_path.split('/')
    model_path = osp.join(user, project, f'model-{run_id}:v0')
    artifact = api.artifact(model_path, type='models')
    model_path = osp.join(artifact.download(), 'model.pt')

    if return_config:
        config = api.run(run_path).config
        return model_path, config
    else:
        return model_path

def get_runs(project_path):
    api = wandb.Api()
    runs = api.runs(project_path)

    return runs

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="conf", config_name="config")
def analyse_dgmrf(config: DictConfig):
    assert 'wandb_run' in config

    model_path, full_config = utils.get_wandb_model(config['wandb_run'], return_config=True)

    wandb_config = eval(full_config['config'].replace("'", '"'))
    wandb_config['data_dir'] = config.data_dir
    wandb_config['output_dir'] = config.output_dir
    wandb_config['wandb_run'] = config.wandb_run

    config = wandb_config
    utils.seed_all(config['seed'])

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


    # initialize model based on given config
    model = setup_model(config, device)

    # load pre-trained model
    model.load_state_dict(torch.load(model_path))

    dir = osp.join(config['output_dir'], 'transition_matrices', config['wandb_run'])
    os.makedirs(dir, exist_ok=True)

    if config.get('n_layers_temporal', 1) > 1:
        for p in range(1, config.get('n_layers_temporal', 1) + 1):
            t_start = 180
            t_end = 182
            F = model.dgmrf.get_transition_matrix(p=p, t_start=t_start, t_end=t_end)
            torch.save(F, osp.join(dir, f'F_p={p}_t={t_start}-{t_end}.pt'))

            t_start = 40
            t_end = 42
            F = model.dgmrf.get_transition_matrix(p=p, t_start=t_start, t_end=t_end)
            torch.save(F, osp.join(dir, f'F_p={p}_t={t_start}-{t_end}.pt'))

            t_start = 250
            t_end = 252
            F = model.dgmrf.get_transition_matrix(p=p, t_start=t_start, t_end=t_end)
            torch.save(F, osp.join(dir, f'F_p={p}_t={t_start}-{t_end}.pt'))

            if model.features is not None:
                torch.save(model.features, osp.join(dir, 'features.pt'))
    else:
        F = model.dgmrf.get_transition_matrix()
        torch.save(F, osp.join(dir, 'F.pt'))

    if model.pos is not None:
        torch.save(model.pos, osp.join(dir, 'pos.pt'))

    # save general properties of the temporal graph
    dataset_dict = load_dataset(config["dataset"], config["data_dir"], device=device)
    temporal_graph = dataset_dict["temporal_graph"]

    F_temporal = ptg.utils.to_dense_adj(temporal_graph.edge_index, max_num_nodes=temporal_graph.num_nodes)[0]
    torch.save(F_temporal, osp.join(dir, 'F_base_graph.pt'))

    adj_normals = ptg.utils.to_dense_adj(temporal_graph.edge_index, edge_attr=temporal_graph.edge_attr,
                                max_num_nodes=temporal_graph.num_nodes)[0]
    torch.save(adj_normals, osp.join(dir, 'adj_normals.pt'))

    if 'true_transition_matrix' in dataset_dict:
        torch.save(dataset_dict['true_transition_matrix'], osp.join(dir, 'true_F.pt'))


if __name__ == "__main__":
    analyse_dgmrf()
