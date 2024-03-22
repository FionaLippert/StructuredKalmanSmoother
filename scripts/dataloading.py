import torch
import torch_geometric as ptg
import os
import json
import pickle
import scipy.linalg as spl
import argparse
from matplotlib import pyplot as plt

from stdgmrf.datasets import toydata
import utils_dgmrf as utils
import constants_dgmrf as constants



parser = argparse.ArgumentParser(description='Generate dataset')

parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--grid_size", type=int, default=30,
        help="Number of grid cells in x and y dimension")
parser.add_argument("--obs_noise_std", type=float, default=0.01,
        help="Std.-dev. of noise for p(y|x)")
parser.add_argument("--obs_ratio", type=float, default=0.7,
        help="Fraction of points to observe")
parser.add_argument("--n_samples", type=int, default=1, help="Number of samples")
parser.add_argument("--time_steps", type=int, default=20, help="Number of time steps")
parser.add_argument("--diff", type=float, default=0.01,
        help="Diffusion coefficient")
parser.add_argument("--advection", type=str, default='constant',
        help="Advection type")
parser.add_argument("--n_transitions", type=int, default=4,
        help="number of transitions per time step in the state space model")
parser.add_argument("--k_max", type=int, default=3,
        help="number of terms to use to approximate matrix exponential")
parser.add_argument("--block_mask", default=False, action='store_true', help="Use block mask instead of random mask")
parser.add_argument("--data_split", type=list, default=0.9, help="fraction of data to use for training, "
                                                                 "the rest is used for validation")



args = parser.parse_args()

# settings
raw_data_dir = '../raw_data'
T = args.time_steps

# generate data
data = toydata.generate_data(args.grid_size, T, diffusion=args.diff, advection=args.advection,
                             obs_noise_std=args.obs_noise_std, obs_ratio=args.obs_ratio, seed=args.seed,
                             n_transitions=args.n_transitions, block_obs=args.block_mask, k_max=args.k_max)

# plotting
graph_list = data["spatiotemporal_graphs"]
if T > 1:
    toydata.plot_spatiotemporal(graph_list, save_to=raw_data_dir)
    toydata.plot_spatiotemporal(graph_list, plot_data=True, save_to=raw_data_dir)
else:
    toydata.plot_spatial(graph_list.to_data_list()[0], save_to=raw_data_dir)


# split data into train and validation set
joint_mask = data['spatiotemporal_graphs']['latent'].mask
n_nodes = args.grid_size * args.grid_size

random_idx = torch.randperm(joint_mask.sum())

# train_idx = torch.isin(data_nodes, train_nodes).nonzero().squeeze()
train_idx = random_idx[:int(joint_mask.sum() * args.data_split)]
print(f'train_idx = {train_idx}')

# val_idx = torch.isin(data_nodes, val_nodes).nonzero().squeeze()
val_idx = random_idx[int(joint_mask.sum() * args.data_split):]
print(f'val_idx = {val_idx}')

data['train_idx'] = train_idx
data['val_idx'] = val_idx


# use ground truth at unobserved nodes for testing
test_idx = (joint_mask == 0).nonzero().squeeze()
data['test_idx'] = test_idx

train_mask = torch.zeros_like(joint_mask)
train_mask[joint_mask.nonzero().squeeze()[train_idx]] = 1

val_mask = torch.zeros_like(joint_mask)
val_mask[joint_mask.nonzero().squeeze()[val_idx]] = 1

test_mask = torch.logical_not(joint_mask)

data['train_masks'] = train_mask.reshape(T, -1)
data['val_masks'] = val_mask.reshape(T, -1)
data['test_masks'] = test_mask.reshape(T, -1)

data['grid_size'] = torch.tensor([args.grid_size, args.grid_size])

# save graph
ds_name = f'advection_{args.grid_size}x{args.grid_size}_obs={args.obs_ratio}_' \
          f'T={T}_diff={args.diff}_adv={args.advection}_ntrans={args.n_transitions}_1block={args.block_mask}_' \
          f'kmax={args.k_max}_{args.seed}'
print(f'Saving dataset {ds_name}')
utils.save_graph_ds(data, args, ds_name)
