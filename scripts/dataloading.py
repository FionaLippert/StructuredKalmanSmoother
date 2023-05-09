import torch
import torch_geometric as ptg
import os
import json
import pickle
import scipy.linalg as spl
import argparse
from matplotlib import pyplot as plt

from structuredKS.datasets import toydata
import utils_dgmrf as utils
import constants_dgmrf as constants



parser = argparse.ArgumentParser(description='Generate dataset')

parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--grid_size", type=int, default=20,
        help="Number of grid cells in x and y dimension")
parser.add_argument("--obs_noise_std", type=float, default=0.01,
        help="Std.-dev. of noise for p(y|x)")
parser.add_argument("--obs_ratio", type=float, default=0.7,
        help="Fraction of points to observe")
parser.add_argument("--n_samples", type=int, default=1, help="Number of samples")
parser.add_argument("--time_steps", type=int, default=6, help="Number of time steps")
parser.add_argument("--diff", type=float, default=0.01,
        help="Diffusion coefficient")
parser.add_argument("--advection", type=str, default='constant',
        help="Advection type")
parser.add_argument("--n_transitions", type=int, default=4,
        help="number of transitions per time step in the state space model")
parser.add_argument("--k_max", type=int, default=1,
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
                             n_transitions=args.n_transitions, block_obs=True)

# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# v = data['velocities'].reshape(2, args.grid_size, args.grid_size)
# img = ax[0].imshow(v[0])
# img = ax[1].imshow(v[1])
# ax[0].axis('off')
# ax[1].axis('off')
# fig.savefig(os.path.join(raw_data_dir, f'velocities.png'), bbox_inches='tight')

# plotting
graph_list = data["spatiotemporal_graphs"] #.to_data_list()
if T > 1:
    toydata.plot_spatiotemporal(graph_list, save_to=raw_data_dir)
    toydata.plot_spatiotemporal(graph_list, plot_data=True, save_to=raw_data_dir)
else:
    toydata.plot_spatial(graph_list.to_data_list()[0], save_to=raw_data_dir)


# cg_mean = data["cg_posterior_mean"]
# fig, ax = plt.subplots(1, T, figsize=(T * 8, 8))
# vmin = cg_mean.min()
# vmax = cg_mean.max()
# cg_mean = cg_mean.reshape(T, args.grid_size, args.grid_size)
# for t in range(T):
#     img = ax[t].imshow(cg_mean[t], vmin=vmin, vmax=vmax)
#     ax[t].axis('off')
#     ax[t].set_title(f't = {t}', fontsize=30)
#
# cbar = fig.colorbar(img, ax=ax, shrink=0.6, aspect=10)
# cbar.ax.tick_params(labelsize=20)


# fig.savefig(os.path.join(raw_data_dir, f'cg_mean.png'), bbox_inches='tight')

#
# def save_graph_ds(save_dict, args, ds_name):
#     ds_dir_path = os.path.join(constants.DS_DIR, ds_name)
#     os.makedirs(ds_dir_path, exist_ok=True)
#
#     for name, data in save_dict.items():
#         fp = os.path.join(ds_dir_path, "{}.pickle".format(name))
#         with open(fp, "wb") as file:
#             pickle.dump(data, file)
#
#     # Dump cmd-line arguments as json in dataset directory
#     json_string = json.dumps(vars(args), sort_keys=True, indent=4)
#     json_path = os.path.join(ds_dir_path, "description.json")
#     with open(json_path, "w") as json_file:
#         json_file.write(json_string)

# split data into train and validation set
joint_mask = data['spatiotemporal_graphs']['latent'].mask
n_nodes = args.grid_size * args.grid_size

nodes = torch.arange(n_nodes).repeat(T)
data_nodes = nodes[joint_mask]

sensor_nodes = data_nodes.unique()
M = sensor_nodes.size(0)
random_idx = torch.randperm(M)

n_train = int(M * args.data_split)
train_nodes = sensor_nodes[random_idx[:n_train]]
val_nodes = sensor_nodes[random_idx[n_train:]]

# TODO: change this back!
random_idx = torch.randperm(joint_mask.sum())

# train_idx = torch.isin(data_nodes, train_nodes).nonzero().squeeze()
train_idx = random_idx[:int(joint_mask.sum() * args.data_split)]
print(f'train_idx = {train_idx}')

# val_idx = torch.isin(data_nodes, val_nodes).nonzero().squeeze()
val_idx = random_idx[int(joint_mask.sum() * args.data_split):]
print(f'val_idx = {val_idx}')

# data_idx = torch.randperm(n_nodes)
# n_train = int(M * args.data_split)
# train_idx = data_idx[:n_train]
# val_idx = data_idx[n_train:]
data['train_idx'] = train_idx
data['val_idx'] = val_idx
data['train_nodes'] = train_nodes
data['val_nodes'] = val_nodes

# use ground truth at unobserved nodes for testing
test_idx = (joint_mask == 0).nonzero().squeeze()
data['test_idx'] = test_idx

data['grid_size'] = torch.tensor([args.grid_size, args.grid_size])

# save graph
ds_name = f'spatiotemporal_{args.grid_size}x{args.grid_size}_obs={args.obs_ratio}_' \
          f'T={T}_diff={args.diff}_adv={args.advection}_ntrans={args.n_transitions}_block={args.block_mask}_{args.seed}'
print(f'Saving dataset {ds_name}')
utils.save_graph_ds(data, args, ds_name)
