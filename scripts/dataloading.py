import torch
import torch_geometric as ptg
import os
import json
import pickle
import scipy.linalg as spl
import argparse

from structuredKS.datasets import toydata
import utils_dgmrf as utils
import constants_dgmrf as constants



parser = argparse.ArgumentParser(description='Generate dataset')

parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--grid_size", type=int, default=20,
        help="Number of grid cells in x and y dimension")
parser.add_argument("--obs_noise_std", type=float, default=0.01,
        help="Std.-dev. of noise for p(y|x)")
parser.add_argument("--obs_ratio", type=float, default=0.9,
        help="Fraction of points to observe")
parser.add_argument("--n_samples", type=int, default=1, help="Number of samples")
parser.add_argument("--time_steps", type=int, default=1, help="Number of time steps")
parser.add_argument("--diff", type=float, default=0.0,
        help="Diffusion coefficient")
parser.add_argument("--advection", type=str, default='None',
        help="Advection type")



args = parser.parse_args()

# settings
raw_data_dir = '../raw_data'
T = args.time_steps

# generate data
data = toydata.generate_data(args.grid_size, T, diffusion=args.diff, advection=args.advection,
                             obs_noise_std=args.obs_noise_std, obs_ratio=args.obs_ratio, seed=args.seed)

# plotting
graph_list = ptg.data.Batch.to_data_list(data["graphs"])
graph_list = data["graphs"].to_data_list()
if T > 1:
    toydata.plot_spatiotemporal(graph_list, save_to=raw_data_dir)
else:
    toydata.plot_spatial(graph_list[0], save_to=raw_data_dir)


def save_graph_ds(save_dict, args, ds_name):
    ds_dir_path = os.path.join(constants.DS_DIR, ds_name)
    os.makedirs(ds_dir_path, exist_ok=True)

    for name, data in save_dict.items():
        fp = os.path.join(ds_dir_path, "{}.pickle".format(name))
        with open(fp, "wb") as file:
            pickle.dump(data, file)

    # Dump cmd-line arguments as json in dataset directory
    json_string = json.dumps(vars(args), sort_keys=True, indent=4)
    json_path = os.path.join(ds_dir_path, "description.json")
    with open(json_path, "w") as json_file:
        json_file.write(json_string)


# save graph
ds_name = f'spatiotemporal_{args.grid_size}x{args.grid_size}_T={T}_diff={args.diff}_adv={args.advection}_{args.seed}'
print(f'Saving dataset {ds_name}')
utils.save_graph_ds(data, args, ds_name)
