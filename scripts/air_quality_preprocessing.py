import numpy as np
import pandas as pd
import os.path as osp
import torch
import torch_geometric as ptg
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import cm, colormaps
from matplotlib.colors import Normalize
import argparse

from structuredKS import utils
import utils_dgmrf

dir = '../datasets/AirQuality'
weather_vars = ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']

parser = argparse.ArgumentParser(description='Generate AirQuality dataset')

parser.add_argument("--t_start", type=int, default=0, help="start index of time series")
parser.add_argument("--t_end", type=int, default=-1, help="end index of time series")
parser.add_argument("--year", type=int, default=2015, help="year to select")
parser.add_argument("--month", type=int, default=3, help="month to select")
parser.add_argument("--data_split", type=list, default=0.9, help="fraction of non-test data to use for training, rest is used for validation")
parser.add_argument("--standardize", default=False, action='store_true', help="Standardize data to mean=0, std=1")
parser.add_argument("--block_mask", default=False, action='store_true', help="Use block mask instead of random mask")
parser.add_argument("--mask_size", type=float, default=0.62, help="mask size in percentage of total area")
parser.add_argument("--log_transform", default=False, action='store_true', help="apply log transform")
parser.add_argument("--variable", type=str, default="PM25", help="Target variable")
parser.add_argument("--obs_ratio", type=float, default=0.9, help="Fraction of points to observe")

args = parser.parse_args()

df_sensors = pd.read_csv(osp.join(dir, 'sensors.csv'))
df_measurements = pd.read_csv(osp.join(dir, 'measurements.csv'))
df_meteo = pd.read_csv(osp.join(dir, 'meteorology.csv'))


pos = np.stack([df_sensors.x.values, df_sensors.y.values]).T

point_data = ptg.data.Data(pos=torch.tensor(pos))

# construct voronoi tessellation
graph_transforms = ptg.transforms.Compose((
    ptg.transforms.Delaunay(),
    ptg.transforms.FaceToEdge(),
    ptg.transforms.Distance()
))
G = graph_transforms(point_data)
edge_weights = 1. / G.edge_attr.squeeze()
G.edge_weight = edge_weights / edge_weights.max()

normals = torch.stack([utils.get_normal(pos[u], pos[v]) for u, v in G.edge_index.T], dim=0)

G.weighted_degrees = utils.weighted_degrees(G)
G.weighted_eigvals = utils.compute_eigenvalues_weighted(G)
G.eigvals = utils.compute_eigenvalues(G)


if args.block_mask:
    # define block mask area
    xmin, xmax, ymin, ymax = G.pos[:, 0].min(), G.pos[:, 0].max(), G.pos[:, 1].min(), G.pos[:, 1].max()
    xmin = xmin + (1 - args.mask_size) * (xmax-xmin)
    xmax = xmax - (1 - args.mask_size) * (xmax-xmin)
    ymin = ymin + (1 - args.mask_size) * (ymax-ymin)
    ymax = ymax - (1 - args.mask_size) * (ymax-ymin)
    test_mask = (G.pos[:, 0] > xmin) & (G.pos[:, 0] < xmax) & (G.pos[:, 1] > ymin) & (G.pos[:, 1] < ymax)

    trainval_nodes = torch.logical_not(test_mask).nonzero().squeeze()
    n_val_nodes = int((1 - args.data_split) * trainval_nodes.numel() * 2)
    random_idx = torch.randperm(trainval_nodes.numel())
    val_nodes = trainval_nodes[random_idx][:n_val_nodes]
    val_mask = torch.zeros_like(test_mask)
    val_mask[val_nodes] = 1


# time processing
df_measurements['timestamp'] = pd.to_datetime(df_measurements['time'])
df_measurements.set_index('timestamp', inplace=True)
df_meteo['timestamp'] = pd.to_datetime(df_meteo['time'])
df_meteo.set_index('timestamp', inplace=True)

# filter time
df_sub = df_measurements[df_measurements.index.year == args.year]
df_sub = df_sub[df_sub.index.month == args.month]
df_meteo = df_meteo[df_meteo.index.year == args.year]
df_meteo = df_meteo[df_meteo.index.month == args.month]

# interpolate meteorology to missing time points
df_meteo.drop('time', inplace=True, axis=1)
df_meteo.drop('weather', inplace=True, axis=1)
df_meteo = df_meteo.groupby('id')[weather_vars]
df_meteo = df_meteo.resample('1H').mean()
df_meteo = df_meteo.interpolate().reset_index()


nodes = torch.arange(G.num_nodes)
spatiotemporal_graphs = []
T = df_sub.time.nunique()
tidx_start = torch.arange(T)[args.t_start]
tidx_end = torch.arange(T)[args.t_end]
T = tidx_end - tidx_start
timestamps = []
all_data = []
all_covariates = []
all_masks = []
all_test_indices = []
all_trainval_indices = []
all_test_masks = []
all_trainval_masks = []
all_val_masks = []
all_train_masks = []

n_masked = 0

variable = f'{args.variable}_Concentration'

# fig, ax = plt.subplots(1, (tidx_end-tidx_start).item()+1, figsize=(2*(tidx_end-tidx_start).item()+1, 2))
G_nx = ptg.utils.convert.to_networkx(G)
# cmap = colormaps.get_cmap('viridis')
# norm = Normalize(vmin=df_sub[variable].min(), vmax=df_sub[variable].max())

for tidx, (ts, df_t) in enumerate(df_sub.groupby('timestamp', sort=True)):
    if tidx >= tidx_start and tidx < tidx_end:

        df_t = df_t[['node_idx', variable]].dropna().sort_values('node_idx')
        mask = torch.isin(nodes, torch.tensor(df_t.node_idx.values), assume_unique=True)

        df_weather = df_meteo.query(f'index == "{ts}"')
        df_weather = pd.merge(df_sensors, df_weather, how='left', left_on='district_id', right_on='id')
        if len(df_weather) != len(nodes):
            print(len(df_weather), len(nodes))
            print(torch.logical_not(torch.isin(nodes, torch.tensor(df_weather.node_idx.values))).nonzero().flatten())

        data = torch.tensor(df_t[variable].values)
        covariates = torch.tensor(df_weather[weather_vars].values) # shape [num_nodes, n_vars]
        all_data.append(data)
        all_masks.append(mask)
        timestamps.append(ts)
        all_covariates.append(covariates)

        if args.block_mask:

            # apply mask for 20% of the time steps
            if ((tidx - tidx_start) > 0.2 * T) and ((tidx - tidx_start) < 0.7 * T):
                # use masked out area for testing
                # all_test_indices.append(torch.logical_and(mask, test_mask)[mask].nonzero().squeeze(-1) + n_masked)
                # use the rest of the observed nodes for training and validation
                all_test_masks.append(torch.logical_and(mask, test_mask))
                # all_trainval_indices.append(torch.logical_and(mask, torch.logical_not(test_mask))[mask].nonzero().squeeze(-1)
                #                             + n_masked)
                all_val_masks.append(torch.logical_and(mask, val_mask))
                all_train_masks.append(torch.logical_and(mask, torch.logical_not(test_mask + val_mask)))
            else:
                # use all observed nodes for training and validation
                # all_trainval_indices.append(torch.arange(mask.sum()) + n_masked)
                all_train_masks.append(mask)

                all_test_masks.append(torch.zeros_like(mask))
                all_val_masks.append(torch.zeros_like(mask))

            n_masked += mask.sum()


all_data = torch.cat(all_data, dim=0)
if args.log_transform:
    all_data = torch.log(all_data + 1e-7)
all_masks = torch.stack(all_masks, dim=0)
joint_mask = all_masks.reshape(-1)
all_covariates = torch.cat(all_covariates, dim=0) # shape [T * num_nodes, n_vars]

if args.block_mask:
    all_test_masks = torch.stack(all_test_masks, dim=0)
    all_train_masks = torch.stack(all_train_masks, dim=0)
    all_val_masks = torch.stack(all_val_masks, dim=0)

    # split data into train, val and test set
    # test_idx = torch.cat(all_test_indices, dim=0)
    # trainval_indices = torch.cat(all_trainval_indices, dim=0)

    # n_trainval = all_trainval_masks.sum()
    # random_idx = torch.randperm(n_trainval)
    # train_idx = all_trainval_masks.reshape(-1).nonzero().squeeze()[random_idx][:int(n_trainval * args.data_split)]
    # val_idx = all_trainval_masks.reshape(-1).nonzero().squeeze()[random_idx][int(n_trainval * args.data_split):]

else:
    n_obs = joint_mask.sum()
    random_idx = torch.randperm(n_obs)
    n_trainval = int(n_obs * args.obs_ratio)
    n_test = n_obs - n_trainval
    test_idx = joint_mask.nonzero().squeeze()[random_idx][:n_test]
    n_train = int(n_trainval * args.data_split)
    train_idx = joint_mask.nonzero().squeeze()[random_idx][n_test:(n_test + n_train)]
    val_idx = joint_mask.nonzero().squeeze()[random_idx][(n_test + n_train):]

    all_test_masks = torch.zeros_like(joint_mask)
    all_test_masks[test_idx] = 1
    all_test_masks = all_test_masks.reshape(T, -1)

    test_mask = all_test_masks.sum(0)

    all_train_masks = torch.zeros_like(joint_mask)
    all_train_masks[train_idx] = 1
    all_train_masks = all_train_masks.reshape(T, -1)

    all_val_masks = torch.zeros_like(joint_mask)
    all_val_masks[val_idx] = 1
    all_test_masks = all_test_masks.reshape(T, -1)


# plot masked sensors on map
fig, ax = plt.subplots(figsize=(10,10))
colors = [(1, 0, 0, 0.7)] * G.num_nodes
for i in test_mask.nonzero().squeeze():
    colors[i] = (0, 0, 1, 0.7)
nx.draw_networkx_nodes(G_nx, pos, node_size=10, node_color=colors, ax=ax, alpha=0.7)
nx.draw_networkx_edges(G_nx, pos, ax=ax, width=0.5, arrowsize=0.1)
fig.savefig(osp.join(dir, 'masked_sensors.png'), dpi=200)


print(f'processed time period between {timestamps[0]} and {timestamps[-1]}')

T = len(timestamps)
print(f'num nodes = {joint_mask.size(0)}')
print(f'number of edges = {G.edge_index.size(1)}')
print(f'obs ratio = {joint_mask.sum() / joint_mask.size(0)}')

print(f'fraction of masked nodes = {test_mask.sum() / G.num_nodes}')
print(f'fraction of test indices = {all_test_masks.sum() / joint_mask.numel()}')

data = {
    "spatial_graph": G,
    "temporal_graph": G,
    "data": all_data,
    "masks": all_masks.reshape(T, -1),
    "test_masks": all_test_masks,
    "train_masks": all_train_masks,
    "val_masks": all_val_masks,
    "covariates": all_covariates,
    # "train_idx": train_idx,
    # "val_idx": val_idx,
    # "test_idx": test_idx
    }

# save graph
ds_name = f'AQ_{args.variable}_T={T}_{args.year}_{args.month}_log={args.log_transform}_1block={args.block_mask}_' \
          f'{args.mask_size if args.block_mask else args.obs_ratio}'
print(f'Saving dataset {ds_name}')
utils_dgmrf.save_graph_ds(data, args, ds_name)

