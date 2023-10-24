import numpy as np
import pandas as pd
import os
import os.path as osp
import torch
import torch_geometric as ptg
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import cm, colormaps, patches
from matplotlib.colors import Normalize
import argparse

from structuredKS import utils
import constants

dir = '../datasets/AirQuality'
# weather_vars = ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']
# weather_vars = ['t2m', 'sp', 'tp', 'u10', 'v10']
weather_vars = ['t2m', 'rh', 'sp', 'u10', 'v10', 'solarpos', 'solarpos_dt'] #, 'dayofyear']

parser = argparse.ArgumentParser(description='Generate AirQuality dataset')

parser.add_argument("--t_start", type=int, default=0, help="start index of time series")
parser.add_argument("--t_end", type=int, default=-1, help="end index of time series")
parser.add_argument("--year", type=int, default=2015, help="year to select")
parser.add_argument("--month", type=int, default=3, help="month to select")
parser.add_argument("--t_blocks", type=int, default=2, help="number of time blocks to mask")
parser.add_argument("--data_split", type=list, default=0.9, help="fraction of non-test data to use for training, rest is used for validation")
parser.add_argument("--standardize", default=False, action='store_true', help="Standardize data to mean=0, std=1")
parser.add_argument("--mask", default='spatial_block', help="Type of mask; spatial_block, all_spatial, all_temporal, forecasting, or random")
parser.add_argument("--mask_size", type=float, default=0.62, help="mask size in percentage of total area")
parser.add_argument("--mask_shift_x", type=float, default=0.0, help="mask shift in x direction")
parser.add_argument("--mask_shift_y", type=float, default=0.0, help="mask shift in y direction")
parser.add_argument("--log_transform", default=True, action='store_true', help="apply log transform")
parser.add_argument("--variable", type=str, default="PM25", help="Target variable")
parser.add_argument("--obs_ratio", type=float, default=0.9, help="Fraction of points to observe")
parser.add_argument("--max_dist", type=float, default=160000, help="Max distance between connected nodes in graph")
parser.add_argument("--n_dummy_sensors", type=int, default=0, help="number of dummy sensors to include")

args = parser.parse_args()

if args.n_dummy_sensors > 0:
    df_sensors = pd.read_csv(osp.join(dir, f'sensors_ndummy={args.n_dummy_sensors}.csv'))
    df_covariates = pd.read_csv(osp.join(dir, f'covariates_{args.year}_{args.month}_ndummy={args.n_dummy_sensors}.csv'))
else:
    df_sensors = pd.read_csv(osp.join(dir, 'sensors.csv'))
    df_covariates = pd.read_csv(osp.join(dir, f'covariates_{args.year}_{args.month}.csv'))

df_measurements = pd.read_csv(osp.join(dir, 'measurements.csv'))


print(f'longitude: min = {df_sensors.longitude.min()}, max = {df_sensors.longitude.max()}')
print(f'latitude: min = {df_sensors.latitude.min()}, max = {df_sensors.latitude.max()}')
print(f'cutoff distance: {(args.max_dist / 1000):.2f} km')


pos = np.stack([df_sensors.x.values, df_sensors.y.values]).T

point_data = ptg.data.Data(pos=torch.tensor(pos))

# construct voronoi tessellation
graph_transforms = ptg.transforms.Compose((
    ptg.transforms.Delaunay(),
    ptg.transforms.FaceToEdge(),
    ptg.transforms.Distance(norm=False)
))
G = graph_transforms(point_data)

mask = G.edge_attr.squeeze() <= args.max_dist
G.edge_attr = G.edge_attr[mask, :]
G.edge_index = G.edge_index[:, mask]

edge_weights = 1. / G.edge_attr.squeeze()
G.edge_weight = edge_weights / edge_weights.max()

normals = torch.stack([utils.get_normal(pos[u], pos[v]) for u, v in G.edge_index.T], dim=0)
G.edge_attr = normals

G.weighted_degrees = utils.weighted_degrees(G)
G.weighted_eigvals = utils.compute_eigenvalues_weighted(G)
G.eigvals = utils.compute_eigenvalues(G)


if args.mask == 'spatial_block':
    # define block mask area
    xmin, xmax, ymin, ymax = G.pos[:, 0].min(), G.pos[:, 0].max(), G.pos[:, 1].min(), G.pos[:, 1].max()
    xmin = xmin + (1 - args.mask_size) * (xmax-xmin) / 2 + args.mask_shift_x * (xmax-xmin)
    xmax = xmax - (1 - args.mask_size) * (xmax-xmin) / 2 + args.mask_shift_x * (xmax-xmin)
    ymin = ymin + (1 - args.mask_size) * (ymax-ymin) / 2 + args.mask_shift_y * (ymax-ymin)
    ymax = ymax - (1 - args.mask_size) * (ymax-ymin) / 2 + args.mask_shift_y * (ymax-ymin)
    test_mask = (G.pos[:, 0] >= xmin) & (G.pos[:, 0] <= xmax) & (G.pos[:, 1] >= ymin) & (G.pos[:, 1] <= ymax)

    trainval_nodes = torch.logical_not(test_mask).nonzero().squeeze()
    n_val_nodes = int((1 - args.data_split) * trainval_nodes.numel() * 2)
    random_idx = torch.randperm(trainval_nodes.numel())
    val_nodes = trainval_nodes[random_idx][:n_val_nodes]
    val_mask = torch.zeros_like(test_mask)
    val_mask[val_nodes] = 1

# time processing
df_measurements['timestamp'] = pd.to_datetime(df_measurements['time'])
df_measurements.set_index('timestamp', inplace=True)
# df_meteo['timestamp'] = pd.to_datetime(df_meteo['time'])
# df_meteo.set_index('timestamp', inplace=True)


# filter time
df_sub = df_measurements[df_measurements.index.year == args.year]
df_sub = df_sub[df_sub.index.month == args.month]
# df_meteo = df_meteo[df_meteo.index.year == args.year]
# df_meteo = df_meteo[df_meteo.index.month == args.month]

df_covariates.set_index('timestamp', inplace=True)

# df_covariates = df_covariates[df_covariates.index.year == args.year]
# df_covariates = df_meteo[df_covariates.index.month == args.month]

# # interpolate meteorology to missing time points
# df_meteo.drop('time', inplace=True, axis=1)
# df_meteo.drop('weather', inplace=True, axis=1)
# df_meteo = df_meteo.groupby('id')[weather_vars]
# df_meteo = df_meteo.resample('1H').mean()
# df_meteo = df_meteo.interpolate().reset_index()
#
# print(df_meteo.head())


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
df_sub[variable] = df_sub[variable].replace(0, np.nan)

# fig, ax = plt.subplots(1, (tidx_end-tidx_start).item()+1, figsize=(2*(tidx_end-tidx_start).item()+1, 2))
G_nx = ptg.utils.convert.to_networkx(G)
# cmap = colormaps.get_cmap('viridis')
# norm = Normalize(vmin=df_sub[variable].min(), vmax=df_sub[variable].max())

fig, ax = plt.subplots(figsize=(10, 10))
test_color = '#0099cc'
train_color = '#cccc00'
colors = [test_color if test_mask[i] else train_color for i in range(len(G_nx))]
nx.draw_networkx_nodes(G_nx, pos, node_size=20, node_color=colors, ax=ax, alpha=0.6)
nx.draw_networkx_edges(G_nx, pos, ax=ax, width=0.5, arrowsize=0.1)
fig.savefig(osp.join(dir, f'sensor_network_{args.max_dist}_ndummy={args.n_dummy_sensors}.png'), dpi=200)



for tidx, (ts, df_t) in enumerate(df_sub.groupby('timestamp', sort=True)):
    if tidx >= tidx_start and tidx < tidx_end:

        df_t = df_t[['node_idx', variable]].dropna().sort_values('node_idx')
        mask = torch.isin(nodes, torch.tensor(df_t.node_idx.values), assume_unique=True)

        # print(ts)

        # df_weather = df_meteo.query(f'timestamp == "{ts}"')
        df_weather = df_covariates.query(f'timestamp == "{ts}"')
        # print(df_weather)
        # print(df_weather.id.nunique(), df_sensors.district_id.nunique())
        # print(torch.isin(torch.tensor(df_sensors.district_id.unique()), torch.tensor(df_weather.id.unique())))
        # df_weather = pd.merge(df_sensors, df_weather, how='left', left_on='district_id', right_on='id')
        # print(df_weather.head())
        # if len(df_weather) != len(nodes):
        #     print(len(df_weather), len(nodes))
        #     print(torch.logical_not(torch.isin(nodes, torch.tensor(df_weather.node_idx.values))).nonzero().flatten())

        data = torch.tensor(df_t[variable].values)
        # print(tidx, data.min())
        covariates = torch.tensor(df_weather.sort_values('node_idx')[weather_vars].values) # shape [num_nodes, n_vars]
        # print(covariates.min(), covariates.max())
        all_data.append(data)
        all_masks.append(mask)
        timestamps.append(ts)
        all_covariates.append(covariates)

        # if args.mask == 'spatial_block':
        #
        #     # apply mask for 20% of the time steps
        #     if ((tidx - tidx_start) > 0.2 * T) and ((tidx - tidx_start) < 0.7 * T):
        #         # use masked out area for testing
        #         # all_test_indices.append(torch.logical_and(mask, test_mask)[mask].nonzero().squeeze(-1) + n_masked)
        #         # use the rest of the observed nodes for training and validation
        #         all_test_masks.append(torch.logical_and(mask, test_mask))
        #         # all_trainval_indices.append(torch.logical_and(mask, torch.logical_not(test_mask))[mask].nonzero().squeeze(-1)
        #         #                             + n_masked)
        #         all_val_masks.append(torch.logical_and(mask, val_mask))
        #         all_train_masks.append(torch.logical_and(mask, torch.logical_not(test_mask + val_mask)))
        #     else:
        #         # use all observed nodes for training and validation
        #         # all_trainval_indices.append(torch.arange(mask.sum()) + n_masked)
        #         all_train_masks.append(mask)
        #
        #         all_test_masks.append(torch.zeros_like(mask))
        #         all_val_masks.append(torch.zeros_like(mask))
        #
        #     n_masked += mask.sum()



all_data = torch.cat(all_data, dim=0)

if args.log_transform:
    all_data = torch.log(all_data + 1e-7)

if args.standardize:
    data_mean, data_std = all_data.mean(), all_data.std()
    all_data = (all_data - data_mean) / data_std
else:
    data_mean = 0
    data_std = 1

all_masks = torch.stack(all_masks, dim=0)
joint_mask = all_masks.reshape(-1)
all_covariates = torch.cat(all_covariates, dim=0) # shape [T * num_nodes, n_vars]

# normalize covariates
all_covariates = all_covariates - all_covariates.min(0).values
all_covariates = all_covariates / all_covariates.max(0).values
all_covariates = all_covariates * 2 - 1 # scale to range (-1, 1)


fig, ax = plt.subplots(figsize=(10, 10))
colors = ['green' if all_masks[:, i].any() else 'red' for i in range(len(G_nx))]
nx.draw_networkx_nodes(G_nx, pos, node_size=20, node_color=colors, ax=ax, alpha=0.7)
nx.draw_networkx_edges(G_nx, pos, ax=ax, width=0.5, arrowsize=0.1)
fig.savefig(osp.join(dir, f'observed_nodes_{args.max_dist}_ndummy={args.n_dummy_sensors}.png'), dpi=200)


def remove_outliers(x, delta=2.0):
    # x has shape [num_nodes, T]
    x = torch.nn.functional.pad(x, (1, 1), 'constant', 0)
    x_forward = x[:, 2:]
    x_backward = x[:, :-2]
    x_center = x[:, 1:-1]

    outliers_forward = (x_forward - x_center).abs() > delta
    outliers_backward = (x_backward - x_center).abs() > delta

    outliers = torch.logical_and(outliers_backward, outliers_forward)
    outliers_nan = torch.logical_or(torch.logical_and(outliers_forward, x_backward.isnan()),
                                    torch.logical_and(outliers_backward, x_forward.isnan()))
    both_nans = torch.logical_and(x_backward.isnan(), x_forward.isnan())

    outliers = torch.logical_or(outliers, outliers_nan)
    mask = torch.logical_or(outliers, both_nans)

    x_center[mask] = np.nan

    return x_center

def remove_constants(x):
    # x has shape [num_nodes, T]
    x = torch.cat([torch.zeros(x.size(0), 1), x], dim=1)
    constants = (x[:, 1:] - x[:, :-1]) == 0
    x = x[:, 1:]
    x[constants] = np.nan

    return x

states = torch.ones(joint_mask.size(), dtype=all_data.dtype) * np.nan
states[joint_mask] = all_data

# clean data
states = remove_outliers(remove_constants(states.reshape(T, -1).transpose(0, 1))).transpose(0, 1).reshape(-1)

# update missing data mask
joint_mask = torch.logical_not(states.isnan())
all_data = states[joint_mask]


if args.mask == 'spatial_block':
    # all_test_masks = torch.stack(all_test_masks, dim=0)
    # all_train_masks = torch.stack(all_train_masks, dim=0)
    # all_val_masks = torch.stack(all_val_masks, dim=0)

    if args.t_blocks == 1:
        tidx = torch.arange(torch.ceil(0.2 * T), torch.ceil(0.7 * T), dtype=torch.long)
    else:
        block_size = int((0.5 * T) / args.t_blocks)
        print(f'length of time blocks = {block_size}')
        random_tidx = torch.randperm(T - block_size)
        starting_points = random_tidx[:args.t_blocks]
        print(f't_blocks start at indices {starting_points}')

        tidx = []
        for s in starting_points:
            tidx.append(torch.arange(s, s + block_size, dtype=torch.long))
        tidx = torch.cat(tidx)

        # tidx1 = torch.arange(torch.ceil(0.2 * T), torch.ceil(0.4 * T), dtype=torch.long)
        # tidx2 = torch.arange(torch.ceil(0.6 * T), torch.ceil(0.8 * T), dtype=torch.long)
        # tidx = torch.cat([tidx1, tidx2])

    all_test_masks = torch.zeros_like(all_masks)
    all_test_masks[tidx, :] = test_mask.view(1, -1).repeat(tidx.size(0), 1)

    all_val_masks = torch.zeros_like(all_masks)
    all_val_masks[tidx, :] = val_mask.view(1, -1).repeat(tidx.size(0), 1)

    all_train_masks = torch.logical_not(torch.logical_or(all_test_masks, all_val_masks))

    print(f'fraction of masked nodes = {test_mask.sum() / G.num_nodes}')

    # plot masked sensors on map
    fig, ax = plt.subplots(figsize=(10, 10))
    c_train = (1, 0, 0, 0.7)
    c_test = (0, 0, 1, 0.7)
    c_train = (57/256, 170/256, 115/256, 0.7)
    c_test = (128/256, 50/256, 116/256, 0.7)
    colors = [c_train] * G.num_nodes
    for i in test_mask.nonzero().squeeze():
        colors[i] = c_test

    nx.draw_networkx_edges(G_nx, pos, ax=ax, width=0.5, arrowsize=0.1, node_size=15)#min_source_margin=0, min_target_margin=0)
    nx.draw_networkx_nodes(G_nx, pos, node_size=15, node_color=colors, ax=ax, alpha=0.9)

    if args.mask == 'spatial_block':
        ax.add_patch(patches.Rectangle((xmin, ymin), min(xmax, G.pos[:, 0].max()) - xmin,
                                       min(ymax, G.pos[:, 1].max()) - ymin,
                                       edgecolor='lightgray', fill=False, lw=2, ls='--'))
    ax.axis('off')
    fig.savefig(osp.join(dir, f'masked_sensors_ndummy={args.n_dummy_sensors}.png'), dpi=400)

elif args.mask == 'all_spatial':
    # mask all nodes for a random set of time points
    random_idx = torch.randperm(T)
    n_trainval = int(T * args.obs_ratio)
    n_test = T - n_trainval
    n_train = int(n_trainval * args.data_split)

    all_test_masks = torch.zeros_like(joint_mask).reshape(T, -1)
    all_test_masks[random_idx[:n_test]] = 1

    all_train_masks = torch.zeros_like(joint_mask).reshape(T, -1)
    all_train_masks[random_idx[n_test:(n_test + n_train)]] = 1

    all_val_masks = torch.zeros_like(joint_mask).reshape(T, -1)
    all_val_masks[random_idx[(n_test + n_train):]] = 1

elif args.mask == 'forecasting':
    # mask all nodes for last few time steps
    n_trainval = int(T * args.obs_ratio)
    n_train = int(n_trainval * args.data_split)
    n_val = n_trainval - n_train

    all_train_masks = torch.zeros_like(joint_mask).reshape(T, -1)
    all_train_masks[:n_train] = 1

    all_val_masks = torch.zeros_like(joint_mask).reshape(T, -1)
    all_val_masks[n_train:n_train + n_val] = 1

    all_test_masks = torch.zeros_like(joint_mask).reshape(T, -1)
    all_test_masks[n_train + n_val:] = 1

elif args.mask == 'all_temporal':
    # mask all time steps for a random set of nodes
    random_idx = torch.randperm(G.num_nodes)
    n_trainval = int(G.num_nodes * args.obs_ratio)
    n_test = G.num_nodes - n_trainval
    n_train = int(n_trainval * args.data_split)

    all_test_masks = torch.zeros_like(joint_mask).reshape(T, -1)
    all_test_masks[:, random_idx[:n_test]] = 1

    all_train_masks = torch.zeros_like(joint_mask).reshape(T, -1)
    all_train_masks[:, random_idx[n_test:(n_test + n_train)]] = 1

    all_val_masks = torch.zeros_like(joint_mask).reshape(T, -1)
    all_val_masks[:, random_idx[(n_test + n_train):]] = 1

else:
    # pick random nodes and time points
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



print(f'processed time period between {timestamps[0]} and {timestamps[-1]}')

T = len(timestamps)
print(f'num nodes = {joint_mask.size(0)}')
print(f'number of edges = {G.edge_index.size(1)}')
print(f'obs ratio = {joint_mask.sum() / joint_mask.size(0)}')

print(f'fraction of test indices = {all_test_masks.sum() / joint_mask.numel()}')

data = {
    "spatial_graph": G,
    "temporal_graph": G,
    "data": all_data,
    "masks": joint_mask.view(T, -1),
    "test_masks": torch.logical_and(joint_mask.view(T, -1), all_test_masks),
    "train_masks": torch.logical_and(joint_mask.view(T, -1), all_train_masks),
    "val_masks": torch.logical_and(joint_mask.view(T, -1), all_val_masks),
    "covariates": all_covariates,
    "data_mean_and_std": torch.tensor([data_mean, data_std])
    # "train_idx": train_idx,
    # "val_idx": val_idx,
    # "test_idx": test_idx
    }

obs_ratio = args.mask_size if args.mask == "spatial_block" else args.obs_ratio



# save graph
ds_name = f'AQ_{args.variable}_T={T}_{args.year}_{args.month}_log={args.log_transform}_norm={args.standardize}_' \
          f'mask={args.mask}_{obs_ratio}_tblocks={args.t_blocks}_ndummy={args.n_dummy_sensors}'
print(f'Saving dataset {ds_name}')
utils.save_dataset(data, args, ds_name, constants.DS_DIR)

# save networkx graph
nx.write_graphml(G_nx, osp.join(constants.DS_DIR, ds_name, 'spatial_graph.graphml'), infer_numeric_types=True)
torch.save(pos, osp.join(constants.DS_DIR, ds_name, 'pos.pt'))