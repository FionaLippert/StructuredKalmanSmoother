import torch
import torch_geometric as ptg
import pandas as pd
import networkx as nx
import os.path as osp
import pickle
import numpy as np
import argparse
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from structuredKS import utils
from utils_dgmrf import save_graph_ds

from pems_plotting import *

utils.seed_all(0)


parser = argparse.ArgumentParser(description='Generate PeMS dataset')

parser.add_argument("--t_start", type=int, default=0, help="start index of time series")
parser.add_argument("--t_end", type=int, default=-1, help="end index of time series")
parser.add_argument("--data_split", type=list, default=[0.7, 0.1, 0.2], help="train, val, test split")
parser.add_argument("--standardize", type=bool, default=True, help="Standardize data to mean=0, std=1")

args = parser.parse_args()

dir = '../datasets/pems'

with open(osp.join(dir, 'processed_nx_graph.pkl'), 'rb') as f:
    G = pickle.load(f)

df_traffic = pd.read_csv(osp.join(dir, 'traffic.csv'))
df_sensors = pd.read_csv(osp.join(dir, 'sensors.csv'))

# assemble graphs
ptg_graph = ptg.utils.convert.from_networkx(G)

pos = torch.stack([ptg_graph.x, ptg_graph.y], dim=1)
normals = torch.stack([utils.get_normal(pos[u], pos[v]) for u, v in ptg_graph.edge_index.T], dim=0)
edge_attr = torch.cat([normals, ptg_graph.weight.unsqueeze(1)], dim=1)

nodes = torch.arange(ptg_graph.num_nodes)
spatiotemporal_graphs = []
T = df_traffic.Timestamp.nunique()
tidx_start = torch.arange(T)[args.t_start]
tidx_end = torch.arange(T)[args.t_end]
joint_mask = []
timestamps = []
for tidx, (ts, df) in enumerate(df_traffic.groupby('Timestamp', sort=True)):
    if tidx >= tidx_start and tidx <= tidx_end:
        df = df[['Node', 'Speed']].dropna().sort_values('Node')
        mask = torch.isin(nodes, torch.tensor(df.Node.values), assume_unique=True)
        data = torch.tensor(df.Speed.values)
        joint_mask.append(mask)
        timestamps.append(ts)

        spatiotemporal_graphs.append(utils.assemble_graph_slice(pos, data, mask, ptg_graph.edge_index,
                                    spatial_edge_attr=ptg_graph.weight.unsqueeze(1),
                                    temporal_edges=ptg_graph.edge_index, temporal_edge_attr=edge_attr))

# split data into train, val and test set
joint_mask = torch.cat(joint_mask, dim=0)
T = len(spatiotemporal_graphs)

nodes = torch.arange(ptg_graph.num_nodes).repeat(T)
data_nodes = nodes[joint_mask]

sensor_nodes = data_nodes.unique()
M = sensor_nodes.size(0)
random_idx = torch.randperm(M)

n_train = int(M * args.data_split[0])
n_val = int(M * args.data_split[1])
n_test = M - n_train - n_val
train_nodes = sensor_nodes[random_idx[:n_train]]
val_nodes = sensor_nodes[random_idx[n_train: n_train + n_val]]
test_nodes = sensor_nodes[random_idx[n_train + n_val: n_train + n_val + n_test]]
print(f'train_nodes = {train_nodes}')

train_idx = torch.isin(data_nodes, train_nodes).nonzero().squeeze()
print(f'train_idx = {train_idx}')

val_idx = torch.isin(data_nodes, val_nodes).nonzero().squeeze()
print(f'val_idx = {val_idx}')

test_idx = torch.isin(data_nodes, test_nodes).nonzero().squeeze()


spatiotemporal_graphs = ptg.data.Batch.from_data_list(spatiotemporal_graphs)
if args.standardize:
    data = spatiotemporal_graphs["data"].x
    mean, std = data.mean(), data.std()
    data_standardized = (data - mean) / std
    spatiotemporal_graphs["data"].x = data_standardized
    # spatiotemporal_graphs["data"].mean = mean
    # spatiotemporal_graphs["data"].std = std


dataset = {'spatiotemporal_graphs': spatiotemporal_graphs,
           'train_idx': train_idx,
           'val_idx': val_idx,
           'test_idx': test_idx,
           'train_nodes': train_nodes,
           'val_nodes': val_nodes,
           'test_nodes': test_nodes,
           # 'timestamps': timestamps
           }

print(dataset)

# save graph
ds_name = f'pems_start={tidx_start}_end={tidx_end}'
print(f'Saving dataset {ds_name}')
dir = save_graph_ds(dataset, args, ds_name)
print(dir)

# fig, ax = plt.subplots(figsize=(15, 10))
# ax_divider = make_axes_locatable(ax)
# cax = ax_divider.append_axes("right", size="7%", pad="2%")
# tidx = 0
# G = spatiotemporal_graphs.get_example(tidx)
# indices = G["latent"].mask.nonzero().squeeze()
# pos = G["latent"].pos.numpy()
# G_t = ptg.data.Data(edge_index=G["latent", "spatial", "latent"].edge_index, num_nodes=G["latent"].num_nodes)
# plot_nodes(dir, G_t, pos, G["data"].x.numpy(),
#            indices, ax, cax, fig, filename='test_plot')
#
# T = spatiotemporal_graphs.num_graphs
# data_indices = spatiotemporal_graphs["latent"].mask.reshape(T, -1)[0].nonzero().squeeze()
# idx_0 = data_indices[0]
# idx_1 = data_indices[1]
# data = {f'node {idx_0}' : spatiotemporal_graphs["latent"].y.reshape(T, -1)[:, idx_0],
#         f'node {idx_1}' : spatiotemporal_graphs["latent"].y.reshape(T, -1)[:, idx_1]}
# fig, ax = plt.subplots(figsize=(10, 6))
# plot_timeseries(dir, data, fig, ax, xticklabels=timestamps, filename='test_timeseries', ylabel='speed')