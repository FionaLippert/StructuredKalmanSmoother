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
import seaborn as sb

from structuredKS import utils
from utils_dgmrf import save_graph_ds

from pems_plotting import *

utils.seed_all(111)
# utils.seed_all(12345)
# utils.seed_all(123)


parser = argparse.ArgumentParser(description='Generate PeMS dataset')

parser.add_argument("--t_start", type=int, default=0, help="start index of time series")
parser.add_argument("--t_end", type=int, default=-1, help="end index of time series")
parser.add_argument("--data_split", type=list, default=[0.7, 0.1, 0.2], help="train, val, test split")
parser.add_argument("--standardize", default=False, action='store_true', help="Standardize data to mean=0, std=1")
parser.add_argument("--variable", type=str, default="Speed", help="Target variable")

args = parser.parse_args()

dir = '../datasets/pems'

with open(osp.join(dir, 'processed_nx_graph.pkl'), 'rb') as f:
    G = pickle.load(f)

df_traffic = pd.read_csv(osp.join(dir, 'traffic.csv'))
df_sensors = pd.read_csv(osp.join(dir, 'sensors.csv'))

# assemble graphs
ptg_graph = ptg.utils.convert.from_networkx(G)
print(ptg_graph.length.sort())
# ptg_graph.edge_weight = 100. / (ptg_graph.length + 1e-2)#/ ptg_graph.weight.max()
ptg_graph.edge_weight = torch.clamp(1 / ptg_graph.travel_time, max=1.0)
print(ptg_graph.edge_weight.min(), ptg_graph.edge_weight.mean(), ptg_graph.edge_weight.max())
fig, ax = plt.subplots()
sb.histplot(ptg_graph.edge_weight.cpu().numpy(), ax=ax)
ax.set_xlim(0, 5)
fig.savefig(osp.join(dir, 'edge_weight_distr.png'), bbox_inches='tight')
ptg_graph.edge_attr = torch.stack([ptg_graph.edge_weight, ptg_graph.lanes / ptg_graph.lanes.max(),
                                   ptg_graph.speed_kph / ptg_graph.speed_kph.max()], dim=1).to(torch.float32)

pos = torch.stack([ptg_graph.x, ptg_graph.y], dim=1)
normals = torch.stack([utils.get_normal(pos[u], pos[v]) for u, v in ptg_graph.edge_index.T], dim=0)
# edge_attr = torch.cat([normals, ptg_graph.weight.unsqueeze(1)], dim=1)
# TODO: add other relevant features such as road type, time of the day, intersection vs road segment (i.e. degree), direction of travel

ptg_graph.pos = pos
# ptg_graph.edge_weight = ptg_graph.edge_weight * ptg_graph.lanes


undir_edge_index, undir_edge_weight = ptg.utils.to_undirected(ptg_graph.edge_index, ptg_graph.edge_weight)
spatial_graph = ptg.data.Data(edge_index=undir_edge_index, edge_weight=undir_edge_weight, pos=ptg_graph.pos)
spatial_graph.weighted_degrees = utils.weighted_degrees(spatial_graph)
spatial_graph.weighted_eigvals = utils.compute_eigenvalues_weighted(spatial_graph)
spatial_graph.eigvals = utils.compute_eigenvalues(spatial_graph)


nodes = torch.arange(ptg_graph.num_nodes)
spatiotemporal_graphs = []
T = df_traffic.Timestamp.nunique()
tidx_start = torch.arange(T)[args.t_start]
tidx_end = torch.arange(T)[args.t_end]
timestamps = []
all_data = []
all_masks = []
for tidx, (ts, df) in enumerate(df_traffic.groupby('Timestamp', sort=True)):
    if tidx >= tidx_start and tidx <= tidx_end:
        df = df[['Node', args.variable]].dropna().sort_values('Node')
        mask = torch.isin(nodes, torch.tensor(df.Node.values), assume_unique=True)
        data = torch.tensor(df[args.variable].values)
        all_data.append(data)
        all_masks.append(mask)
        timestamps.append(ts)

        # spatiotemporal_graphs.append(utils.assemble_graph_slice(pos, data, mask, undir_edge_index,
        #                             spatial_edge_attr=torch.stack(undir_edge_attr, dim=1),
        #                             temporal_edges=ptg_graph.edge_index, temporal_edge_attr=normals))

all_data = torch.cat(all_data, dim=0)
all_masks = torch.stack(all_masks, dim=0)

print(f'processed time period between {timestamps[0]} and {timestamps[-1]}')


# split data into train, val and test set
joint_mask = all_masks.reshape(-1)
T = len(timestamps)
print(f'num nodes = {joint_mask.size(0)}, num sensors = {joint_mask.sum()}')
print(f'directed edges = {ptg_graph.edge_index.size(1)}, undirected edges = {undir_edge_index.size(1)}')
print(f'obs ratio = {joint_mask.sum() / joint_mask.size(0)}')

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

train_idx = torch.isin(data_nodes, train_nodes).nonzero().squeeze()
val_idx = torch.isin(data_nodes, val_nodes).nonzero().squeeze()
test_idx = torch.isin(data_nodes, test_nodes).nonzero().squeeze()


# spatiotemporal_graphs = ptg.data.Batch.from_data_list(spatiotemporal_graphs)
if args.standardize:
    # data = spatiotemporal_graphs["data"].x
    # mean, std = data.mean(), data.std()
    # data_standardized = (data - mean) / std
    # spatiotemporal_graphs["data"].x = data_standardized
    # spatiotemporal_graphs["data"].mean = mean
    # spatiotemporal_graphs["data"].std = std
    print(f'standardize data')
    mean, std = all_data.mean(), all_data.std()
    all_data = (all_data - mean) / std
else:
    mean, std = torch.tensor(0), torch.tensor(1)




dataset = {
    # 'spatiotemporal_graphs': spatiotemporal_graphs,
           'spatial_graph': spatial_graph,
           'temporal_graph': ptg_graph,
           'train_idx': train_idx,
           'val_idx': val_idx,
           'test_idx': test_idx,
           'train_nodes': train_nodes,
           'val_nodes': val_nodes,
           'test_nodes': test_nodes,
           'timestamps': timestamps,
           'data': all_data,
           'masks': all_masks,
           'data_mean': mean,
           'data_std': std
           }


# save graph
ds_name = f'pems_start={tidx_start}_end={tidx_end}_{args.variable}'
print(f'Saving dataset {ds_name}')
dir = save_graph_ds(dataset, args, ds_name)
print(dir)

fig, ax = plt.subplots(figsize=(15, 10))
# ax_divider = make_axes_locatable(ax)
# cax = ax_divider.append_axes("right", size="7%", pad="2%")
tidx = 0
# G = spatiotemporal_graphs.get_example(tidx)
indices = all_masks[0].nonzero().squeeze()
pos = ptg_graph.pos.numpy()
# G_t = ptg.data.Data(edge_index=G["latent", "spatial", "latent"].edge_index, num_nodes=G["latent"].num_nodes)
# G_t = ptg.data.Data(edge_index=G["latent", "temporal", "latent"].edge_index, num_nodes=G["latent"].num_nodes)
# plot_nodes(dir, G_t, pos, G["data"].x.numpy(),
#            indices, ax, cax, fig, filename='test_plot')

node_colors = torch.cat([torch.ones(len(train_nodes)), torch.ones(len(val_nodes)) * 2, torch.ones(len(test_nodes)) * 3])
plot_nodes(dir, ptg_graph, pos, np.ones(len(ptg_graph)),
           np.arange(len(ptg_graph)), fig, ax, cax=None, unobs_color=(0.5, 0.5, 0.5, 0),
           node_alpha=0.5, node_size=1, edge_width=0.2, arrowsize=1)

ax.scatter(pos[train_nodes, 0], pos[train_nodes, 1], alpha=0.8, color='red', label='training', s=10)
ax.scatter(pos[val_nodes, 0], pos[val_nodes, 1], alpha=0.8, color='orange', label='validation', s=10)
ax.scatter(pos[test_nodes, 0], pos[test_nodes, 1], alpha=0.8, color='green', label='testing', s=10)
ax.legend()

plt.savefig(osp.join(dir, f'node_split.png'), dpi=500)#, transparent=True)

fig, ax = plt.subplots(figsize=(15, 10))
ax.scatter(pos[indices, 0], pos[indices, 1], alpha=0.2)
for idx in indices:
    ax.text(pos[idx, 0], pos[idx, 1], str(idx.item()), fontsize=8)
fig.savefig(osp.join(dir, f'observed_node_labels.png'), dpi=500)
#
# T = spatiotemporal_graphs.num_graphs
# data_indices = spatiotemporal_graphs["latent"].mask.reshape(T, -1)[0].nonzero().squeeze()
# idx_0 = data_indices[0]
# idx_1 = data_indices[1]
# data = {f'node {idx_0}' : spatiotemporal_graphs["latent"].y.reshape(T, -1)[:, idx_0],
#         f'node {idx_1}' : spatiotemporal_graphs["latent"].y.reshape(T, -1)[:, idx_1]}
# fig, ax = plt.subplots(figsize=(10, 6))
# plot_timeseries(dir, data, fig, ax, xticklabels=timestamps, filename='test_timeseries', ylabel='speed')