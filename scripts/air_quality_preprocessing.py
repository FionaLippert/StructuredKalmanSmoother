import numpy as np
import pandas as pd
import os.path as osp
import torch
import torch_geometric as ptg
import networkx as nx
from matplotlib import pyplot as plt
import argparse

from structuredKS import utils

dir = '../datasets/AirQuality'

parser = argparse.ArgumentParser(description='Generate AirQuality dataset')

parser.add_argument("--t_start", type=int, default=0, help="start index of time series")
parser.add_argument("--t_end", type=int, default=-1, help="end index of time series")
parser.add_argument("--year", type=int, default=2015, help="year to select")
parser.add_argument("--month", type=int, default=1, help="month to select")
parser.add_argument("--data_split", type=list, default=[0.7, 0.1, 0.2], help="train, val, test split")
parser.add_argument("--standardize", default=False, action='store_true', help="Standardize data to mean=0, std=1")
parser.add_argument("--variable", type=str, default="CO_Concentration", help="Target variable")

args = parser.parse_args()

df_sensors = pd.read_csv(osp.join(dir, 'sensors.csv'))
df_measurements = pd.read_csv(osp.join(dir, 'measurements.csv'))

print(df_sensors.x)

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

fig, ax = plt.subplots(figsize=(10, 10))
G_nx = ptg.utils.convert.to_networkx(G)
nx.draw_networkx_nodes(G_nx, pos, node_size=10, ax=ax)
nx.draw_networkx_edges(G_nx, pos, ax=ax, width=2, arrowsize=2)
fig.savefig(osp.join(dir, 'graph.png'))





# time processing
df_measurements['timestamp'] = pd.to_datetime(df_measurements['time'])
df_measurements.set_index('timestamp', inplace=True)

# filter time
df_sub = df_measurements[df_measurements.index.year == args.year]
df_sub = df_sub[df_sub.index.month == args.month]

nodes = torch.arange(G.num_nodes)
spatiotemporal_graphs = []
T = df_sub.time.nunique()
tidx_start = torch.arange(T)[args.t_start]
tidx_end = torch.arange(T)[args.t_end]
timestamps = []
all_data = []
all_masks = []
for tidx, (ts, df_t) in enumerate(df_sub.groupby('timestamp', sort=True)):
    if tidx >= tidx_start and tidx <= tidx_end:
        df_t = df_t[['node_idx', args.variable]].dropna().sort_values('node_idx')
        mask = torch.isin(nodes, torch.tensor(df_t.node_idx.values), assume_unique=True)
        data = torch.tensor(df_t[args.variable].values)
        all_data.append(data)
        all_masks.append(mask)
        timestamps.append(ts)

all_data = torch.cat(all_data, dim=0)
all_masks = torch.stack(all_masks, dim=0)

print(f'processed time period between {timestamps[0]} and {timestamps[-1]}')


# split data into train, val and test set
joint_mask = all_masks.reshape(-1)
T = len(timestamps)
print(f'num nodes = {joint_mask.size(0)}')
print(f'number of edges = {G.edge_index.size(1)}')
print(f'obs ratio = {joint_mask.sum() / joint_mask.size(0)}')

