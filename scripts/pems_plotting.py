import torch
import torch_geometric as ptg
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm, colormaps
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import networkx as nx
import os.path as osp

def plot_nodes(dir, G, pos, values, indices, fig, ax, cax=None, vmin=None, vmax=None, filename=None,
              node_size=50, node_alpha=0.8, edge_alpha=0.8, cmap_name='viridis', plot_title=None, unobs_color=(0, 0, 0, 1),
               edge_width=1, arrowsize=5, mark_indices=None):


    if vmin is None: vmin = np.min(values)
    if vmax is None: vmax = np.max(values)

    cmap = colormaps.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    colors = [unobs_color for v in range(G.num_nodes)]
    for idx, val in zip(indices, values):
        colors[idx] = cmap(norm(val))


    G_nx = ptg.utils.convert.to_networkx(G)
    nx.draw_networkx_nodes(G_nx, pos, node_size=node_size, node_color=colors,
                           alpha=node_alpha, ax=ax)
    nx.draw_networkx_edges(G_nx, pos, ax=ax, alpha=edge_alpha, width=edge_width, arrowsize=arrowsize),
                           #connectionstyle='arc3,rad=0.2')

    if mark_indices is not None:
        ax.scatter(pos[mark_indices, 0], pos[mark_indices, 1], c='none', s=3*node_size, edgecolors='black')


    if plot_title is not None:
        ax.set_title(plot_title)

    # adding realworld map to the background
    # ctx.add_basemap(ax=ax, crs='epsg:4326')
    ax.set_axis_off()

    if cax is not None:
        cbar = fig.colorbar(cm.ScalarMappable(norm, cmap), orientation='vertical', cax=cax)
            # cbar.ax.tick_params(labelsize=10)

    if filename is not None:
        plt.savefig(osp.join(dir, f'{filename}.png'), dpi=500)#, transparent=True)

def plot_timeseries(dir, data, fig, ax, xticklabels=None, filename=None, xlabel=None, ylabel=None):

    for k, v in data.items():
        ax.plot(v, label=k)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=90)

    ax.legend()
    ax.set(xlabel=xlabel, ylabel=ylabel)

    if filename is not None:
        fig.savefig(osp.join(dir, f'{filename}.png'), dpi=500, transparent=True)
