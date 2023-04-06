import torch
import torch_geometric as ptg
from torch_sparse import SparseTensor
import networkx as nx
from matplotlib import pyplot as plt
import os
from structuredKS import utils
from time import perf_counter
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

def Matrn_G(L, tau, kappa2, gamma):
    G = tau * (kappa2 * torch.eye(L.shape[0]) + L)**gamma
    return G

def DGMRF_G(D, A, alpha, beta):
    G = alpha * D + beta * A
    return G

def diag_G(L, sigma):
    G = torch.eye(L.shape[0]) / sigma
    return G


def generate_observations(states, noise_std, obs_ratio):
    n_obs = int(obs_ratio * states.size(0))
    jdx, _ = torch.randperm(states.size(0))[:n_obs].sort()
    H = torch.eye(states.size(0))
    H = H[jdx, :]

    noise = noise_std * torch.randn(states.size())
    data = states + noise

    return data, H

def get_mask(H):
    return H.sum(0).bool()

def mask_observations(data, H):
    return H @ data


def combine_graphs(graph_list, shift_pos=False):
    pos = []
    edge_index = []
    num_nodes = 0
    pos_max = torch.tensor([0, 0])
    for tidx, graph in enumerate(graph_list):
        pos.append(graph.pos + pos_max) # shape [num_nodes, 2]
        edge_index.append(graph.edge_index + num_nodes) # shape [2, num_edges]
        num_nodes += graph.num_nodes
        if shift_pos:
            pos_max[0] += graph.pos[:, 0].max() + int(0.01 * graph.num_nodes)
    pos = torch.cat(pos, dim=0)
    edge_index = torch.cat(edge_index, dim=1)
    return pos, edge_index

def diffusion_coeffs(adj, d):
    # assume dt=1, cell_size=1
    # otherwise: return adj * d * dt / cell_size
    return adj * d

def advection_coeffs(edge_velocities):
    # assume dt=1, face_length=1, cell_size=1
    return -0.5 * edge_velocities

def transition_matrix(coeffs):
    # add self-edges
    diag = torch.diag(1 - coeffs.sum(1))
    F = coeffs + diag

    return F

def construct_joint_F(transition_matrix, T):
    joint_rows = []
    joint_cols = []
    rows, cols = transition_matrix.coalesce().indices()
    vals = transition_matrix.coalesce().values()
    N = transition_matrix.size(0)

    # off-diagonal blocks
    for t in range(T-1):
        joint_rows.append(rows + (t+1) * N)
        joint_cols.append(cols + t * N)

    joint_indices = torch.stack([torch.cat(joint_rows), torch.cat(joint_cols)], dim=0)
    joint_vals = vals.repeat(T-1)
    joint_F = torch.sparse_coo_tensor(joint_indices, joint_vals, size=(N*T, N*T))

    return joint_F

def sparse_identity(dim):
    indices = torch.arange(dim).repeat(2, 1)
    values = torch.ones(dim)
    id = torch.sparse_coo_tensor(indices, values, size=(dim, dim))

    return id


def construct_joint_G(initial_G, transition_G, T):
    joint_indices = [initial_G.coalesce().indices()]
    transition_indices = transition_G.coalesce().indices()
    N = transition_G.size(0)

    for t in range(T-1):
        joint_indices.append(transition_indices + (t+1) * N)

    joint_indices = torch.cat(joint_indices, dim=1)
    joint_vals = torch.cat([initial_G.coalesce().values(), transition_G.coalesce().values().repeat(T-1)])
    joint_G = torch.sparse_coo_tensor(joint_indices, joint_vals, size=(N*T, N*T))

    return joint_G



def get_normal(u, v, max=None):
    dx = v[0] - u[0]
    dy = v[1] - u[1]

    # take care of periodic boundaries
    if abs(dx) == max:
        dx = -dx
    if abs(dy) == max:
        dy = -dy

    normal = torch.tensor([dx, dy])
    norm = torch.pow(normal, 2).sum().sqrt()
    normal = normal / norm
    return normal

# def assemble_graph(latent_states, latent_pos, grid_size, T, data, spatial_edges, temporal_edges, spatiotemporal_edges,
#                    spatiotemporal_attr, observation_edges, observation_weights, observation_noise_std):
#
#     graph = ptg.data.HeteroData()
#
#     # latent node properties
#     graph['latent'].x = latent_states
#     graph['latent'].pos = latent_pos
#     graph['latent'].N = grid_size**2
#     graph['latent'].grid_size = grid_size
#     graph['latent'].T = T
#
#     # data node properties
#     graph['data'].x = data
#     graph['data'].noise_std = observation_noise_std
#
#     # edges
#     graph['latent', 'spatial', 'latent'].edge_index = spatial_edges
#     graph['latent', 'temporal', 'latent'].edge_index = temporal_edges
#
#     graph['latent', 'spatiotemporal', 'latent'].edge_index = spatiotemporal_edges
#     graph['latent', 'spatiotemporal', 'latent'].edge_attr = spatiotemporal_attr
#
#     graph['latent', 'observation', 'data'].edge_index = observation_edges
#     graph['latent', 'observation', 'data'].edge_weight = observation_weights
#
#     return graph



def generate_data(grid_size, T, diffusion=0, advection='zero', obs_noise_std=0.01, obs_ratio=0.8,
                  transition_noise_std=0.1, seed=0, n_transitions=1):
    # set seed for reproducibility
    torch.manual_seed(seed)

    # setup lattice graph
    N = grid_size * grid_size
    g = nx.DiGraph()
    g = nx.grid_2d_graph(grid_size, grid_size, periodic=True, create_using=g)
    node_pos = torch.tensor(list(g.nodes()))
    node_dict = dict(zip(list(g.nodes()), range(len(g))))
    normals = torch.stack([get_normal(u, v, max=grid_size - 1) for u, v in g.edges()], dim=0)

    # adjacency matrix
    adj = torch.tensor(nx.adjacency_matrix(g).todense())
    edges, _ = ptg.utils.dense_to_sparse(adj)

    # constructing spatial graphs
    D = torch.diag(adj.sum(0))
    L = D - adj
    P_0 = Matrn_G(L, 1, 0, 1)
    P_t = torch.eye(N) / transition_noise_std

    # only remove self loops if not diagonal??
    transition_edges, _ = ptg.utils.dense_to_sparse(P_t)
    # transition_edges, _ = ptg.utils.remove_self_loops(ptg.utils.dense_to_sparse(P_t))

    # define dynamics
    F_coeffs = torch.zeros(N, N)
    identity = True
    velocities = torch.zeros(2, len(g))
    if diffusion:
        print(f'use diffusion coefficient = {diffusion}')
        F_coeffs += diffusion_coeffs(adj, diffusion)
        identity = False
    if not advection == 'zero':
        if advection == 'constant':
            velocities = torch.tensor([-0.3, 0.3]).unsqueeze(1).repeat(1, N)
        else:
            P_v = Matrn_G(L, 1000, 0, 1)
            velocities, _, _ = utils.sample_GMRF(P_v, torch.zeros(2, N))
            velocities = 0.1 * torch.tensor([[np.sin(2 * np.pi * x / grid_size),
                                              np.sin(2 * np.pi * y / grid_size)] for x, y in g.nodes()]).T
            print(velocities.min(), velocities.max())
            print(velocities)
        edge_velocities = {(u, v): (normals[i] * 0.5 * (velocities[:, node_dict[u]] + velocities[:, node_dict[v]])).sum()
                           for i, (u, v) in enumerate(g.edges())}
        nx.set_edge_attributes(g, edge_velocities, name='velocity')
        edge_velocities = torch.tensor(nx.adjacency_matrix(g, weight='velocity').todense())
        F_coeffs += advection_coeffs(edge_velocities)
        identity = False
    print(F_coeffs)
    if identity:
        print(f'use copy operator (identity) as dynamics')

    F = transition_matrix(F_coeffs)
    F = torch.linalg.matrix_power(F, n_transitions)

    # define joint distribution
    joint_F = construct_joint_F(F.to_sparse(), T)
    joint_Q = construct_joint_G(P_0.to_sparse(), P_t.to_sparse(), T)
    joint_G = (joint_Q @ (sparse_identity(N * T) - joint_F))
    print(joint_G.to_dense().min())
    mean = torch.zeros(N * T)

    # draw samples
    # states, precision_matrix, cov_matrix = utils.sample_GMRF(joint_G.to_dense(), mean)

    # generate observations and construct graph for each time step
    spatiotemporal_graphs = []
    spatial_graphs = []
    y = []
    data = []
    H = []
    states = []

    transition_noise_distr = MultivariateNormal(loc=torch.zeros(N), precision_matrix=P_t.transpose(0, 1) @ P_t)
    initial_noise_distr = MultivariateNormal(loc=torch.zeros(N), precision_matrix=P_0.transpose(0, 1) @ P_0)
    for t in range(T):
        # states_t = states[t * N:(t + 1) * N]
        if t == 0:
            states_t = mean[:N] + initial_noise_distr.sample()
        else:
            states_t = F @ states_t + transition_noise_distr.sample()

        # generate observations
        data_t, H_t = generate_observations(states_t, obs_noise_std, obs_ratio)
        y_t = mask_observations(data_t, H_t)
        mask_t = get_mask(H_t)
        observation_edges = H_t.to_sparse().coalesce().indices()

        # construct graph with spatial edges
        spatial_graphs.append(utils.assemble_graph_0(states_t, node_pos, grid_size, data_t, mask_t, edges,
                                         observation_edges, obs_noise_std))
        # spatial_graphs[-1]["latent", "spatial", "latent"].eigvals = utils.compute_eigenvalues(
        #     sp_graph_t["latent", "spatial", "latent"])
        # spatial_graphs.append(sp_graph_t)

        # construct graph with spatiotemporal edges
        if t > 0:
            spatiotemporal_graphs.append(utils.assemble_graph_t(states_t, node_pos, grid_size, data_t, mask_t,
                            transition_edges, edges, observation_edges, obs_noise_std, temporal_edge_attr=normals))
            # st_graph_t["latent", "spatial", "latent"].eigvals = utils.compute_eigenvalues(
            #     st_graph_t["latent", "spatial", "latent"])
            # spatiotemporal_graphs.append(st_graph_t)
            # print(sp_graph_t)
            # print(spatiotemporal_graphs[-1])
        else:
            spatiotemporal_graphs.append(spatial_graphs[-1])
        y.append(y_t)
        data.append(data_t)
        H.append(H_t.to_sparse())
        states.append(states_t)

    y = torch.cat(y, dim=0)
    data = torch.cat(data, dim=0)
    states = torch.cat(states, dim=0)
    H = utils.sparse_block_diag(*H)
    mask = get_mask(H.to_dense())
    R_inv = sparse_identity(H.size(0)) / (obs_noise_std**2)

    # compute true posterior
    joint_precision = joint_G.transpose(0, 1) @ joint_G
    posterior_prec, posterior_eta = utils.compute_posterior(joint_precision.to_dense(), mean, y, H, R_inv)
    posterior_mean = torch.inverse(posterior_prec) @ posterior_eta

    joint_edges = joint_G.coalesce().indices()
    joint_weights = joint_G.coalesce().values()
    joint_obs_edges = H.coalesce().indices()
    joint_graph = utils.assemble_joint_graph(states, grid_size, T, y, joint_edges, joint_obs_edges,
                                             obs_noise_std, joint_edge_attr=joint_weights)
    joint_graph['latent'].prior_mean = mean.unsqueeze(1)
    joint_graph['latent'].true_posterior_mean = posterior_mean

    spatial_graphs = ptg.data.Batch.from_data_list(spatial_graphs)
    spatiotemporal_graphs = ptg.data.Batch.from_data_list(spatiotemporal_graphs)
    if not advection == 'zero':
        spatiotemporal_graphs['latent'].velocities = velocities.T

    # print(spatiotemporal_graphs)

    # joint graph that is compatible with original dgmrf code
    graph_y = ptg.data.Data(edge_index=spatial_graphs["latent", "spatial", "latent"].edge_index, x=data.unsqueeze(1),
                            mask=mask, pos=node_pos.repeat(T, 1), T=T, grid_size=grid_size)
    graph_y.eigvals = utils.compute_eigenvalues(graph_y)

    graph_post_mean = graph_y.clone()
    graph_post_mean.x = posterior_mean

    return {"spatial_graphs": spatial_graphs,
            "spatiotemporal_graphs": spatiotemporal_graphs,
            "joint_graph": joint_graph,
            "graph_y": graph_y,
            "graph_post_true_mean": graph_post_mean,
            "velocities": velocities}


def time_func(func, *inputs):
    ts = perf_counter()
    output = func(*inputs)
    te = perf_counter()
    print(f'function {func.__name__} took {te - ts}s')
    return output


def plot_spatiotemporal(graphs, save_to=''):
    T = len(graphs)
    fig, ax = plt.subplots(1, T, figsize=(T * 8, 8))
    vmin = min([g['latent'].x.min() for g in graphs])
    vmax = max([g['latent'].x.max() for g in graphs])
    for t in range(T):
        graph = graphs[t]['latent']
        x_t = graph.x.reshape(graph.grid_size, graph.grid_size)
        img = ax[t].imshow(x_t, vmin=vmin, vmax=vmax)
        ax[t].axis('off')
        ax[t].set_title(f't = {t}', fontsize=30)

    cbar = fig.colorbar(img, ax=ax, shrink=0.6, aspect=10)
    cbar.ax.tick_params(labelsize=20)

    if os.path.isdir(save_to):
        fig.savefig(os.path.join(save_to, f'latent_states.png'), bbox_inches='tight')

def plot_spatial(graph, save_to=''):
    fig, ax = plt.subplots(figsize=(8, 8))
    x = graph['latent'].x.reshape(graph['latent'].grid_size, graph['latent'].grid_size)
    img = ax.imshow(x, vmin=x.min(), vmax=x.max())
    ax.axis('off')

    cbar = fig.colorbar(img, ax=ax, shrink=0.6, aspect=10)
    cbar.ax.tick_params(labelsize=20)

    if os.path.isdir(save_to):
        fig.savefig(os.path.join(save_to, f'latent_states.png'), bbox_inches='tight')

