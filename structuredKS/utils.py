import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import scipy.linalg as spl
import torch_geometric as ptg

def compute_posterior(precision, mean, y, H, R_inv):
    if mean.dim() == 1:
        mean = mean.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    eta = precision @ mean
    post_eta = eta + H.transpose(0, 1) @ R_inv @ y
    post_prec = precision + H.transpose(0, 1) @ R_inv @ H
    return post_prec, post_eta

def sample_GMRF(G, mean):
    prec = G.transpose(0, 1) @ G
    dist = MultivariateNormal(loc=mean, precision_matrix=prec)
    x = dist.sample().unsqueeze(1)
    return x, prec, dist.covariance_matrix

def assemble_graph_t(latent_states, latent_pos, grid_size, data,
                     spatial_edges, temporal_edges, observation_edges, observation_noise_std, **kwargs):

    graph = ptg.data.HeteroData()

    # latent node properties
    graph['latent'].x = latent_states
    graph['latent'].pos = latent_pos
    graph['latent'].grid_size = grid_size
    if 'covariates' in kwargs:
        graph['latent'].covariates = kwargs['covariates']

    # data node properties
    graph['data'].x = data
    graph['data'].noise_std = observation_noise_std

    # spatial graph
    graph['latent', 'spatial', 'latent'].edge_index = spatial_edges
    if 'spatial_edge_attr' in kwargs:
        graph['latent', 'spatial', 'latent'].edge_attr = kwargs['spatial_edge_attr']

    # temporal graph
    graph['latent', 'temporal', 'latent'].edge_index = temporal_edges
    if 'temporal_edge_attr' in kwargs:
        graph['latent', 'temporal', 'latent'].edge_attr = kwargs['temporal_edge_attr']

    # observation graph
    graph['latent', 'observation', 'data'].edge_index = observation_edges
    if 'observation_weights' in kwargs:
        graph['latent', 'observation', 'data'].edge_weights = kwargs['observation_weights']

    return graph

def assemble_joint_graph(latent_states, grid_size, T, data,
                     joint_edges, observation_edges, observation_noise_std, **kwargs):

    graph = ptg.data.HeteroData()

    # latent node properties
    graph['latent'].x = latent_states
    graph['latent'].grid_size = grid_size
    graph['latent'].T = T

    # data node properties
    graph['data'].x = data
    graph['data'].noise_std = observation_noise_std

    # joint graph
    graph['latent', 'precision', 'latent'].edge_index = joint_edges
    if 'joint_edge_attr' in kwargs:
        graph['latent', 'precision', 'latent'].edge_attr = kwargs['joint_edge_attr']

    # observation graph
    graph['latent', 'observation', 'data'].edge_index = observation_edges
    if 'observation_weights' in kwargs:
        graph['latent', 'observation', 'data'].edge_weights = kwargs['observation_weights']

    return graph


def assemble_graph_0(latent_states, latent_pos, grid_size, data,
                     spatial_edges, observation_edges, observation_noise_std, **kwargs):

    graph = ptg.data.HeteroData()

    # latent node properties
    graph['latent'].x = latent_states
    graph['latent'].pos = latent_pos
    graph['latent'].grid_size = grid_size
    if 'covariates' in kwargs:
        graph['latent'].covariates = kwargs['covariates']

    # data node properties
    graph['data'].x = data
    graph['data'].noise_std = observation_noise_std

    # spatial graph
    graph['latent', 'spatial', 'latent'].edge_index = spatial_edges
    if 'spatial_edge_attr' in kwargs:
        graph['latent', 'spatial', 'latent'].edge_attr = kwargs['spatial_edge_attr']

    # observation graph
    graph['latent', 'observation', 'data'].edge_index = observation_edges
    if 'observation_weights' in kwargs:
        graph['latent', 'observation', 'data'].edge_weights = kwargs['observation_weights']

    return graph

# Computes all eigenvalues of the diffusion weight matrix D^(-1)A
def compute_eigenvalues(graph):
    adj_matrix = ptg.utils.to_dense_adj(graph.edge_index)[0]
    node_degrees = ptg.utils.degree(graph.edge_index[0])

    adj_matrix_norm = adj_matrix / node_degrees.unsqueeze(1)
    adj_eigvals = spl.eigvals(adj_matrix_norm.cpu().numpy()).real

    return torch.tensor(adj_eigvals, dtype=torch.float32)

# Computes all eigenvalues of the adjacency matrix A
def compute_eigenvalues_A(graph):
    adj_matrix = ptg.utils.to_dense_adj(graph.edge_index)[0]
    adj_eigvals = spl.eigvals(adj_matrix.cpu().numpy()).real

    return torch.tensor(adj_eigvals, dtype=torch.float32)




def conv2matrix(kernel, img_shape, zero_padding=1):
    """
    Converts a convolution kernel to the corresponding Toeplitz matrix, using zero padding to retain original size.
    Note that when multiplying the flattened input image with the Toeplitz matrix, no explicit zero padding is required.
    :param kernel: (out_channels, in_channels, kernel_height, kernel_width)
    :param img_shape: (in_channels, img_height, img_width)
    :param zero_padding: width of zero padding
    :return: Toeplitz matrix M (out_channels * img_height * img_width, in_channels * img_height * img_width)
    """

    # TODO: write test case for this!

    assert img_shape[0] == kernel.shape[1]
    assert len(img_shape[1:]) == len(kernel.shape[2:])

    padded_img_shape = (img_shape[0], img_shape[1] + 2 * zero_padding, img_shape[2] + 2 * zero_padding)

    M = torch.zeros((kernel.shape[0],
                     *((torch.tensor(padded_img_shape[1:]) - torch.tensor(kernel.shape[2:])) + 1),
                     *padded_img_shape))

    for i in range(M.shape[1]):
        for j in range(M.shape[2]):
            M[:, i, j, :, i:i + kernel.shape[2], j:j + kernel.shape[3]] = kernel

    M = M.flatten(0, len(kernel.shape[2:])).flatten(1)

    pads = [zero_padding] * 4
    mask = F.pad(torch.ones(img_shape), pads).flatten()
    mask = mask.bool()
    M = M[:, mask]

    return M

def block2flat(block_matrix):

    """
    Flatten a block matrix with separate block dimensions
    :param block_matrix: (..., outer_height, outer_width, block_height, block_width)
    :return: flat_matrix (..., outer_height * block_height, outer_width * block_width)
    """

    h, w, bh, bw = block_matrix.shape[-4:]

    return block_matrix.transpose(-3, -2).reshape(-1, h * bh, w * bw)

def flat2block(flat_matrix, block_height, block_width):

    """
    Separate flat matrix into blocks
    :param flat_matrix: (..., outer_height * block_height, outer_width * block_width)
    :return: block_matrix (..., outer_height, outer_width, block_height, block_width)
    """

    assert flat_matrix.shape[-2] % block_height == 0 and flat_matrix.shape[-1] % block_width == 0

    outer_height = flat_matrix.shape[-2] // block_height
    outer_width = flat_matrix.shape[-1] // block_width

    return flat_matrix.reshape(-1, outer_height, outer_width, block_height, block_width).transpose(-3, -2)

def sparse_zero_padding(sparse_matrix, padding=[0, 0, 0, 0]):
    """
    adds zero padding to a sparse matrix
    :param sparse_matrix: torch.sparse_coo_tensor
    :param padding: number of rows/columns to pad with zeros on the top, bottom, left, right
    :return: padded sparse matrix
    """

    indices = sparse_matrix.coalesce().indices() + torch.tensor([[padding[0], padding[2]]]).T
    size = (sparse_matrix.size(0) + sum(padding[:2]), sparse_matrix.size(1) + sum(padding[2:]))
    padded_matrix = torch.sparse_coo_tensor(indices, sparse_matrix.coalesce().values(), size=size)

    return padded_matrix



def sparse_block_diag(*sparse_matrices, padding=[0, 0, 0, 0]):
    """
    combine sparse matrices into one big blockdiagonal matrix that can be used for batched sparse matrix multiplication
    :param sparse_matrices: list of torch.sparse_coo_tensor's
    :param padding: number of rows/columns to pad with zeros on the top, bottom, left, right
    :return: block diagonal torch.sparse_coo_tensor
    """

    values = torch.cat([a.coalesce().values() for a in sparse_matrices])
    dims = torch.tensor([a.size() for a in sparse_matrices]) # size (len(sparse_matrices), 2)

    indices = torch.cat([sparse_matrices[idx].coalesce().indices() +
                         torch.tensor([dims[:idx, 0].sum() + padding[0], dims[:idx, 1].sum() + padding[2]]).unsqueeze(1)
                         for idx in torch.arange(len(sparse_matrices))], dim=1)
    size = (dims[:, 0].sum() + sum(padding[:2]), dims[:, 1].sum() + sum(padding[2:]))
    block_diag = torch.sparse_coo_tensor(indices, values, size=size)

    return block_diag

def sparse_block_diag_repeat(sparse_matrix, n_repeat, padding=[0, 0, 0, 0]):
    """
    construct big blockdiagonal matrix by repeating the same matrix n_repeat times
    :param sparse_matrices: list of torch.sparse_coo_tensor's
    :param n_repeat: number of repetitions
    :param padding: number of rows/columns to pad with zeros on the top, bottom, left, right
    :return: block diagonal torch.sparse_coo_tensor
    """

    values = sparse_matrix.coalesce().values().repeat(n_repeat)
    size = sparse_matrix.size()

    indices = torch.cat([sparse_matrix.coalesce().indices() +
                         torch.tensor([r * size[0] + padding[0], r * size[1] + padding[2]]).unsqueeze(1)
                         for r in torch.arange(n_repeat)], dim=1)
    size = (size[0] * n_repeat + sum(padding[:2]), size[1] * n_repeat + sum(padding[2:]))
    block_diag = torch.sparse_coo_tensor(indices, values, size=size)

    return block_diag


def reverse_nested_block_diag(nested_block_matrix, outer_shape, inner_shape, padding=[0, 0, 0, 0]):
    cm = nested_block_matrix.coalesce()

    n_outer = int(cm.size(0) / outer_shape[0])
    n_inner = int(outer_shape[0] / inner_shape[0])
    new_outer_shape = n_outer * torch.tensor(inner_shape)
    new_outer_shape[0] = new_outer_shape[0] + sum(padding[:2])
    new_outer_shape[1] = new_outer_shape[1] + sum(padding[2:])

    b_outer = torch.div(cm.indices()[0], outer_shape[0], rounding_mode='floor')
    b_inner = torch.div(cm.indices()[0] - b_outer * outer_shape[0],
                        inner_shape[0], rounding_mode='floor')

    base_indices = cm.indices() - b_outer * torch.tensor(outer_shape).unsqueeze(1) - b_inner * torch.tensor(
        inner_shape).unsqueeze(1)

    new_indices = base_indices + b_inner * new_outer_shape.unsqueeze(1) \
                               + b_outer * torch.tensor(inner_shape).unsqueeze(1) \
                               + torch.tensor([padding[0], padding[2]]).unsqueeze(1)

    size = (n_inner * int(new_outer_shape[0]), n_inner * int(new_outer_shape[1]))
    new = torch.sparse_coo_tensor(new_indices, cm.values(), size=size).coalesce()

    return new

def sparse_cat(*sparse_matrices, dim=0):
    """
    concatenate sparse matrices into one big sparse matrix
    :param sparse_matrices: list of torch.sparse_coo_tensor's
    :param dim: dimension along which matrices are concatenated
    :return: concatenation torch.sparse_coo_tensor
    """

    other_dim = abs(dim-1)
    N = sparse_matrices[0].size(other_dim)
    assert torch.all(torch.tensor([m.size(other_dim) == N for m in sparse_matrices]))

    values = torch.cat([a.coalesce().values() for a in sparse_matrices])
    dims = torch.tensor([a.size(dim) for a in sparse_matrices])

    indices = torch.cat([m.coalesce().indices() + dims[:idx].sum() * torch.tensor([[other_dim, dim]]).T
                         for idx, m in enumerate(torch.arange(len(sparse_matrices)))], dim=1)

    size = (other_dim * dims.sum() + dim * N, dim * dims.sum() + other_dim * N)
    result = torch.sparse_coo_tensor(indices, values, size=size)

    return result


def construct_U(*transition_matrices):
    T = len(transition_matrices) + 1
    N = transition_matrices[0].size(0)
    ones = torch.ones(N)

    indices = torch.stack([torch.arange(T * N), torch.arange(T * N)])
    U = torch.sparse_coo_tensor(indices, -torch.ones(T * N))

    for t, F in enumerate(transition_matrices):
        indices = [[t * N + k for k in range(N)], [k for k in range(N)]]
        proj_l = torch.sparse_coo_tensor(indices, ones, size=(T * N, N))

        indices = [[k for k in range(N)], [(t + 1) * N + k for k in range(N)]]
        proj_r = torch.sparse_coo_tensor(indices, ones, size=(N, T * N))

        A = torch.sparse.mm(proj_l, F)
        U = U + torch.sparse.mm(A, proj_r)
    return U


def construct_Omega(precision_matrices, transition_matrices):
    T = len(precision_matrices)
    N = precision_matrices[0].size(0)
    ones = torch.ones(N)

    omega = torch.sparse_coo_tensor(size=(T * N, T * N))

    for t in range(T - 1):
        # off-diagonal blocks
        indices = [[(t + 1) * N + k for k in range(N)], [k for k in range(N)]]
        proj_l = torch.sparse_coo_tensor(indices, ones, size=(T * N, N))

        indices = [[k for k in range(N)], [t * N + k for k in range(N)]]
        proj_r = torch.sparse_coo_tensor(indices, ones, size=(N, T * N))

        offdiag_block = -torch.sparse.mm(precision_matrices[t + 1], transition_matrices[t])
        offdiag = torch.sparse.mm(torch.sparse.mm(proj_l, offdiag_block), proj_r)

        # diagonal blocks
        diag_block = precision_matrices[t] + torch.sparse.mm(transition_matrices[t], -offdiag_block)
        diag = torch.sparse.mm(torch.sparse.mm(proj_r.transpose(0, 1), diag_block), proj_r)

        omega = omega + diag + offdiag + offdiag.transpose(0, 1)

    # last time slice
    indices = [[k for k in range(N)], [(T - 1) * N + k for k in range(N)]]
    proj = torch.sparse_coo_tensor(indices, ones, size=(N, T * N))
    diag = torch.sparse.mm(torch.sparse.mm(proj.transpose(0, 1), precision_matrices[-1]), proj)
    omega = omega + diag

    return omega
