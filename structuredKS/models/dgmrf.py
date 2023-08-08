import torch
from torch.nn.parameter import Parameter
import torch_geometric as ptg
import copy
import numpy as np
import scipy.stats as sps
from structuredKS import cg_batch
import pytorch_lightning as pl
from timeit import default_timer as timer

from structuredKS.utils import crps_score, int_score
from structuredKS.models.KS import KalmanSmoother



# def new_graph(like_graph, new_x=None):
#     graph = copy.copy(like_graph) # Shallow copy
#     graph.x = new_xt
#     return graph

def get_num_nodes(edge_index):
    return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0


class DGMRFActivation(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight_param = torch.nn.parameter.Parameter(
                2*torch.rand(1) - 1.) # U(-1,1)

        # For log-det
        self.n_training_samples = config["n_training_samples"]
        self.last_input = None

    @property
    def activation_weight(self):
        return torch.nn.functional.softplus(self.weight_param)

    def forward(self, x, edge_index, transpose, with_bias):
        self.last_input = x.detach()
        return torch.nn.functional.prelu(x, self.activation_weight)

    def log_det(self):
        # Computes log-det for last input fed to forward
        n_negative = (self.last_input < 0.).sum().to(torch.float32)
        return (1./self.n_training_samples)*n_negative*torch.log(self.activation_weight)


class DGMRFLayer(ptg.nn.MessagePassing):
    def __init__(self, config, graph, vi_layer=False):
        super(DGMRFLayer, self).__init__(aggr="add", node_dim=-1)

        self.edge_index = graph['edge_index']
        self.num_nodes = get_num_nodes(self.edge_index)
        self.degrees = ptg.utils.degree(self.edge_index[0])

        self.alpha1_param = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)
        self.alpha2_param = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)

        if config["use_bias"]:
            self.bias = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)
        else:
            self.bias = None

        if config["log_det_method"] == "eigvals":
            assert 'eigvals' in graph, ("Dataset not pre-processed with eigenvalues")
            self.adj_eigvals = graph['eigvals']
            self.eigvals_log_det = True
        elif config["log_det_method"] == "dad":
            assert 'dad_traces' in graph, (
                "Dataset not pre-processed with DAD traces")
            dad_traces = graph['dad_traces']

            # Complete vector to use in power series for log-det-computation
            k_max = len(dad_traces)
            self.power_ks = torch.arange(k_max) + 1
            self.power_series_vec = (dad_traces * torch.pow(-1., (self.power_ks + 1))
                                     ) / self.power_ks
        else:
            assert False, "Unknown log-det method"

        self.log_degrees = torch.log(self.degrees)
        self.sum_log_degrees = torch.sum(self.log_degrees)  # For determinant

        # Degree weighting parameter (can not be fixed for vi)
        self.fixed_gamma = (not vi_layer) and bool(config["fix_gamma"])
        if self.fixed_gamma:
            self.gamma_param = config["gamma_value"] * torch.ones(1)
        else:
            self.gamma_param = torch.nn.parameter.Parameter(2. * torch.rand(1, ) - 1)

        # edge_log_degrees contains log(d_i) of the target node of each edge
        self.edge_log_degrees = self.log_degrees[self.edge_index[1]]
        self.edge_log_degrees_transpose = self.log_degrees[self.edge_index[0]]

    @property
    def degree_power(self):
        if self.fixed_gamma:
            return self.gamma_param
        else:
            # Forcing gamma to be in (0,1)
            return torch.sigmoid(self.gamma_param)

    @property
    def self_weight(self):
        # Forcing alpha1 to be positive is no restriction on the model
        return torch.exp(self.alpha1_param)

    @property
    def neighbor_weight(self):
        # Second parameter is (alpha2 / alpha1)
        return self.self_weight * torch.tanh(self.alpha2_param)

    # def weight_self_representation(self, x):
    #     # Representation of same node weighted with degree (taken to power)
    #     return (x.view(-1, self.num_nodes) * torch.exp(
    #         self.degree_power * self.log_degrees)).view(-1, 1)

    def weight_self_representation(self, x):
        # Representation of same node weighted with degree (taken to power)
        # x has shape [..., num_nodes]
        return x * torch.exp(self.degree_power * self.log_degrees)

    def forward(self, x, transpose, with_bias):
        # x has shape [num_nodes, n_samples]
        weighted_repr = self.weight_self_representation(x) # shape [..., num_nodes]

        aggr = (self.self_weight * weighted_repr) + (self.neighbor_weight * self.propagate(
            self.edge_index, x=x, transpose=transpose)) # Shape [..., num_nodes]

        if self.bias and with_bias:
            aggr += self.bias

        return aggr

    def message(self, x_j, transpose):
        # x_j are neighbor features
        if transpose:
            log_degrees = self.edge_log_degrees_transpose
        else:
            log_degrees = self.edge_log_degrees

        edge_weights = torch.exp((self.degree_power - 1) * log_degrees)
        # if self.dist_weighted:
        #     edge_weights = edge_weights * self.dist_edge_weights

        # weighted_messages = x_j.view(-1, edge_weights.shape[0]) * edge_weights
        weighted_messages = x_j * edge_weights

        return weighted_messages

    def log_det(self):
        if self.eigvals_log_det:
            # Eigenvalue-based method
            eigvals = self.neighbor_weight[0] * self.adj_eigvals + self.self_weight[0]
            agg_contrib = torch.sum(torch.log(torch.abs(eigvals)))  # from (aI+aD^-1A)
            degree_contrib = self.degree_power * self.sum_log_degrees  # From D^gamma
            return agg_contrib + degree_contrib
        else:
            # Power series method, using DAD traces
            alpha_contrib = self.num_nodes * self.alpha1_param
            gamma_contrib = self.degree_power * self.sum_log_degrees
            dad_contrib = torch.sum(self.power_series_vec * \
                                    torch.pow(torch.tanh(self.alpha2_param), self.power_ks))
            return alpha_contrib + gamma_contrib + dad_contrib


class DGMRFLayerMultiChannel(ptg.nn.MessagePassing):
    def __init__(self, config, graph, vi_layer=False, T=1, shared='all', weighted=False):
        super(DGMRFLayerMultiChannel, self).__init__(aggr="add", node_dim=-1)

        self.edge_index = graph['edge_index']
        self.num_nodes = get_num_nodes(self.edge_index)
        self.T = T
        self.shared = shared

        if weighted:
            self.edge_weights = graph.get('edge_weight', torch.ones(self.edge_index.size(1)))
            self.degrees = graph.get('weighted_degrees', ptg.utils.degree(self.edge_index[0], num_nodes=self.num_nodes))
        else:
            self.edge_weights = torch.ones(self.edge_index.size(1))
            self.degrees = ptg.utils.degree(self.edge_index[0], num_nodes=self.num_nodes)

        print(f'edge weights = {self.edge_weights}')


        if self.shared == 'dynamics':
            # same parameters for dynamics transition noise
            self.alpha1_param = torch.nn.parameter.Parameter(2.*torch.rand(2, 1) - 1)
            self.alpha2_param = torch.nn.parameter.Parameter(2.*torch.rand(2, 1) - 1)
        elif self.shared == 'all':
            # same parameters for all time steps
            self.alpha1_param = torch.nn.parameter.Parameter(2. * torch.rand(1, 1) - 1)
            self.alpha2_param = torch.nn.parameter.Parameter(2. * torch.rand(1, 1) - 1)
        else:
            # different parameters for all time steps
            self.alpha1_param = torch.nn.parameter.Parameter(2.*torch.rand(self.T, 1) - 1)
            self.alpha2_param = torch.nn.parameter.Parameter(2.*torch.rand(self.T, 1) - 1)

        if config["use_bias"]:
            if self.shared == 'dynamics':
                # spatial bias is zero for t > 0
                self.bias_param = torch.nn.parameter.Parameter(2.*torch.rand(1,) - 1)
            else:
                self.bias_param = torch.nn.parameter.Parameter(2.*torch.rand(self.alpha1_param.size()) - 1)
        else:
            self.bias_param = None

        if config["log_det_method"] == "eigvals":
            assert 'eigvals' in graph, (
                "Dataset not pre-processed with eigenvalues")
            self.adj_eigvals = graph.get('weighted_eigvals', graph['eigvals']) if weighted else graph['eigvals']
            self.eigvals_log_det = True
        elif config["log_det_method"] == "dad":
            assert 'dad_traces' in graph, (
                "Dataset not pre-processed with DAD traces")
            dad_traces = graph.get('weighted_dad_traces', graph['dad_traces']) if weighted else graph['dad_traces']

            # Complete vector to use in power series for log-det-computation
            k_max = len(dad_traces)
            self.power_ks = torch.arange(k_max) + 1 # [k_max]
            self.power_series_vec = (dad_traces * torch.pow(-1., (self.power_ks + 1))
                                     ) / self.power_ks # [k_max]
        else:
            assert False, "Unknown log-det method"

        self.log_degrees = torch.log(self.degrees).unsqueeze(0)
        self.sum_log_degrees = torch.sum(self.log_degrees)  # For determinant

        # Degree weighting parameter (can not be fixed for vi)
        self.fixed_gamma = (not vi_layer) and bool(config["fix_gamma"])
        if self.fixed_gamma:
            self.gamma_param = config["gamma_value"] * torch.ones(self.T, 1)
        else:
            self.gamma_param = torch.nn.parameter.Parameter(2. * torch.rand(self.alpha1_param.size()) - 1)

        # edge_log_degrees contains log(d_i) of the target node of each edge
        self.edge_log_degrees = self.log_degrees[:, self.edge_index[1]]
        self.edge_log_degrees_transpose = self.log_degrees[:, self.edge_index[0]]

    @property
    def gamma(self):
        if self.shared == 'dynamics':
            return torch.cat([self.gamma_param[0].unsqueeze(0),
                              self.gamma_param[1].unsqueeze(0).repeat(self.T - 1, 1)], dim=0)
        else:
            return self.gamma_param

    @property
    def alpha1(self):
        if self.shared == 'dynamics':
            # use same parameters for time steps 1,...,T
            return torch.cat([self.alpha1_param[0].unsqueeze(0),
                              self.alpha1_param[1].unsqueeze(0).repeat(self.T - 1, 1)], dim=0)
        else:
            return self.alpha1_param

    @property
    def alpha2(self):
        if self.shared == 'dynamics':
            # use same parameters for time steps 1,...,T
            return torch.cat([self.alpha2_param[0].unsqueeze(0),
                              self.alpha2_param[1].unsqueeze(0).repeat(self.T - 1, 1)], dim=0)
        else:
            return self.alpha1_param

    @property
    def bias(self):
        if self.shared == 'dynamics':
            # return torch.cat([self.bias_param[0].unsqueeze(0),
            #                   self.bias_param[1].unsqueeze(0).repeat(self.T - 1, 1)], dim=0)
            return torch.cat([self.bias_param.unsqueeze(0), torch.zeros(self.T - 1, 1)], dim=0)
        # else:
        #if self.shared == 'dynamics':
            # use same parameters for time steps 1,...,T
        #    return torch.cat([self.bias_param[0].unsqueeze(0),
        #                      self.bias_param[1].unsqueeze(0).repeat(self.T - 1, 1)], dim=0)
        else:
            return self.bias_param

    @property
    def degree_power(self):
        if self.fixed_gamma:
            return self.gamma_param
        else:
            # Forcing gamma to be in (0,1)
            return torch.sigmoid(self.gamma)

    @property
    def self_weight(self):
        # Forcing alpha1 to be positive is no restriction on the model
        return torch.exp(self.alpha1)

    @property
    def neighbor_weight(self):
        # Second parameter is (alpha2 / alpha1)
        return self.self_weight * torch.tanh(self.alpha2)

    def weight_self_representation(self, x):
        # Representation of same node weighted with degree (taken to power)
        # x has shape [..., num_nodes]
        # [..., T, num_nodes] * [T, 1] * [1, num_nodes]
        return x * torch.exp(self.degree_power * self.log_degrees)

    def forward(self, x, transpose, with_bias):
        # x has shape [..., T, num_nodes]
        weighted_repr = self.weight_self_representation(x) # shape [..., T, num_nodes]

        # [T, 1] * [..., T, num_nodes] + ([channels, 1] * [..., T, num_nodes]
        aggr = (self.self_weight * weighted_repr) + (self.neighbor_weight * self.propagate(
            self.edge_index, x=x, transpose=transpose)) # Shape [..., num_nodes]
        # TODO: fix this for no bias case
        if with_bias and self.bias_param is not None:
            aggr += self.bias

        return aggr

    def message(self, x_j, transpose):
        # x_j are neighbor features of size [..., T, num_edges]
        if transpose:
            log_degrees = self.edge_log_degrees_transpose
        else:
            log_degrees = self.edge_log_degrees

        # shape [channels, num_edges]
        edge_weights = torch.exp((self.degree_power - 1) * log_degrees) * self.edge_weights.unsqueeze(0)
        # if self.dist_weighted:
        #     edge_weights = edge_weights * self.dist_edge_weights

        # weighted_messages = x_j.view(-1, edge_weights.shape[0]) * edge_weights
        # [..., T, num_edges] * [T, num_edges]
        weighted_messages = x_j * edge_weights

        return weighted_messages

    def log_det(self):
        if self.eigvals_log_det:
            # Eigenvalue-based method
            # [T, 1] * [1, n_eigvals] + [T, 1]
            eigvals = self.neighbor_weight * self.adj_eigvals.unsqueeze(0) + self.self_weight # [T, n_eigvals]
            agg_contrib = torch.log(torch.abs(eigvals)).sum(1).unsqueeze(1)  # [T, 1]
            degree_contrib = self.degree_power * self.sum_log_degrees  # From D^gamma
            channel_log_dets = agg_contrib + degree_contrib
        else:
            # Power series method, using DAD traces
            alpha_contrib = self.num_nodes * self.alpha1 # [channels, 1]
            gamma_contrib = self.degree_power * self.sum_log_degrees # [T, 1]
            dad_contrib = (self.power_series_vec.unsqueeze(0) * \
                                    torch.pow(torch.tanh(self.alpha2), self.power_ks)).sum(1).unsqueeze(1)
            channel_log_dets = alpha_contrib + gamma_contrib + dad_contrib

        if self.shared == 'all':
            # same log det for all time points
            return channel_log_dets.sum() * self.T
        else:
            return channel_log_dets.sum()




class DGMRF(torch.nn.Module):
    def __init__(self, config, graph, T=1, shared='all', weighted=False):
        super(DGMRF, self).__init__()

        self.edge_index = graph['edge_index']
        self.num_nodes = get_num_nodes(self.edge_index)
        self.T = T

        layer_list = []
        for layer_i in range(config["n_layers"]):
            layer_list.append(DGMRFLayerMultiChannel(config, graph, vi_layer=False, T=T,
                                                     shared=shared, weighted=weighted))

            # Optionally add non-linearities between hidden layers
            # TODO: make this work for multi channel case
            # if config["non_linear"] and (layer_i < (config["n_layers"]-1)):
            #     layer_list.append(DGMRFActivation(config))

        self.layers = torch.nn.ModuleList(layer_list)

    def forward(self, x, transpose=False, with_bias=True, **kwargs):

        if transpose:
            # Transpose operation means reverse layer order
            layer_iter = reversed(self.layers)
        else:
            layer_iter = self.layers

        for layer in layer_iter:
            x = layer(x, transpose, with_bias)

        return x

    def log_det(self):
        return sum([layer.log_det() for layer in self.layers])

    def get_matrices(self):
        input = torch.eye(self.num_nodes).reshape(self.num_nodes, 1, self.num_nodes).repeat(1, self.T, 1)
        print('get_matrices', input.size())
        all_Q = self.forward(input, transpose=True, with_bias=False)

        Q_0 = all_Q[:, 0].squeeze()
        Q_t = all_Q[:, 1].squeeze()

        return Q_0, Q_t

    def get_inv_matrices(self):
        Q_0, Q_t = self.get_matrices()

        Q_0_inv = torch.inverse(Q_0)
        Q_t_inv = torch.inverse(Q_t)

        return Q_0_inv, Q_t_inv


class DiagonalModel(torch.nn.Module):
    def __init__(self, config, graph, T=1, shared='all', **kwargs):
        super(DiagonalModel, self).__init__()

        self.edge_index = graph['edge_index']
        self.num_nodes = get_num_nodes(self.edge_index)
        self.T = T

        self.shared = shared

        self.precision_param = Parameter(torch.rand(2, 1) if self.shared == 'dynamics' else torch.rand(1))
        self.bias_param = Parameter(torch.rand(2, 1) * 2 - 1 if self.shared == 'dynamics' else torch.rand(1) * 2 - 1)

    @property
    def precision(self):
        if self.shared == 'dynamics':
            # use same parameters for time steps 1,...,T
            return torch.cat([self.precision_param[0], self.precision_param[1].repeat(self.T - 1)]).unsqueeze(-1)
        else:
            # use same parameters for all time steps 0, 1, ..., T
            return self.precision_param.repeat(self.T).unsqueeze(-1)

    @property
    def bias(self):
        if self.shared == 'dynamics':
            # use same parameters for time steps 1,...,T
            return torch.cat([self.bias_param[0], self.bias_param[1].repeat(self.T - 1)]).unsqueeze(-1)
        else:
            # use same parameters for all time steps 0, 1, ..., T
            return self.bias_param.repeat(self.T).unsqueeze(-1)

    def forward(self, x, with_bias=True, **kwargs):
        # x has shape [..., T, num_nodes]

        x = self.precision * x
        if with_bias:
            x = x + self.bias

        return x

    def log_det(self):
        return self.num_nodes * self.precision.sum()

    def get_matrices(self):

        Q_0 = torch.eye(self.num_nodes) * self.precision[0]
        Q_t = torch.eye(self.num_nodes) * self.precision[-1]

        return Q_0, Q_t

    def get_inv_matrices(self):

        Q_0_inv = torch.eye(self.num_nodes) / self.precision[0]
        Q_t_inv = torch.eye(self.num_nodes) / self.precision[-1]

        return Q_0_inv, Q_t_inv


class JointDGMRF(torch.nn.Module):
    def __init__(self, config, spatial_graph, temporal_graph=None, T=1, shared='dynamics',
                 weighted=False, features=None):
        super().__init__()

        if config.get('diag_noise_model', False):
            self.dgmrf = DiagonalModel(config, spatial_graph, T=T, shared=shared)
        else:
            self.dgmrf = DGMRF(config, spatial_graph, T=T, shared=shared, weighted=weighted)
        self.T = T

        if temporal_graph is not None:
            self.dynamics = torch.nn.ModuleList([
                TemporalDGMRF(config, spatial_graph, temporal_graph, T=T, shared=shared, features=features)
                for _ in range(config.get('n_layers_temporal', 1))])

    def forward(self, x, transpose=False, with_bias=True, **kwargs):
        # x has shape [num_samples, T, num_nodes]
        out = x
        if hasattr(self, 'dynamics') and not transpose:
            for p, layer in enumerate(self.dynamics):
                out = out - layer(x, with_bias=with_bias, p=p+1) #v=kwargs.get('v', None))

        z = self.dgmrf(out, transpose=transpose, with_bias=with_bias)

        out = z

        if hasattr(self, 'dynamics') and transpose:
            # for layer in reversed(self.dynamics):
            for p, layer in enumerate(self.dynamics):
                out = out - layer(z, transpose=True, with_bias=with_bias, p=p+1) #v=kwargs.get('v', None))

        return out

    def get_matrices(self):

        return self.dgmrf.get_matrices()

    def get_inv_matrices(self):

        return self.dgmrf.get_inv_matrices()

    def get_transition_matrix(self, p=1):
        assert len(self.dynamics) >= p

        return self.dynamics[p-1].get_matrix()

    def log_det(self):
        return self.dgmrf.log_det()


class DirectedDiffusionModel(ptg.nn.MessagePassing):
    """
    Compute x_{t+1} = Fx_t, where F is based on discretized diffusion

    (for now we assume a regular lattice with cell_size = 1x1)
    """

    def __init__(self, config, graph):
        super(DirectedDiffusionModel, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.K = config.get('diff_K', 1)
        print(f'K = {self.K}')
        self.diff_param_forward = torch.nn.Parameter(2 * torch.rand(self.K,) - 1)
        self.diff_param_backward = torch.nn.Parameter(2 * torch.rand(self.K, ) - 1)
        self.edge_index = graph['edge_index']
        self.edge_index_backward = self.edge_index.flip(0)
        self.edge_weights = graph.get('edge_weight', torch.ones(self.edge_index.size(1)))


    @property
    def diff_coef_forward(self):
        # force diffusion coefficient to be positive
        # return torch.exp(self.diff_param_forward)
        return self.diff_param_forward

    @property
    def diff_coef_backward(self):
        # force diffusion coefficient to be positive
        # return torch.exp(self.diff_param_backward)
        return self.diff_param_backward


    def forward(self, x, transpose=False, **kwargs):

        # k=0 (i.e. identity map)
        x_k = x
        out = x
        sign = 1

        for k in range(self.K):
            # TODO: normalize by (weighted) degree?
            Dx_k, Ax_k = self.propagate(self.edge_index, x=x_k, edge_weight=self.edge_weights)
            DTx_k, ATx_k = self.propagate(self.edge_index_backward, x=x_k, edge_weight=self.edge_weights)

            if transpose:
                # _, ATx_k = self.propagate(self.edge_index_backward, x=x)
                # x_k = Dx_k - ATx_k
                x_k_forward = Dx_k - ATx_k
                x_k_backward = DTx_k - Ax_k
            else:
                x_k_forward = Dx_k - Ax_k
                x_k_backward = DTx_k - ATx_k


            # add k-th term: diff_k * (-1)^k * Lx_k = diff_k * (-1)^k (D - A)x_k
            sign = -sign
            out = out + sign * (self.diff_coef_forward[k] * x_k_forward + self.diff_coef_backward[k] * x_k_backward)

        return out

    def message(self, x_i, x_j, edge_weight):
        # construct messages to node i for each edge (j,i)

        # diffusion from self
        # msg_i = self.diff_coef * x_i
        # diffusion from neighbours
        # msg_j = self.diff_coef * x_j

        # msg = self.diff_coef * (x_j - x_i)
        # TODO: adjust if graph has edge weights
        return edge_weight * torch.stack([x_i, x_j], dim=0)

class DiffusionModel(ptg.nn.MessagePassing):
    """
    Compute x_{t+1} = Fx_t, where F is based on discretized diffusion

    (for now we assume a regular lattice with cell_size = 1x1)
    """

    def __init__(self, config, graph):
        super(DiffusionModel, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.K = config.get('diff_K', 1)
        print(f'K = {self.K}')
        self.diff_param = torch.nn.Parameter(2 * torch.rand(self.K,) - 1)
        self.ar_weight = torch.nn.Parameter(torch.zeros(1,))
        self.edge_index = graph['edge_index']
        self.edge_index_backward = self.edge_index.flip(0)
        self.edge_weights = graph.get('edge_weight', torch.ones(self.edge_index.size(1)))


    @property
    def diff_coef(self):
        # force diffusion coefficient to be positive
        return self.diff_param
    #
    # @property
    # def diff_coef_backward(self):
    #     # force diffusion coefficient to be positive
    #     # return torch.exp(self.diff_param_backward)
    #     return self.diff_param_backward


    def forward(self, x, transpose=False, **kwargs):

        # k=0 (i.e. identity map)
        x_k = x
        out = self.ar_weight * x
        sign = 1

        for k in range(self.K):
            # TODO: normalize by (weighted) degree?
            Dx_k, Ax_k = self.propagate(self.edge_index, x=x_k, edge_weight=self.edge_weights)
            DTx_k, ATx_k = self.propagate(self.edge_index_backward, x=x_k, edge_weight=self.edge_weights)

            if transpose:
                # _, ATx_k = self.propagate(self.edge_index_backward, x=x)
                # x_k = Dx_k - ATx_k
                x_k = Dx_k - ATx_k
            else:
                x_k = Dx_k - Ax_k

            # add k-th term: diff_k * (-1)^k * Lx_k = diff_k * (-1)^k (D - A)x_k
            sign = -sign
            out = out + sign * self.diff_coef[k] * x_k

        return out

    def message(self, x_i, x_j, edge_weight):
        # construct messages to node i for each edge (j,i)
        return edge_weight * torch.stack([x_i, x_j], dim=0)


class FlowModel(ptg.nn.MessagePassing):
    """
    Compute x_{t+1} = Fx_t, where F is based on discretized diffusion

    (for now we assume a regular lattice with cell_size = 1x1)
    """

    def __init__(self, config, graph):
        super(FlowModel, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.param_self = torch.nn.Parameter(torch.ones(1,))
        # self.param_flow = torch.nn.Parameter(2 * torch.rand(1,) - 1)
        # self.param_backward = torch.nn.Parameter(2 * torch.rand(1, ) - 1)
        self.edge_index = graph['edge_index']
        self.edge_index_backward = self.edge_index.flip(0)
        self.edge_weights = graph.get('edge_weight', torch.ones(self.edge_index.size(1)))
        self.edge_attr = graph.get('edge_attr', torch.ones(self.edge_index.size(1), 1))

        # self.edge_mlp = torch.nn.Sequential(torch.nn.Linear(self.edge_attr.size(1), 1, bias=False))#,
        #                                     #torch.nn.Sigmoid())
        self.edge_mlp = torch.nn.Sequential(torch.nn.Linear(self.edge_attr.size(1), 10, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(10, 1))


    # @property
    # def coef_self(self):
    #     # force diffusion coefficient to be positive
    #     return torch.exp(self.param_forward)

    # @property
    # def coef_flow(self):
    #     # force diffusion coefficient to be positive
    #     return torch.exp(self.param_flow)

    # @property
    # def coef_backward(self):
    #     # force diffusion coefficient to be positive
    #     return torch.exp(self.param_backward)


    def forward(self, x, transpose=False, **kwargs):

        _, A_in_x = self.propagate(self.edge_index, x=x, edge_weight=self.edge_weights, edge_attr=self.edge_attr)
        D_out_x, A_out_x = self.propagate(self.edge_index_backward, x=x, edge_weight=self.edge_weights,
                                          edge_attr=self.edge_attr)

        if transpose:
            out = self.param_self * x + A_out_x - D_out_x
        else:
            out = self.param_self * x + A_in_x - D_out_x

        return out

    def message(self, x_i, x_j, edge_weight, edge_attr):
        # construct messages to node i for each edge (j,i)
        weights = self.edge_mlp(edge_attr).reshape(1, 1, -1)

        # return torch.stack([edge_weight * x_i, edge_weight * x_j], dim=0)
        return torch.stack([weights * x_i, weights * x_j], dim=0)

class ARModel(torch.nn.Module):
    """
    Compute x_{t+1} = theta * x_t
    """

    def __init__(self):
        super(ARModel, self).__init__()

        self.weight = torch.nn.Parameter(torch.ones(1,))
        
    def forward(self, x, **kwargs):

        return self.weight * x


class ARModelMultiChannel(torch.nn.Module):
    """
    Compute x_{t+1} = theta * x_t
    """

    def __init__(self, T=1):
        super(ARModelMultiChannel, self).__init__()

        self.weight = torch.nn.Parameter(torch.rand(1, T, 1))

    def forward(self, x, **kwargs):
        
        return self.weight * x
        


class GNNAdvection(ptg.nn.MessagePassing):
    def __init__(self, config, temporal_graph, n_features=0, **kwargs):
        super(GNNAdvection, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.edge_index = temporal_graph['edge_index']
        self.edge_index_transpose = self.edge_index.flip(0)
        edge_attr = temporal_graph['edge_attr'] #[:2] # normal vectors
        edge_weights = temporal_graph.get('edge_weight', torch.ones(edge_attr.size(0)))
        self.edge_features = torch.cat([edge_attr, edge_weights.unsqueeze(1)], dim=1).to(torch.float32)

        self.edge_dim = self.edge_features.size(1)
        self.n_features = n_features

        hidden_dim = config.get('GNN_hidden_dim', 8)

        self.edge_mlp = torch.nn.Sequential(torch.nn.Linear(self.edge_dim + n_features, hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, 2),
                                            torch.nn.Tanh())

        self.diff_param = torch.nn.Parameter(2 * torch.rand(1,) - 1)

    @property
    def diff_coeff(self):
        return torch.pow(self.diff_param, 2)

    def forward(self, x, transpose=False, features=None, **kwargs):
        # x has shape [num_samples, T, num_nodes]
        #print(f'features min = {features.min()}')
        if features is None:
            agg_i, agg_j = self.propagate(self.edge_index, x=x, edge_attr=self.edge_features,
                                          transpose=transpose)
        else:
            assert features.size(-1) == self.n_features
            agg_i, agg_j = self.propagate(self.edge_index, x=x, edge_attr=self.edge_features,
                                          node_attr=features.transpose(1, 2), transpose=transpose)

        if transpose:
            if features is None:
                _, aggT_j = self.propagate(self.edge_index_transpose, x=x, edge_attr=self.edge_features,
                                           transpose=transpose)
            else:
                assert features.size(-1) == self.n_features
                _, aggT_j = self.propagate(self.edge_index_transpose, x=x, edge_attr=self.edge_features,
                                           node_attr=features.transpose(1, 2), transpose=transpose)

            new_x = x + aggT_j + agg_i
        else:
            new_x = x + agg_j + agg_i

        
        # print(f'new x = {new_x}')

        return new_x


    def message(self, x_i, x_j, edge_attr, transpose, node_attr_i=None, node_attr_j=None):
        # if transpose:
        #       inputs = torch.cat([edge_attr, node_attr_j.squeeze(0).T, node_attr_i.squeeze(0).T], dim=-1)
        # else:
        #       inputs = torch.cat([edge_attr, node_attr_i.squeeze(0).T, node_attr_j.squeeze(0).T], dim=-1)
        if node_attr_i is None or node_attr_j is None:
            inputs = edge_attr
        else:
            edge_attr = edge_attr.unsqueeze(0).repeat(node_attr_i.size(0), 1, 1)
            #if transpose:
            #    inputs = torch.cat([edge_attr, node_attr_j.transpose(1, 2), node_attr_i.transpose(1, 2)], dim=-1)
            #else:
            #    inputs = torch.cat([edge_attr, node_attr_i.transpose(1, 2), node_attr_j.transpose(1, 2)], dim=-1)
            #if transpose:
            #    inputs = torch.cat([edge_attr, node_attr_i.transpose(1, 2)], dim=-1)
            #else:
            #    inputs = torch.cat([edge_attr, node_attr_i.transpose(1, 2)], dim=-1)
            inputs = torch.cat([edge_attr, node_attr_i.transpose(1, 2)], dim=-1)
        # inputs = torch.cat([inputs.reshape(1, 1, *inputs.shape).repeat(x_i.size(0), x_i.size(1), 1, 1),
        #                     input_states_i.unsqueeze(-1) + input_states_j.unsqueeze(-1)], dim=-1)

        #print(f'inputs min = {inputs.min()}, max = {inputs.max()}')
        coeffs = self.edge_mlp(inputs) #.squeeze(-1)

        #print(f'coeffs min = {coeffs.min()}, max = {coeffs.max()}')

        msg_i = (coeffs[..., 1] - self.diff_coeff) * x_i
        msg_j = (coeffs[..., 0] + self.diff_coeff) * x_j

        return torch.stack([msg_i, msg_j], dim=0)

class GNNTransition(ptg.nn.MessagePassing):
    def __init__(self, config, temporal_graph, n_features, **kwargs):
        super(GNNTransition, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.edge_index = temporal_graph['edge_index']
        self.edge_index_transpose = self.edge_index.flip(0)
        edge_attr = temporal_graph['edge_attr'] #[:2] # normal vectors
        edge_weights = temporal_graph.get('edge_weight', torch.ones(edge_attr.size(0)))
        self.edge_features = torch.cat([edge_attr, edge_weights.unsqueeze(1)], dim=1).to(torch.float32)

        self.edge_dim = self.edge_features.size(1)
        self.n_features = n_features

        hidden_dim = config.get('GNN_hidden_dim', 8)

        self.edge_mlp = torch.nn.Sequential(torch.nn.Linear(self.edge_dim + n_features, hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, 1),
                                            # torch.nn.Tanh()
                                            )

        # self.diff_param = torch.nn.Parameter(2 * torch.rand(1,) - 1)
    #
    # @property
    # def diff_coeff(self):
    #     return torch.pow(self.diff_param, 2)

    def forward(self, x, transpose=False, features=None):
        #assert features.size(-1) == self.n_features
        # x has shape [num_samples, T, num_nodes]
        # features has shape [T, num_nodes, num_features]
        edge_index = self.edge_index_transpose if transpose else self.edge_index
        
        if features is None:
            agg = self.propagate(edge_index, x=x, edge_attr=self.edge_features, transpose=transpose)
        else:
            agg = self.propagate(edge_index, x=x, node_attr=features.transpose(1, 2),
                             edge_attr=self.edge_features, transpose=transpose)

        new_x = x + agg # TODO: also use self-weight? i.e. F_ii != 1

        return new_x


    def message(self, x_j, edge_attr, node_attr_i=None, node_attr_j=None, transpose=False):
        # edge_attr has shape [num_edges, num_features]
        # covariates has shape [T, num_features, num_edges]

        #TODO: check if this works now
        if node_attr_i is None or node_attr_j is None:
            inputs = edge_attr.unsqueeze(0)
        else:
            edge_attr = edge_attr.unsqueeze(0).repeat(node_attr_i.size(0), 1, 1)
            inputs = torch.cat([edge_attr, (node_attr_j + node_attr_i).transpose(1, 2)], dim=-1)

        coeffs = self.edge_mlp(inputs).squeeze(-1)

        msg = coeffs * x_j

        return msg


class AdvectionDiffusionModel(ptg.nn.MessagePassing):
    """
    Compute x_{t+1} = Fx_t, where F is based on discretized advection

    (for now we assume a regular lattice with cell_size = 1x1)
    """

    def __init__(self, config, graph):
        super(AdvectionDiffusionModel, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.velocity = torch.nn.Parameter(2 * torch.rand(2, ) - 1)
        self.diff_param = torch.nn.Parameter(2 * torch.rand(1, ) - 1)
        self.edge_index = graph['edge_index']
        self.edge_index_transpose = self.edge_index.flip(0)
        self.edge_attr = graph['edge_attr'] # normal vectors at cell boundaries

    @property
    def diff_coeff(self):
        # force diffusion coefficient to be positive
        return torch.pow(self.diff_param, 2)

    def forward(self, x, transpose=False, **kwargs):
        agg_i, agg_j = self.propagate(self.edge_index, x=x, edge_attr=self.edge_attr)

        if transpose:
            agg_i_T, agg_j_T = self.propagate(self.edge_index_transpose, x=x, edge_attr=self.edge_attr)
            update = agg_j_T + agg_i
        else:
            update = agg_j + agg_i

        return x + update

    def message(self, x_i, x_j, edge_attr):
        # construct messages to node i for each edge (j,i)
        # edge_attr has shape [num_edges, 2]
        # velocity has shape [2]
        adv_coef = -0.5 * (edge_attr * self.velocity).sum(1)
        msg_i = (adv_coef - self.diff_coeff) * x_i
        msg_j = (adv_coef + self.diff_coeff) * x_j
        # msg = -0.5 * (edge_attr * self.velocity).sum(1) * (x_j + x_i) + self.diff_coeff * (x_j - x_i)
        return torch.stack([msg_i, msg_j], dim=0)

class InhomogeneousAdvectionDiffusionModel(ptg.nn.MessagePassing):
    """
    Compute x_{t+1} = Fx_t, where F is based on discretized advection and diffusion
    Advection varies over nodes.

    (for now we assume a regular lattice with cell_size = 1x1)
    """

    def __init__(self, config, graph):
        super(InhomogeneousAdvectionDiffusionModel, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.diff_param = torch.nn.Parameter(2 * torch.rand(1, ) - 1)
        self.edge_index = graph['edge_index']
        self.edge_index_transpose = self.edge_index.flip(0)
        self.edge_attr = graph['edge_attr']  # normal vectors at cell boundaries

    @property
    def diff_coeff(self):
        # force diffusion coefficient to be positive
        return torch.exp(self.diff_param)

    def forward(self, x, transpose=False, **kwargs):
        if transpose:
            edge_index = self.edge_index_transpose
        else:
            edge_index = self.edge_index

        return x + self.propagate(edge_index, x=x, v=kwargs.get('v'), edge_attr=self.edge_attr)

    def message(self, x_i, x_j, v_i, v_j, edge_attr):
        # construct messages to node i for each edge (j,i)
        # edge_attr has shape [num_edges, 2]
        # x has shape [num_samples, T, num_edges]
        # velocity has shape [2, num_samples, T, num_edges]
        velocity = 0.5 * (v_i + v_j).permute(1, 2, 3, 0) # shape [num_samples, T, num_edges, 2]
        msg = -0.5 * (edge_attr * velocity).sum(-1) * (x_j + x_i) + self.diff_coeff * (x_j - x_i)
        return msg

class AdvectionModel(ptg.nn.MessagePassing):
    def __init__(self, config, graph):
        super(AdvectionModel, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.velocity = torch.nn.Parameter(2 * torch.rand(2, ) - 1)
        self.edge_index = graph['edge_index']
        self.edge_index_transpose = self.edge_index.flip(0)
        self.edge_attr = graph['edge_attr'] # normal vectors at cell boundaries

    def forward(self, x, transpose=False, **kwargs):
        agg_i, agg_j = self.propagate(self.edge_index, x=x, edge_attr=self.edge_attr)

        if transpose:
            agg_i_T, agg_j_T = self.propagate(self.edge_index_transpose, x=x, edge_attr=self.edge_attr)
            update = agg_j_T + agg_i
        else:
            update = agg_j + agg_i

        return x + update


    def message(self, x_i, x_j, edge_attr):
        # construct messages to node i for each edge (j,i)
        # edge_attr has shape [num_edges, 2]
        # velocity has shape [2]
        coef = -0.5 * (edge_attr * self.velocity).sum(1)
        msg_i = coef * x_i
        msg_j = coef * x_j
        # msg = -0.5 * (edge_attr * self.velocity).sum(1) * (x_j + x_i)
        return torch.stack([msg_i, msg_j], dim=0)


class TemporalDGMRF(torch.nn.Module):
    def __init__(self, config, spatial_graph, temporal_graph, features=None, **kwargs):
        super().__init__()

        self.num_nodes = get_num_nodes(spatial_graph['edge_index'])

        self.features = features
        self.use_features = config.get('use_features_dynamics', False) and features is not None
        self.n_features = self.features.size(-1) if self.use_features else 0

        self.vi_layer = kwargs.get('vi_layer', False)
        if self.vi_layer:
            self.transition_type = config.get('vi_transition_type', 'identity')
        else:
            self.transition_type = config.get('transition_type', 'identity')

        self.n_transitions = config.get('n_transitions', 1)
        self.shared_dynamics = kwargs.get('shared', 'dynamics')

        # setup transition model
        if self.transition_type == 'diffusion':
            self.transition_models = torch.nn.ModuleList([DiffusionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'directed_diffusion':
            self.transition_models = torch.nn.ModuleList([DirectedDiffusionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'advection':
            self.transition_models = torch.nn.ModuleList([AdvectionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'GNN_advection':
            self.transition_models = torch.nn.ModuleList([GNNAdvection(config, temporal_graph, self.n_features)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'advection+diffusion':
            self.transition_models = torch.nn.ModuleList([AdvectionDiffusionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'inhomogeneous_advection+diffusion':
            self.transition_models = torch.nn.ModuleList([InhomogeneousAdvectionDiffusionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'GNN':
            #assert self.use_features_dynamics
            self.transition_models = torch.nn.ModuleList([GNNTransition(config, temporal_graph,
                                                                        self.n_features, **kwargs)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'flow':
            self.transition_models = torch.nn.ModuleList([FlowModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        else: # self.transition_type == "AR":
            T = 1 if self.shared_dynamics == 'dynamics' else (kwargs.get('T', 2) - 1)
            self.transition_models = torch.nn.ModuleList([ARModelMultiChannel(T=T) for _ in range(self.n_transitions)])

        if kwargs.get('use_dynamics_bias', True) and not self.vi_layer:
            print('use dynamics bias')
            if self.use_features:
                hidden_dim = config.get('MLP_hidden_dim', 8)
                self.bias_net = torch.nn.Sequential(torch.nn.Linear(self.n_features, hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, 1))
            # elif self.shared_dynamics:
            else:
                self.bias_param = torch.nn.parameter.Parameter(torch.rand(1,) - 0.5)
            # else:
            #     self.bias_param = torch.nn.parameter.Parameter(torch.rand(kwargs['T'],) - 0.5)
        else:
            self.bias_param = torch.zeros(1)

    def bias(self, features=None):
        if self.use_features:
            return self.bias_net(features)
        else:
            return self.bias_param

    def forward(self, x, transpose=False, with_bias=True, p=1, **kwargs):
        # computes e=Fx
        # x has shape [n_samples, T, num_nodes]
        # p is the auto-regressive order

        if transpose:
            states = x[:, p:]
            features = self.features[p:] if self.use_features else None
            layers = reversed(self.transition_models)
        else:
            states = x[:, :-p]
            features = self.features[:-p] if self.use_features else None
            layers = self.transition_models

        if with_bias:
            if self.use_features:
                bias = self.bias_net(features).squeeze(-1).unsqueeze(0) # shape [1, T, num_nodes]
            else:
                bias = self.bias_param.reshape(1, -1, 1)

            states = states + bias

        for layer in layers:
            if hasattr(self, 'transition_models'):
                # if self.transition_type == 'inhomogeneous_advection+diffusion':
                #     v = kwargs.get('v').unsqueeze(2).repeat(1, 1, states.size(1), 1)  # shape [2, n_samples, T-1, num_nodes]
                #     states = self.transition_model(states, v=v, transpose=transpose)
                # else:
                states = layer(states, transpose=transpose, features=features)

            # if with_bias:
            #     states = states + self.bias
        # states = torch.cat([torch.zeros_like(x[:, 0]).unsqueeze(1), states], dim=1)

        if transpose:
            states = torch.cat([states, torch.zeros_like(x[:, x.size(1)-p:]).reshape(x.size(0), p, -1)], dim=1)
        else:
            states = torch.cat([torch.zeros_like(x[:, :p]).reshape(x.size(0), p, -1), states], dim=1)

        # Fx = x - states
        # return Fx

        return states

    def get_matrix(self, dtype=torch.float32):
        # F_t = self.forward(torch.eye(num_nodes).reshape(1, 1, num_nodes, num_nodes), transpose=True, with_bias=False)
        F_t = self.forward(torch.eye(self.num_nodes).reshape(self.num_nodes, 1, self.num_nodes).repeat(1, 2, 1),
                transpose=True, with_bias=False, p=1)[:, 0, :].squeeze()

        return F_t.to(dtype)


# # TODO: write parent class for all jointDGMRFs defined by a dict of DGMRFs and time associated indices
# class ParallelDGMRF(torch.nn.Module):
#     def __init__(self, graphs, config):
#         super().__init__()
#
#         self.T = len(graphs)
#
#         # define DGMRF for initial precision
#         self.dgmrf_list = torch.nn.ModuleList([DGMRF(graphs.get_example(i)["latent", "spatial", "latent"], config)
#                            for i in torch.arange(self.T)])
#
#
#     def forward(self, x, transpose=False, with_bias=False):
#         # computes z = Px
#
#         # x should have shape [T, n_samples, num_nodes]
#
#         z = [self.dgmrf_list[i](x[i], transpose, with_bias) for i in torch.arange(self.T)]
#         z = torch.stack(z, dim=0) # shape [T, n_samples, num_nodes]
#
#         return z
#
#     def log_det(self):
#         log_det_list = [dgmrf.log_det() for dgmrf in self.dgmrf_list]
#         return sum(log_det_list)

# class JointVI(torch.nn.Module):
#     def __init__(self, graph_list, time_intervals, initial_guess, config, dynamics_graph=None):
#         super().__init__()
#
#         self.vi_list = torch.nn.ModuleList([VariationalDist(config, g, initial_guess[i])
#                                             for i, g in enumerate(graph_list)])
#         self.time_intervals = time_intervals
#
#         if not dynamics_graph is None:
#             self.dynamics = TemporalDGMRF(dynamics_graph, config)
#
#     def sample(self):
#
#         # TODO: first draw standard sample, push it through F, and then feed it to VI distributions
#         #  integrate dynamics into VariationalDist?
#         # if hasattr(self, 'dynamics'):
#         #     x = self.dynamics(x)
#         #
#         # z = [self.dgmrf_list[i](x[:, ti], transpose, with_bias) for i, ti in enumerate(self.time_intervals)]
#         # z = torch.cat(z, dim=1)  # shape [n_samples, T, num_nodes]
#         #
#         # return z
#
#         return None
#
#     def log_det(self):
#         return sum([len(ti) * self.vi_list[i].log_det() for i, ti in enumerate(self.time_intervals)])


# class ParallelVI(torch.nn.Module):
#     def __init__(self, graphs, initial_guess, config):
#         super().__init__()
#
#         # define DGMRF for initial precision
#         # self.vi_list = torch.nn.ModuleList([VariationalDist(config, graphs.get_example(i)["latent", "spatial", "latent"], initial_guess[i])
#         #                 for i in torch.arange(len(graphs))])
#
#
#     def sample(self):
#         # computes z = Pe
#
#         x = [vi.sample() for vi in self.vi_list]
#         x = torch.stack(x, dim=1) # shape [n_samples, T, num_nodes]
#
#         return x
#
#     @property
#     def mean_param(self):
#         return torch.cat([vi.mean_param for vi in self.vi_list], dim=0)
#
#     def log_det(self):
#         log_det_list = [vi.log_det() for vi in self.vi_list]
#         return sum(log_det_list)
#
#     @torch.no_grad()
#     def posterior_estimate(self):
#         posteriors = [vi.posterior_estimate() for vi in self.vi_list]
#         post_mean, post_std = list(zip(*posteriors))
#         post_mean = torch.cat(post_mean, dim=0) # shape [T * n_nodes]
#         post_std = torch.cat(post_std, dim=0)  # shape [T * n_nodes]
#         return post_mean, post_std



class VariationalDist(torch.nn.Module):
    def __init__(self, config, graph, initial_guess, temporal_graph=None, n_features=0, T=1, shared='all'):
        super().__init__()

        # Dimensionality of distribution (num_nodes of graph)
        self.dim = get_num_nodes(graph['edge_index'])
        self.T = T

        # Standard amount of samples (must be fixed to be efficient)
        self.n_samples = config["n_training_samples"]
        self.n_post_samples = config["n_post_samples"]

        # Variational distribution, Initialize with observed y
        self.mean_param = torch.nn.parameter.Parameter(initial_guess)
        self.diag_param = torch.nn.parameter.Parameter(2 * torch.rand(self.T, self.dim) - 1.)  # U(-1,1)
        self.layers = torch.nn.ModuleList([DGMRFLayerMultiChannel(config, graph, T=T, shared=shared, vi_layer=True,
                                                                  weighted=config.get('weighted_vi', False))
                            for _ in range(config["vi_layers"])])

        if temporal_graph is not None:
            self.dynamics = TemporalDGMRF(config, graph, temporal_graph, T=T, shared=shared, vi_layer=True)


        if config["vi_layers"] > 0:
            self.post_diag_param = torch.nn.parameter.Parameter(
                2 * torch.rand(self.T, self.dim) - 1.)

        if config["use_features"] and n_features > 0:
            # Additional variational distribution for linear coefficients
            self.coeff_mean_param = torch.nn.parameter.Parameter(torch.randn(1, n_features))
            self.coeff_diag_param = torch.nn.parameter.Parameter(
                2*torch.rand(1, n_features) - 1.) # U(-1,1)
            # self.coeff_mean_param = torch.nn.parameter.Parameter(torch.randn(self.T, n_features))
            # self.coeff_diag_param = torch.nn.parameter.Parameter(
            #     2 * torch.rand(self.T, n_features) - 1.)  # U(-1,1)

            self.coeff_inv_std = config["coeff_inv_std"]

    @property
    def std(self):
        # Note: Only std before layers
        return torch.nn.functional.softplus(self.diag_param)

    @property
    def post_diag(self):
        # Diagonal of diagonal matrix applied after layers
        return torch.nn.functional.softplus(self.post_diag_param)

    @property
    def coeff_std(self):
        return torch.nn.functional.softplus(self.coeff_diag_param)

    def sample(self):
        # standard_sample = torch.randn(self.n_samples, self.dim)
        samples = torch.randn(self.n_samples, self.T, self.dim)

        if hasattr(self, 'dynamics'):
            samples = samples - self.dynamics(samples, with_bias=False)

        samples = self.std * samples # [T, dim] * [samples, T, dim]

        for layer in self.layers:
            propagated = layer(samples, transpose=False, with_bias=False)
            samples = propagated

        if self.layers:
            # Apply post diagonal matrix
            samples = self.post_diag.unsqueeze(0) * samples # [1, T, dim] * [samples, T, dim]
        samples = samples + self.mean_param.unsqueeze(0) # [samples, T, dim] + [1, T, dim]
        return samples # shape (n_samples, T, dim)

    def forward(self, x, transpose=False):

        if transpose:
            return self.P_transpose(x)
        else:
            return self.P(x)

    def P(self, x):
        # apply factor P to vector x
        # x has shape [nbatch, T, dim]

        if hasattr(self, 'dynamics'):
            # multiply with \Tilde{F}
            out = x - self.dynamics(x, with_bias=False)
        else:
            out = x

        # multiply with diagonal matrix
        out = self.std * out  # [T, dim] * [nbatch, T, dim]

        # multiply with \Tilde{S}
        for layer in self.layers:
            out = layer(out, transpose=False, with_bias=False)

        if self.layers:
            # multiply with post diagonal matrix
            out = self.post_diag.unsqueeze(0) * out # [1, T, dim] * [nbatch, T, dim]

        return out # shape (nbatch, T, dim)


    def P_transpose(self, x):
        # apply factor P^T to vector x
        # x has shape [nbatch, T, dim]

        if self.layers:
            # multiply with post diagonal matrix
            out = self.post_diag.unsqueeze(0) * x # [1, T, dim] * [nbatch, T, dim]
        else:
            out = x

        # multiply with \Tilde{S}
        for layer in self.layers:
            out = layer(out, transpose=True, with_bias=False)

        # multiply with diagonal matrix
        out = self.std * out # [T, dim] * [nbatch, T, dim]

        if hasattr(self, 'dynamics'):
            # multiply with \Tilde{F}
            out = out - self.dynamics(out, transpose=True, with_bias=False)

        return out # shape (nbatch, T, dim)


    def log_det(self):
        layers_log_det = sum([layer.log_det() for layer in self.layers])
        std_log_det = torch.sum(torch.log(self.std))
        total_log_det = 2.0*std_log_det + 2.0*layers_log_det

        if self.layers:
            post_diag_log_det = torch.sum(torch.log(self.post_diag))
            total_log_det = total_log_det + 2.0*post_diag_log_det

        return total_log_det

    def sample_coeff(self, n_samples):
        standard_sample = torch.randn(n_samples, *self.coeff_mean_param.shape)
        samples = (self.coeff_std * standard_sample) + self.coeff_mean_param
        return samples # [n_samples, 1, n_features]

    def log_det_coeff(self):
        return 2.0*torch.sum(torch.log(self.coeff_std))

    def ce_coeff(self):
        # Compute Cross-entropy term (CE between VI-dist and coeff prior)
        return -0.5*(self.coeff_inv_std**2)*torch.sum(
                torch.pow(self.coeff_std, 2) + torch.pow(self.coeff_mean_param, 2))

    @torch.no_grad()
    def posterior_estimate(self, noise_var):
        # Compute mean and marginal std of distribution (posterior estimate)
        # only for x-part, not for beta-part (linear model for covariates)

        # Marginal std. (MC estimate)
        mc_sample_list = []
        cur_mc_samples = 0
        while cur_mc_samples < self.n_post_samples:
            mc_sample_list.append(self.sample())
            cur_mc_samples += self.n_samples
        mc_samples = torch.cat(mc_sample_list, dim=0)[:self.n_post_samples]

        # MC estimate of variance using known population mean
        post_var_x = torch.mean(torch.pow(mc_samples - self.mean_param, 2), dim=0)
        # Posterior std.-dev. for y
        post_std = torch.sqrt(post_var_x + noise_var)

        return self.mean_param.detach(), post_std # [channels, dim]

class ObservationModel(ptg.nn.MessagePassing):
    """
    Apply observation model to latent states: y = Hx
    """

    def __init__(self, graph):
        super(ObservationModel, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.edge_index = graph['edge_index']

        if 'edge_weight' in graph:
            self.edge_weights = graph['edge_weight']
        else:
            self.edge_weights = torch.ones_like(self.edge_index[0])


        self.mask = ptg.utils.degree(self.edge_index[0], num_nodes=get_num_nodes(self.edge_index)).bool()
        #self.mask = ptg.utils.degree(self.edge_index[1], num_nodes=num_nodes).bool()


    def forward(self, x):

        y = self.propagate(self.edge_index, x=x, edge_weights=self.edge_weights)

        return y[:, self.mask]

    def message(self, x_j, edge_weights):
        # construct messages to node i for each edge (j,i)
        msg = edge_weights * x_j
        return msg


class SpatiotemporalInference(pl.LightningModule):

    def __init__(self, config, initial_guess, data, joint_mask, spatial_graph, temporal_graph=None,
                 T=1, gt=None, features=None, **kwargs):
        #def __init__(self, graphs, initial_guess, config):
        super(SpatiotemporalInference, self).__init__()
        self.save_hyperparameters()

        self.config = config

        # model settings
        self.learning_rate = config.get('lr', 0.01)

        self.T = T
        self.num_nodes = get_num_nodes(spatial_graph['edge_index'])
        self.N = self.T * self.num_nodes
        print(f'N={self.N}, T={self.T}')

        self.y = data
        self.mask = joint_mask # shape [T * num_nodes]
        self.y_masked = torch.zeros(self.mask.size(), dtype=self.y.dtype) #* np.nan
        self.y_masked[self.mask] = self.y

        self.pos = spatial_graph.get('pos', None)
        if not gt is None:
            self.gt = gt # shape [T * num_nodes]

        if config["learn_noise_std"]:
            self.obs_noise_param = torch.nn.parameter.Parameter(torch.tensor(config["noise_std"]))
        else:
            self.obs_noise_param = torch.tensor(config["noise_std"])

        self.use_dynamics = config.get("use_dynamics", False)
        self.independent_time = config.get("independent_time", False)
        self.use_hierarchy = config.get("use_hierarchy", False)
        self.use_vi_dynamics = config.get('use_vi_dynamics', False)

        self.features = features

        self.use_features = config.get("use_features", False) and features is not None
        self.use_features_dynamics = config.get("use_features_dynamics", False) and features is not None

        self.data_mean = kwargs.get('data_mean', 0)
        self.data_std = kwargs.get('data_std', 1)

        self.true_post_mean = kwargs.get('true_post_mean', None)
        self.true_post_std = kwargs.get('true_post_std', None)

        # model components
        if self.use_dynamics:

            shared = 'none' if config.get('independent_time', True) else 'dynamics'
            features = self.features.reshape(self.T, self.num_nodes, -1) if self.use_features_dynamics else None
            self.dgmrf = JointDGMRF(config, spatial_graph, temporal_graph, T=self.T, shared=shared,
                                    weighted=config.get('weighted_dgmrf', False), features=features)

            self.input_shape = [self.T, self.num_nodes]
            shared_vi = 'none'

            # if self.use_hierarchy:
            #     # use DGMRF with shared parameters across time for latent v
            #     # self.dgmrf_vx = DGMRF(graphs.get_example(0)["latent", "spatial", "latent"], config)
            #     # self.vi_dist_vx = VariationalDist(config, graphs.get_example(0)["latent", "spatial", "latent"],
            #     #                                   torch.zeros(self.N // self.T))
            #     # self.dgmrf_vy = DGMRF(graphs.get_example(0)["latent", "spatial", "latent"], config)
            #     # self.vi_dist_vy = VariationalDist(config, graphs.get_example(0)["latent", "spatial", "latent"],
            #     #                                   torch.zeros(self.N // self.T))
            #     self.dgmrf_vx = DGMRF(config, spatial_graph, T=1, weighted=config.get('weighted_dgmrf', False))
            #     self.vi_dist_vx = VariationalDist(config, spatial_graph, torch.zeros(self.num_nodes), T=1)
            #     self.dgmrf_vy = DGMRF(config, spatial_graph, T=1, weighted=config.get('weighted_dgmrf', False))
            #     self.vi_dist_vy = VariationalDist(config, spatial_graph, torch.zeros(self.num_nodes), T=1)

        elif self.independent_time:
            # treat time steps independently, with separate DGMRF for each time step
            self.dgmrf = DGMRF(config, spatial_graph, T=self.T, shared='none', weighted=config.get('weighted_dgmrf', False))
            self.input_shape = [self.T, self.num_nodes]
            shared_vi = 'none'
        else:
            # use a single DGMRF, with parameters shared across time steps
            self.dgmrf = DGMRF(config, spatial_graph, T=self.T, shared='all', weighted=config.get('weighted_dgmrf', False))
            self.input_shape = [self.T, self.num_nodes]
            shared_vi = 'all'

        if not config.get('use_KS', False):
            self.vi_dist = VariationalDist(config, spatial_graph, initial_guess.reshape(self.T, -1),
                                           T=self.T, shared=shared_vi,
                                           temporal_graph=(temporal_graph if self.use_vi_dynamics else None),
                                           n_features=(self.features.size(-1) if self.use_features else 0))

        # self.obs_model = ObservationModel(graphs["latent", "observation", "data"], self.N)
        self.obs_model = lambda x: x[:, self.mask]

        self.n_training_samples = config["n_training_samples"]


    def get_name(self):
        return 'SpatiotemporalDGMRF'

    @property
    def noise_var(self):
        return self.obs_noise_param**2

    @property
    def log_noise_std(self):
        return 0.5 * self.noise_var.log()

    def all_H(self, mask):
        # mask has shape [T * num_nodes]
        # output should have shape [T, num_observed_nodes, num_nodes]
        # obs_nodes = mask.reshape(self.T, -1).sum(0).nonzero().squeeze()

        # stacked_H = torch.eye(self.num_nodes).unsqueeze(0).repeat(self.T, 1, 1)
        # stacked_H = stacked_H[:, obs_nodes, :] # nodes that are observed at any time point

        # sub_mask = (stacked_H @ mask.view(self.T, -1, 1).to(torch.float32)).squeeze(-1)
        # unobs_nodes = (sub_mask - 1).nonzero()

        identity = torch.eye(self.num_nodes)
        all_H = []

        # adjust for each time step
        for t in range(self.T):
            # get observed nodes for time t
            jdx = mask.view(self.T, -1)[t].to(torch.float32).flatten().nonzero().squeeze()

            all_H.append(identity[jdx, :].unsqueeze(0))
            # jdx = (sub_mask[t] - 1).nonzero().squeeze()

            # stacked_H[t_idx, n_idx] = 0

        return all_H

    def stacked_H(self, mask):
        # mask has shape [T * num_nodes]
        # output should have shape [T, num_observed_nodes, num_nodes]
        obs_nodes = mask.reshape(self.T, -1).sum(0).nonzero().squeeze()

        stacked_H = torch.eye(self.num_nodes).unsqueeze(0).repeat(self.T, 1, 1)
        stacked_H = stacked_H[:, obs_nodes, :] # nodes that are observed at any time point

        sub_mask = (stacked_H @ mask.view(self.T, -1, 1).to(torch.float32)).squeeze(-1)
        unobs_nodes = (sub_mask - 1).nonzero()

        # adjust for each time step
        for tidx, nidx in unobs_nodes:
            # get unobserved nodes for time t
            stacked_H[tidx, nidx] = 0

        return stacked_H

    def _reconstruction_loss(self, x, mask):
        # x has shape [n_samples, T * n_nodes]
        # y_hat = self.obs_model(x)
        # residuals = self.y[index] - y_hat[:, index]
        residuals = self.y_masked[..., mask] - x[..., mask]
        # rec_loss = torch.mean(torch.pow(residuals, 2))
        rec_loss = torch.sum(torch.pow(residuals, 2))

        return rec_loss

    def _joint_log_likelihood(self, x, mask, v=None):
        # x has shape [n_samples, T, n_nodes]
        # v has shape [2, n_samples, n_nodes]
        # N = x.size(-1) * x.size(-2)

        # compute log-likelihood of samples given prior p(x)
        Gx = self.dgmrf(x, v=v)  # shape (n_samples, T, n_nodes)
        # prior_ll = (-0.5 * torch.sum(torch.pow(Gx, 2)) + self.dgmrf.log_det()) / self.N

        prior_ll = (-0.5 * torch.sum(torch.pow(Gx, 2)) / self.n_training_samples + self.dgmrf.log_det())

        ## compute log-likelihood of latent field v
        # if self.use_hierarchy:
        #     Gvx = self.dgmrf_vx(v[0]) #.reshape(-1, self.N))
        #     Gvy = self.dgmrf_vy(v[1]) #.reshape(-1, self.N))
        #     vx_ll = (-0.5 * torch.sum(torch.pow(Gvx, 2)) + self.dgmrf_vx.log_det()) / (self.num_nodes)
        #     vy_ll = (-0.5 * torch.sum(torch.pow(Gvy, 2)) + self.dgmrf_vy.log_det()) / (self.num_nodes)
        #     prior_ll = prior_ll + vx_ll + vy_ll

        x = x.reshape(self.n_training_samples, -1)  # shape [n_samples, T * num_nodes]

        # compute data log-likelihood given samples
        if self.use_features:
            coeffs = self.vi_dist.sample_coeff(self.n_training_samples).unsqueeze(1) # [n_samples, 1, n_features]
            x = x + coeffs @ self.features.transpose(0, 1)

        # rec_loss = self._reconstruction_loss(x, mask)

        rec_loss = self._reconstruction_loss(x, mask) / self.n_training_samples

        data_ll = -0.5 * rec_loss / self.noise_var - self.log_noise_std * mask.sum()

        # scale by total number of nodes in space and time
        # data_ll = data_ll / self.N
        # prior_ll = prior_ll / self.N

        self.log("train_rec_loss", rec_loss.item(), sync_dist=True)
        self.log("train_prior_ll", prior_ll.item(), sync_dist=True)
        self.log("train_data_ll", data_ll.item(), sync_dist=True)

        return prior_ll + data_ll

    def KS_ELBO(self, mu_0, Q_0, mu_s, cov_s, cov_s_lag, F_t, Q_t, data, stacked_H):
        # mu_0 has shape [num_nodes]
        # Q_0 has shape [num_nodes, num_nodes]
        # mu_s has shape [T, num_nodes]
        # cov_s has shape [T, num_nodes, num_nodes]
        # cov_s_lag has shape[T-1, num_nodes, num_nodes]
        # F_t, Q_t have shape [T-1, num_nodes, num_nodes]

        # compute E_{x|y}[p(x)]
        mu_0 = mu_0.unsqueeze(-1)
        mu_s = mu_s.unsqueeze(-1)
        E0 = torch.trace(Q_0 @ (cov_s[0] + mu_s[0] @ mu_s[0].transpose(0, 1))) \
              - 2 * mu_0.transpose(0, 1) @ Q_0 @ mu_s[0] \
              + mu_0.transpose(0, 1) @ Q_0 @ mu_0

        Et = (Q_t @ (cov_s[1:] + mu_s[1:] @ mu_s[1:].transpose(-1, -2))).diagonal(offset=0, dim1=-1, dim2=-2) \
             - 2 * (Q_t @ F_t @ (cov_s_lag + mu_s[:-1] @ mu_s[1:].transpose(-1, -2))).diagonal(offset=0, dim1=-1, dim2=-2) \
             + (Q_t @ F_t @ (cov_s[:-1] + mu_s[:-1] @ mu_s[:-1].transpose(-1, -2))).diagonal(offset=0, dim1=-1, dim2=-2)

        prior_term = -0.5 * (E0 + Et.sum()) + self.dgmrf.log_det()

        # compute E_{x|y}[p(y|x)]
        data = data.unsqueeze(-1)
        data_term = -0.5 * (((stacked_H @ data).transpose(-1, -2) @ stacked_H @ data) / self.noise_var \
                            + ((stacked_H @ (cov_s + mu_s @ mu_s.transpose(-1, -2)) @ stacked_H.transpose(-1, -2)) \
                               / self.noise_var).diagonal(offset=0, dim1=-1, dim2=-2)
                            - (2 * (stacked_H @ data).transpose(-1, -2) @ stacked_H @ mu_s) / self.noise_var).sum() \
                    - stacked_H.sum() * self.log_noise_std

        return prior_term + data_term



    def training_step(self, train_mask):

        # self.train_mask = torch.zeros_like(self.mask)
        # self.train_mask[self.mask.nonzero()[training_index.squeeze()]] = 1
        self.train_mask = train_mask.squeeze(0)

        if self.config.get('use_KS', False):
            Q_0, Q_t = self.dgmrf.get_matrices() # all have shape [num_nodes, num_nodes]
            #TODO: allow for time-varying F_t (depending on covariates)?
            F_t = self.dgmrf.get_transition_matrix()
            # assume mu_0 and c_t are all zero
            #TODO: get mu_0 and c_t from STDGMRF
            mu_0 = torch.zeros(self.num_nodes)
            # R_t = self.noise_var * torch.eye(self.num_nodes)

            Q_0_inv, Q_t_inv = self.dgmrf.get_inv_matrices()

            data = (self.y_masked * self.train_mask).reshape(self.T, -1)

            all_H = self.all_H(self.train_mask)
            stacked_H = self.stacked_H(self.train_mask)

            ks = KalmanSmoother(mu_0.unsqueeze(0), Q_0_inv.unsqueeze(0), F_t.unsqueeze(0),
                                Q_t_inv.unsqueeze(0), all_H, torch.ones(1,) * self.noise_var)
            mu_s, cov_s, cov_s_lag = ks.smoother(data.unsqueeze(0))

            Q_t = Q_t.unsqueeze(0).repeat(self.T - 1, 1, 1)
            F_t = F_t.unsqueeze(0).repeat(self.T - 1, 1, 1)

            elbo = self.KS_ELBO(mu_0, Q_0, mu_s.squeeze(0), cov_s.squeeze(0), cov_s_lag.squeeze(0),
                                F_t, Q_t, data, stacked_H)

            self.log("train_elbo", elbo.item(), sync_dist=True)

        else:
            # sample from variational distribution
            samples = self.vi_dist.sample()  # shape [n_samples, T, num_nodes]
            # N = samples.size(-1) * samples.size(-2)

            # compute entropy of variational distribution
            # vi_entropy = 0.5 * self.vi_dist.log_det() / self.N
            vi_entropy = 0.5 * self.vi_dist.log_det() # TODO: check effect on results in submitted paper!

            # if self.use_hierarchy:
            #     v_samples = torch.stack([self.vi_dist_vx.sample(), self.vi_dist_vy.sample()], dim=0).reshape(
            #         2, self.n_training_samples, -1) # .reshape(2, self.n_training_samples, self.T, -1)
            #     joint_ll = self._joint_log_likelihood(samples, training_index.squeeze(), v=v_samples)
            #     vi_entropy = vi_entropy + 0.5 * (self.vi_dist_vx.log_det() + self.vi_dist_vy.log_det()) / (self.num_nodes)
            # else:
            #     joint_ll = self._joint_log_likelihood(samples, training_index.squeeze())
            joint_ll = self._joint_log_likelihood(samples, self.train_mask)

            if self.use_features:
                vi_entropy = vi_entropy + 0.5 * self.vi_dist.log_det_coeff() # log-det for coefficients
                joint_ll = joint_ll + self.vi_dist.ce_coeff() # cross entropy term for coefficients


            elbo = (joint_ll + vi_entropy) / self.N

            self.log("train_elbo", elbo.item(), sync_dist=True)
            self.log("vi_entropy", vi_entropy.item(), sync_dist=True)

        return -1. * elbo


    def validation_step(self, val_mask, *args):

        if self.config.get('use_KS', False):
            pass
        else:
            samples = self.vi_dist.sample() # shape [n_samples, T, num_nodes]
            samples = samples.reshape(self.n_training_samples, -1) # shape [n_samples, T * num_nodes]

            if self.use_features:
                coeffs = self.vi_dist.sample_coeff(self.n_training_samples).unsqueeze(1) # [n_samples, 1, n_features]
                samples = samples + coeffs @ self.features.transpose(0, 1)

            rec_loss = self._reconstruction_loss(samples, val_mask.squeeze(0)) / (val_mask.sum() * self.n_training_samples)

            self.log("val_rec_loss", rec_loss.item(), sync_dist=True)

    def test_step(self, test_mask, *args):
    
        if self.config.get('use_KS', False):
            self.KS_inference(test_mask.squeeze(0), split='test')
        else:
            self.DGMRF_inference(test_mask.squeeze(0), split='test')

    def DGMRF_inference(self, test_mask, split='test'):

        # posterior inference using variational distribution
        self.vi_mean, self.vi_std = self.vi_dist.posterior_estimate(self.noise_var)
        mean = self.vi_mean.flatten()
        std = self.vi_std.flatten()

        # use held out part of data to evaluate predictions
        data_mask = torch.logical_and(self.mask, torch.logical_not(test_mask))  # all data except test data

        # y_masked = torch.zeros_like(self.y_masked)
        # y_masked[data_mask] = self.y_masked[data_mask]
        y_masked = self.y_masked * data_mask

        # def preconditioner(x, transpose=False):
        #     # x has shape [nbatch, T, n_nodes]
        #
        #     #diag = 1 + torch.rand(1, 1, x.size(-1))
        #     diag = torch.ones(1, 1, x.size(-1)) * 0.2
        #
        #     return x * diag
        #
        # def preconditioner2(x, transpose=False):
        #     input = torch.eye(self.num_nodes).unsqueeze(1).repeat(1, self.T, 1) # shape [n_nodes, T, n_nodes]
        #
        #     Gx = self.dgmrf(input, with_bias=False)
        #
        #     #print(f'Gx min = {Gx.min()}, max = {Gx.max()}')
        #     GtGx = self.dgmrf(Gx, transpose=True, with_bias=False)  # has shape [n_nodes, T, n_nodes]
        #
        #     #print(f'GtGx min = {GtGx.min()}, max = {GtGx.max()}')
        #     if self.noise_var is not None:
        #         out = GtGx + (data_mask.to(torch.float64) / self.noise_var).view(1, self.T, -1) * input
        #     else:
        #         out = GtGx
        #
        #     #print(f'out min = {out.min()}, max = {out.max()}')
        #
        #     diag = torch.diagonal(out, dim1=0, dim2=2)
        #     #print(diag.min(), diag.max())
        #
        #     return x / diag.view(1, self.T, self.num_nodes).sqrt()



        self.post_mean, self.post_std, niter = posterior_inference(self.dgmrf, y_masked.reshape(1, *self.input_shape),
                                                        data_mask, self.config, self.noise_var, self.vi_mean,
                                                        features=self.features, verbose=False,
                                                        preconditioner=None)
                                                        #preconditioner=preconditioner)


        self.log('niter_cg', niter, sync_dist=True)

        if hasattr(self, 'gt'):
            target = self.gt[test_mask]
        else:
            target = self.y_masked[test_mask]

        residuals_vi = (target - mean[test_mask])
        residuals_cg = target - self.post_mean.squeeze(-1)[test_mask]

        self.log(f"{split}_mae_vi", residuals_vi.abs().mean().item(), sync_dist=True)
        self.log(f"{split}_rmse_vi", torch.pow(residuals_vi, 2).mean().sqrt().item(), sync_dist=True)
        #self.log(f"{split}_mape_vi", (residuals_vi / target).abs().mean().item(), sync_dist=True)

        self.log(f"{split}_mae", residuals_cg.abs().mean().item(), sync_dist=True)
        self.log(f"{split}_rmse", torch.pow(residuals_cg, 2).mean().sqrt().item(), sync_dist=True)
        #self.log(f"{split}_mape", (residuals_cg / target).abs().mean().item(), sync_dist=True)

        pred_mean_np = self.post_mean.squeeze(-1)[test_mask].cpu().numpy()
        pred_std_np = self.post_std.squeeze(-1)[test_mask].cpu().numpy()
        target_np = target.cpu().numpy()

        self.log(f"{split}_crps", crps_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)
        self.log(f"{split}_int_score", int_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)

        if self.true_post_mean is not None:
            target = self.true_post_mean.reshape(-1)[test_mask]
            residuals = target - self.post_mean.squeeze(-1)[test_mask]
            self.log(f"{split}_mae_mean", residuals.abs().mean().item(), sync_dist=True)
            self.log(f"{split}_rmse_mean", torch.pow(residuals, 2).mean().sqrt().item(), sync_dist=True)

            target_np = target.cpu().numpy()

            self.log(f"{split}_crps_mean", crps_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)
            self.log(f"{split}_int_score_mean", int_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)

            if self.true_post_std is not None:
                target_std = self.true_post_std.reshape(-1)[test_mask]
                residuals_std = target_std - self.post_std.squeeze(-1)[test_mask]
                self.log(f"{split}_mae_std", residuals_std.abs().mean().item(), sync_dist=True)
                self.log(f"{split}_rmse_std", torch.pow(residuals_std, 2).mean().sqrt().item(), sync_dist=True)



    def predict_step(self, predict_mask, *args):

        if self.config.get('use_KS', False):
            return self.KS_prediction(predict_mask.squeeze(0))
        else:
            return self.DGMRF_prediction(predict_mask.squeeze(0))

    def DGMRF_prediction(self, predict_mask):

        #F_t = self.dgmrf.get_transition_matrix()
        #print(f'transition matrix = {F_t}')
        #svdvals = torch.linalg.svdvals(F_t)
        #print(f'svd vals min = {svdvals.min()}, max = {svdvals.max()}')
        
        data_mask = torch.logical_and(self.mask, torch.logical_not(predict_mask))

        y_masked = torch.zeros_like(self.y_masked)
        y_masked[data_mask] = self.y_masked[data_mask]

        if not (hasattr(self, 'vi_mean') or hasattr(self, 'vi_std')):
            self.vi_mean, self.vi_std = self.vi_dist.posterior_estimate(self.noise_var)

        if not (hasattr(self, 'cg_mean') or hasattr(self, 'cg_std')):
            self.post_mean, self.post_std, niter = posterior_inference(self.dgmrf, y_masked.reshape(1, *self.input_shape),
                                                            data_mask, self.config, self.noise_var, self.vi_mean,
                                                            features=self.features,
                                                            preconditioner=None)
                                                            #preconditioner=preconditioner2)

        if not hasattr(self, 'cg_samples'):
            self.post_samples = sample_posterior(10, self.dgmrf, y_masked.reshape(1, *self.input_shape),
                                                 data_mask, self.config, self.noise_var, self.vi_mean, features=self.features,
                                                 preconditioner=None)
                                                 #preconditioner=self.vi_dist)

            # if self.use_features:
            #     post_samples_x = self.post_samples[:, :self.N]
            #     post_samples_beta = self.post_samples[:, self.N:]
            #     self.post_samples = post_samples_x + self.features @ post_samples_beta


        return {'post_mean': self.post_mean.reshape(self.T, self.num_nodes),
                'post_std': self.post_std.reshape(self.T, self.num_nodes),
                'post_samples': self.post_samples.reshape(-1, self.T, self.num_nodes),
                'vi_mean': self.vi_mean,
                'vi_std':  self.vi_std,
                'data': y_masked.reshape(self.T, self.num_nodes),
                'gt': self.gt if hasattr(self, 'gt') else self.y_masked,
                'predict_mask': predict_mask}


    def KS_inference(self, test_mask, split):
        # use held out part of data to evaluate predictions
        data_mask = torch.logical_and(self.mask, torch.logical_not(test_mask))  # all data except test data

        # y_masked = torch.zeros_like(self.y_masked)
        # y_masked[data_mask] = self.y_masked[data_mask]
        data = (self.y_masked * data_mask).reshape(self.T, -1)
        # data = self.y_masked.reshape(self.T, -1)

        # assume mu_0 and c_t are all zero
        mu_0 = torch.zeros(self.num_nodes)

        F_t = self.dgmrf.get_transition_matrix()
        Q_0_inv, Q_t_inv = self.dgmrf.get_inv_matrices()

        all_H = self.all_H(test_mask)

        ks = KalmanSmoother(mu_0.unsqueeze(0), Q_0_inv.unsqueeze(0), F_t.unsqueeze(0),
                            Q_t_inv.unsqueeze(0), all_H, torch.ones(1, ) * self.noise_var)
        mu_s, cov_s, cov_s_lag = ks.smoother(data.unsqueeze(0))

        self.post_mean = mu_s.reshape(-1)
        self.post_std = torch.diagonal(cov_s, dim1=-2, dim2=-1).squeeze(0).reshape(-1)

        if hasattr(self, 'gt'):
            target = self.gt[test_mask]
        else:
            target = self.y_masked[test_mask]

        residuals = (target - self.post_mean[test_mask])

        self.log(f"{split}_mae", residuals.abs().mean().item(), sync_dist=True)
        self.log(f"{split}_rmse", torch.pow(residuals, 2).mean().sqrt().item(), sync_dist=True)
        self.log(f"{split}_mape", (residuals / target).abs().mean().item(), sync_dist=True)

        pred_mean_np = self.post_mean[test_mask].cpu().numpy()
        pred_std_np = self.post_std[test_mask].cpu().numpy()
        target_np = target.cpu().numpy()

        self.log(f"{split}_crps", crps_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)
        self.log(f"{split}_int_score", int_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)


    def KS_prediction(self, predict_mask):
        # assume mu_0 and c_t are all zero

        data_mask = torch.logical_and(self.mask, torch.logical_not(predict_mask))

        data = torch.zeros_like(self.y_masked)
        data[data_mask] = self.y_masked[data_mask]
        # data = self.y_masked * data_mask
        data = data.reshape(self.T, -1)

        mu_0 = torch.zeros(self.num_nodes)

        F_t = self.dgmrf.get_transition_matrix()
        Q_0_inv, Q_t_inv = self.dgmrf.get_inv_matrices()

        all_H = self.all_H(predict_mask)

        ks = KalmanSmoother(mu_0.unsqueeze(0), Q_0_inv.unsqueeze(0), F_t.unsqueeze(0),
                            Q_t_inv.unsqueeze(0), all_H, torch.ones(1, ) * self.noise_var)
        mu_s, cov_s, cov_s_lag = ks.smoother(data.unsqueeze(0))


        return {'ks_mean': mu_s.squeeze(0),
                'ks_std': torch.diagonal(cov_s, dim1=-2, dim2=-1).squeeze(0),
                'data': data,
                'gt': self.gt if hasattr(self, 'gt') else self.y_masked,
                'predict_mask': predict_mask}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

#
# def vi_loss(dgmrf, vi_dist, graph_y, config):
#     vi_samples = vi_dist.sample()
#     vi_log_det = vi_dist.log_det()
#     vi_dist.sample_batch.x = vi_samples.reshape(-1,1)
#     # Column vector of node values for all samples
#
#     g = dgmrf(vi_dist.sample_batch) # Shape (n_training_samples*n_nodes, 1)
#
#     # Construct loss
#     l1 = 0.5*vi_log_det
#     l2 = -graph_y.n_observed*config["log_noise_std"]
#     l3 = dgmrf.log_det()
#     l4 = -(1./(2. * config["n_training_samples"])) * torch.sum(torch.pow(g,2))
#
#     if config["use_features"]:
#         vi_coeff_samples = vi_dist.sample_coeff(config["n_training_samples"])
#         # Mean from a VI sample (x + linear feature model)
#         vi_samples = vi_samples + vi_coeff_samples@graph_y.features.transpose(0,1)
#
#         # Added term when using additional features
#         vi_coeff_log_det = vi_dist.log_det_coeff()
#         entropy_term = 0.5*vi_coeff_log_det
#         ce_term = vi_dist.ce_coeff()
#
#         l1 = l1 + entropy_term
#         l4 = l4 + ce_term
#
#     l5 = -(1./(2. * torch.exp(2.*config["log_noise_std"]) *\
#         config["n_training_samples"]))*torch.sum(torch.pow(
#             (vi_samples - graph_y.x.flatten()), 2)[:, graph_y.mask])
#
#     elbo = l1 + l2 + l3 + l4 + l5
#     loss = (-1./graph_y.num_nodes)*elbo
#     return loss




# Solve Q_tilde x = rhs using Conjugate Gradient
def cg_solve(rhs, dgmrf, mask, T, config, noise_var=None, features=None,
             verbose=False, initial_guess=None, return_info=False, preconditioner=None, regularizer=0):
    # rhs has shape [n_batch, T * n_nodes]

    n_batch = rhs.size(0)
    N = mask.numel()

    if preconditioner is not None:
        #print('apply preconditioner P^T to rhs')
        rhs = preconditioner(rhs.view(n_batch, T, -1), transpose=True).view(n_batch, -1)

    # CG requires more precision for numerical stability
    rhs = rhs.to(torch.float64)

    if initial_guess is None:
        initial_guess = torch.zeros_like(rhs)

    b = rhs + regularizer * initial_guess

    # Batch linear operator
    def Q_tilde_batched(x):
        # x has shape (n_batch, T * n_nodes, 1)

        x = x.reshape(n_batch, T, -1)

        if preconditioner is not None:
            #print('apply preconditioner P to x')
            x = preconditioner(x, transpose=False)

        Gx = dgmrf(x, with_bias=False)
        GtGx = dgmrf(Gx, transpose=True, with_bias=False) # has shape [n_batch, T, n_nodes]

        # compute Omega^+ @ x (i.e. posterior precision matrix multiplied x)
        # res = GtGx.view(n_batch, N, 1)
        #
        # if noise_var is not None:
        #     res = res + (mask.to(torch.float64) / noise_var).view(1, N, 1) * x

        if noise_var is not None:
            res = GtGx + (mask.to(torch.float64) / noise_var).view(1, T, -1) * x
        else:
            res = GtGx

        res = res + regularizer * x # regularization

        if preconditioner is not None:
            #print('apply preconditioner P^T to GtGx')
            res = preconditioner(res, transpose=True)

        res = res.view(n_batch, N, 1)

        return res

    if config["use_features"] and features is not None:
        # features has shape [T, n_nodes, n_features]
        masked_features = features.to(torch.float64) * mask.to(torch.float64).view(N, 1) # shape [n_nodes * T, n_features]
        masked_features_cov = masked_features.transpose(0,1)@masked_features # shape [n_features, n_features]

        def Q_tilde_batched_with_features(x):
            # x has shape (n_batch, N + n_features, 1)
            node_x = x[:,:N]
            coeff_x = x[:,N:]

            top_res1 = Q_tilde_batched(node_x) # shape [n_batch, N, 1]
            top_res2 = (masked_features @ coeff_x) / noise_var # shape [n_batch, N, 1]

            bot_res1 = (masked_features.transpose(0,1) @ node_x) / noise_var # shape [n_batch, n_features, 1]
            bot_res2 = (masked_features_cov @ coeff_x) / noise_var +\
                    (config["coeff_inv_std"]**2)*coeff_x # shape [n_batch, n_features, 1]

            res = torch.cat([top_res1 + top_res2, bot_res1 + bot_res2], dim=1) # shape [n_batch, N + n_features, 1]

            return res

        Q_tilde_func = Q_tilde_batched_with_features
    else:
        Q_tilde_func = Q_tilde_batched

    solution, cg_info = cg_batch.cg_batch(Q_tilde_func, b.unsqueeze(-1), X0=initial_guess.unsqueeze(-1),
                                          rtol=config["inference_rtol"], maxiter=config["max_cg_iter"], verbose=verbose)

    residuals = rhs + regularizer * solution.squeeze(-1) - Q_tilde_func(solution).squeeze(-1)
    res_norm = torch.linalg.norm(residuals, dim=1)
    cg_info["res_norm"] = res_norm

    if config["use_features"] and features is not None:
        solution_x = solution[:, :N]
        solution_beta = solution[:, N:]

        if preconditioner is not None:
            solution_x = preconditioner(solution_x.view(n_batch, T, -1), transpose=False).view(n_batch, -1)

        solution = solution_x + features @ solution_beta
    else:
        if preconditioner is not None:
            #print('apply preconditioner P to solution')
            solution = preconditioner(solution.view(n_batch, T, -1), transpose=False).view(n_batch, -1)

    if verbose:
        print("CG finished in {} iterations, solution optimal: {}".format(
            cg_info["niter"], cg_info["optimal"]))

    if return_info:
        return solution.to(torch.float32), cg_info
    else:
        return solution.to(torch.float32)

def get_bias(dgmrf, input_shape):
    bias = dgmrf(torch.zeros(input_shape))
    return bias

# @torch.no_grad()
# def sample_posterior(n_samples, dgmrf, graph_y, config, rtol, verbose=False):
#     # Construct RHS using Papandeous and Yuille method
#     bias = get_bias(dgmrf, graph_y)
#     std_gauss1 = torch.randn(n_samples, graph_y.num_nodes, 1) - bias # Bias offset
#     std_gauss2 = torch.randn(n_samples, graph_y.num_nodes, 1)
#
#     std_gauss1_graphs = ptg.data.Batch.from_data_list(
#         [new_graph(graph_y, new_x=sample) for sample in std_gauss1])
#     rhs_sample1 = dgmrf(std_gauss1_graphs, transpose=True, with_bias=False)
#     rhs_sample1 = rhs_sample1.reshape(-1, graph_y.num_nodes, 1)
#
#     float_mask = graph_y.mask.to(torch.float32).unsqueeze(1)
#     y_masked = (graph_y.x * float_mask).unsqueeze(0)
#     gauss_masked = std_gauss2 * float_mask.unsqueeze(0)
#     rhs_sample2 = (1./torch.exp(2.*config["log_noise_std"]))*y_masked +\
#         (1./torch.exp(2.*config["log_noise_std"]))*gauss_masked
#
#     rhs_sample = rhs_sample1 + rhs_sample2
#     # Shape (n_samples, n_nodes, 1)
#
#     if config["features"]:
#         # Change rhs to also sample coefficients
#         n_features = graph_y.features.shape[1]
#         std_gauss1_coeff = torch.randn(n_samples, n_features, 1)
#         rhs_sample1_coeff = config["coeff_inv_std"]*std_gauss1_coeff
#
#         rhs_sample2_coeff = graph_y.features.transpose(0,1)@rhs_sample2
#         rhs_sample_coeff = rhs_sample1_coeff + rhs_sample2_coeff
#
#         rhs_sample = torch.cat((rhs_sample, rhs_sample_coeff), dim=1)
#
#     # Solve using Conjugate gradient
#     samples = cg_solve(rhs_sample, dgmrf, graph_y, config,
#             rtol=rtol, verbose=verbose)
#
#     return samples

@torch.no_grad()
def sample_posterior(n_samples, dgmrf, data, mask, config, noise_var, initial_guess, return_info=False, features=None,
                     preconditioner=None):
    # Construct RHS using Papandeous and Yuille method
    bias = get_bias(dgmrf, data.size())
    T, num_nodes = data.shape[1:]
    std_gauss1 = torch.randn(n_samples, T, num_nodes)
    std_gauss2 = torch.randn(n_samples, T * num_nodes)

    # G^T @ (z_1 - b)
    rhs_sample1 = dgmrf(std_gauss1 - bias, transpose=True, with_bias=False).reshape(n_samples, -1)

    # H^T @ R^{-1} @ (y + z_2)
    rhs_sample2 = (mask.to(torch.float32) * data.to(torch.float32).reshape(-1, T * num_nodes)
                   + mask.to(torch.float32) * noise_var.sqrt() * std_gauss2) / noise_var

    rhs_sample = rhs_sample1 + rhs_sample2


    if config["use_features"] and features is not None:
        # Change rhs to also sample coefficients
        n_features = features.size(-1)
        std_gauss1_coeff = torch.randn(n_samples, n_features, 1)
        rhs_sample1_coeff = config["coeff_inv_std"] * std_gauss1_coeff

        rhs_sample2_coeff = features.transpose(0,1) @ rhs_sample2.view(n_samples, T * num_nodes, 1)
        rhs_sample_coeff = rhs_sample1_coeff + rhs_sample2_coeff


        rhs_sample = torch.cat((rhs_sample, rhs_sample_coeff.squeeze(-1)), dim=1)

        # samples, cg_info = cg_solve(rhs_sample, dgmrf, mask, data.size(1), config,
        #                             noise_var=noise_var, verbose=False, return_info=True, features=features,
        #                             preconditioner=preconditioner)

    initial_guess = initial_guess.repeat(n_samples, 1).reshape(rhs_sample.size())
    res_norm = torch.ones(data.size(0)) * float("inf")
    rhs_norm = torch.linalg.norm(rhs_sample, dim=1)
    k = 0
    cg_niter = 0
    regularizer = config.get('cg_regularizer', 0)
    while (res_norm > config.get('outer_rtol', 1e-7) * rhs_norm).any() and (k < config.get('max_outer_iter', 100)):
        samples, cg_info = cg_solve(rhs_sample, dgmrf, mask, data.size(1), config,
                                      noise_var=noise_var, verbose=False, return_info=True,
                                      initial_guess=initial_guess, features=features,
                                      preconditioner=preconditioner, regularizer=regularizer)

        k = k + 1
        cg_niter += cg_info["niter"]
        res_norm = cg_info["res_norm"]
        print(f'relative residual norm after outer iteration {k} = {res_norm / rhs_norm}')
        initial_guess = samples.squeeze(-1)
        regularizer = 0.1 * regularizer

    cg_info["niter"] = cg_niter

    if return_info:
        return samples, cg_info
    else:
        return samples


@torch.no_grad()
def sample_prior(n_samples, dgmrf, data_shape, config, preconditioner=None):

    bias = get_bias(dgmrf, data_shape)
    std_gauss = torch.randn(n_samples, data_shape[1], data_shape[2])

    rhs_sample = dgmrf(std_gauss - bias, transpose=True, with_bias=False) # G^T @ (z - b)

    # if preconditioner is not None:
    #     rhs_sample = preconditioner(rhs_sample, transpose=True)

    # Solve using Conjugate gradient
    samples = cg_solve(rhs_sample.reshape(1, -1), dgmrf, torch.ones(data_shape[1]*data_shape[2]), data_shape[1], config,
            verbose=False, preconditioner=preconditioner)

    # if preconditioner is not None:
    #     samples = preconditioner(samples, transpose=False)

    return samples



@torch.no_grad()
def posterior_mean(dgmrf, data, mask, config, noise_var, initial_guess, verbose=False, preconditioner=None, features=None,
                   return_info=False):
    # data has shape [n_batch, T, num_nodes]
    bias = get_bias(dgmrf, data.size())
    eta = -1. * dgmrf(bias, transpose=True, with_bias=False)  # -G^T @ b

    masked_y = mask.to(torch.float32).reshape(1, *data.shape[1:]) * data.to(
        torch.float32)  # H^T @ y (has shape [1, T, num_nodes])
    mean_rhs = eta + masked_y / noise_var  # eta + H^T @ R^{-1} @ y

    mean_rhs = mean_rhs.reshape(data.size(0), -1)

    if config["use_features"] and features is not None:
        rhs_append = (features.transpose(0, 1) @ masked_y.reshape(-1, 1)) / noise_var
        mean_rhs = torch.cat([mean_rhs, rhs_append.view(1, -1).repeat(mean_rhs.size(0), 1)],
                             dim=1)  # shape [n_batch, T * num_nodes + n_features]

        # run CG iterations
        initial_guess = torch.zeros(*mean_rhs.size(), 1)

        post_mean, cg_info = cg_solve(mean_rhs, dgmrf, mask, data.size(1), config,
                                      noise_var=noise_var, verbose=verbose, return_info=True, features=features,
                                      initial_guess=initial_guess,
                                      preconditioner=preconditioner)

        post_mean = post_mean[0]
    else:
        # run CG iterations
        initial_guess = initial_guess.reshape(mean_rhs.size())
        res_norm = torch.ones(data.size(0)) * float("inf")
        rhs_norm = torch.linalg.norm(mean_rhs, dim=1)
        k = 0
        cg_niter = 0
        regularizer = config.get('cg_regularizer', 0)
        while (res_norm > config.get('outer_rtol', 1e-7) * rhs_norm).any() and (k < config.get('max_outer_iter', 100)):
            post_mean, cg_info = cg_solve(mean_rhs, dgmrf, mask, data.size(1), config,
                                          noise_var=noise_var, verbose=verbose, return_info=True,
                                          initial_guess=initial_guess,
                                          preconditioner=preconditioner, regularizer=regularizer)

            k = k + 1
            cg_niter += cg_info["niter"]
            res_norm = cg_info["res_norm"]
            print(f'relative residual norm after outer iteration {k} = {res_norm / rhs_norm}')
            initial_guess = post_mean.squeeze(-1)
            regularizer = 0.1 * regularizer

        post_mean = post_mean[0]
        cg_info["niter"] = cg_niter

    if return_info:
        return post_mean, cg_info
    else:
        return post_mean

@torch.no_grad()
def posterior_inference(dgmrf, data, mask, config, noise_var, initial_guess, features=None,
                        verbose=False, return_time=False, preconditioner=None):
    # data has shape [n_batch, T, num_nodes]

    print(f'compute posterior mean')

    start = timer()
    post_mean, cg_info = posterior_mean(dgmrf, data, mask, config, noise_var, initial_guess, return_info=True, features=features,
                                        preconditioner=preconditioner, verbose=verbose)
    end = timer()
    time_per_iter = (end - start) / cg_info["niter"]
    mean_niter = cg_info["niter"]

    print(f'draw samples from posterior')

    posterior_samples_list = []
    cur_post_samples = 0
    niter_list = []
    while cur_post_samples < config["n_post_samples"]:
        samples, cg_info = sample_posterior(config["n_training_samples"], dgmrf, data, mask, config,
                                            noise_var, initial_guess, return_info=True, features=features,
                                            preconditioner=preconditioner)
        posterior_samples_list.append(samples)
        cur_post_samples += config["n_training_samples"]
        niter_list.append(cg_info["niter"])

    posterior_samples = torch.cat(posterior_samples_list, dim=0)[:config["n_post_samples"]]

    # MC estimate of variance using known population mean
    post_var_x = torch.mean(torch.pow(posterior_samples - post_mean, 2), dim=0)
    # Posterior std.-dev. for y
    post_std = torch.sqrt(post_var_x + noise_var)

    if return_time:
        return post_mean, post_std, time_per_iter, torch.tensor(niter_list, dtype=torch.float32).mean()
    else:
        return post_mean, post_std, mean_niter


# @torch.no_grad()
# def posterior_inference(dgmrf, graph_y, config):
#     # Posterior mean
#     graph_bias = new_graph(graph_y, new_x=get_bias(dgmrf, graph_y))
#     Q_mu = -1.*dgmrf(graph_bias, transpose=True, with_bias=False) # Q@mu = -G^T@b
#
#     masked_y = graph_y.mask.to(torch.float32).unsqueeze(1) * graph_y.x
#     mean_rhs = Q_mu + (1./torch.exp(2.*config["log_noise_std"])) * masked_y
#
#     # if config["features"]:
#     #     rhs_append = (1./torch.exp(2.*config["log_noise_std"]))*\
#     #         graph_y.features.transpose(0,1)@masked_y
#     #
#     #     mean_rhs = torch.cat((mean_rhs, rhs_append), dim=0)
#
#     post_mean = cg_solve(mean_rhs.unsqueeze(0), dgmrf, graph_y, config,
#             config["inference_rtol"], verbose=True)[0]
#
#     # if config["features"]:
#     #     # CG returns posterior mean of both x and coeff., compute posterior
#     #     post_mean_x = post_mean[:graph_y.num_nodes]
#     #     post_mean_beta = post_mean[graph_y.num_nodes:]
#     #
#     #     post_mean = post_mean_x + graph_y.features@post_mean_beta
#     #
#     #     # Plot posterior mean for x alone
#     #     graph_post_mean_x = new_graph(graph_y, new_x=post_mean_x)
#     #     # vis.plot_graph(graph_post_mean_x, name="post_mean_x", title="X Posterior Mean")
#
#     graph_post_mean = new_graph(graph_y, new_x=post_mean)
#
#     # Posterior samples and marginal variances
#     # Batch sampling
#     posterior_samples_list = []
#     cur_post_samples = 0
#     while cur_post_samples < config["n_post_samples"]:
#         posterior_samples_list.append(sample_posterior(config["n_training_samples"],
#             dgmrf, graph_y, config, config["inference_rtol"], verbose=True))
#         cur_post_samples += config["n_training_samples"]
#
#     posterior_samples = torch.cat(posterior_samples_list,
#            dim=0)[:config["n_post_samples"]]
#
#     if config["features"]:
#         # Include linear feature model to posterior samples
#         post_samples_x = posterior_samples[:,:graph_y.num_nodes]
#         post_samples_coeff = posterior_samples[:,graph_y.num_nodes:]
#
#         posterior_samples = post_samples_x + graph_y.features@post_samples_coeff
#
#     # MC estimate of variance using known population mean
#     post_var_x = torch.mean(torch.pow(posterior_samples - post_mean, 2), dim=0)
#     # Posterior std.-dev. for y
#     post_std = torch.sqrt(post_var_x + torch.exp(2.*config["log_noise_std"]))
#
#     graph_post_std = new_graph(graph_y, new_x=post_std)
#     graph_post_sample = new_graph(graph_y)
#
#     # # Plot posterior samples
#     # for sample_i, post_sample in enumerate(
#     #         posterior_samples[:config["plot_post_samples"]]):
#     #     graph_post_sample.x = post_sample
#     #     vis.plot_graph(graph_post_sample, name="post_sample",
#     #             title="Posterior sample {}".format(sample_i))
#
#     return graph_post_mean, graph_post_std
#
#

