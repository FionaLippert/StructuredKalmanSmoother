import torch
import torch_geometric as ptg
import copy
import numpy as np
import scipy.stats as sps
from structuredKS import cg_batch
import pytorch_lightning as pl



def new_graph(like_graph, new_x=None):
    graph = copy.copy(like_graph) # Shallow copy
    graph.x = new_x
    return graph

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
            self.degrees = graph.get('weighted_degrees', )
        else:
            self.edge_weights = torch.ones(self.edge_index.size(1))
            self.degrees = ptg.utils.degree(self.edge_index[0], num_nodes=self.num_nodes)

        print(f'edge weights = {self.edge_weights}')
        print(f'(weighted) degrees = {self.degrees}')


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
        if with_bias and self.bias is not None:
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

        layer_list = []
        for layer_i in range(config["n_layers"]):
            layer_list.append(DGMRFLayerMultiChannel(config, graph, vi_layer=False, T=T,
                                                     shared=shared, weighted=weighted))
            # layer_list.append(DGMRFLayer(graph, config))

            # Optionally add non-linearities between hidden layers
            # TODO: make this work for multi channel case
            # if config["non_linear"] and (layer_i < (config["n_layers"]-1)):
            #     layer_list.append(DGMRFActivation(config))

        self.layers = torch.nn.ModuleList(layer_list)

    def forward(self, x, transpose=False, with_bias=True, **kwargs):
        # x = data.x

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


class JointDGMRF(torch.nn.Module):
    def __init__(self, config, spatial_graph, temporal_graph=None, T=1, shared='dynamics', weighted=False):
        super().__init__()

        # self.dgmrf_list = torch.nn.ModuleList([DGMRF(g, config) for g in graph_list])
        # self.time_intervals = time_intervals
        self.dgmrf = DGMRF(config, spatial_graph, T=T, shared=shared, weighted=weighted)

        if temporal_graph is not None:
            self.dynamics = TemporalDGMRF(config, spatial_graph, temporal_graph, T=T, shared=shared)

    def forward(self, x, transpose=False, with_bias=True, **kwargs):
        # x has shape [num_samples, T, num_nodes]
        if hasattr(self, 'dynamics') and not transpose:
            x = self.dynamics(x, with_bias=with_bias, v=kwargs.get('v', None))

        # z = [self.dgmrf_list[i](x[:, ti], transpose, with_bias) for i, ti in enumerate(self.time_intervals)]
        # z = torch.cat(z, dim=1)  # shape [num_samples, T, num_nodes]
        z = self.dgmrf(x, transpose, with_bias)

        if hasattr(self, 'dynamics') and transpose:
            z = self.dynamics(z, transpose=True, with_bias=with_bias, v=kwargs.get('v', None))

        return z

    def log_det(self):
        # return sum([len(ti) * self.dgmrf_list[i].log_det() for i, ti in enumerate(self.time_intervals)])
        return self.dgmrf.log_det()


class DirectedDiffusionModel(ptg.nn.MessagePassing):
    """
    Compute x_{t+1} = Fx_t, where F is based on discretized diffusion

    (for now we assume a regular lattice with cell_size = 1x1)
    """

    def __init__(self, config, graph):
        super(DiffusionModel, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

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
        return edge_weight.reshape(1, 1, 1, -1) * torch.stack([x_i, x_j], dim=0)

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
        out = x
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
        return edge_weight.reshape(1, 1, 1, -1) * torch.stack([x_i, x_j], dim=0)

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

        self.weight = torch.nn.Parameter(torch.ones(1, T, 1))

    def forward(self, x, **kwargs):
        return self.weight * x



class GNNAdvection(ptg.nn.MessagePassing):
    def __init__(self, config, temporal_graph, **kwargs):
        super(GNNAdvection, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.edge_index = temporal_graph['edge_index']
        self.edge_index_transpose = self.edge_index.flip(0)
        self.edge_attr = temporal_graph['edge_attr'] #[:2] # normal vectors

        self.edge_dim = self.edge_attr.size(-1)

        self.edge_mlp = torch.nn.Sequential(torch.nn.Linear(self.edge_dim, 10),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(10, 2),
                                            torch.nn.Tanh())

        self.diff_param = torch.nn.Parameter(2 * torch.rand(1,) - 1)

        # self.v_mlp = torch.nn.Sequential(torch.nn.Linear(self.node_dim, 10),
        #                                     torch.nn.ReLU(),
        #                                     torch.nn.Linear(10, 2))

    @property
    def diff_coeff(self):
        return torch.pow(self.diff_param, 2)

    def forward(self, x, transpose=False):
        # x has shape [num_samples, T, num_nodes]
        if transpose:
            edge_index = self.edge_index_transpose
        else:
            edge_index = self.edge_index
        return x + self.propagate(edge_index, x=x, edge_attr=self.edge_attr, transpose=transpose)


    def message(self, x_i, x_j, edge_attr, transpose):
        # if transpose:
        #       inputs = torch.cat([edge_attr, node_attr_j.squeeze(0).T, node_attr_i.squeeze(0).T], dim=-1)
        # else:
        #       inputs = torch.cat([edge_attr, node_attr_i.squeeze(0).T, node_attr_j.squeeze(0).T], dim=-1)
        inputs = edge_attr
        coeffs = self.edge_mlp(inputs) #.squeeze(-1)
        msg = (coeffs[:, 0] + self.diff_coeff) * x_j + (coeffs[:, 1] - self.diff_coeff) * x_i
        return msg

    # def message(self, x_i, x_j, edge_attr, node_attr_i, node_attr_j):
    #     v_i = self.v_mlp(node_attr_i.squeeze(0).T)
    #     v_j = self.v_mlp(node_attr_j.squeeze(0).T)
    #     v = 0.5 * (v_i + v_j)
    #     msg = -0.5 * (edge_attr * v).sum(1) * (x_j + x_i)
    #     return msg

class GNNTransition(ptg.nn.MessagePassing):
    def __init__(self, config, spatial_graph, temporal_graph, **kwargs):
        super(GNNTransition, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.edge_index = temporal_graph['edge_index']
        self.edge_index_transpose = self.edge_index.flip(0)
        self.edge_attr = temporal_graph['edge_attr']
        self.node_attr = spatial_graph['pos']
        self.scale = self.node_attr.max()

        self.edge_dim = self.edge_attr.size(-1)
        self.node_dim = self.node_attr.size(-1)

        self.edge_mlp = torch.nn.Sequential(torch.nn.Linear(self.edge_dim + 2*self.node_dim, 10),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(10, 1))



    def forward(self, x, transpose=False):
        # x has shape [num_samples, T, num_nodes]
        if transpose:
            edge_index = self.edge_index_transpose
        else:
            edge_index = self.edge_index
        return x + self.propagate(edge_index, x=x, edge_attr=self.edge_attr,
                                  node_attr=self.node_attr.T.unsqueeze(0) / self.scale, transpose=transpose)


    def message(self, x_i, x_j, edge_attr, node_attr_i, node_attr_j, transpose):
        if transpose:
              inputs = torch.cat([edge_attr, node_attr_j.squeeze(0).T, node_attr_i.squeeze(0).T], dim=-1)
        else:
              inputs = torch.cat([edge_attr, node_attr_i.squeeze(0).T, node_attr_j.squeeze(0).T], dim=-1)
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
    def __init__(self, config, spatial_graph, temporal_graph, **kwargs):
        super().__init__()
        self.transition_type = config.get('transition_type', 'identity')
        self.n_transitions = config.get('n_transitions', 1)

        self.shared_dynamics = kwargs.get('shared', 'dynamics')

        # setup transition model
        if self.transition_type == 'diffusion':
            self.transition_models = torch.nn.ModuleList([DiffusionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'advection':
            self.transition_models = torch.nn.ModuleList([AdvectionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'GNN_advection':
            self.transition_models = torch.nn.ModuleList([GNNAdvection(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'advection+diffusion':
            self.transition_models = torch.nn.ModuleList([AdvectionDiffusionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'inhomogeneous_advection+diffusion':
            self.transition_models = torch.nn.ModuleList([InhomogeneousAdvectionDiffusionModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'GNN':
            self.transition_models = torch.nn.ModuleList([GNNTransition(config, spatial_graph, temporal_graph, **kwargs)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == 'flow':
            self.transition_models = torch.nn.ModuleList([FlowModel(config, temporal_graph)
                                                          for _ in range(self.n_transitions)])
        elif self.transition_type == "AR":
            T = 1 if self.shared_dynamics == 'dynamics' else (kwargs.get('T', 2) - 1)
            self.transition_models = torch.nn.ModuleList([ARModelMultiChannel(T=T) for _ in range(self.n_transitions)])

        if kwargs.get('use_dynamics_bias', True):
            if self.shared_dynamics:
                self.bias = torch.nn.parameter.Parameter(torch.rand(1,))
            else:
                self.bias = torch.nn.parameter.Parameter(torch.rand(kwargs['T'],))
        else:
            self.bias = torch.zeros(1)

    def forward(self, x, transpose=False, with_bias=True, **kwargs):
        # computes e=Fx
        # x has shape [n_samples, T, num_nodes]

        if transpose:
            states = x[:, 1:]
        else:
            states = x[:, :-1]

        if with_bias:
            states = states + self.bias.reshape(1, -1, 1)

        for l in range(self.n_transitions):
            # TODO: use different parameters for each transition layer?
            # TODO: if so, order needs to be reversed for transpose model
            if hasattr(self, 'transition_models'):
                # if self.transition_type == 'inhomogeneous_advection+diffusion':
                #     v = kwargs.get('v').unsqueeze(2).repeat(1, 1, states.size(1), 1)  # shape [2, n_samples, T-1, num_nodes]
                #     states = self.transition_model(states, v=v, transpose=transpose)
                # else:
                states = self.transition_models[l](states, transpose=transpose)

            # if with_bias:
            #     states = states + self.bias
        # states = torch.cat([torch.zeros_like(x[:, 0]).unsqueeze(1), states], dim=1)
        if transpose:
            states = torch.cat([states, torch.zeros_like(x[:, -1]).unsqueeze(1)], dim=1)
        else:
            states = torch.cat([torch.zeros_like(x[:, 0]).unsqueeze(1), states], dim=1)

        Fx = x - states

        return Fx


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
    def __init__(self, config, graph, initial_guess, T=1, shared='all'):
        super().__init__()

        # Dimensionality of distribution (num_nodes of graph)
        self.dim = get_num_nodes(graph['edge_index'])
        self.T = T

        # Standard amount of samples (must be fixed to be efficient)
        self.n_samples = config["n_training_samples"]
        self.n_post_samples = config["n_post_samples"]
        # self.noise_std = noise_std #config["noise_std"]
        # self.log_noise_std = np.log(config["noise_std"])

        # # Variational distribution, Initialize with observed y
        # self.mean_param = torch.nn.parameter.Parameter(initial_guess)
        # self.diag_param = torch.nn.parameter.Parameter(2*torch.rand(self.dim) - 1.) # U(-1,1)

        # self.layers = torch.nn.ModuleList([DGMRFLayer(graph, config, vi_layer=True)
        #                                    for _ in range(config["vi_layers"])])

        # Variational distribution, Initialize with observed y
        self.mean_param = torch.nn.parameter.Parameter(initial_guess)
        self.diag_param = torch.nn.parameter.Parameter(2 * torch.rand(self.T, self.dim) - 1.)  # U(-1,1)
        self.layers = torch.nn.ModuleList([DGMRFLayerMultiChannel(config, graph, T=T, shared=shared, vi_layer=True,
                                                                  weighted=config.get('weighted_vi', False))
                            for _ in range(config["vi_layers"])])
        # self.layers = torch.nn.ModuleList([DGMRFLayer(graph, config, vi_layer=True)
        #                                    for _ in range(config["vi_layers"])])
        if config["vi_layers"] > 0:
            # self.post_diag_param = torch.nn.parameter.Parameter(
            #     2*torch.rand(self.dim) - 1.)
            self.post_diag_param = torch.nn.parameter.Parameter(
                2 * torch.rand(self.T, self.dim) - 1.)

        # Reuse same batch with different x-values
        # self.sample_batch = ptg.data.Batch.from_data_list([
        #     ptg.data.Data(edge_index=graph.edge_index, num_nodes=self.dim) for _ in range(self.n_samples)])

        if config["features"]:
            # Additional variational distribution for linear coefficients
            # TODO adjust for multi-channel setting
            n_features = graph.features.shape[1]
            self.coeff_mean_param = torch.nn.parameter.Parameter(torch.randn(n_features))
            self.coeff_diag_param = torch.nn.parameter.Parameter(
                2*torch.rand(n_features) - 1.) # U(-1,1)

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
        standard_sample = torch.randn(self.n_samples, self.T, self.dim)
        samples = self.std * standard_sample # [T, dim] * [samples, T, dim]

        for layer in self.layers:
            propagated = layer(samples, transpose=False, with_bias=False)
            samples = propagated

        if self.layers:
            # Apply post diagonal matrix
            samples  = self.post_diag.unsqueeze(0) * samples # [1, T, dim] * [samples, T, dim]
        samples = samples + self.mean_param.unsqueeze(0) # [samples, T, dim] + [1, T, dim]
        return samples # shape (n_samples, T, dim)

    def log_det(self):
        layers_log_det = sum([layer.log_det() for layer in self.layers])
        std_log_det = torch.sum(torch.log(self.std))
        total_log_det = 2.0*std_log_det + 2.0*layers_log_det

        if self.layers:
            post_diag_log_det = torch.sum(torch.log(self.post_diag))
            total_log_det = total_log_det + 2.0*post_diag_log_det

        return total_log_det

    def sample_coeff(self, n_samples):
        # TODO: adjust to multi-channel setting
        standard_sample = torch.randn(self.coeff_mean_param.shape[0], n_samples)
        samples = (self.coeff_std * standard_sample) + self.coeff_mean_param
        return samples # shape (n_features, n_samples)

    def log_det_coeff(self):
        return 2.0*torch.sum(torch.log(self.coeff_std))

    def ce_coeff(self):
        # TODO: adjust to multi-channel setting
        # Compute Cross-entropy term (CE between VI-dist and coeff prior)
        return -0.5*(self.coeff_inv_std**2)*torch.sum(
                torch.pow(self.coeff_std, 2) + torch.pow(self.coeff_mean_param, 2))

    @torch.no_grad()
    def posterior_estimate(self, noise_var):
        # Compute mean and marginal std of distribution (posterior estimate)

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


def crps_score(pred_mean, pred_std, target):
    # Inputs should be numpy arrays
    z = (target - pred_mean)/pred_std

    crps = pred_std*((1./np.sqrt(np.pi)) - 2*sps.norm.pdf(z) - z*(2*sps.norm.cdf(z) - 1))
    return (-1.)*np.mean(crps) # Negative crps, so lower is better

def int_score(pred_mean, pred_std, target, alpha=0.05):
    lower_std, upper_std = sps.norm.interval(1.-alpha)
    lower = pred_mean + pred_std*lower_std
    upper = pred_mean + pred_std*upper_std

    int_score = (upper - lower) + (2/alpha)*(lower-target)*(target < lower) +\
        (2/alpha)*(target-upper)*(target > upper)

    return np.mean(int_score)

class SpatiotemporalInference(pl.LightningModule):

    def __init__(self, config, initial_guess, data, joint_mask, spatial_graph, temporal_graph=None,
                 T=1, gt=None, **kwargs):
        #def __init__(self, graphs, initial_guess, config):
        super(SpatiotemporalInference, self).__init__()
        self.save_hyperparameters()

        self.config = config

        # model settings
        self.learning_rate = config.get('lr', 0.01)

        # self.T = len(graphs)
        self.T = T
        self.num_nodes = get_num_nodes(spatial_graph['edge_index'])
        # self.N = graphs["latent"].num_nodes
        self.N = self.T * self.num_nodes
        print(f'N={self.N}, T={self.T}')

        # self.y = graphs["data"].x
        # self.mask = graphs["latent"].mask
        self.y = data
        self.mask = joint_mask # shape [T * num_nodes]
        self.y_masked = torch.ones(self.mask.size(), dtype=self.y.dtype) * np.nan
        self.y_masked[self.mask] = self.y

        # self.pos = graphs.get_example(0)["latent"].pos
        self.pos = spatial_graph.get('pos', None)
        # if hasattr(graphs["latent"], 'x'):
        #     self.gt = graphs["latent"].x
        if not gt is None:
            self.gt = gt # shape [T * num_nodes]


        # self.noise_std = config.get("noise_std")
        if config["learn_noise_std"]:
            self.obs_noise_param = torch.nn.parameter.Parameter(torch.tensor(config["noise_std"]))
        else:
            self.obs_noise_param = torch.tensor(config["noise_std"])

        # self.log_noise_std = np.log(self.noise_std)

        self.use_dynamics = config.get("use_dynamics", False)
        self.use_hierarchy = config.get("use_hierarchy", False)
        self.independent_time = config.get("independent_time", False)

        self.data_mean = kwargs.get('data_mean', 0)
        self.data_std = kwargs.get('data_std', 1)

        # model components
        if self.use_dynamics:
            # use dynamic prior with transition matrix F_t
            # graph_list = [graphs.get_example(0)["latent", "spatial", "latent"],
            #               graphs.get_example(1)["latent", "spatial", "latent"]]
            # time_intervals = [torch.tensor([0]), torch.arange(1, self.T)]
            # dynamics_graph = graphs.get_example(1)["latent", "temporal", "latent"]
            # self.dgmrf = JointDGMRF(graph_list, time_intervals, config, dynamics_graph, pos=self.pos)
            # self.vi_dist = ParallelVI(graphs, initial_guess, config)

            # self.dgmrf = DGMRF(spatial_graph, config, num_channels=self.T, shared_dynamics=True)
            shared = 'none' if config.get('independent_time', True) else 'dynamics'
            self.dgmrf = JointDGMRF(config, spatial_graph, temporal_graph, T=self.T, shared=shared,
                                    weighted=config.get('weighted_dgmrf', False))

            self.input_shape = [self.T, self.num_nodes]
            shared_vi = 'none'

            if self.use_hierarchy:
                # use DGMRF with shared parameters across time for latent v
                # self.dgmrf_vx = DGMRF(graphs.get_example(0)["latent", "spatial", "latent"], config)
                # self.vi_dist_vx = VariationalDist(config, graphs.get_example(0)["latent", "spatial", "latent"],
                #                                   torch.zeros(self.N // self.T))
                # self.dgmrf_vy = DGMRF(graphs.get_example(0)["latent", "spatial", "latent"], config)
                # self.vi_dist_vy = VariationalDist(config, graphs.get_example(0)["latent", "spatial", "latent"],
                #                                   torch.zeros(self.N // self.T))
                self.dgmrf_vx = DGMRF(config, spatial_graph, T=1, weighted=config.get('weighted_dgmrf', False))
                self.vi_dist_vx = VariationalDist(config, spatial_graph, torch.zeros(self.num_nodes), T=1)
                self.dgmrf_vy = DGMRF(config, spatial_graph, T=1, weighted=config.get('weighted_dgmrf', False))
                self.vi_dist_vy = VariationalDist(config, spatial_graph, torch.zeros(self.num_nodes), T=1)

        elif self.independent_time:
            # treat time steps independently, with separate DGMRF for each time step
            # graph_list = [graphs.get_example(t)["latent", "spatial", "latent"] for t in range(self.T)]
            # time_intervals = [torch.tensor([t]) for t in range(self.T)]
            # self.dgmrf = JointDGMRF(graph_list, time_intervals, config)
            self.dgmrf = DGMRF(config, spatial_graph, T=self.T, shared='none', weighted=config.get('weighted_dgmrf', False))
            # self.vi_dist = ParallelVI(graphs, initial_guess, config)
            # self.vi_dist = VariationalDist(config, spatial_graph, initial_guess.reshape(self.T, -1), num_channels=self.T)

            self.input_shape = [self.T, self.num_nodes]
            shared_vi = 'none'
        else:
            # use a single DGMRF, with parameters shared across time steps
            self.dgmrf = DGMRF(config, spatial_graph, T=self.T, shared='all', weighted=config.get('weighted_dgmrf', False))
            # self.vi_dist = VariationalDist(config, spatial_graph, initial_guess, num_channels=self.T)

            # self.input_shape = [1, self.num_nodes]
            self.input_shape = [self.T, self.num_nodes]
            shared_vi = 'all'

        # shared_vi = "all"
        self.vi_dist = VariationalDist(config, spatial_graph, initial_guess.reshape(self.T, -1),
                                       T=self.T, shared=shared_vi)

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

    def _reconstruction_loss(self, x, index):
        # x has shape [n_samples, T * n_nodes]
        y_hat = self.obs_model(x)
        residuals = self.y[index] - y_hat[:, index]
        rec_loss = torch.mean(torch.pow(residuals, 2))

        return rec_loss

    def _joint_log_likelihood(self, x, index, v=None):
        # x has shape [n_samples, T, n_nodes]
        # v has shape [2, n_samples, n_nodes]
        # N = x.size(-1) * x.size(-2)

        # compute log-likelihood of samples given prior p(x)
        Gx = self.dgmrf(x, v=v)  # shape (n_samples, T, n_nodes)
        prior_ll = (-0.5 * torch.sum(torch.pow(Gx, 2)) + self.dgmrf.log_det()) / self.N

        ## compute log-likelihood of latent field v
        if self.use_hierarchy:
            Gvx = self.dgmrf_vx(v[0]) #.reshape(-1, self.N))
            Gvy = self.dgmrf_vy(v[1]) #.reshape(-1, self.N))
            vx_ll = (-0.5 * torch.sum(torch.pow(Gvx, 2)) + self.dgmrf_vx.log_det()) / (self.num_nodes)
            vy_ll = (-0.5 * torch.sum(torch.pow(Gvy, 2)) + self.dgmrf_vy.log_det()) / (self.num_nodes)
            prior_ll = prior_ll + vx_ll + vy_ll

        # compute data log-likelihood given samples
        x = x.reshape(self.n_training_samples, -1)  # shape [n_samples, T * num_nodes]
        rec_loss = self._reconstruction_loss(x, index)
        data_ll = -0.5 * rec_loss / self.noise_var - self.log_noise_std

        self.log("train_rec_loss", rec_loss.item(), sync_dist=True)
        self.log("train_prior_ll", prior_ll.item(), sync_dist=True)
        self.log("train_data_ll", data_ll.item(), sync_dist=True)

        return prior_ll + data_ll



    def training_step(self, training_index):
        #training_mask = batch['training_mask']

        # sample from variational distribution
        samples = self.vi_dist.sample()  # shape [n_samples, T, num_nodes]
        # N = samples.size(-1) * samples.size(-2)

        # compute entropy of variational distribution
        vi_entropy = 0.5 * self.vi_dist.log_det() / self.N

        if self.use_hierarchy:
            v_samples = torch.stack([self.vi_dist_vx.sample(), self.vi_dist_vy.sample()], dim=0).reshape(
                2, self.n_training_samples, -1) # .reshape(2, self.n_training_samples, self.T, -1)
            joint_ll = self._joint_log_likelihood(samples, training_index.squeeze(), v=v_samples)
            vi_entropy = vi_entropy + 0.5 * (self.vi_dist_vx.log_det() + self.vi_dist_vy.log_det()) / (self.num_nodes)
        else:
            joint_ll = self._joint_log_likelihood(samples, training_index.squeeze())

        elbo = joint_ll + vi_entropy

        self.log("train_elbo", elbo.item(), sync_dist=True)
        self.log("vi_entropy", vi_entropy.item(), sync_dist=True)

        return -elbo


    def validation_step(self, val_index, *args):

        samples = self.vi_dist.sample()

        pd_check = (samples * self.dgmrf(samples)).reshape(self.n_training_samples, -1).sum(1)
        if not (pd_check > 0).all():
            print(f'pd check failed: min value = {pd_check.min()}')


        samples = samples.reshape(self.n_training_samples, -1)
        rec_loss = self._reconstruction_loss(samples, val_index.squeeze())

        self.log("val_rec_loss", rec_loss.item(), sync_dist=True)

        # vi_mean, vi_std = self.vi_dist.posterior_estimate()
        # val_dict = {'vi_mean': vi_mean,
        #             'vi_std': vi_std,
        #             'val_index': val_index}

        # return val_dict

    def test_step(self, test_index, *args):
        # posterior inference using variational distribution
        vi_mean, vi_std = self.vi_dist.posterior_estimate(self.noise_var)
        mean = vi_mean.flatten()
        std = vi_std.flatten()

        data = torch.zeros(self.mask.size(), dtype=self.y.dtype)
        data[self.mask] = self.y # use self.y_masked?
        # TODO: fix CG (dist becomes negative)

        # cg_mean = posterior_mean(self.dgmrf, data.reshape(1, *self.input_shape),
        #                            self.mask, self.config, self.noise_var, initial_guess=vi_mean)

        cg_mean, cg_std = posterior_inference(self.dgmrf, data.reshape(1, *self.input_shape),
                                              self.mask, self.config, self.noise_var)#, initial_guess=vi_mean)

        if hasattr(self, 'gt'):
            # use unobserved nodes to evaluate predictions
            # test_index indexes latent space
            test_mask = torch.logical_not(self.mask)
            gt_mean = self.gt[test_mask]# * self.data_std + self.data_mean
            residuals_vi = (self.gt[test_mask] - mean[test_mask])# * self.data_std + self.data_mean
            residuals_cg = (self.gt[test_mask] - cg_mean[test_mask])# * self.data_std + self.data_mean

            self.log("test_mae_vi", residuals_vi.abs().mean().item(), sync_dist=True)
            self.log("test_rmse_vi", torch.pow(residuals_vi, 2).mean().sqrt().item(), sync_dist=True)
            self.log("test_mse_vi", torch.pow(residuals_vi, 2).mean().item(), sync_dist=True)
            self.log("test_mape_vi", (residuals_vi / gt_mean).abs().mean().item(), sync_dist=True)

            self.log("test_mae_cg", residuals_cg.abs().mean().item(), sync_dist=True)
            self.log("test_rmse_cg", torch.pow(residuals_cg, 2).mean().sqrt().item(), sync_dist=True)
            self.log("test_mse_cg", torch.pow(residuals_cg, 2).mean().item(), sync_dist=True)
            self.log("test_mape_cg", (residuals_cg / gt_mean).abs().mean().item(), sync_dist=True)

            pred_mean_np = mean[test_mask].cpu().numpy()
            pred_std_np = std[test_mask].cpu().numpy()
            target_np = self.gt[test_mask].cpu().numpy()

            self.log("test_crps_vi", crps_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)
            self.log("test_int_score_vi", int_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)

            pred_mean_np = cg_mean[test_mask].cpu().numpy()
            pred_std_np = cg_std[test_mask].cpu().numpy()
            target_np = self.gt[test_mask].cpu().numpy()

            self.log("test_crps_cg", crps_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)
            self.log("test_int_score_cg", int_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)

        else:
            # use held out part of data to evaluate predictions
            # test_index indexes data space
            masked_mean = self.obs_model(mean.unsqueeze(0)).squeeze(0)
            masked_std = self.obs_model(std.unsqueeze(0)).squeeze(0)
            masked_mean_cg = self.obs_model(cg_mean.unsqueeze(0)).squeeze(0)
            data = self.y[test_index]# * self.data_std + self.data_mean
            residuals_vi = (self.y[test_index] - masked_mean[test_index])# * self.data_std + self.data_mean
            residuals_cg = (self.y[test_index] - masked_mean_cg[test_index])# * self.data_std + self.data_mean

            self.log("test_mae_vi", residuals_vi.abs().mean().item(), sync_dist=True)
            self.log("test_rmse_vi", torch.pow(residuals_vi, 2).mean().sqrt().item(), sync_dist=True)
            self.log("test_mape_vi", (residuals_vi / data).abs().mean().item(), sync_dist=True)

            self.log("test_mae_cg", residuals_cg.abs().mean().item(), sync_dist=True)
            self.log("test_rmse_cg", torch.pow(residuals_cg, 2).mean().sqrt().item(), sync_dist=True)
            self.log("test_mape_cg", (residuals_cg / data).abs().mean().item(), sync_dist=True)

            # TODO: std from posterior_estimate can't be used as is, or can it??
            pred_mean_np = masked_mean[test_index].cpu().numpy()
            pred_std_np = masked_std[test_index].cpu().numpy()
            target_np = self.y[test_index].cpu().numpy()

            self.log("test_crps", crps_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)
            self.log("test_int_score", int_score(pred_mean_np, pred_std_np, target_np), sync_dist=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def vi_loss(dgmrf, vi_dist, graph_y, config):
    vi_samples = vi_dist.sample()
    vi_log_det = vi_dist.log_det()
    vi_dist.sample_batch.x = vi_samples.reshape(-1,1)
    # Column vector of node values for all samples

    g = dgmrf(vi_dist.sample_batch) # Shape (n_training_samples*n_nodes, 1)

    # Construct loss
    l1 = 0.5*vi_log_det
    l2 = -graph_y.n_observed*config["log_noise_std"]
    l3 = dgmrf.log_det()
    l4 = -(1./(2. * config["n_training_samples"])) * torch.sum(torch.pow(g,2))

    if config["features"]:
        vi_coeff_samples = vi_dist.sample_coeff(config["n_training_samples"])
        # Mean from a VI sample (x + linear feature model)
        vi_samples = vi_samples + vi_coeff_samples@graph_y.features.transpose(0,1)

        # Added term when using additional features
        vi_coeff_log_det = vi_dist.log_det_coeff()
        entropy_term = 0.5*vi_coeff_log_det
        ce_term = vi_dist.ce_coeff()

        l1 = l1 + entropy_term
        l4 = l4 + ce_term

    l5 = -(1./(2. * torch.exp(2.*config["log_noise_std"]) *\
        config["n_training_samples"]))*torch.sum(torch.pow(
            (vi_samples - graph_y.x.flatten()), 2)[:, graph_y.mask])

    elbo = l1 + l2 + l3 + l4 + l5
    loss = (-1./graph_y.num_nodes)*elbo
    return loss




# Solve Q_tilde x = rhs using Conjugate Gradient
def cg_solve(rhs, dgmrf, mask, T, config, noise_var=None, verbose=False, initial_guess=None):
    # rhs has shape [n_batch, T * n_nodes]
    # n_nodes = graph_y.num_nodes
    # x_dummy = torch.zeros(rhs.shape[0],n_nodes,1)
    # graph_list = [new_graph(graph_y, new_x=x_part) for x_part in x_dummy]
    # graph_batch = ptg.data.Batch.from_data_list(graph_list)

    n_batch = rhs.size(0)
    N = rhs.size(1)

    # CG requires more precision for numerical stability
    rhs = rhs.to(torch.float64)
    #input_shape = rhs.size()

    # Batch linear operator
    def Q_tilde_batched(x):
        # x has shape (n_batch, T * n_nodes, 1)
        # Implicitly applies posterior precision matrix Q_tilde to a vector x
        # y = ((G^T)G + (sigma^-2)(I_m))x

        # Modify graph batch
        # graph_batch.x = x.reshape(-1,1)

        Gx = dgmrf(x.reshape(n_batch, T, -1), with_bias=False)
        # Gx = dgmrf(x.reshape(n_batch, -1), with_bias=False)
        GtGx = dgmrf(Gx, transpose=True, with_bias=False) # has shape [n_batch, T, n_nodes]

        # noise_add = mask.to(torch.float64) / (config["noise_std"]**2) # shape [T * n_nodes]

        # compute Omega^+ @ x (i.e. posterior precision matrix multiplied x)
        res = GtGx.view(n_batch, N, 1) #+ noise_add.view(1, N, 1) * x # shape [n_batch, T * n_nodes, 1]
        # res = GtGx.unsqueeze(-1) + noise_add.view(1, -1, 1) * x  # shape [n_batch, T * n_nodes, 1]

        if noise_var is not None:
            res = res + (mask.to(torch.float64) / noise_var).view(1, N, 1) * x

        return res

    # if config["features"]:
    #     # Feature matrix with 0-rows for masked nodes
    #     masked_features = graph_y.features * graph_y.mask.to(torch.float64).unsqueeze(1)
    #     masked_features_cov = masked_features.transpose(0,1)@masked_features
    #
    #     noise_precision = 1/torch.exp(2.*config["log_noise_std"])
    #
    #     def Q_tilde_batched_with_features(x):
    #         # x has shape (n_batch, n_nodes+n_features, 1)
    #         node_x = x[:,:n_nodes]
    #         coeff_x = x[:,n_nodes:]
    #
    #         top_res1 = Q_tilde_batched(node_x)
    #         top_res2 = noise_precision*masked_features@coeff_x
    #
    #         bot_res1 = noise_precision*masked_features.transpose(0,1)@node_x
    #         bot_res2 = noise_precision*masked_features_cov@coeff_x +\
    #                 (config["coeff_inv_std"]**2)*coeff_x
    #
    #         res = torch.cat((
    #             top_res1 + top_res2,
    #             bot_res1 + bot_res2,
    #             ), dim=1)
    #
    #         return res

        # Q_tilde_func = Q_tilde_batched_with_features
    # else:
    Q_tilde_func = Q_tilde_batched

    solution, cg_info = cg_batch.cg_batch(Q_tilde_func, rhs.unsqueeze(-1), X0=initial_guess,
                                          rtol=config["inference_rtol"], maxiter=config["max_cg_iter"], verbose=verbose)

    if verbose:
        print("CG finished in {} iterations, solution optimal: {}".format(
            cg_info["niter"], cg_info["optimal"]))

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
def sample_posterior(n_samples, dgmrf, data, mask, config, noise_var):
    # Construct RHS using Papandeous and Yuille method
    bias = get_bias(dgmrf, data.size())
    T, num_nodes = data.shape[1:]
    std_gauss1 = torch.randn(n_samples, T, num_nodes)
    std_gauss2 = torch.randn(n_samples, T, num_nodes)

    rhs_sample1 = dgmrf(std_gauss1 - bias, transpose=True, with_bias=False) # G^T @ (z_1 - b)

    float_mask = mask.to(torch.float32).reshape(1, T, num_nodes)

    rhs_sample2 = float_mask * (data + noise_var.sqrt() * std_gauss2) / noise_var # H^T @ R^{-1} @ (y + z_2)

    rhs_sample = rhs_sample1 + rhs_sample2
    # Shape (n_samples, n_nodes, 1)

    # if config["features"]:
    #     # Change rhs to also sample coefficients
    #     n_features = graph_y.features.shape[1]
    #     std_gauss1_coeff = torch.randn(n_samples, n_features, 1)
    #     rhs_sample1_coeff = config["coeff_inv_std"]*std_gauss1_coeff
    #
    #     rhs_sample2_coeff = graph_y.features.transpose(0,1)@rhs_sample2
    #     rhs_sample_coeff = rhs_sample1_coeff + rhs_sample2_coeff
    #
    #     rhs_sample = torch.cat((rhs_sample, rhs_sample_coeff), dim=1)

    # Solve using Conjugate gradient
    samples = cg_solve(rhs_sample.reshape(n_samples, -1), dgmrf, mask, data.size(1), config,
                       noise_var=noise_var, verbose=False)

    return samples


@torch.no_grad()
def sample_prior(n_samples, dgmrf, data_shape, config):

    bias = get_bias(dgmrf, data_shape)
    std_gauss = torch.randn(n_samples, data_shape[1], data_shape[2])

    rhs_sample = dgmrf(std_gauss - bias, transpose=True, with_bias=False) # G^T @ (z - b)

    # Solve using Conjugate gradient
    samples = cg_solve(rhs_sample.reshape(1, -1), dgmrf, torch.ones(data_shape[1]*data_shape[2]), data_shape[1], config,
            verbose=False)

    return samples



@torch.no_grad()
def posterior_mean(dgmrf, data, mask, config, noise_var, initial_guess=None):
    # data has shape [n_batch, T, num_nodes]
    bias = get_bias(dgmrf, data.size())
    print(f'bias = {bias}')
    eta = -1. * dgmrf(bias, transpose=True, with_bias=False) # -G^T @ b
    print(f'eta = {eta}')
    masked_y = mask.to(torch.float32).reshape(1, *data.shape[1:]) * data # H^T @ y

    mean_rhs = eta + masked_y / noise_var # eta + H^T @ R^{-1} @ y

    # mean_rhs = eta

    mean_rhs = mean_rhs.reshape(data.size(0), -1)

    post_mean = cg_solve(mean_rhs.reshape(data.size(0), -1), dgmrf, mask, data.size(1), config,
                         noise_var=noise_var, verbose=True)[0] #,
                         # initial_guess=initial_guess.reshape(*mean_rhs.size(), 1))[0]

    return post_mean

@torch.no_grad()
def posterior_inference(dgmrf, data, mask, config, noise_var, initial_guess=None, verbose=False):
    # data has shape [n_batch, T, num_nodes]
    bias = get_bias(dgmrf, data.size())
    eta = -1. * dgmrf(bias, transpose=True, with_bias=False) # -G^T @ b

    masked_y = mask.to(torch.float32).reshape(1, *data.shape[1:]) * data # H^T @ y

    mean_rhs = eta + masked_y / noise_var # eta + H^T @ R^{-1} @ y

    # mean_rhs = eta

    mean_rhs = mean_rhs.reshape(data.size(0), -1)

    print(f'compute posterior mean')
    post_mean = cg_solve(mean_rhs.reshape(data.size(0), -1), dgmrf, mask, data.size(1), config,
                         noise_var=noise_var, verbose=False)[0] #,
                         # initial_guess=initial_guess.reshape(*mean_rhs.size(), 1))[0]

    print(f'draw samples from posterior')
    posterior_samples_list = []
    cur_post_samples = 0
    while cur_post_samples < config["n_post_samples"]:
        posterior_samples_list.append(sample_posterior(config["n_training_samples"], dgmrf, data, mask, config, noise_var))
        cur_post_samples += config["n_training_samples"]

    posterior_samples = torch.cat(posterior_samples_list, dim=0)[:config["n_post_samples"]]

    # MC estimate of variance using known population mean
    post_var_x = torch.mean(torch.pow(posterior_samples - post_mean, 2), dim=0)
    # Posterior std.-dev. for y
    post_std = torch.sqrt(post_var_x + noise_var)

    return post_mean, post_std


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

