


class DirectedDiffusionModel(ptg.nn.MessagePassing):
    """
    Compute x_{t+1} = Fx_t, where F is based on discretized diffusion

    (for now we assume a regular lattice with cell_size = 1x1)
    """

    def __init__(self, config, graph):
        super(DirectedDiffusionModel, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.K = config.get('diff_K', 1)
        print(f'K = {self.K}')
        self.diff_param_forward = torch.nn.Parameter(2 * torch.rand(self.K, ) - 1)
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
        self.diff_param = torch.nn.Parameter(2 * torch.rand(self.K, ) - 1)
        self.ar_weight = torch.nn.Parameter(torch.zeros(1, ))
        self.edge_index = graph['edge_index']
        self.edge_index_backward = self.edge_index.flip(0)
        self.edge_weights = graph.get('edge_weight', torch.ones(self.edge_index.size(1)))

    @property
    def diff_coef(self):
        # force diffusion coefficient to be positive
        return self.diff_param

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


class StaticTransition(MessagePassing):
    """
    Used to learn static transition function F
    """

    def __init__(self, indices, **kwargs):
        """
        :param indices: row and column indices of non-zero entries of transition matrix F
        :param kwargs: optional key word arguments, such as initial_weights
        """
        super(StaticTransition, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.transition_graph = indices
        self.num_edges = self.transition_graph.size(1)

        # initialize edge weights corresponding to identity matrix
        self.identity_weights = torch.zeros(self.num_edges) + \
                                torch.eq(self.transition_graph[0], self.transition_graph[1])
        weights = kwargs.get('initial_weights', self.identity_weights)
        self.edge_weights = nn.parameter.Parameter(weights)

    def reset_weights(self, weights=None):
        with torch.no_grad():
            if weights is None:
                self.edge_weights = nn.parameter.Parameter(self.identity_weights)
            else:
                assert weights.size() == self.edge_weights.size()
                self.edge_weights = nn.parameter.Parameter(weights)

    def forward(self, node_states):

        node_features = self.propagate(self.transition_graph, x=node_states, edge_attr=self.edge_weights)

        return node_features

    def message(self, x_j, edge_attr):
        # construct messages to node i for each edge (j,i)

        msg = edge_attr * x_j

        return msg

    def to_dense(self):
        F = tg.utils.to_dense_adj(self.transition_graph, edge_attr=self.edge_weights)
        return F


class FlowModel(ptg.nn.MessagePassing):
    """
    Compute x_{t+1} = Fx_t, where F is based on discretized diffusion

    (for now we assume a regular lattice with cell_size = 1x1)
    """

    def __init__(self, config, graph):
        super(FlowModel, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.param_self = torch.nn.Parameter(torch.ones(1, ))
        self.edge_index = graph['edge_index']
        self.edge_index_backward = self.edge_index.flip(0)
        self.edge_weights = graph.get('edge_weight', torch.ones(self.edge_index.size(1)))
        self.edge_attr = graph.get('edge_attr', torch.ones(self.edge_index.size(1), 1))

        self.edge_mlp = torch.nn.Sequential(torch.nn.Linear(self.edge_attr.size(1), 10, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(10, 1))

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

        return torch.stack([weights * x_i, weights * x_j], dim=0)


class ARModel(torch.nn.Module):
    """
    Compute x_{t+1} = theta * x_t
    """

    def __init__(self):
        super(ARModel, self).__init__()

        self.weight = torch.nn.Parameter(torch.ones(1, ))

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
        edge_attr = temporal_graph['edge_attr']  # [:2] # normal vectors
        edge_weights = temporal_graph.get('edge_weight', torch.ones(edge_attr.size(0)))
        self.edge_features = torch.cat([edge_attr, edge_weights.unsqueeze(1)], dim=1).to(torch.float32)

        self.edge_dim = self.edge_features.size(1)
        self.n_features = n_features

        hidden_dim = config.get('GNN_hidden_dim', 8)

        self.edge_mlp = torch.nn.Sequential(torch.nn.Linear(self.edge_dim + n_features, hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, 2),
                                            torch.nn.Tanh())

        self.diff_param = torch.nn.Parameter(2 * torch.rand(1, ) - 1)

    @property
    def diff_coeff(self):
        return torch.pow(self.diff_param, 2)

    def forward(self, x, transpose=False, features=None, **kwargs):
        # x has shape [num_samples, T, num_nodes]

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

        return new_x

    def message(self, x_i, x_j, edge_attr, transpose, node_attr_i=None, node_attr_j=None):

        if node_attr_i is None or node_attr_j is None:
            inputs = edge_attr
        else:
            edge_attr = edge_attr.unsqueeze(0).repeat(node_attr_i.size(0), 1, 1)

            inputs = torch.cat([edge_attr, node_attr_i.transpose(1, 2)], dim=-1)

        coeffs = self.edge_mlp(inputs)  # .squeeze(-1)

        msg_i = (coeffs[..., 1] - self.diff_coeff) * x_i
        msg_j = (coeffs[..., 0] + self.diff_coeff) * x_j

        return torch.stack([msg_i, msg_j], dim=0)


class GNNTransition(ptg.nn.MessagePassing):
    def __init__(self, config, temporal_graph, n_features, **kwargs):
        super(GNNTransition, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.edge_index = temporal_graph['edge_index']
        self.edge_index_transpose = self.edge_index.flip(0)
        edge_attr = temporal_graph['edge_attr']  # [:2] # normal vectors
        edge_weights = temporal_graph.get('edge_weight', torch.ones(edge_attr.size(0)))
        self.edge_features = torch.cat([edge_attr, edge_weights.unsqueeze(1)], dim=1).to(torch.float32)

        self.edge_dim = self.edge_features.size(1)
        self.n_features = n_features

        hidden_dim = config.get('GNN_hidden_dim', 8)

        self.edge_mlp = torch.nn.Sequential(torch.nn.Linear(self.edge_dim + n_features, hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, 1),
                                            )

    def forward(self, x, transpose=False, features=None):
        # assert features.size(-1) == self.n_features
        # x has shape [num_samples, T, num_nodes]
        # features has shape [T, num_nodes, num_features]
        edge_index = self.edge_index_transpose if transpose else self.edge_index

        if features is None:
            agg = self.propagate(edge_index, x=x, edge_attr=self.edge_features, transpose=transpose)
        else:
            agg = self.propagate(edge_index, x=x, node_attr=features.transpose(1, 2),
                                 edge_attr=self.edge_features, transpose=transpose)

        new_x = x + agg

        return new_x

    def message(self, x_j, edge_attr, node_attr_i=None, node_attr_j=None, transpose=False):
        # edge_attr has shape [num_edges, num_features]
        # covariates has shape [T, num_features, num_edges]

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
        self.edge_attr = graph['edge_attr']  # normal vectors at cell boundaries

        self.config = config

    @property
    def diff_coeff(self):
        # force diffusion coefficient to be positive
        return torch.pow(self.diff_param, 2)

    def forward(self, x, transpose=False, **kwargs):
        # compute F_t @ x

        update = x
        out = x

        factor = 1

        for k in range(self.config.get('k_max', 1)):
            # approximate matrix exponential up to order k_max

            factor = factor / (k + 1)

            agg_i, agg_j = self.propagate(self.edge_index, x=update, edge_attr=self.edge_attr)

            if transpose:
                agg_i_T, agg_j_T = self.propagate(self.edge_index_transpose, x=update, edge_attr=self.edge_attr)
                update = agg_j_T + agg_i
            else:
                update = agg_j + agg_i

            out = out + factor * update

        return out

    def message(self, x_i, x_j, edge_attr):
        # construct messages to node i for each edge (j,i)
        # edge_attr has shape [num_edges, 2]
        # velocity has shape [2]
        adv_coef = -0.5 * (edge_attr * self.velocity).sum(1)
        msg_i = (adv_coef - self.diff_coeff) * x_i
        msg_j = (adv_coef + self.diff_coeff) * x_j

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
        velocity = 0.5 * (v_i + v_j).permute(1, 2, 3, 0)  # shape [num_samples, T, num_edges, 2]
        msg = -0.5 * (edge_attr * velocity).sum(-1) * (x_j + x_i) + self.diff_coeff * (x_j - x_i)
        return msg


class AdvectionModel(ptg.nn.MessagePassing):
    def __init__(self, config, graph):
        super(AdvectionModel, self).__init__(aggr='add', flow="target_to_source", node_dim=-1)

        self.velocity = torch.nn.Parameter(2 * torch.rand(2, ) - 1)
        self.edge_index = graph['edge_index']
        self.edge_index_transpose = self.edge_index.flip(0)
        self.edge_attr = graph['edge_attr']  # normal vectors at cell boundaries

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

        return torch.stack([msg_i, msg_j], dim=0)
