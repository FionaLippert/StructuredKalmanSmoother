import torch
from torch import nn
from torch_geometric.nn import MessagePassing
import torch_geometric as tg
import torch.nn.functional as F


class StaticTransition(MessagePassing):
    """
    Used to learn static transition function F
    """

    def __init__(self, indices, **kwargs):
        """
        :param indices: row and column indices of non-zero entries of transition matrix F
        :param kwargs: optional key word arguments, such as initial_weights
        """
        super(StaticTransition, self).__init__(aggr='add', flow="target_to_source", node_dim=1)

        torch.manual_seed(kwargs.get('seed', 1234))

        self.transition_graph = indices
        self.num_edges = self.transition_graph.size(1)

        # initialize edge weights corresponding to identity matrix
        self.identity_weights = torch.zeros(self.num_edges) + \
                                torch.eq(self.transition_graph[0], self.transition_graph[1])
        weights = kwargs.get('initial_weights', self.identity_weights)
        self.edge_weights = nn.parameter.Parameter(weights)

        # self.reset_parameters()

    def reset_weights(self, weights=None):
        with torch.no_grad():
            if weights is None:
                self.edge_weights = nn.parameter.Parameter(self.identity_weights)
            else:
                assert weights.size() == self.edge_weights.size()
                self.edge_weights = nn.parameter.Parameter(weights)

    def reset_parameters(self):
        raise NotImplementedError

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

class GNN(MessagePassing):
    """
    Message passing in temporal graph. Used to learn transition function F
    Can the same class be used to learn Q?
    """

    def __init__(self, n_layers, n_node_features, n_edge_features, node_dim, edge_dim, **kwargs):
        super(GNN, self).__init__(aggr='add', node_dim=0)

        seed = kwargs.get('seed', 1234)
        torch.manual_seed(seed)

        self.n_layers = n_layers

        self.node_embedding = ...   # node_dim --> node_features
        self.edge_embedding = ...   # edge_dim --> edge_features
        self.node_mlp = ...         # node_features --> node_features
        self.edge_mlp = ...         # edge_features --> edge_features
        self.out_mlp = ...          # edge_features --> F_block_dim x F_block_dim (or flattened version of if)

        self.reset_parameters()

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, data):
        node_features = self.node_embedding(data.x)
        edge_features = self.edge_embedding(data.edge_attr)
        row, col = data.edge_index

        # message passing through graph
        for l in range(self.n_layers):
            # update nodes
            node_features = self.propagate(data.edge_index, x=node_features, edge_attr=data.edge_attr)
            # update edges
            edge_features = self.edge_mlp(torch.cat([node_features[row], node_features[col], edge_features], dim=-1))

        # compute outputs (i.e. elements of transition matrix)
        # if n_layers=0, F_ij depends only on the embedding of the provided edge features
        out = self.out_mlp(torch.cat([node_features[row], node_features[col], edge_features], dim=-1))

        # use node_features to get diagonal elements F_ii ? Or include it as self-edge?

        return out

    def message(self, x_i, x_j, edge_attr):
        # construct messages to node i for each edge (j,i)

        inputs = [x_i, x_j, edge_attr]
        inputs = torch.cat(inputs, dim=1)

        msg = self.msg_mlp(inputs)
        msg = F.relu(msg)

        return msg

    def update(self, aggr_out):
        out = self.node_mlp(aggr_out)

        return out



