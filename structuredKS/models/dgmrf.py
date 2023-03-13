import torch
import torch_geometric as ptg
import copy
from structuredKS import cg_batch

def new_graph(like_graph, new_x=None):
    graph = copy.copy(like_graph) # Shallow copy
    graph.x = new_x
    return graph

def get_num_nodes(graph):
    if hasattr(graph, "num_nodes"):
        return graph.num_nodes
    else:
        return int(graph.edge_index.max()) + 1 if graph.edge_index.numel() > 0 else 0


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
    def __init__(self, graph, config, vi_layer=False):
        super(DGMRFLayer, self).__init__(aggr="add")

        self.num_nodes = get_num_nodes(graph)

        self.degrees = ptg.utils.degree(graph.edge_index[0])
        self.edge_index = graph.edge_index

        self.alpha1_param = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)
        self.alpha2_param = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)

        if config["use_bias"]:
            self.bias = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)
        else:
            self.bias = None

        if config["log_det_method"] == "eigvals":
            assert hasattr(graph, "eigvals"), (
                "Dataset not pre-processed with eigenvalues")
            self.adj_eigvals = graph.eigvals
            self.eigvals_log_det = True
        elif config["log_det_method"] == "dad":
            assert hasattr(graph, "dad_traces"), (
                "Dataset not pre-processed with DAD traces")
            dad_traces = graph.dad_traces

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
        self.edge_log_degrees = self.log_degrees[graph.edge_index[1]]
        self.edge_log_degrees_transpose = self.log_degrees[graph.edge_index[0]]

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

    def weight_self_representation(self, x):
        # Representation of same node weighted with degree (taken to power)
        return (x.view(-1, self.num_nodes) * torch.exp(
            self.degree_power * self.log_degrees)).view(-1, 1)

    def forward(self, x, transpose, with_bias):
        weighted_repr = self.weight_self_representation(x)

        aggr = (self.self_weight * weighted_repr) + (self.neighbor_weight*self.propagate(
            self.edge_index, x=x, transpose=transpose)) # Shape (n_nodes*n_graphs,1)

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

        weighted_messages = x_j.view(-1, edge_weights.shape[0]) * edge_weights

        return weighted_messages.view(-1, 1)

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




class DGMRF(torch.nn.Module):
    def __init__(self, graph, config):
        super(DGMRF, self).__init__()

        layer_list = []
        for layer_i in range(config["n_layers"]):
            layer_list.append(DGMRFLayer(graph, config))

            # Optionally add non-linearities between hidden layers
            if config["non_linear"] and (layer_i < (config["n_layers"]-1)):
                layer_list.append(DGMRFActivation(config))

        self.layers = torch.nn.ModuleList(layer_list)

    def forward(self, x, transpose=False, with_bias=True):
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


class SpatialDGMRF(torch.nn.Module):
    def __init__(self, graph_0, graph_t, config):
        super().__init__()

        # define DGMRF for initial precision
        self.dgmrf_0 = DGMRF(graph_0, config)

        # define DGMRF for precision Q_t of dynamics error term
        self.dgmrf_t = DGMRF(graph_t, config)


    def forward(self, graphs, transpose=False, with_bias=False):
        # computes z = Pe

        # treat t=0 separately
        z_0 = self.dgmrf_0(graphs.get_example(0)["latent"].x, transpose, with_bias)
        # treat all t>0 the same
        graphs_t = ptg.data.Batch.from_data_list(graphs.index_select(torch.arange(1, graphs.num_graphs)))
        z_t = self.dgmrf_t(graphs_t["latent"].x, transpose, with_bias)

        z = torch.cat([z_0, z_t], dim=0)

        return z

    def log_det(self, T):
        return self.dgmrf_0.log_det() + (T-1) * self.dgmrf_t.log_det()



class VariationalDist(torch.nn.Module):
    def __init__(self, config, graph_y):
        super().__init__()

        # Dimensionality of distribution (num_nodes of graph)
        self.dim = graph_y.num_nodes

        # Standard amount of samples (must be fixed to be efficient)
        self.n_samples = config["n_training_samples"]

        # Variational distribution, Initialize with observed y
        self.mean_param = torch.nn.parameter.Parameter(graph_y.mask*graph_y.x[:,0])
        self.diag_param = torch.nn.parameter.Parameter(
                2*torch.rand(self.dim) - 1.) # U(-1,1)

        self.layers = torch.nn.ModuleList([DGMRFLayer(graph_y, config, vi_layer=True)
                                           for _ in range(config["vi_layers"])])
        if config["vi_layers"] > 0:
            self.post_diag_param = torch.nn.parameter.Parameter(
                2*torch.rand(self.dim) - 1.)

        # Reuse same batch with different x-values
        self.sample_batch = ptg.data.Batch.from_data_list([new_graph(graph_y)
                                    for _ in range(self.n_samples)])

        if config["features"]:
            # Additional variational distribution for linear coefficients
            n_features = graph_y.features.shape[1]
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
        standard_sample = torch.randn(self.n_samples, self.dim)
        ind_samples = self.std * standard_sample

        self.sample_batch.x = ind_samples.reshape(-1,1) # Stack all
        for layer in self.layers:
            propagated = layer(self.sample_batch.x, self.sample_batch.edge_index,
                    transpose=False, with_bias=False)
            self.sample_batch.x = propagated

        samples = self.sample_batch.x.reshape(self.n_samples, -1)
        if self.layers:
            # Apply post diagonal matrix
            samples  = self.post_diag * samples
        samples = samples + self.mean_param # Add mean last (not changed by layers)
        return samples # shape (n_samples, n_nodes)

    def log_det(self):
        layers_log_det = sum([layer.log_det() for layer in self.layers])
        std_log_det = torch.sum(torch.log(self.std))
        total_log_det = 2.0*std_log_det + 2.0*layers_log_det

        if self.layers:
            post_diag_log_det = torch.sum(torch.log(self.post_diag))
            total_log_det = total_log_det + 2.0*post_diag_log_det

        return total_log_det

    def sample_coeff(self, n_samples):
        standard_sample = torch.randn(n_samples, self.coeff_mean_param.shape[0])
        samples = (self.coeff_std * standard_sample) + self.coeff_mean_param
        return samples # shape (n_samples, n_features)

    def log_det_coeff(self):
        return 2.0*torch.sum(torch.log(self.coeff_std))

    def ce_coeff(self):
        # Compute Cross-entropy term (CE between VI-dist and coeff prior)
        return -0.5*(self.coeff_inv_std**2)*torch.sum(
                torch.pow(self.coeff_std, 2) + torch.pow(self.coeff_mean_param, 2))

    @torch.no_grad()
    def posterior_estimate(self, graph_y, config):
        # Compute mean and marginal std of distribution (posterior estimate)
        # Mean
        graph_post_mean = new_graph(graph_y,
                new_x=self.mean_param.detach().unsqueeze(1))

        # Marginal std. (MC estimate)
        mc_sample_list = []
        cur_mc_samples = 0
        while cur_mc_samples < config["n_post_samples"]:
            mc_sample_list.append(self.sample())
            cur_mc_samples += self.n_samples
        mc_samples = torch.cat(mc_sample_list, dim=0)[:config["n_post_samples"]]

        # MC estimate of variance using known population mean
        post_var_x = torch.mean(torch.pow(mc_samples - self.mean_param, 2), dim=0)
        # Posterior std.-dev. for y
        post_std = torch.sqrt(post_var_x + torch.exp(2.*config["log_noise_std"])).unsqueeze(1)

        graph_post_std = new_graph(graph_y, new_x=post_std)

        return graph_post_mean, graph_post_std

class ObservationModel(ptg.nn.MessagePassing):
    """
    Apply observation model to latent states: y = Hx
    """

    def __init__(self):
        super(ObservationModel, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, x, graph):

        if hasattr(graph, "edge_weight"):
            edge_weights = graph.edge_weight.view(-1, 1)
        else:
            edge_weights = torch.ones_like(graph.edge_index[0]).view(-1, 1)
        y = self.propagate(graph.edge_index, x=x, edge_weights=edge_weights)

        # apply mask based on incoming edges
        num_nodes = x.size(0)
        mask = ptg.utils.degree(graph.edge_index[0], num_nodes=num_nodes).bool()

        return y[mask]

    def message(self, x_j, edge_weights):
        # construct messages to node i for each edge (j,i)
        msg = edge_weights * x_j
        return msg


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

def get_bias(dgmrf, graph_y):
    zero_graph = new_graph(graph_y, new_x=torch.zeros(graph_y.num_nodes, 1))
    bias = dgmrf(zero_graph)
    return bias


# Solve Q_tilde x = rhs using Conjugate Gradient
def cg_solve(rhs, dgmrf, graph_y, config, rtol, verbose=False):
    # Only create the graph_batch once, then we can just replace x
    n_nodes = graph_y.num_nodes
    x_dummy = torch.zeros(rhs.shape[0],n_nodes,1)
    graph_list = [new_graph(graph_y, new_x=x_part) for x_part in x_dummy]
    graph_batch = ptg.data.Batch.from_data_list(graph_list)

    # CG requires more precision for numerical stability
    rhs = rhs.to(torch.float64)

    # Batch linear operator
    def Q_tilde_batched(x):
        # x has shape (n_batch, n_nodes, 1)
        # Implicitly applies posterior precision matrix Q_tilde to a vector x
        # y = ((G^T)G + (sigma^-2)(I_m))x

        # Modify graph batch
        graph_batch.x = x.reshape(-1,1)

        Gx = dgmrf(graph_batch, with_bias=False)
        graph_batch.x = Gx
        GtGx = dgmrf(graph_batch, transpose=True, with_bias=False)
        #shape (n_batch*n_nodes, 1)

        noise_add = (1./torch.exp(2.*config["log_noise_std"])) * graph_y.mask.to(torch.float32)
        # Shape (n_nodes,)

        res = GtGx.view(-1,n_nodes,1) + noise_add.view(1,n_nodes,1)*x
        # Shape (n_batch, n_nodes,1)
        return res

    if config["features"]:
        # Feature matrix with 0-rows for masked nodes
        masked_features = graph_y.features * graph_y.mask.to(torch.float64).unsqueeze(1)
        masked_features_cov = masked_features.transpose(0,1)@masked_features

        noise_precision = 1/torch.exp(2.*config["log_noise_std"])

        def Q_tilde_batched_with_features(x):
            # x has shape (n_batch, n_nodes+n_features, 1)
            node_x = x[:,:n_nodes]
            coeff_x = x[:,n_nodes:]

            top_res1 = Q_tilde_batched(node_x)
            top_res2 = noise_precision*masked_features@coeff_x

            bot_res1 = noise_precision*masked_features.transpose(0,1)@node_x
            bot_res2 = noise_precision*masked_features_cov@coeff_x +\
                    (config["coeff_inv_std"]**2)*coeff_x

            res = torch.cat((
                top_res1 + top_res2,
                bot_res1 + bot_res2,
                ), dim=1)

            return res

        Q_tilde_func = Q_tilde_batched_with_features
    else:
        Q_tilde_func = Q_tilde_batched

    solution, cg_info = cg_batch.cg_batch(Q_tilde_func, rhs, rtol=rtol)

    if verbose:
        print("CG finished in {} iterations, solution optimal: {}".format(
            cg_info["niter"], cg_info["optimal"]))

    return solution.to(torch.float32)

@torch.no_grad()
def sample_posterior(n_samples, dgmrf, graph_y, config, rtol, verbose=False):
    # Construct RHS using Papandeous and Yuille method
    bias = get_bias(dgmrf, graph_y)
    std_gauss1 = torch.randn(n_samples, graph_y.num_nodes, 1) - bias # Bias offset
    std_gauss2 = torch.randn(n_samples, graph_y.num_nodes, 1)

    std_gauss1_graphs = ptg.data.Batch.from_data_list(
        [new_graph(graph_y, new_x=sample) for sample in std_gauss1])
    rhs_sample1 = dgmrf(std_gauss1_graphs, transpose=True, with_bias=False)
    rhs_sample1 = rhs_sample1.reshape(-1, graph_y.num_nodes, 1)

    float_mask = graph_y.mask.to(torch.float32).unsqueeze(1)
    y_masked = (graph_y.x * float_mask).unsqueeze(0)
    gauss_masked = std_gauss2 * float_mask.unsqueeze(0)
    rhs_sample2 = (1./torch.exp(2.*config["log_noise_std"]))*y_masked +\
        (1./torch.exp(2.*config["log_noise_std"]))*gauss_masked

    rhs_sample = rhs_sample1 + rhs_sample2
    # Shape (n_samples, n_nodes, 1)

    if config["features"]:
        # Change rhs to also sample coefficients
        n_features = graph_y.features.shape[1]
        std_gauss1_coeff = torch.randn(n_samples, n_features, 1)
        rhs_sample1_coeff = config["coeff_inv_std"]*std_gauss1_coeff

        rhs_sample2_coeff = graph_y.features.transpose(0,1)@rhs_sample2
        rhs_sample_coeff = rhs_sample1_coeff + rhs_sample2_coeff

        rhs_sample = torch.cat((rhs_sample, rhs_sample_coeff), dim=1)

    # Solve using Conjugate gradient
    samples = cg_solve(rhs_sample, dgmrf, graph_y, config,
            rtol=rtol, verbose=verbose)

    return samples


@torch.no_grad()
def posterior_inference(dgmrf, graph_y, config):
    # Posterior mean
    graph_bias = new_graph(graph_y, new_x=get_bias(dgmrf, graph_y))
    Q_mu = -1.*dgmrf(graph_bias, transpose=True, with_bias=False) # Q@mu = -G^T@b

    masked_y = graph_y.mask.to(torch.float32).unsqueeze(1) * graph_y.x
    mean_rhs = Q_mu + (1./torch.exp(2.*config["log_noise_std"])) * masked_y

    if config["features"]:
        rhs_append = (1./torch.exp(2.*config["log_noise_std"]))*\
            graph_y.features.transpose(0,1)@masked_y

        mean_rhs = torch.cat((mean_rhs, rhs_append), dim=0)

    post_mean = cg_solve(mean_rhs.unsqueeze(0), dgmrf, graph_y, config,
            config["inference_rtol"], verbose=True)[0]

    if config["features"]:
        # CG returns posterior mean of both x and coeff., compute posterior
        post_mean_x = post_mean[:graph_y.num_nodes]
        post_mean_beta = post_mean[graph_y.num_nodes:]

        post_mean = post_mean_x + graph_y.features@post_mean_beta

        # Plot posterior mean for x alone
        graph_post_mean_x = new_graph(graph_y, new_x=post_mean_x)
        # vis.plot_graph(graph_post_mean_x, name="post_mean_x", title="X Posterior Mean")

    graph_post_mean = new_graph(graph_y, new_x=post_mean)

    # Posterior samples and marginal variances
    # Batch sampling
    posterior_samples_list = []
    cur_post_samples = 0
    while cur_post_samples < config["n_post_samples"]:
        posterior_samples_list.append(sample_posterior(config["n_training_samples"],
            dgmrf, graph_y, config, config["inference_rtol"], verbose=True))
        cur_post_samples += config["n_training_samples"]

    posterior_samples = torch.cat(posterior_samples_list,
           dim=0)[:config["n_post_samples"]]

    if config["features"]:
        # Include linear feature model to posterior samples
        post_samples_x = posterior_samples[:,:graph_y.num_nodes]
        post_samples_coeff = posterior_samples[:,graph_y.num_nodes:]

        posterior_samples = post_samples_x + graph_y.features@post_samples_coeff

    # MC estimate of variance using known population mean
    post_var_x = torch.mean(torch.pow(posterior_samples - post_mean, 2), dim=0)
    # Posterior std.-dev. for y
    post_std = torch.sqrt(post_var_x + torch.exp(2.*config["log_noise_std"]))

    graph_post_std = new_graph(graph_y, new_x=post_std)
    graph_post_sample = new_graph(graph_y)

    # # Plot posterior samples
    # for sample_i, post_sample in enumerate(
    #         posterior_samples[:config["plot_post_samples"]]):
    #     graph_post_sample.x = post_sample
    #     vis.plot_graph(graph_post_sample, name="post_sample",
    #             title="Posterior sample {}".format(sample_i))

    return graph_post_mean, graph_post_std



