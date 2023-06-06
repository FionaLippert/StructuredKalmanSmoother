# Replacing String with another string
import torch
from structuredKS.models import layers, dgmrf
import torch_geometric as tg
import pytest

torch.manual_seed(0)

@ pytest.fixture
def diagonal_transition():
    F = torch.diag(torch.rand(5))
    sparse_pattern = F.to_sparse_coo().coalesce()
    G = layers.StaticTransition(sparse_pattern.indices(), initial_weights=sparse_pattern.values())

    return F, G

@ pytest.fixture
def asymmetric_transition():
    F = torch.diag(torch.ones(5))
    F[0, 2:4] += torch.ones(2)
    F[1, 3] = 1
    F[2, 0] = 1
    F[3, 4] = 1
    F[4, 2] = 1
    sparse_pattern = F.to_sparse_coo().coalesce()
    G = layers.StaticTransition(sparse_pattern.indices(), initial_weights=sparse_pattern.values())

    return F, G

@ pytest.fixture
def asymmetric_dgmrf_transition():
    F = torch.zeros((5, 5))
    F[0, 2:4] += torch.ones(2)
    F[1, 3] = 1
    F[2, 0] = 1
    F[3, 4] = 1
    F[4, 2] = 1
    edge_index, _ = tg.utils.dense_to_sparse(F)
    edge_attr = torch.rand(edge_index.size(1), 2)
    G = tg.data.Data(edge_index=edge_index, num_nodes=5, edge_attr=edge_attr)

    return F, G

@ pytest.fixture
def symmetric_dgmrf_graph():
    G = torch.eye(5)
    G[0, 2:4] += torch.ones(2)
    G[2:4, 0] += torch.ones(2)
    edge_index, _ = tg.utils.dense_to_sparse(G)
    G = tg.data.Data(edge_index=edge_index, eigvals=0, num_nodes=5)

    return G, 5

def test_diagonal_transition(diagonal_transition):
    F, G = diagonal_transition

    input = torch.rand(5)
    mm = (F @ input.unsqueeze(-1)).squeeze()
    gnn = G(input.unsqueeze(0)).squeeze()

    assert torch.eq(mm, gnn).all()

def test_asymmetric_transition(asymmetric_transition):
    F, G = asymmetric_transition
    input = torch.ones(F.size(0))
    mm = (F @ input.unsqueeze(-1)).squeeze()
    gnn = G(input.unsqueeze(0)).squeeze()

    assert torch.eq(mm, gnn).all()

def test_sparse_to_dense(asymmetric_transition):
    F, G = asymmetric_transition
    F_from_G = G.to_dense()

    assert torch.eq(F, F_from_G).all()

def test_diffusion_transpose(asymmetric_dgmrf_transition):
    F, G = asymmetric_dgmrf_transition
    diff = dgmrf.DiffusionModel({}, G.to_dict())
    input = torch.rand(F.size(0))

    diff_matrix = diff(torch.eye(F.size(0)))

    diff_x_T = diff(input, transpose=True)

    diff_x_matrix_T = diff_matrix @ input

    assert torch.allclose(diff_x_T, diff_x_matrix_T)

def test_advection_diffusion_transpose(asymmetric_dgmrf_transition):
    F, G = asymmetric_dgmrf_transition
    model = dgmrf.AdvectionDiffusionModel({}, G.to_dict())
    input = torch.rand(F.size(0))

    matrix = model(torch.eye(F.size(0)))
    out = model(input, transpose=True)
    out_matrix = matrix @ input

    assert torch.allclose(out, out_matrix)

config = {
          "noise_std": 0.001,
          "n_layers": 1,
          "n_transitions": 1,
          "non_linear": False,
          "fix_gamma": False,
          "log_det_method": "eigvals",
          "use_bias": True,
          "features": False,
          "use_dynamics": False,
          "independent_time": False,
          "use_hierarchy": False,
          "transition_type": "identity",
          "inference_rtol": 1e-7,
          "max_cg_iter": 500
}

def test_joint_dgmrf_time_invariant(symmetric_dgmrf_graph):
    graph_G, num_nodes = symmetric_dgmrf_graph
    T = 3

    model = dgmrf.DGMRF(config, graph_G.to_dict(), T=T, shared='all')

    x = torch.rand(1, T, num_nodes)

    Gx = model(x, with_bias=False)
    xTGx = x.reshape(1, -1) @ Gx.reshape(-1, 1)
    print(xTGx)

    GTx = model(x, transpose=True, with_bias=False)
    xTGTx = x.reshape(1, -1) @ GTx.reshape(-1, 1)

    assert torch.allclose(xTGx, xTGTx)

def test_CG_zero_mean_joint_dgmrf_time_invariant(symmetric_dgmrf_graph):
    graph_G, num_nodes = symmetric_dgmrf_graph
    T = 3

    config['use_bias'] = False # mean=0

    model = dgmrf.DGMRF(config, graph_G.to_dict(), T=T, shared='all')
    bias = dgmrf.get_bias(model, (1, T, num_nodes))
    eta = -1. * model(bias, transpose=True, with_bias=False)  # -G^T @ b

    cg_mean = dgmrf.cg_solve(eta.reshape(1, -1), model, torch.zeros(T * num_nodes), T, config,
                         config["inference_rtol"], verbose=True)[0]

    assert torch.allclose(torch.zeros_like(cg_mean), cg_mean, rtol=1e-04, atol=1e-7)

def test_CG_nonzero_mean_joint_dgmrf_time_invariant(symmetric_dgmrf_graph):
    graph_G, num_nodes = symmetric_dgmrf_graph
    T = 3

    model = dgmrf.DGMRF(config, graph_G.to_dict(), T=T, shared='all')

    mean = torch.rand(1, T, num_nodes)
    Gmu = model(mean, with_bias=False)
    eta = model(Gmu, transpose=True, with_bias=False).reshape(1, -1)

    cg_mean = dgmrf.cg_solve(eta, model, torch.zeros(T * num_nodes), T, config,
                         config["inference_rtol"], verbose=True)

    assert torch.allclose(mean.reshape(-1), cg_mean.reshape(-1), rtol=1e-04, atol=1e-7)

def test_CG_nonzero_mean_joint_dgmrf_independent_time(symmetric_dgmrf_graph):
    graph_G, num_nodes = symmetric_dgmrf_graph
    T = 3

    model = dgmrf.DGMRF(config, graph_G.to_dict(), T=T, shared='none')

    mean = torch.rand(1, T, num_nodes)
    Gmu = model(mean, with_bias=False)
    eta = model(Gmu, transpose=True, with_bias=False).reshape(1, -1)

    cg_mean = dgmrf.cg_solve(eta, model, torch.zeros(T * num_nodes), T, config,
                         config["inference_rtol"], verbose=True)

    assert torch.allclose(mean.reshape(-1), cg_mean.reshape(-1), rtol=1e-04, atol=1e-7)

def test_CG_nonzero_mean_joint_dgmrf_AR(symmetric_dgmrf_graph, asymmetric_dgmrf_transition):
    F, graph_F = asymmetric_dgmrf_transition
    graph_G, num_nodes = symmetric_dgmrf_graph
    T = 3

    config['transition_type'] = 'AR'
    model = dgmrf.JointDGMRF(config, graph_G.to_dict(), graph_F.to_dict(), T=T, shared='dynamics')

    mean = torch.rand(1, T, num_nodes)
    Gmu = model(mean, with_bias=False)
    eta = model(Gmu, transpose=True, with_bias=False).reshape(1, -1)

    cg_mean = dgmrf.cg_solve(eta, model, torch.zeros(T * num_nodes), T, config,
                         config["inference_rtol"], verbose=True)

    assert torch.allclose(mean.reshape(-1), cg_mean.reshape(-1), rtol=1e-04, atol=1e-7)


def test_CG_nonzero_mean_joint_dgmrf_diffusion(symmetric_dgmrf_graph, asymmetric_dgmrf_transition):
    F, graph_F = asymmetric_dgmrf_transition
    graph_G, num_nodes = symmetric_dgmrf_graph
    T = 3

    config['transition_type'] = 'diffusion'
    model = dgmrf.JointDGMRF(config, graph_G.to_dict(), graph_F.to_dict(), T=T, shared='dynamics')

    mean = torch.rand(1, T, num_nodes)
    Gmu = model(mean, with_bias=False)
    eta = model(Gmu, transpose=True, with_bias=False).reshape(1, -1)

    cg_mean = dgmrf.cg_solve(eta, model, torch.zeros(T * num_nodes), T, config,
                         config["inference_rtol"], verbose=True)

    assert torch.allclose(mean.reshape(-1), cg_mean.reshape(-1), rtol=1e-04, atol=1e-7)

def test_CG_nonzero_mean_joint_dgmrf_advection_diffusion(symmetric_dgmrf_graph, asymmetric_dgmrf_transition):
    F, graph_F = asymmetric_dgmrf_transition
    graph_G, num_nodes = symmetric_dgmrf_graph
    T = 3

    config['transition_type'] = 'advection+diffusion'
    model = dgmrf.JointDGMRF(config, graph_G.to_dict(), graph_F.to_dict(), T=T, shared='dynamics')

    mean = torch.rand(1, T, num_nodes)
    Gmu = model(mean, with_bias=False)
    eta = model(Gmu, transpose=True, with_bias=False).reshape(1, -1)

    cg_mean = dgmrf.cg_solve(eta, model, torch.zeros(T * num_nodes), T, config,
                         config["inference_rtol"], verbose=True)

    assert torch.allclose(mean.reshape(-1), cg_mean.reshape(-1), rtol=1e-04, atol=1e-7)

def test_CG_nonzero_mean_joint_dgmrf_advection_diffusion_2layers(symmetric_dgmrf_graph, asymmetric_dgmrf_transition):
    F, graph_F = asymmetric_dgmrf_transition
    graph_G, num_nodes = symmetric_dgmrf_graph
    T = 3

    config['transition_type'] = 'advection+diffusion'
    config['n_layers_temporal'] = 2
    model = dgmrf.JointDGMRF(config, graph_G.to_dict(), graph_F.to_dict(), T=T, shared='dynamics')

    mean = torch.rand(1, T, num_nodes)
    Gmu = model(mean, with_bias=False)
    eta = model(Gmu, transpose=True, with_bias=False).reshape(1, -1)

    cg_mean = dgmrf.cg_solve(eta, model, torch.zeros(T * num_nodes), T, config,
                         config["inference_rtol"], verbose=True)

    assert torch.allclose(mean.reshape(-1), cg_mean.reshape(-1), rtol=1e-04, atol=1e-7)



def test_joint_dgmrf_independent_time(symmetric_dgmrf_graph):
    graph_G, num_nodes = symmetric_dgmrf_graph
    T = 3

    time_ranges = [torch.tensor([t]) for t in range(T)]
    # model = dgmrf.JointDGMRF([graph_G for _ in range(T)], time_ranges, config)
    model = dgmrf.DGMRF(config, graph_G.to_dict(), T=T, shared='none')

    x = torch.rand(1, T, num_nodes)

    Gx = model(x, with_bias=False)
    xTGx = x.reshape(1, -1) @ Gx.reshape(-1, 1)

    GTx = model(x, transpose=True, with_bias=False)
    xTGTx = x.reshape(1, -1) @ GTx.reshape(-1, 1)

    assert torch.allclose(xTGx, xTGTx)


def test_joint_dgmrf_identity(symmetric_dgmrf_graph, asymmetric_dgmrf_transition):
    F, graph_F = asymmetric_dgmrf_transition
    graph_G, num_nodes = symmetric_dgmrf_graph
    T = 3

    # time_ranges = [torch.arange(T)]
    config['transition_type'] = 'identity'
    # model = dgmrf.JointDGMRF([graph_G for _ in range(T)], time_ranges, config, graph_F)
    model = dgmrf.JointDGMRF(config, graph_G.to_dict(), graph_F.to_dict(), T=T, shared='dynamics')

    x = torch.rand(1, T, num_nodes)

    Gx = model(x, with_bias=False)
    xTGx = x.reshape(1, -1) @ Gx.reshape(-1, 1)

    GTx = model(x, transpose=True, with_bias=False)
    xTGTx = x.reshape(1, -1) @ GTx.reshape(-1, 1)

    assert torch.allclose(xTGx, xTGTx)

def test_joint_dgmrf_diffusion(symmetric_dgmrf_graph, asymmetric_dgmrf_transition):
    F, graph_F = asymmetric_dgmrf_transition
    graph_G, num_nodes = symmetric_dgmrf_graph
    T = 3

    # time_ranges = [torch.arange(T)]
    config['transition_type'] = 'diffusion'
    # model = dgmrf.JointDGMRF([graph_G for _ in range(T)], time_ranges, config, graph_F)
    model = dgmrf.JointDGMRF(config, graph_G.to_dict(), graph_F.to_dict(), T=T, shared='dynamics')

    x = torch.rand(1, T, num_nodes)

    Gx = model(x, with_bias=False)
    xTGx = x.reshape(1, -1) @ Gx.reshape(-1, 1)

    GTx = model(x, transpose=True, with_bias=False)
    xTGTx = x.reshape(1, -1) @ GTx.reshape(-1, 1)

    assert torch.allclose(xTGx, xTGTx)

def test_joint_dgmrf_advection_diffusion(symmetric_dgmrf_graph, asymmetric_dgmrf_transition):
    F, graph_F = asymmetric_dgmrf_transition
    graph_G, num_nodes = symmetric_dgmrf_graph
    T = 3

    time_ranges = [torch.arange(T)]
    config['transition_type'] = 'advection+diffusion'
    # model = dgmrf.JointDGMRF([graph_G for _ in range(T)], time_ranges, config, graph_F)
    model = dgmrf.JointDGMRF(config, graph_G.to_dict(), graph_F.to_dict(), T=T, shared='dynamics')

    x = torch.rand(1, T, num_nodes)

    Gx = model(x, with_bias=False)
    xTGx = x.reshape(1, -1) @ Gx.reshape(-1, 1)

    GTx = model(x, transpose=True, with_bias=False)
    xTGTx = x.reshape(1, -1) @ GTx.reshape(-1, 1)

    assert torch.allclose(xTGx, xTGTx)


def test_joint_dgmrf_GNNadvection(symmetric_dgmrf_graph, asymmetric_dgmrf_transition):
    F, graph_F = asymmetric_dgmrf_transition
    graph_G, num_nodes = symmetric_dgmrf_graph
    T = 3


    time_ranges = [torch.arange(T)]
    config['transition_type'] = 'GNN_advection'
    graph_G.pos = torch.rand(graph_F.num_nodes, 2, dtype=torch.float32)
    graph_F.edge_attr = torch.rand(graph_F.num_edges, 2, dtype=torch.float32)
    # model = dgmrf.JointDGMRF([graph_G for _ in range(T)], time_ranges, config, graph_F)
    model = dgmrf.JointDGMRF(config, graph_G.to_dict(), graph_F.to_dict(), T=T, shared='dynamics')

    x = torch.rand(1, T, num_nodes)

    Gx = model(x, with_bias=False)
    xTGx = x.reshape(1, -1) @ Gx.reshape(-1, 1)

    GTx = model(x, transpose=True, with_bias=False)
    xTGTx = x.reshape(1, -1) @ GTx.reshape(-1, 1)

    print(xTGx, xTGTx)

    assert torch.allclose(xTGx, xTGTx)

