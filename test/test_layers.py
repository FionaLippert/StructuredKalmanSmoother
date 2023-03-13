# Replacing String with another string
import torch
from structuredKS.models import layers
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
    sparse_pattern = F.to_sparse_coo().coalesce()
    G = layers.StaticTransition(sparse_pattern.indices(), initial_weights=sparse_pattern.values())

    return F, G

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
