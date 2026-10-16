import torch
from graph_ml.models.mlp import MLP
from graph_ml.models.gcn import GCN
from graph_ml.models.graphsage import GraphSAGE

def test_mlp_shape():
    model = MLP(4)
    out = model(torch.randn(5, 4))
    assert out.shape == (5,)

def test_gcn_shape():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)
    assert GCN(4)(x, edge_index).shape == (3,)

def test_graphsage_shape():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)
    assert GraphSAGE(4)(x, edge_index).shape == (3,)
