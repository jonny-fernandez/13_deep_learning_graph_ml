"""GCN for binary starter shapes or multiclass Cora node classification."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 1, dropout: float = 0.4):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_classes <= 0 or not 0 <= dropout < 1:
            raise ValueError('Invalid GCN architecture configuration')
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = float(dropout)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1) if self.num_classes == 1 else x
