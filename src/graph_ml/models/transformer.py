"""Small position-aware Transformer encoder for fixed, completed HAR windows."""
import torch
from torch import nn


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, model_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, num_classes: int = 1, max_length: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        if model_dim % num_heads or min(input_dim, model_dim, num_layers, num_classes, max_length) < 1:
            raise ValueError('Invalid Transformer dimensions')
        self.projection = nn.Linear(input_dim, model_dim)
        self.position = nn.Embedding(max_length, model_dim)
        layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads,
                                           dim_feedforward=model_dim * 2,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, num_classes))
        self.num_classes = num_classes
        self.max_length = max_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] > self.max_length:
            raise ValueError('Expected [batch, sequence<=max_length, channels]')
        positions = torch.arange(x.shape[1], device=x.device)
        h = self.encoder(self.projection(x) + self.position(positions).unsqueeze(0))
        logits = self.head(h.mean(dim=1))
        return logits.squeeze(-1) if self.num_classes == 1 else logits
