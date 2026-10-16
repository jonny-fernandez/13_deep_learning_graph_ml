"""LSTM window classifier; backward compatible with the original binary shape test."""
import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1,
                 num_classes: int = 1, dropout: float = 0.15):
        super().__init__()
        if min(input_dim, hidden_dim, num_layers, num_classes) < 1:
            raise ValueError('Positive model dimensions required')
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('Expected [batch, sequence, channels]')
        _, (hidden, _) = self.lstm(x)
        logits = self.head(self.dropout(hidden[-1]))
        return logits.squeeze(-1) if self.num_classes == 1 else logits
