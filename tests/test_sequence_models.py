import torch
from graph_ml.models.lstm import LSTMClassifier
from graph_ml.models.transformer import TransformerClassifier
from graph_ml.evaluation.model_selection import complexity_justified

def test_lstm_shape():
    model = LSTMClassifier(input_dim=5)
    assert model(torch.randn(4, 10, 5)).shape == (4,)

def test_transformer_shape():
    model = TransformerClassifier(input_dim=5)
    assert model(torch.randn(4, 10, 5)).shape == (4,)

def test_complexity_must_clear_improvement_threshold():
    assert complexity_justified(0.80, 0.83, 0.02)
    assert not complexity_justified(0.80, 0.805, 0.02)
