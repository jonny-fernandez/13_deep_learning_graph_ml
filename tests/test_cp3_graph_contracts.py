"""Offline CP3 contract tests; synthetic fixtures are NOT dataset performance evidence."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from graph_ml.cp3.data import GraphTask, construct_graph_task
from graph_ml.cp3.training import fit_graph_model, validation_node_saliency


def synthetic_cora():
    n = 1640
    masks = [torch.zeros(n, dtype=torch.bool) for _ in range(3)]
    masks[0][:140], masks[1][140:640], masks[2][640:] = True, True, True
    y = (torch.arange(n) % 7).long()
    y[640:] = -777  # sealed test labels intentionally invalid; loaders must not inspect.
    edge = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    g = SimpleNamespace(x=torch.zeros(n, 1433), edge_index=edge, y=y,
                        train_mask=masks[0], val_mask=masks[1], test_mask=masks[2])
    meta = {'n_nodes': n, 'n_edges_directed_entries': 4, 'raw_file_sha256': {'foo': 'bar'},
            'train_mask_count': 140, 'val_mask_count': 500}
    return g, meta


def test_public_mask_contract_does_not_export_sealed_labels():
    g, meta = synthetic_cora()
    task = construct_graph_task(g, meta, {'foo': 'bar'})
    assert task.test_count_sealed == 1000
    assert (task.y_train.min().item(), task.y_val.max().item()) == (0, 6)
    assert not hasattr(task, 'y_test') and not hasattr(task, 'test_idx')
    assert 'test_labels' not in task.metadata
    assert len(task.train_idx) == 140 and len(task.val_idx) == 500


def test_mismatched_source_or_overlap_fails_closed():
    g, meta = synthetic_cora()
    with pytest.raises(ValueError, match='provenance'):
        construct_graph_task(g, meta, {'foo': 'altered'})
    g.test_mask[0] = True
    with pytest.raises(ValueError, match='overlap'):
        construct_graph_task(g, meta, {'foo': 'bar'})


def test_wrong_masks_fail_closed():
    g, meta = synthetic_cora()
    g.val_mask[140] = False
    with pytest.raises(ValueError, match='public split'):
        construct_graph_task(g, meta, {'foo': 'bar'})


class TinyGraphClassifier(torch.nn.Module):
    """Dependency-free synthetic graph-shape mock. Not a GNN or a Cora result."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1433, 7)
    def forward(self, x, edge_index):
        return self.linear(x)


def tiny_task():
    torch.manual_seed(42)
    return GraphTask(x=torch.randn(28, 1433), edge_index=torch.tensor([[0,1],[1,2]]),
                     train_idx=torch.arange(14), val_idx=torch.arange(14,28),
                     y_train=(torch.arange(14) % 7), y_val=(torch.arange(14) % 7),
                     test_count_sealed=10, metadata={})


def test_masked_training_and_validation_checkpoint(tmp_path):
    task = tiny_task()
    result = fit_graph_model(task=task, model=TinyGraphClassifier(), name='synthetic_test',
                             output_dir=tmp_path, epochs=2, patience=2)
    assert result['status'] == 'validation_only_test_sealed'
    assert result['train_loss_target_count'] == 14 and result['sealed_test_count'] == 10
    assert result['selected_epoch'] in (1,2)
    assert result['checkpoint_sha256'] and (tmp_path / 'synthetic_test_best.pt').exists()
    assert result['metrics_validation']['n_val'] == 14


def test_gradient_attribution_is_predicted_class_only():
    task = tiny_task()
    saliency = validation_node_saliency(task, TinyGraphClassifier(), top_k=5)
    assert saliency['node_id'] == 14
    assert len(saliency['feature_indices_zero_based']) == 5
    assert 'predicted_topic_class' in saliency
    assert 'label' not in saliency


def test_invalid_graph_class_id_rejected(tmp_path):
    task = tiny_task()
    task.y_val[0] = 99
    with pytest.raises(ValueError, match='class'):
        fit_graph_model(task=task, model=TinyGraphClassifier(), name='invalid', output_dir=tmp_path, epochs=1)


def test_actual_graph_shapes_when_pyg_installed():
    pytest.importorskip('torch_geometric')
    from graph_ml.models.gcn import GCN
    from graph_ml.models.graphsage import GraphSAGE
    x = torch.randn(10, 1433)
    edges = torch.tensor([[0,1,2,3,4,5,6],[1,2,3,4,5,6,7]], dtype=torch.long)
    for cls in (GCN, GraphSAGE):
        model = cls(1433, hidden_dim=8, num_classes=7)
        y = model(x, edges)
        assert y.shape == (10, 7)
        y[:2].sum().backward()
        assert all(p.grad is not None for p in model.parameters())
        model.eval()
        assert model(x, torch.empty(2,0,dtype=torch.long)).shape == (10,7)


def test_cp2_lineage_gate_on_expected_report(tmp_path):
    pytest.importorskip('torch_geometric')
    from graph_ml.cp3.runner import require_cp2_report
    with pytest.raises(FileNotFoundError):
        require_cp2_report(tmp_path, 'cp1_digest')
    cp2 = {'checkpoint': 'CP2', 'status': 'validation_complete_awaiting_review', 'seed': 42,
           'cp1_report_sha256': 'cp1_digest',
           'selected_experiments': ['bank_mlp','har_lstm','har_transformer'],
           'results': {n:{'status':'validation_only_test_sealed'} for n in ('bank_mlp','har_lstm','har_transformer')}}
    path = tmp_path/'outputs'/'cp2'/'cp2_validation_report.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(cp2))
    assert len(require_cp2_report(tmp_path,'cp1_digest')[1]) == 64
    with pytest.raises(ValueError):
        require_cp2_report(tmp_path, 'wrong')
    cp2['results']['har_lstm']['status'] = 'failed'
    path.write_text(json.dumps(cp2))
    with pytest.raises(ValueError):
        require_cp2_report(tmp_path, 'cp1_digest')
