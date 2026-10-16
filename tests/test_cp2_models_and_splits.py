"""Offline CP2 regression tests; no public dataset/network required."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from graph_ml.models.mlp import MLP
from graph_ml.models.lstm import LSTMClassifier
from graph_ml.models.transformer import TransformerClassifier
from graph_ml.cp2.data import subject_partition
from graph_ml.cp2.training import fit_model, threshold_analysis, set_seed


def test_existing_binary_shape_contracts_preserved():
    assert MLP(4)(torch.randn(3, 4)).shape == (3,)
    assert LSTMClassifier(9)(torch.randn(3, 12, 9)).shape == (3,)
    assert TransformerClassifier(9)(torch.randn(3, 12, 9)).shape == (3,)


@pytest.mark.parametrize('model', [LSTMClassifier(input_dim=9, hidden_dim=12, num_classes=6),
                                   TransformerClassifier(input_dim=9, model_dim=16, num_heads=4,
                                                         num_layers=1, num_classes=6, max_length=128)])
def test_multiclass_shape_and_gradients(model):
    x = torch.randn(4, 12, 9)
    output = model(x)
    assert output.shape == (4, 6)
    loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 2, 4, 5]))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_transformer_rejects_long_windows():
    with pytest.raises(ValueError):
        TransformerClassifier(9, max_length=4)(torch.randn(1, 5, 9))


def test_har_subject_split_matches_cp1_ids_and_disjoint():
    subjects = np.array([1, 2, 2, 3, 1, 3])
    train, val = subject_partition(subjects, [1, 2], [3])
    assert train.tolist() == [True, True, True, False, True, False]
    assert val.tolist() == [False, False, False, True, False, True]
    with pytest.raises(ValueError):
        subject_partition(subjects, [1, 3], [3, 2])
    with pytest.raises(ValueError):
        subject_partition(subjects, [1], [3])  # unexpected missing subject 2


def test_threshold_analysis_is_validation_only_and_detects_no_positives_at_point_five():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([.1, .2, .3, .4, .45])
    result = threshold_analysis(y, p)
    assert result['default_0p5']['positive_recall'] == 0
    assert result['validation_selected_f1_threshold']['positive_recall'] > 0
    assert 'validation only' in result['warning']


def test_bounded_binary_training_retains_sealed_test_and_writes_checkpoint(tmp_path):
    set_seed(42)
    x_train = np.random.RandomState(42).normal(size=(48, 5)).astype(np.float32)
    y_train = (x_train[:, 0] > 0).astype(np.int64)
    x_val = np.random.RandomState(43).normal(size=(20, 5)).astype(np.float32)
    y_val = (x_val[:, 0] > 0).astype(np.int64)
    model = MLP(5, hidden_dim=8)
    result = fit_model('test_bank', model, x_train, y_train, x_val, y_val,
                       classes=2, primary_metric='pr_auc_average_precision',
                       output_dir=tmp_path / 'checkpoints', epochs=2,
                       batch_size=16)
    assert result['selected_epoch'] in (1, 2)
    assert result['n_validation'] == 20
    assert 'threshold_analysis_validation' in result
    assert (tmp_path / 'checkpoints' / 'test_bank_best.pt').exists()


def test_bounded_sequence_training_with_multiclass(tmp_path):
    rng = np.random.RandomState(42)
    x_train = rng.normal(size=(18, 8, 3)).astype(np.float32)
    y_train = np.tile(np.arange(3), 6).astype(np.int64)
    x_val = rng.normal(size=(9, 8, 3)).astype(np.float32)
    y_val = np.tile(np.arange(3), 3).astype(np.int64)
    model = LSTMClassifier(input_dim=3, hidden_dim=8, num_classes=3)
    result = fit_model('test_seq', model, x_train, y_train, x_val, y_val,
                       classes=3, primary_metric='f1_macro',
                       output_dir=tmp_path / 'checkpoints', epochs=2,
                       batch_size=9)
    assert result['metrics_validation']['n_val'] == 9
    assert result['parameter_count'] > 0
    assert 'threshold_analysis_validation' not in result


def test_bank_cp2_preprocessing_train_only_and_locked_source_order(tmp_path):
    import pandas as pd
    from graph_ml.cp2.data import bank_arrays
    rows = 40
    source = pd.DataFrame({
        'age': np.arange(rows) + 20,
        'job': ['admin.' if i < 28 else 'technician' for i in range(rows)],
        'month': ['may'] * rows,
        'duration': np.arange(rows) + 100,
        'y': ['yes' if i % 8 == 0 else 'no' for i in range(rows)],
    })
    target = tmp_path / 'bank-additional-full.csv'
    source.to_csv(target, sep=';', index=False)
    cp1 = {'n_train': 28, 'n_validation': 6, 'n_test_sealed': 6}
    xtr, ytr, xval, yval, meta = bank_arrays(target, cp1)
    assert xtr.shape[0] == 28 and xval.shape[0] == 6
    assert meta['prediction_time'].startswith('pre-contact')
    assert meta['input_dim'] == xtr.shape[1]
    assert not np.isnan(xtr).any() and not np.isnan(xval).any()
    # No test observations or duration field enter the CP2 matrix.
    assert len(ytr) + len(yval) == 34 < len(source)
    with pytest.raises(ValueError):
        bank_arrays(target, {**cp1, 'n_train': 27})


def test_har_cp2_reads_only_train_files_and_matches_cp1_subjects(tmp_path):
    from graph_ml.cp2.data import har_arrays
    root = tmp_path / 'UCI HAR Dataset'
    signal_dir = root / 'train' / 'Inertial Signals'
    signal_dir.mkdir(parents=True)
    subjects = np.array([1, 1, 2, 2, 3, 3])
    np.savetxt(root / 'train' / 'subject_train.txt', subjects, fmt='%d')
    np.savetxt(root / 'train' / 'y_train.txt', [1, 2, 1, 2, 1, 2], fmt='%d')
    for kind in ('total_acc', 'body_acc', 'body_gyro'):
        for axis in 'xyz':
            np.savetxt(signal_dir / f'{kind}_{axis}_train.txt',
                       np.arange(6 * 128, dtype=np.float32).reshape(6, 128) / 100.0)
    cp1 = {'n_train': 4, 'n_validation': 2,
           'data_contract': {'train_group_ids': [1, 2], 'val_group_ids': [3],
                             'n_subject_train': 2, 'n_subject_val': 1}}
    xtr, ytr, xval, yval, meta = har_arrays(root, cp1)
    assert xtr.shape == (4, 128, 9) and xval.shape == (2, 128, 9)
    assert len(ytr) == 4 and len(yval) == 2 and meta['subject_disjoint']
    assert np.isfinite(xtr).all() and np.isfinite(xval).all()
    # Explicitly no official test/ directory exists; loader must not access it.
    assert not (root / 'test').exists()
