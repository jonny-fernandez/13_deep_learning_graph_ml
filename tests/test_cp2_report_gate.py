import json
from pathlib import Path
import pytest
from graph_ml.cp2.runner import require_cp1_report


def _sample():
    metric = {'f1_macro': .45}
    task = lambda: {'baseline': 'LogisticRegression', 'status': 'validation_only_test_sealed', 'n_train': 5, 'n_validation': 3,
                    'n_test_sealed': 2, 'metrics_validation': metric}
    tasks = {name: task() for name in ('bank', 'har', 'cora')}
    tasks['bank']['data_contract'] = {'excluded': ['duration', 'y']}
    return {'checkpoint': 'CP1', 'seed': 42, 'tasks': tasks}


def test_requires_completed_cp1_report(tmp_path):
    with pytest.raises(FileNotFoundError):
        require_cp1_report(tmp_path)
    path = tmp_path / 'outputs' / 'cp1' / 'cp1_validation_report.json'
    path.parent.mkdir(parents=True)
    data = _sample()
    path.write_text(json.dumps(data))
    report, digest = require_cp1_report(tmp_path)
    assert report['seed'] == 42 and len(digest) == 64
    data['tasks']['har']['status'] = 'partial'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        require_cp1_report(tmp_path)


def test_end_to_end_cp2_runner_on_synthetic_only(monkeypatch, tmp_path):
    """Integration wiring only; results synthetic and never presented as benchmarks."""
    import numpy as np
    from graph_ml.cp2 import runner
    data = _sample()
    data['tasks']['bank']['metrics_validation']['pr_auc_average_precision'] = .12
    data['tasks']['har']['metrics_validation']['f1_macro'] = .30
    data['tasks']['cora']['metrics_validation']['f1_macro'] = .30
    out = tmp_path / 'outputs' / 'cp1' / 'cp1_validation_report.json'
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(data))
    monkeypatch.setattr(runner, '_check_cp1_archive_hash', lambda *args: None)
    rng = np.random.RandomState(42)
    bank_train = rng.normal(size=(30, 4)).astype('float32')
    bank_val = rng.normal(size=(12, 4)).astype('float32')
    har_train = rng.normal(size=(24, 8, 9)).astype('float32')
    har_val = rng.normal(size=(12, 8, 9)).astype('float32')
    monkeypatch.setattr(runner, 'bank_arrays', lambda *args: (
        bank_train, np.tile([0, 1], 15), bank_val, np.tile([0, 1], 6), {'input_dim': 4}))
    monkeypatch.setattr(runner, 'har_arrays', lambda *args: (
        har_train, np.tile(np.arange(6), 4), har_val, np.tile(np.arange(6), 2), {'input_dim': 9}))
    runner.run_all(tmp_path, epochs=1, batch_size=12)
    report = json.loads((tmp_path / 'outputs' / 'cp2' / 'cp2_validation_report.json').read_text())
    assert report['status'] == 'validation_complete_awaiting_review'
    assert set(report['results']) == {'bank_mlp', 'har_lstm', 'har_transformer'}
    assert all(row['status'] == 'validation_only_test_sealed' for row in report['results'].values())
