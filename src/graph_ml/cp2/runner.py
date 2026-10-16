"""CP2 comparisons against frozen CP1 baseline, using validation labels only."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from importlib.metadata import version
import numpy as np
import torch
from graph_ml.cp1.common import sha256_file
from graph_ml.cp2.data import bank_arrays, har_arrays
from graph_ml.cp2.training import set_seed, fit_model
from graph_ml.models.mlp import MLP
from graph_ml.models.lstm import LSTMClassifier
from graph_ml.models.transformer import TransformerClassifier


def require_cp1_report(root: Path) -> tuple[dict, str]:
    path = root / 'outputs' / 'cp1' / 'cp1_validation_report.json'
    if not path.is_file():
        raise FileNotFoundError(f'Run CP1 first and retain its report: {path}')
    cp1 = json.loads(path.read_text(encoding='utf-8'))
    if (cp1.get('checkpoint') != 'CP1' or cp1.get('seed') != 42 or
            set(cp1.get('tasks', {})) != {'bank', 'har', 'cora'} or
            any(t.get('status') != 'validation_only_test_sealed' for t in cp1['tasks'].values()) or
            cp1['tasks']['bank']['data_contract']['excluded'] != ['duration', 'y']):
        raise ValueError('CP1 report missing, incomplete or not the approved frozen split; do not retrain on different splits')
    for k in ('bank', 'har', 'cora'):
        t = cp1['tasks'][k]
        if (t['n_train'] <= 0 or t['n_validation'] <= 0 or t['n_test_sealed'] <= 0 or
                not isinstance(t.get('baseline'), str) or
                not (0 <= t['metrics_validation']['f1_macro'] <= 1)):
            raise ValueError('CP1 baseline report contains invalid counts/metrics')
    return cp1, sha256_file(path)


def _persist(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(partial, path)


def _check_cp1_archive_hash(root: Path, task: str, cp1_task: dict) -> None:
    name = 'bank_marketing/bank_marketing.zip' if task == 'bank' else 'har/har_uci.zip'
    path = root / 'data' / 'raw' / name
    if not path.is_file() or sha256_file(path) != cp1_task['source_archive_sha256']:
        raise ValueError(f'{task} source archive missing/changed. Re-run CP1 and inspect its results before CP2.')


def run_all(root: Path, *, epochs: int = 6, batch_size: int = 256,
            device: str = 'cpu', models: tuple[str, ...] = ('bank_mlp', 'har_lstm', 'har_transformer')):
    cp1, baseline_hash = require_cp1_report(root)
    if not models or len(models) != len(set(models)) or not set(models).issubset({'bank_mlp', 'har_lstm', 'har_transformer'}):
        raise ValueError('Invalid CP2 experiment choice')
    if epochs < 1 or epochs > 100 or batch_size < 1 or batch_size > 2048:
        raise ValueError('Invalid resource settings')
    if device != 'cpu' and not (device == 'cuda' and torch.cuda.is_available()):
        raise ValueError('Requested device unavailable; use --device cpu')
    set_seed(42)
    out_dir = root / 'outputs' / 'cp2'
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'cp2_validation_report.json'
    result = {'project': '13_deep_learning_graph_ml', 'checkpoint': 'CP2',
              'status': 'partial_validation_only', 'note': 'No sealed test features or labels used; CP1 baselines are frozen',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'seed': 42,
              'platform': platform.platform(), 'python': sys.version.split()[0],
              'pytorch': version('torch'), 'numpy': version('numpy'), 'device': device,
              'torch_threads': torch.get_num_threads(),
              'cp1_report_sha256': baseline_hash,
              'selected_experiments': list(models), 'results': {}}
    for name in models:
        print(f'CP2 {name}: training features/labels only, selecting by VALIDATION...', flush=True)
        if name == 'bank_mlp':
            _check_cp1_archive_hash(root, 'bank', cp1['tasks']['bank'])
            csv = root / 'data' / 'raw' / 'bank_marketing' / 'bank-additional-full.csv'
            x_train, y_train, x_val, y_val, metadata = bank_arrays(csv, cp1['tasks']['bank'])
            set_seed(42)
            model = MLP(input_dim=metadata['input_dim'], hidden_dim=64)
            task, primary = 'bank', 'pr_auc_average_precision'
            baseline_score = cp1['tasks'][task]['metrics_validation'][primary]
        else:
            _check_cp1_archive_hash(root, 'har', cp1['tasks']['har'])
            folder = root / 'data' / 'raw' / 'har' / 'UCI HAR Dataset'
            x_train, y_train, x_val, y_val, metadata = har_arrays(folder, cp1['tasks']['har'])
            set_seed(42)
            if name == 'har_lstm':
                model = LSTMClassifier(input_dim=9, hidden_dim=32, num_layers=1, num_classes=6, dropout=.1)
            else:
                model = TransformerClassifier(input_dim=9, model_dim=32, num_heads=4,
                                              num_layers=1, num_classes=6, max_length=128, dropout=.1)
            task, primary = 'har', 'f1_macro'
            baseline_score = cp1['tasks'][task]['metrics_validation'][primary]
        record = fit_model(name, model, x_train, y_train, x_val, y_val,
                           classes=2 if task == 'bank' else 6, primary_metric=primary,
                           output_dir=out_dir / 'checkpoints', epochs=epochs,
                           batch_size=batch_size, seed=42, device=device)
        record['task'] = task
        record['data_contract'] = metadata
        record['baseline'] = {'model': cp1['tasks'][task]['baseline'],
                              'metric': primary, 'validation_score': baseline_score,
                              'baseline_report_sha256': baseline_hash}
        record['delta_vs_baseline_validation'] = record['metrics_validation'][primary] - baseline_score
        record['interpretation'] = 'Only compares within the same task and split. HAR representations differ: engineered vs raw windows.'
        result['results'][name] = record
        _persist(report_path, result)
        print(f'CP2 {name}: val_{primary}={record["metrics_validation"][primary]:.4f} '
              f'(CP1 {baseline_score:.4f}); delta={record["delta_vs_baseline_validation"]:+.4f}', flush=True)
    if len(result['results']) == len(models) and set(models) == {'bank_mlp', 'har_lstm', 'har_transformer'}:
        result['status'] = 'validation_complete_awaiting_review'
    else:
        result['status'] = 'partial_validation_only'
    _persist(report_path, result)
    print(f'CP2 RUN FINISHED ({result["status"]}): {report_path}', flush=True)
    print('Final test partitions remain sealed. Please share this JSON for CP2 review.', flush=True)
