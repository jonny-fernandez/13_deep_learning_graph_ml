"""One command: optional pytest, obtain approved public datasets, fit three baselines."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from .common import SEED, sha256_file, validation_metrics
from .downloads import BANK_URL, HAR_URL, bank_csv, ensure_download, prepare_har
from .bank import load_bank, bank_estimator
from .har import load_har, har_estimator
from .cora import load_cora, cora_estimator


def run_one(name: str, splits, estimator, archive_hash: str | None):
    print(f'[{name}] Training traditional baseline on TRAIN only; scoring VALIDATION only...', flush=True)
    start = time.perf_counter()
    estimator.fit(splits.x_train, splits.y_train)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probabilities = estimator.predict_proba(splits.x_val)
    inference_seconds = time.perf_counter() - start
    labels = [int(i) for i in estimator.classes_]
    if len(labels) < 2:
        raise ValueError('Training split has fewer than two classes')
    metrics = validation_metrics(splits.y_val, probabilities, labels=labels)
    result = {'task': name, 'status': 'validation_only_test_sealed',
              'n_train': int(len(splits.y_train)), 'n_validation': int(len(splits.y_val)),
              'n_test_sealed': int(splits.n_sealed_test), 'train_duration_seconds': round(training_seconds, 5),
              'validation_predict_duration_seconds': round(inference_seconds, 5),
              'baseline': type(estimator[-1]).__name__, 'metrics_validation': metrics,
              'data_contract': splits.metadata, 'source_archive_sha256': archive_hash,
              'preprocessing': 'sklearn Pipeline fitted on training partition only'}
    print(f'[{name}] train={result["n_train"]} val={result["n_validation"]} test_sealed={result["n_test_sealed"]}'
          f' | validation macro-F1={metrics["f1_macro"]:.4f} | train={training_seconds:.2f}s', flush=True)
    return result


def run_all(project_root: Path, tasks: list[str]):
    from importlib.metadata import version
    output = project_root / 'outputs' / 'cp1'
    output.mkdir(parents=True, exist_ok=True)
    result = {'project': '13_deep_learning_graph_ml', 'checkpoint': 'CP1',
              'note': 'Only validation metrics; held-out test NOT evaluated',
              'seed': SEED, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'python': sys.version.split()[0],
              'versions': {lib: version(lib) for lib in ('torch', 'torch-geometric', 'scikit-learn', 'pandas', 'numpy')},
              'tasks': {}}
    for name in tasks:
        if name == 'bank':
            archive = ensure_download(BANK_URL, project_root / 'data' / 'raw' / 'bank_marketing' / 'bank_marketing.zip')
            csv = bank_csv(archive, project_root / 'data' / 'raw' / 'bank_marketing' / 'bank-additional-full.csv')
            s = load_bank(csv)
            r = run_one('bank', s, bank_estimator(s.x_train), sha256_file(archive))
        elif name == 'har':
            archive = ensure_download(HAR_URL, project_root / 'data' / 'raw' / 'har' / 'har_uci.zip')
            folder = prepare_har(archive, project_root / 'data' / 'raw' / 'har' / 'UCI HAR Dataset')
            s = load_har(folder)
            r = run_one('har', s, har_estimator(), sha256_file(archive))
        elif name == 'cora':
            s = load_cora(project_root / 'data' / 'raw' / 'cora')
            r = run_one('cora', s, cora_estimator(), None)
        else:
            raise ValueError('Unknown dataset task')
        result['tasks'][name] = r
        # Save partial verified results: user can recover from a failed later download.
        _atomic_json(output / 'cp1_validation_report.json', result)
    _atomic_json(output / 'cp1_validation_report.json', result)
    print(f'CP1 RUN COMPLETE (not checkpoint approval): {output / "cp1_validation_report.json"}')
    print('Final test sets remain sealed. Send terminal output to review before marking CP1 complete.')


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)
