"""CP3 auditable local run: candidate GNNs, no-edge ablation, lineage and saliency."""
from __future__ import annotations
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import torch
from graph_ml.cp1.common import sha256_file
from graph_ml.cp2.runner import require_cp1_report
from graph_ml.cp2.training import set_seed
from graph_ml.models.gcn import GCN
from graph_ml.models.graphsage import GraphSAGE
from .data import load_locked_cora
from .training import fit_graph_model, validation_node_saliency


def require_cp2_report(root: Path, expected_cp1_sha: str) -> tuple[dict, str]:
    path = root / 'outputs' / 'cp2' / 'cp2_validation_report.json'
    if not path.is_file():
        raise FileNotFoundError('CP2 validation report missing; finish CP2 first')
    report = json.loads(path.read_text(encoding='utf-8'))
    if (report.get('checkpoint') != 'CP2' or
            report.get('status') != 'validation_complete_awaiting_review' or
            report.get('seed') != 42 or report.get('cp1_report_sha256') != expected_cp1_sha or
            set(report.get('selected_experiments', [])) != {'bank_mlp', 'har_lstm', 'har_transformer'} or
            set(report.get('results', {})) != {'bank_mlp', 'har_lstm', 'har_transformer'} or
            any(v.get('status') != 'validation_only_test_sealed' for v in report['results'].values())):
        raise ValueError('CP2 lineage/validation gate failed. Do not mix experiment splits.')
    return report, sha256_file(path)


def atomic_report(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def make_model(kind: str, hidden: int, dropout: float):
    if kind == 'gcn':
        return GCN(1433, hidden_dim=hidden, num_classes=7, dropout=dropout)
    if kind == 'graphsage':
        return GraphSAGE(1433, hidden_dim=hidden, num_classes=7, dropout=dropout)
    raise ValueError('Unknown graph architecture')


def optional_mlflow(report: dict, root: Path):
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError('MLflow is optional and not installed. Run without --mlflow or install mlflow in .venv.') from exc
    uri = (root / 'outputs' / 'cp3' / 'mlruns').resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment('p13_cora_validation_only')
    refs = {}
    for name, row in report['results'].items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({'task': 'cora_node_classification', 'split': 'public_transductive_validation_only',
                             'cp1_sha256': report['cp1_report_sha256'], 'cp2_sha256': report['cp2_report_sha256']})
            mlflow.log_params({'architecture': row['model'], 'edges': row['edge_mode'],
                               'seed': row['training_config']['seed'], 'epochs_completed': row['epochs_completed'],
                               'params': row['parameter_count']})
            mlflow.log_metrics({'validation_macro_f1': row['metrics_validation']['f1_macro'],
                                'validation_log_loss': row['metrics_validation']['log_loss'],
                                'training_seconds': row['training_duration_seconds']})
            refs[name] = mlflow.active_run().info.run_id
    return {'status': 'enabled', 'tracking_uri': uri, 'run_ids': refs}


def run_all(root: Path, *, epochs: int = 45, patience: int = 10,
            device: str = 'cpu', mlflow_enabled: bool = False) -> None:
    cp1, cp1_sha = require_cp1_report(root)
    _, cp2_sha = require_cp2_report(root, cp1_sha)
    if epochs < 1 or epochs > 300 or patience < 1 or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Invalid CP3 settings/device')
    if device not in {'cpu', 'cuda'}:
        raise ValueError('Unknown device')
    set_seed(42)
    task = load_locked_cora(root / 'data' / 'raw' / 'cora', cp1['tasks']['cora']['data_contract'])
    if (len(task.y_train), len(task.y_val), task.test_count_sealed) != (
            cp1['tasks']['cora']['n_train'], cp1['tasks']['cora']['n_validation'], cp1['tasks']['cora']['n_test_sealed']):
        raise ValueError('Cora split differs from CP1 frozen baseline')
    baseline_score = cp1['tasks']['cora']['metrics_validation']['f1_macro']
    out = root / 'outputs' / 'cp3'
    report_path = out / 'cp3_validation_report.json'
    report = {'project': '13_deep_learning_graph_ml', 'checkpoint': 'CP3',
              'status': 'partial_validation_only', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'seed': 42, 'platform': platform.platform(), 'python': sys.version.split()[0],
              'torch': version('torch'), 'torch_geometric': version('torch-geometric'),
              'device': device, 'torch_threads': torch.get_num_threads(),
              'cp1_report_sha256': cp1_sha, 'cp2_report_sha256': cp2_sha,
              'cora_baseline_validation_macro_f1': baseline_score,
              'task_contract': task.metadata, 'results': {}, 'candidate_selection': None,
              'mlflow_tracking': {'status': 'not_requested', 'note': 'Local JSON and checkpoint hashes are always recorded.'}}
    # Small, predeclared, deterministic candidate list. No final-test-based choices.
    candidates = [('gcn_h32_d03', 'gcn', 32, .3), ('gcn_h64_d05', 'gcn', 64, .5),
                  ('graphsage_h32_d03', 'graphsage', 32, .3), ('graphsage_h64_d05', 'graphsage', 64, .5)]
    for name, family, hidden, dropout in candidates:
        print(f'CP3 train {name}: train-mask labels only; validation selection...', flush=True)
        set_seed(42)
        m = make_model(family, hidden, dropout)
        row = fit_graph_model(task=task, model=m, name=name, output_dir=out / 'checkpoints',
                              epochs=epochs, patience=patience, seed=42, use_edges=True, device=device)
        row['config'] = {'architecture': family, 'hidden_dim': hidden, 'dropout': dropout}
        row['baseline_validation_f1_macro'] = baseline_score
        row['delta_vs_cora_baseline_validation'] = row['metrics_validation']['f1_macro'] - baseline_score
        row['checkpoint_local_path'] = str(Path('outputs') / 'cp3' / 'checkpoints' / (name + '_best.pt'))
        report['results'][name] = row
        atomic_report(report_path, report)
        print(f'CP3 {name}: val_macro_f1={row["metrics_validation"]["f1_macro"]:.4f} '
              f'(CP1 baseline {baseline_score:.4f})', flush=True)
    # Selection is by observed validation macro-F1, then fewer parameters on exact ties.
    chosen = max(candidates, key=lambda c: (report['results'][c[0]]['metrics_validation']['f1_macro'],
                                             -report['results'][c[0]]['parameter_count']))
    best_name, family, hidden, dropout = chosen
    print(f'CP3 selected on VALIDATION ONLY: {best_name}. Training matched no-edge ablation...', flush=True)
    set_seed(42)
    no_edges = fit_graph_model(task=task, model=make_model(family, hidden, dropout),
                               name=best_name + '_no_edges', output_dir=out / 'checkpoints',
                               epochs=epochs, patience=patience, seed=42, use_edges=False, device=device)
    no_edges['config'] = {'architecture': family, 'hidden_dim': hidden, 'dropout': dropout}
    no_edges['checkpoint_local_path'] = str(Path('outputs') / 'cp3' / 'checkpoints' / (best_name + '_no_edges_best.pt'))
    report['results'][best_name + '_no_edges'] = no_edges
    report['candidate_selection'] = {'selected_name': best_name, 'selection_metric': 'validation_macro_f1',
             'full_graph_validation_score': report['results'][best_name]['metrics_validation']['f1_macro'],
             'no_edges_validation_score': no_edges['metrics_validation']['f1_macro'],
             'graph_edge_ablation_delta': report['results'][best_name]['metrics_validation']['f1_macro'] - no_edges['metrics_validation']['f1_macro'],
             'warning': 'The no-edge model is retrained, not an inference-time edge removal. Performance deltas are validation-only and not causal.'}
    # Load selected checkpoint weights into same architecture; no test labels ever involved.
    ckpt = out / 'checkpoints' / f'{best_name}_best.pt'
    selected_model = make_model(family, hidden, dropout)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    selected_model.load_state_dict(state['state_dict'])
    report['validation_node_attribution'] = validation_node_saliency(task, selected_model)
    if mlflow_enabled:
        report['mlflow_tracking'] = optional_mlflow(report, root)
    report['status'] = 'validation_complete_awaiting_review'
    atomic_report(report_path, report)
    print(f'CP3 RUN FINISHED ({report["status"]}): {report_path}', flush=True)
    print('Test labels remain sealed; send JSON for review before CP4.', flush=True)
