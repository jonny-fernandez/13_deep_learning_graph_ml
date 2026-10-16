"""Pre-unlock lineage and artifact integrity gates. NO TEST DATA IS LOADED HERE."""
from __future__ import annotations
import json
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile
import os
from pathlib import Path
from graph_ml.cp1.common import sha256_file
from graph_ml.cp2.runner import require_cp1_report
from graph_ml.cp1.downloads import _unique_member

MODELS = ('bank_logistic', 'bank_mlp', 'har_logistic', 'har_lstm', 'har_transformer',
          'cora_logistic', 'cora_gcn', 'cora_graphsage')
CKPTS = {
    'bank_mlp': ('cp2', 'bank_mlp'),
    'har_lstm': ('cp2', 'har_lstm'),
    'har_transformer': ('cp2', 'har_transformer'),
    'cora_gcn': ('cp3', 'gcn_h64_d05'),
    'cora_graphsage': ('cp3', 'graphsage_h64_d05'),
}

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def _fail(msg: str) -> None:
    raise ValueError('CP4 PRE-UNLOCK GATE: ' + msg)

def _verify_cached_bank_csv_matches_official_archive(archive: Path, extracted: Path) -> None:
    """Compare bytes, not labels. Avoid altered cached CSV changing final test data."""
    if not extracted.is_file():
        _fail('Previously extracted Bank CSV is missing')
    with ZipFile(archive) as outer:
        direct=[x for x in outer.infolist() if not x.is_dir() and Path(x.filename).name=='bank-additional-full.csv']
        if len(direct)>1:
            _fail('Ambiguous Bank Marketing CSV in archive')
        if direct:
            data=outer.read(direct[0])
        else:
            nested=_unique_member(outer,'bank-additional.zip')
            if nested.file_size>30*1024*1024:
                _fail('Oversize nested Bank Marketing ZIP')
            with ZipFile(BytesIO(outer.read(nested))) as inner:
                data=inner.read(_unique_member(inner,'bank-additional-full.csv'))
    if len(data)>30*1024*1024 or sha256(data).hexdigest()!=sha256_file(extracted):
        _fail('Cached Bank CSV differs from verified frozen source archive')


def required_checkpoint_path(root: Path, alias: str) -> Path:
    stage, name = CKPTS[alias]
    return root / 'outputs' / stage / 'checkpoints' / (name + '_best.pt')

def build_lock(root: Path) -> dict:
    """No final-test input paths or labels are referenced by this routine."""
    cp1, h1 = require_cp1_report(root)
    p2 = root / 'outputs' / 'cp2' / 'cp2_validation_report.json'
    if not p2.is_file():
        _fail('CP2 report missing')
    cp2 = _load_json(p2)
    h2 = sha256_file(p2)
    if (cp2.get('checkpoint') != 'CP2' or cp2.get('status') != 'validation_complete_awaiting_review'
            or cp2.get('seed') != 42 or cp2.get('cp1_report_sha256') != h1
            or set(cp2.get('results', {})) != {'bank_mlp','har_lstm','har_transformer'}
            or set(cp2.get('selected_experiments', [])) != {'bank_mlp','har_lstm','har_transformer'}):
        _fail('CP2 frozen report gate failed')
    p3 = root / 'outputs' / 'cp3' / 'cp3_validation_report.json'
    if not p3.is_file():
        _fail('CP3 report missing')
    cp3 = _load_json(p3)
    h3 = sha256_file(p3)
    if (cp3.get('checkpoint') != 'CP3' or cp3.get('status') != 'validation_complete_awaiting_review'
            or cp3.get('seed') != 42 or cp3.get('cp1_report_sha256') != h1
            or cp3.get('cp2_report_sha256') != h2):
        _fail('CP3 does not match CP1/CP2 report hashes or validated state')
    names = {'gcn_h32_d03', 'gcn_h64_d05', 'graphsage_h32_d03',
             'graphsage_h64_d05', 'gcn_h64_d05_no_edges'}
    if set(cp3.get('results', {})) != names:
        _fail('CP3 graph experiment inventory changed')
    selection = cp3.get('candidate_selection') or {}
    full = {k:cp3['results'][k]['metrics_validation']['f1_macro'] for k in names if not k.endswith('_no_edges')}
    if (selection.get('selected_name') != 'gcn_h64_d05'
            or max(full, key=full.get) != 'gcn_h64_d05'
            or abs(selection.get('full_graph_validation_score', -1) - full['gcn_h64_d05']) > 1e-10):
        _fail('Graph selection changed or lacks validation justification')
    if (cp3.get('task_contract', {}).get('test_nodes_sealed') != 1000
            or cp1['tasks']['cora']['n_test_sealed'] != 1000):
        _fail('Cora frozen split mismatch')
    for task, rel in [('bank', 'bank_marketing/bank_marketing.zip'), ('har', 'har/har_uci.zip')]:
        p = root / 'data' / 'raw' / rel
        if not p.is_file() or sha256_file(p) != cp1['tasks'][task]['source_archive_sha256']:
            _fail(f'{task} source archive not present or changed')
    _verify_cached_bank_csv_matches_official_archive(
        root/'data'/'raw'/'bank_marketing'/'bank_marketing.zip',
        root/'data'/'raw'/'bank_marketing'/'bank-additional-full.csv')
    cora_root = root / 'data' / 'raw' / 'cora' / 'Cora' / 'raw'
    for name, h in cp1['tasks']['cora']['data_contract']['raw_file_sha256'].items():
        p = cora_root / name
        if not p.is_file() or sha256_file(p) != h:
            _fail(f'Cora raw source changed: {name}')
    rows = {}
    for alias, (stage, name) in CKPTS.items():
        item = cp2['results'][name] if stage == 'cp2' else cp3['results'][name]
        if item.get('status') != 'validation_only_test_sealed':
            _fail(f'{name}: checkpoint selected against wrong split')
        path = required_checkpoint_path(root, alias)
        if not path.is_file():
            _fail(f'Missing model checkpoint {path}')
        digest = sha256_file(path)
        if stage == 'cp3' and digest != item.get('checkpoint_sha256'):
            _fail(f'CP3 checkpoint content changed: {name}')
        if path.stat().st_size != item['checkpoint_bytes']:
            _fail(f'Checkpoint size changed: {name}')
        rows[alias] = {'name': name, 'checkpoint_sha256': digest,
                       'selected_epoch': item['selected_epoch'], 'parameter_count': item['parameter_count'],
                       'validation_f1_macro': item['metrics_validation']['f1_macro'],
                       'validation_selection_metric': item.get('selection_metric', 'f1_macro')}
    if cp2['results']['bank_mlp']['selection_metric'] != 'pr_auc_average_precision':
        _fail('Bank MLP original selection criterion changed')
    bank_threshold = cp2['results']['bank_mlp']['threshold_analysis_validation']['validation_selected_f1_threshold']['threshold']
    if not isinstance(bank_threshold, (float, int)) or not 0 < bank_threshold < 1:
        _fail('Bank validation-selected threshold is invalid')
    if any(cp2['results'][n]['selection_metric'] != 'f1_macro' for n in ('har_lstm', 'har_transformer')):
        _fail('HAR model selection criterion changed')
    return {'project': '13_deep_learning_graph_ml', 'checkpoint': 'CP4',
            'lock_version': 1, 'status': 'PRE_FINAL_TEST_LOCKED',
            'seed': 42, 'cp1_report_sha256': h1, 'cp2_report_sha256': h2,
            'cp3_report_sha256': h3,
            'final_test_model_inventory': list(MODELS),
            'bank_mlp_operating_threshold_selected_on_validation': bank_threshold,
            'checkpoint_inventory': rows,
            'decision': {'bank': 'report LR vs MLP, use validation PR-AUC for MLP selection',
                         'har': 'report LR vs LSTM vs Transformer; all are pre-specified, no post-test selection',
                         'cora': 'GCN gcn_h64_d05 is validation-selected winner; GraphSAGE h64 is secondary predeclared comparator',
                         'test_use': 'One-time reporting only. Do not select, refit, or retune after test scores.'},
            'sealed_test_sizes': {t:cp1['tasks'][t]['n_test_sealed'] for t in ('bank','har','cora')}}

def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False, sort_keys=True)+'\n', encoding='utf-8')
    os.replace(tmp, path)

def ensure_lock(root: Path) -> tuple[dict, str]:
    current = build_lock(root)
    path = root / 'outputs' / 'cp4' / 'cp4_selection_lock.json'
    if path.exists():
        existing = _load_json(path)
        if existing != current:
            _fail('Existing selection lock differs from source/checkpoints; do NOT unlock tests')
    else:
        atomic_json(path, current)
    return current, sha256_file(path)
