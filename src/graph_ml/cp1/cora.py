"""Cora public-mask transductive node classification, graph-free baseline."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .common import Splits, assert_disjoint, sha256_file


def load_cora(root) -> Splits:
    from torch_geometric.datasets import Planetoid
    graph = Planetoid(root=str(root), name='Cora', split='public')[0]
    x = graph.x.detach().cpu().numpy()
    mask = {k: getattr(graph, k).detach().cpu().numpy().astype(bool) for k in ('train_mask', 'val_mask', 'test_mask')}
    assert_disjoint(np.flatnonzero(mask['train_mask']), np.flatnonzero(mask['val_mask']), np.flatnonzero(mask['test_mask']))
    if not np.isfinite(x).all() or len(mask['train_mask']) != x.shape[0]:
        raise ValueError('Invalid Cora node features/masks')
    raw_folder = Path(root) / 'Cora' / 'raw'
    source_file_hashes = {p.name: sha256_file(p) for p in sorted(raw_folder.glob('ind.cora.*'))}
    if len(source_file_hashes) != 8:
        raise ValueError('Cora raw provenance incomplete; expected 8 official Planetoid files')
    # Note: no test labels or test performance are inspected here.
    tr, va = mask['train_mask'], mask['val_mask']
    # Slice target tensor first: never export or score sealed test-node labels.
    y_train = graph.y[graph.train_mask].detach().cpu().numpy()
    y_val = graph.y[graph.val_mask].detach().cpu().numpy()
    return Splits(x[tr], x[va], y_train, y_val, int(mask['test_mask'].sum()),
                  {'task': 'cora', 'split': 'Planetoid public fixed masks, transductive nodes and edges',
                   'target': '7-way research topic class for a paper node',
                   'prediction_time': 'static citation graph snapshot (not future-link prediction)',
                   'n_nodes': int(graph.num_nodes), 'n_edges_directed_entries': int(graph.edge_index.shape[1]),
                   'n_features_raw': int(x.shape[1]), 'raw_file_sha256': source_file_hashes, 'train_mask_count': int(tr.sum()),
                   'val_mask_count': int(va.sum()),
                   'transductive_warning': 'test node features/topology are observable, but only train node labels may be used for fitting'})


def cora_estimator():
    # train-mask-only scaling; graph-free logistic regression on node features
    return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=700, random_state=42))
