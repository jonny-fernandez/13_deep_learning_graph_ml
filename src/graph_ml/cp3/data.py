"""Sealed Cora task: observe all node features/topology, only train and val labels.

This is a *transductive* static-graph experiment, not inductive/generalization to
unseen future nodes. The Planetoid source holds all labels internally; only
train/validation-label slices leave this construction routine.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from graph_ml.cp1.common import sha256_file


@dataclass
class GraphTask:
    x: torch.Tensor
    edge_index: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    y_train: torch.Tensor
    y_val: torch.Tensor
    test_count_sealed: int
    metadata: dict


def construct_graph_task(graph, cp1_meta: dict, source_hashes: dict) -> GraphTask:
    """Reject changed source/masks and do not provide any test labels to trainers."""
    x, edges = graph.x, graph.edge_index
    n = x.shape[0]
    if x.ndim != 2 or x.shape[1] != 1433 or x.dtype not in (torch.float32, torch.float64):
        raise ValueError('Cora feature dimensions/dtype differ from frozen task')
    if not torch.isfinite(x).all() or edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('Corrupt graph features/edge index')
    if edges.shape[1] == 0 or edges.min() < 0 or edges.max() >= n:
        raise ValueError('Invalid graph topology')
    masks = [getattr(graph, f'{name}_mask') for name in ('train', 'val', 'test')]
    if any(m.dtype != torch.bool or m.shape != (n,) for m in masks):
        raise ValueError('Incorrect public node masks')
    train, val, test = masks
    if torch.any(train & val) or torch.any(train & test) or torch.any(val & test):
        raise ValueError('Node masks overlap; leakage risk')
    if (int(train.sum()), int(val.sum()), int(test.sum())) != (140, 500, 1000):
        raise ValueError('Expected Cora public split 140/500/1000')
    if (n != cp1_meta.get('n_nodes') or edges.shape[1] != cp1_meta.get('n_edges_directed_entries')
            or source_hashes != cp1_meta.get('raw_file_sha256')
            or int(train.sum()) != cp1_meta.get('train_mask_count')
            or int(val.sum()) != cp1_meta.get('val_mask_count')):
        raise ValueError('Cora data provenance/split differ from frozen CP1 contract')
    train_idx, val_idx = train.nonzero(as_tuple=False).reshape(-1), val.nonzero(as_tuple=False).reshape(-1)
    # No computation ever indexes graph.y with test_mask or unmasked/full graph.y.
    y_train = graph.y[train_idx].detach().clone().to(dtype=torch.long)
    y_val = graph.y[val_idx].detach().clone().to(dtype=torch.long)
    if (y_train.min() < 0 or y_train.max() >= 7 or y_val.min() < 0 or y_val.max() >= 7):
        raise ValueError('Unexpected train/validation topic labels')
    return GraphTask(x.detach().clone().float(), edges.detach().clone(), train_idx, val_idx,
                     y_train, y_val, int(test.sum()), {
                         'dataset': 'Cora Planetoid public transductive graph',
                         'n_nodes': n, 'n_edge_index_entries': int(edges.shape[1]),
                         'n_features': int(x.shape[1]), 'classes': 7,
                         'train_nodes': int(train.sum()), 'validation_nodes': int(val.sum()),
                         'test_nodes_sealed': int(test.sum()),
                         'graph_visibility': 'All node FEATURES and edges visible (including unlabeled test nodes). Only training LABELS used for loss.',
                         'source_file_sha256': source_hashes})


def load_locked_cora(root: Path, cp1_meta: dict) -> GraphTask:
    # PyG may internally load all label values, but only train/val targets are exported.
    from torch_geometric.datasets import Planetoid
    graph = Planetoid(root=str(root), name='Cora', split='public')[0]
    raw = root / 'Cora' / 'raw'
    hashes = {p.name: sha256_file(p) for p in sorted(raw.glob('ind.cora.*'))}
    if len(hashes) != 8:
        raise ValueError('Missing or changed official Cora raw files')
    return construct_graph_task(graph, cp1_meta, hashes)
