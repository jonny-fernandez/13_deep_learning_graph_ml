"""CP3 CPU-first full-graph training, strictly masked targets and validation selection."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from graph_ml.cp1.common import sha256_file, validation_metrics
from graph_ml.cp2.training import set_seed
from .data import GraphTask


def check_task(task: GraphTask) -> None:
    n = len(task.x)
    if (task.x.ndim != 2 or task.edge_index.shape[0] != 2 or
            task.x.shape[1] != 1433 or n <= 0 or
            task.train_idx.numel() != len(task.y_train) or
            task.val_idx.numel() != len(task.y_val) or
            task.train_idx.numel() == 0 or task.val_idx.numel() == 0 or
            not set(task.train_idx.tolist()).isdisjoint(set(task.val_idx.tolist()))):
        raise ValueError('CP3 graph training target/mask contract failed')
    if task.y_train.min() < 0 or task.y_train.max() >= 7 or task.y_val.min() < 0 or task.y_val.max() >= 7:
        raise ValueError('Unexpected graph classes')


def graph_probabilities(model: nn.Module, x, edges, val_idx):
    model.eval()
    with torch.inference_mode():
        logits = model(x, edges)
        if logits.shape != (x.shape[0], 7):
            raise ValueError('Graph model must output a logit per node per class')
        return torch.softmax(logits[val_idx], dim=-1).cpu().numpy()


def fit_graph_model(*, task: GraphTask, model: nn.Module, name: str, output_dir: Path,
                    epochs: int = 45, patience: int = 10, lr: float = .01,
                    seed: int = 42, use_edges: bool = True, device: str = 'cpu') -> dict:
    check_task(task)
    if epochs < 1 or epochs > 300 or patience < 1 or not (0 < lr < 1):
        raise ValueError('Invalid bounded graph-training settings')
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Graph device unavailable')
    set_seed(seed)
    dest = torch.device(device)
    model = model.to(dest)
    x = task.x.to(dest)
    edge_index = (task.edge_index if use_edges else torch.empty((2, 0), dtype=torch.long)).to(dest)
    idx_train, idx_val = task.train_idx.to(dest), task.val_idx.to(dest)
    targets_train = task.y_train.to(dest)
    # Training code does NOT receive test indices or test labels.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=.5)
    history, selected, state, best_epoch, stale = [], -1.0, None, 0, 0
    start = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index)
        if logits.shape != (x.shape[0], 7):
            raise ValueError('Graph model must produce seven-class node logits')
        # Loss only reads the public 140 labeled training nodes.
        loss = nn.functional.cross_entropy(logits[idx_train], targets_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        probs = graph_probabilities(model, x, edge_index, idx_val)
        metrics = validation_metrics(task.y_val.cpu().numpy(), probs, labels=list(range(7)))
        score = metrics['f1_macro']
        history.append({'epoch': epoch, 'train_loss': float(loss.item()),
                        'validation_f1_macro': float(score), 'learning_rate': optimizer.param_groups[0]['lr']})
        scheduler.step(score)
        if score > selected + 1e-6:
            selected, state, best_epoch, stale = score, deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f'[{name}] epoch {epoch}/{epochs} train_loss={loss.item():.4f} val_macro_F1={score:.4f}', flush=True)
        if stale >= patience:
            print(f'[{name}] validation early stop after {epoch} epochs', flush=True)
            break
    seconds = time.perf_counter() - start
    if state is None:
        raise RuntimeError('No graph model state selected')
    model.load_state_dict(state)
    begin = time.perf_counter()
    probs = graph_probabilities(model, x, edge_index, idx_val)
    inference_seconds = time.perf_counter() - begin
    metrics = validation_metrics(task.y_val.cpu().numpy(), probs, labels=list(range(7)))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f'{name}_best.pt'
    torch.save({'state_dict': state, 'seed': seed, 'selected_epoch': best_epoch,
                'architecture': type(model).__name__, 'edges_used': use_edges,
                'selection': 'validation_macro_f1'}, checkpoint)
    return {'experiment_name': name, 'model': type(model).__name__,
            'status': 'validation_only_test_sealed', 'edge_mode': 'full_transductive' if use_edges else 'no_edges_self_features_only',
            'train_loss_target_count': int(len(task.y_train)), 'validation_target_count': int(len(task.y_val)),
            'sealed_test_count': task.test_count_sealed, 'selected_epoch': best_epoch,
            'epochs_completed': len(history), 'parameter_count': int(sum(p.numel() for p in model.parameters())),
            'training_duration_seconds': round(seconds, 5), 'validation_predict_duration_seconds': round(inference_seconds, 5),
            'metrics_validation': metrics, 'learning_curve': history,
            'checkpoint_local_path': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint),
            'checkpoint_bytes': checkpoint.stat().st_size,
            'training_config': {'epochs_max': epochs, 'patience': patience, 'learning_rate': lr,
                                'seed': seed, 'device': device, 'optimizer': 'AdamW',
                                'weight_decay': .0005, 'full_batch': True, 'grad_clip_norm': 1.0}}


def validation_node_saliency(task: GraphTask, model: nn.Module, *, use_edges: bool = True, top_k: int = 10) -> dict:
    """Local gradient*input of predicted class; explanation, NOT causal effects.

    Uses ONE validation node's prediction and its features only, never its label.
    """
    model.eval()
    model = model.cpu()
    x = task.x.detach().clone().cpu().requires_grad_(True)
    edges = task.edge_index.cpu() if use_edges else torch.empty((2,0), dtype=torch.long)
    i = int(task.val_idx[0])
    logits = model(x, edges)
    predicted = int(logits[i].argmax().item())
    score = logits[i, predicted]
    grad = torch.autograd.grad(score, x, retain_graph=False)[0]
    weights = (grad[i] * x.detach()[i]).abs().detach().cpu().numpy()
    top = np.argsort(-weights, kind='stable')[:top_k]
    return {'node_id': i, 'predicted_topic_class': predicted,
            'method': 'absolute local gradient-times-input on selected node features, selected PREDICTED class',
            'feature_indices_zero_based': [int(k) for k in top],
            'attribution_magnitudes': [float(weights[k]) for k in top],
            'interpretation_limitations': 'Gradient sensitivity is not a causal or faithful global explanation; topology/neighbor features can affect the result. Validation node selected deterministically; its label is not used.'}
