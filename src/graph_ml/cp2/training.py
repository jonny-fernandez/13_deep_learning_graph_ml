"""Bounded, deterministic CPU-first PyTorch validation experiment loop."""
from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from graph_ml.cp1.common import validation_metrics
from sklearn.metrics import average_precision_score, f1_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU-first; Windows DataLoader worker processes disabled (num_workers=0).
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))


def probability_matrix(model: nn.Module, x: np.ndarray, classes: int, batch_size: int,
                       device: torch.device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    prob = []
    with torch.inference_mode():
        for (xb,) in loader:
            logits = model(xb.to(device))
            if classes == 2:
                positive = torch.sigmoid(logits.view(-1)).cpu().numpy()
                prob.append(np.stack([1.0 - positive, positive], axis=-1))
            else:
                prob.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(prob, axis=0)


def threshold_analysis(y, probability_positive):
    """Validation-only secondary operating-point analysis, never test tuning."""
    p = np.asarray(probability_positive)
    if p.shape != np.asarray(y).shape:
        raise ValueError('Incorrect binary prediction dimensions')
    out = []
    for threshold in np.linspace(.05, .95, 19):
        predictions = (p >= threshold).astype(int)
        out.append({'threshold': round(float(threshold), 4),
                    'positive_recall': float(np.sum((predictions == 1) & (y == 1)) / max(1, np.sum(y == 1))),
                    'f1_positive': float(f1_score(y, predictions, zero_division=0))})
    best = max(out, key=lambda d: (d['f1_positive'], -abs(d['threshold'] - .5)))
    return {'default_0p5': next(k for k in out if abs(k['threshold'] - .5) < 1e-5),
            'validation_selected_f1_threshold': best,
            'warning': 'Threshold selected on validation only; performance is not an unbiased test estimate.'}


def fit_model(name: str, model: nn.Module, x_train: np.ndarray, y_train: np.ndarray,
              x_val: np.ndarray, y_val: np.ndarray, *, classes: int,
              primary_metric: str, output_dir: Path, epochs: int = 6, batch_size: int = 256,
              learning_rate: float = 0.001, patience: int = 2, seed: int = 42,
              device: str = 'cpu') -> dict:
    if epochs < 1 or batch_size < 1 or classes < 2 or patience < 1:
        raise ValueError('Invalid CP2 training config')
    if primary_metric not in ('pr_auc_average_precision', 'f1_macro'):
        raise ValueError('Unsupported model-selection metric')
    if len(x_train) != len(y_train) or len(x_val) != len(y_val) or len(y_val) == 0:
        raise ValueError('Mismatched/empty train-validation split')
    if not np.isin(np.unique(y_train), np.arange(classes)).all() or not np.isin(np.unique(y_val), np.arange(classes)).all():
        raise ValueError('Unexpected class IDs')
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_device = torch.device(device)
    model.to(selected_device)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=batch_size, shuffle=True, generator=loader_generator,
                        num_workers=0)
    criterion = nn.BCEWithLogitsLoss() if classes == 2 else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=1)
    history, highest, best_epoch, best_state, stale = [], -float('inf'), None, None, 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(selected_device), yb.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits.reshape(-1), yb.float()) if classes == 2 else criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.detach().item()) * len(yb)
            seen += len(yb)
        probs = probability_matrix(model, x_val, classes, batch_size, selected_device)
        metrics = validation_metrics(y_val, probs, labels=list(range(classes)))
        score = metrics.get(primary_metric)
        if score is None or not np.isfinite(score):
            raise ValueError('Invalid primary validation metric')
        history.append({'epoch': epoch, 'training_loss': running_loss / seen,
                        'validation_primary_metric': score,
                        'validation_macro_f1': metrics['f1_macro'],
                        'learning_rate': optimizer.param_groups[0]['lr']})
        print(f'[{name}] epoch={epoch}/{epochs} loss={running_loss/seen:.4f} '
              f'val_{primary_metric}={score:.4f}', flush=True)
        scheduler.step(score)
        if score > highest + 1e-6:
            highest, best_epoch, best_state, stale = score, epoch, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    training_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError('Training did not produce a valid model state')
    model.load_state_dict(best_state)
    infer_start = time.perf_counter()
    probs = probability_matrix(model, x_val, classes, batch_size, selected_device)
    inference_seconds = time.perf_counter() - infer_start
    metrics = validation_metrics(y_val, probs, labels=list(range(classes)))
    checkpoint = output_dir / f'{name}_best.pt'
    torch.save({'state_dict': best_state, 'model_name': name, 'seed': seed,
                'selected_epoch': best_epoch, 'primary_validation_metric': primary_metric}, checkpoint)
    result = {'model': name, 'status': 'validation_only_test_sealed', 'selected_epoch': best_epoch,
              'epochs_completed': len(history), 'selection_metric': primary_metric,
              'best_validation_selection_score': highest,
              'n_train': int(len(y_train)), 'n_validation': int(len(y_val)),
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'training_duration_seconds': round(training_seconds, 5),
              'validation_predict_duration_seconds': round(inference_seconds, 5),
              'checkpoint_local_path': str(checkpoint.relative_to(output_dir.parent.parent)),
              'checkpoint_bytes': checkpoint.stat().st_size,
              'metrics_validation': metrics, 'learning_curve': history,
              'training_config': {'epochs_max': epochs, 'patience': patience,
                                  'batch_size': batch_size, 'learning_rate': learning_rate,
                                  'weight_decay': 0.0001, 'gradient_clip_norm': 1.,
                                  'optimizer': 'AdamW', 'scheduler': 'ReduceLROnPlateau',
                                  'seed': seed, 'device': str(selected_device),
                                  'data_loader_workers': 0}}
    if classes == 2:
        result['threshold_analysis_validation'] = threshold_analysis(y_val, probs[:, 1])
    return result
