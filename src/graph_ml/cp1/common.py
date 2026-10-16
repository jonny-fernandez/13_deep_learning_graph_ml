"""Dataset contracts and validation-only reporting; final test labels stay sealed."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, log_loss, precision_recall_fscore_support,
    roc_auc_score,
)

SEED = 42


@dataclass
class Splits:
    """ONLY train/validation feature and label arrays. No test arrays/labels."""
    x_train: object
    x_val: object
    y_train: np.ndarray
    y_val: np.ndarray
    n_sealed_test: int
    metadata: dict


def chronological_boundaries(n: int, train_ratio: float = .70, val_ratio: float = .15) -> tuple[np.ndarray, np.ndarray, int]:
    """Preserve source order and quarantine final test rows."""
    if n < 30 or not (0 < train_ratio < 1) or not (0 < val_ratio < 1 - train_ratio):
        raise ValueError("Invalid dataset length or chronological split ratios")
    a, b = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    train_idx, val_idx = np.arange(a), np.arange(a, b)
    if train_idx[-1] >= val_idx[0] or b >= n:
        raise AssertionError("Non-disjoint or empty splits")
    return train_idx, val_idx, n - b


def assert_disjoint(train, val, test=None) -> None:
    a, b = set(map(int, train)), set(map(int, val))
    if not a or not b or not a.isdisjoint(b):
        raise ValueError("Train/validation groups or nodes overlap (or are empty)")
    if test is not None:
        c = set(map(int, test))
        if not c or not a.isdisjoint(c) or not b.isdisjoint(c):
            raise ValueError("Test groups or nodes overlap training/validation")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validation_metrics(y_true, p, *, labels: list[int]) -> dict:
    """All metrics are on the validation split. No sealed test reference."""
    y = np.asarray(y_true, dtype=int)
    probs = np.asarray(p, dtype=float)
    if len(y) == 0 or probs.shape != (len(y), len(labels)):
        raise ValueError("Probability matrix shape does not match validation targets/classes")
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid validation probabilities")
    # float32 softmax/sigmoid rounding can trigger spurious sklearn log_loss
    # 'probabilities do not sum to 1' warnings. Validate first, then normalize
    # in float64 without changing the class ranking or evaluation split.
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError('Invalid probabilities outside [0, 1]')
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = np.asarray(labels)[probs.argmax(axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y, preds, labels=labels, zero_division=0)
    result = {
        'n_val': int(len(y)), 'accuracy': float(accuracy_score(y, preds)),
        'balanced_accuracy': float(balanced_accuracy_score(y, preds)),
        'f1_macro': float(f1_score(y, preds, labels=labels, average='macro', zero_division=0)),
        'log_loss': float(log_loss(y, probs, labels=labels)),
        'labels': [int(i) for i in labels],
        'per_class': {str(k): {'precision': float(a), 'recall': float(b), 'f1': float(c), 'support': int(d)}
                      for k, a, b, c, d in zip(labels, precision, recall, f1, support)},
        'confusion_matrix': confusion_matrix(y, preds, labels=labels).tolist(),
    }
    if len(labels) == 2:
        # Fixed positive label=1, fixed decision threshold=0.5.
        if labels != [0, 1]:
            raise ValueError('Binary labels must be [0, 1]')
        if len(np.unique(y)) == 2:
            result['roc_auc'] = float(roc_auc_score(y, probs[:, 1]))
        else:
            result['roc_auc'] = None
        result['pr_auc_average_precision'] = float(average_precision_score(y, probs[:, 1]))
    return result
