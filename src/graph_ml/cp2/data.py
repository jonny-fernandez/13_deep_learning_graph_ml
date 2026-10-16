"""CP2 training-only data views. No held-out test features/labels accessed."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from graph_ml.cp1.bank import load_bank, bank_estimator
from graph_ml.cp1.har import har_sequence_train_data


def bank_arrays(csv_file, cp1_task):
    split = load_bank(csv_file)
    if (len(split.y_train), len(split.y_val), split.n_sealed_test) != (
            cp1_task['n_train'], cp1_task['n_validation'], cp1_task['n_test_sealed']):
        raise ValueError('Bank CP1 split mismatch; refusing to compare experiments')
    pipeline = bank_estimator(split.x_train)
    transform = pipeline.named_steps['preprocess']
    x_train = transform.fit_transform(split.x_train, split.y_train)
    x_val = transform.transform(split.x_val)
    if sparse.issparse(x_train):
        x_train = x_train.toarray()
    if sparse.issparse(x_val):
        x_val = x_val.toarray()
    if x_train.size > 10_000_000 or x_val.size > 10_000_000:
        raise ValueError('Unexpected dense feature size; review before memory allocation')
    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
        raise ValueError('Nonfinite transformed Bank features')
    return x_train, split.y_train.astype(np.int64), x_val, split.y_val.astype(np.int64), {
        'input_dim': int(x_train.shape[1]), 'preprocessing': 'CP1 encoding/scaling fit TRAIN only',
        'prediction_time': 'pre-contact; duration excluded',
        'split': split.metadata['split'], 'source_rows': split.metadata['source_rows']}


def subject_partition(subject_ids, train_ids, val_ids):
    subjects = np.asarray(subject_ids, dtype=int)
    train_ids, val_ids = set(map(int, train_ids)), set(map(int, val_ids))
    if not train_ids or not val_ids or train_ids & val_ids:
        raise ValueError('CP1 train/validation subject sets overlap or are empty')
    if set(map(int, subjects)) != train_ids | val_ids:
        raise ValueError('HAR subject IDs differ from locked CP1 baseline report')
    train_mask = np.isin(subjects, list(train_ids))
    val_mask = np.isin(subjects, list(val_ids))
    if train_mask.sum() == 0 or val_mask.sum() == 0 or np.any(train_mask & val_mask):
        raise ValueError('Empty/overlapping HAR subject partitions')
    return train_mask, val_mask


def har_arrays(directory, cp1_task):
    # Only training folder is read. HAR test features/labels remain sealed.
    x = har_sequence_train_data(directory)
    y = np.atleast_1d(np.loadtxt(directory / 'train' / 'y_train.txt', dtype=int)) - 1
    subjects = np.atleast_1d(np.loadtxt(directory / 'train' / 'subject_train.txt', dtype=int))
    if not (x.shape == (len(y), 128, 9) and len(y) == len(subjects)):
        raise ValueError('HAR raw sequence/train label/subject mismatch')
    if not np.isin(y, np.arange(6)).all() or not np.isfinite(x).all():
        raise ValueError('HAR feature/target schema failure')
    meta = cp1_task['data_contract']
    train, val = subject_partition(subjects, meta['train_group_ids'], meta['val_group_ids'])
    if (int(train.sum()), int(val.sum())) != (cp1_task['n_train'], cp1_task['n_validation']):
        raise ValueError('HAR CP1 subject split sizes differ')
    # Fit channel normalization only on samples from training subjects.
    means = x[train].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    stds = x[train].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    stds = np.maximum(stds, 1e-6)
    x_train = (x[train] - means) / stds
    x_val = (x[val] - means) / stds
    return x_train.astype(np.float32), y[train].astype(np.int64), x_val.astype(np.float32), y[val].astype(np.int64), {
        'input_dim': 9, 'sequence_length': 128,
        'preprocessing': 'nine channel mean/std from TRAIN subjects only',
        'subject_disjoint': True, 'n_subject_train': meta['n_subject_train'],
        'n_subject_val': meta['n_subject_val'],
        'prediction_time': 'after 128-sample sensor window ends; not future forecasting',
        'representation_note': 'CP1 logistic regression uses engineered 561 features; CP2 models learn from the same completed raw windows'}
