"""HAR activity-window classification: subject-held-out validation, not forecasting."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .common import Splits, assert_disjoint, sha256_file

SIGNALS = tuple(f'{kind}_{axis}' for kind in ('total_acc', 'body_acc', 'body_gyro') for axis in 'xyz')


def har_sequence_train_data(directory):
    """For CP2: each sample is one completed 128-step sensor window."""
    sequences = []
    for kind in ('total_acc', 'body_acc', 'body_gyro'):
        for axis in 'xyz':
            name = f'{kind}_{axis}_train.txt'
            sequences.append(np.atleast_2d(np.loadtxt(directory / 'train' / 'Inertial Signals' / name, dtype=np.float32)))
    x = np.stack(sequences, axis=-1)
    if x.shape[1:] != (128, 9):
        raise ValueError('HAR inertial shape should be (N, 128, 9)')
    return x


def load_har(directory) -> Splits:
    X = np.loadtxt(directory / 'train' / 'X_train.txt', dtype=np.float32)
    y = np.loadtxt(directory / 'train' / 'y_train.txt', dtype=int) - 1
    subject = np.loadtxt(directory / 'train' / 'subject_train.txt', dtype=int)
    test_subject = np.loadtxt(directory / 'test' / 'subject_test.txt', dtype=int)
    if len(X) != len(y) or len(y) != len(subject) or X.shape[1] != 561 or not np.isin(y, np.arange(6)).all():
        raise ValueError('HAR train feature/label/subject contract failure')
    train, val = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=42).split(X, y, groups=subject))
    assert_disjoint(np.unique(subject[train]), np.unique(subject[val]), np.unique(test_subject))
    return Splits(X[train], X[val], y[train], y[val], len(test_subject),
                  {'task': 'har', 'source_rows_training': len(y), 'split': 'official subject-disjoint train/test; train-group 80/20 validation',
                   'target': 'six activity classes (0-5)', 'prediction_time': 'after 128-sample window ends',
                   'n_subject_train': int(len(set(subject[train]))), 'n_subject_val': int(len(set(subject[val]))),
                   'n_subject_sealed_test': int(len(set(test_subject))),
                   'n_features_engineered': int(X.shape[1]),
                   'train_source_sha256': sha256_file(directory / 'train' / 'X_train.txt'),
                   'val_group_ids': sorted(set(map(int, subject[val]))),
                   'train_group_ids': sorted(set(map(int, subject[train])))})


def har_estimator():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))
