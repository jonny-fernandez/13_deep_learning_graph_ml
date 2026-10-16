"""Pre-contact Bank Marketing prediction with chronological split."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .common import Splits, chronological_boundaries

FORBIDDEN_COLUMNS = frozenset({'duration', 'y'})


def load_bank(path) -> Splits:
    df = pd.read_csv(path, sep=';')
    expected = {'y', 'duration', 'job', 'age', 'month'}
    if not expected.issubset(df):
        raise ValueError(f'Unexpected Bank Marketing schema; missing {expected - set(df)}')
    ti, vi, n_test = chronological_boundaries(len(df))
    development = df.iloc[:(len(ti) + len(vi))]
    if not development['y'].isin(['yes', 'no']).all() or development.isna().any().any():
        raise ValueError('Unexpected Bank Marketing train/validation target or missing data')
    X_train = df.iloc[ti].drop(columns=list(FORBIDDEN_COLUMNS)).copy()
    X_val = df.iloc[vi].drop(columns=list(FORBIDDEN_COLUMNS)).copy()
    assert not FORBIDDEN_COLUMNS.intersection(X_train.columns)
    assert not FORBIDDEN_COLUMNS.intersection(X_val.columns)
    y_train = (df.iloc[ti]['y'] == 'yes').to_numpy(dtype=int)
    y_val = (df.iloc[vi]['y'] == 'yes').to_numpy(dtype=int)
    return Splits(X_train, X_val, y_train, y_val, n_test,
                  {'task': 'bank', 'source_rows': len(df), 'split': 'source row order: first 70% train, next 15% validation, last 15% sealed',
                   'target': 'y (yes=1)', 'prediction_time': 'before campaign phone call',
                   'excluded': sorted(FORBIDDEN_COLUMNS), 'n_features_raw': int(X_train.shape[1]),
                   'validation_positive_rate': float(np.mean(y_val)), 'train_positive_rate': float(np.mean(y_train))})


def bank_estimator(X: pd.DataFrame) -> Pipeline:
    categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical),
    ])
    return Pipeline([('preprocess', preprocess), ('model', LogisticRegression(max_iter=400, random_state=42))])
