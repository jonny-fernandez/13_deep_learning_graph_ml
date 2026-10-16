"""CP1 offline guards: no external downloads or live API calls."""
import io
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
from graph_ml.cp1.common import assert_disjoint, chronological_boundaries, validation_metrics
from graph_ml.cp1.bank import load_bank, bank_estimator
from graph_ml.cp1.downloads import ensure_download, bank_csv, prepare_har
from graph_ml.cp1.har import load_har, har_sequence_train_data


def test_chronological_split_is_ordered_and_sealed():
    tr, va, nt = chronological_boundaries(100)
    assert len(tr) == 70 and len(va) == 15 and nt == 15
    assert tr[-1] + 1 == va[0]
    assert va[-1] == 84


def test_overlap_rejected():
    assert_disjoint([1, 2], [3], [4, 5])
    with pytest.raises(ValueError):
        assert_disjoint([1, 2], [2], [4])
    with pytest.raises(ValueError):
        assert_disjoint([1, 2], [3], [2, 4])


def test_bank_excludes_call_duration_and_learns_categories_on_training_only(tmp_path):
    n = 120
    df = pd.DataFrame({'age': [25 + (i % 15) for i in range(n)],
                       'job': ['only_validation' if 84 <= i < 102 else 'admin' for i in range(n)],
                       'month': ['jan'] * n, 'duration': [100000 if i >= 84 else 1 for i in range(n)],
                       'y': ['yes' if i % 3 == 0 else 'no' for i in range(n)]})
    path = tmp_path / 'bank-additional-full.csv'
    df.to_csv(path, sep=';', index=False)
    splits = load_bank(path)
    assert 'duration' not in splits.x_train and 'duration' not in splits.x_val
    assert splits.n_sealed_test == 18
    model = bank_estimator(splits.x_train)
    model.fit(splits.x_train, splits.y_train)
    assert model.predict_proba(splits.x_val).shape == (18, 2)
    categorical = model.named_steps['preprocess'].named_transformers_['categorical']
    learned = categorical.named_steps['onehot'].categories_[0]
    assert 'only_validation' not in learned


def test_invalid_target_fails_closed(tmp_path):
    df = pd.DataFrame({'age': [20] * 40, 'job': ['a'] * 40, 'month': ['jan'] * 40,
                       'duration': [2] * 40, 'y': ['unknown'] * 40})
    path = tmp_path / 'data.csv'
    df.to_csv(path, sep=';', index=False)
    with pytest.raises(ValueError, match='target'):
        load_bank(path)


def test_metrics_are_validation_only_and_class_specific():
    metrics = validation_metrics(np.array([0, 1, 1, 0]),
                                 np.array([[.9, .1], [.2, .8], [.1, .9], [.8, .2]]), labels=[0, 1])
    assert metrics['n_val'] == 4
    assert metrics['roc_auc'] == 1.0
    assert metrics['f1_macro'] == 1.0
    assert metrics['confusion_matrix'] == [[2, 0], [0, 2]]
    assert 'test' not in metrics


def test_malformed_probabilities_rejected():
    with pytest.raises(ValueError, match='Invalid'):
        validation_metrics([0, 1], [[.8, .1], [.2, .8]], labels=[0, 1])


def test_multiclass_metrics_without_binary_auc():
    m = validation_metrics([0, 1, 2], [[.9, .05, .05], [.05, .9, .05], [.05, .05, .9]], labels=[0, 1, 2])
    assert m['f1_macro'] == 1.0
    assert 'roc_auc' not in m


def test_downloader_never_accepts_http(tmp_path):
    with pytest.raises(ValueError, match='Unapproved'):
        ensure_download('http://example.com/untrusted.zip', tmp_path / 'x.zip')


def test_bank_zip_member_selection_without_path_traversal(tmp_path):
    archive = tmp_path / 'bank.zip'
    with ZipFile(archive, 'w') as z:
        z.writestr('bank-additional/bank-additional-full.csv', 'age;job;duration;y\n22;admin.;32;no\n')
        z.writestr('../../malicious.txt', 'untrusted')
    csv = bank_csv(archive, tmp_path / 'safe' / 'data.csv')
    assert csv.is_file()
    assert not (tmp_path / 'malicious.txt').exists()



def test_bank_nested_uci_archive_structure(tmp_path):
    """Actual UCI bank+marketing.zip nests bank-additional.zip before the CSV."""
    outer_path = tmp_path / 'bank+marketing.zip'
    inner_buffer = io.BytesIO()
    valid_csv = 'age;job;duration;y\n22;admin.;32;no\n'
    with ZipFile(inner_buffer, 'w') as inner:
        inner.writestr('bank-additional/bank-additional-full.csv', valid_csv)
        inner.writestr('../../malicious.txt', 'should never be extracted')
    with ZipFile(outer_path, 'w') as outer:
        outer.writestr('bank.zip', b'unrelated dataset')
        outer.writestr('bank-additional.zip', inner_buffer.getvalue())
    output = tmp_path / 'raw' / 'bank-additional-full.csv'
    assert bank_csv(outer_path, output) == output
    assert output.read_text() == valid_csv
    assert not (tmp_path / 'malicious.txt').exists()
    # Cache works without re-extracting (and without touching the archive).
    outer_path.unlink()
    assert bank_csv(outer_path, output) == output


def test_bank_nested_official_quoted_semicolon_header(tmp_path):
    """UCI publishes quoted headers; the loader must accept actual CSV syntax."""
    columns = ('age', 'job', 'marital', 'education', 'default', 'housing',
               'loan', 'contact', 'month', 'day_of_week', 'duration',
               'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate',
               'cons.price.idx', 'cons.conf.idx', 'euribor3m',
               'nr.employed', 'y')
    header = ';'.join(f'"{c}"' for c in columns)
    row = ';'.join(('22', '"admin."', '"single"', '"university.degree"',
                    '"no"', '"no"', '"no"', '"cellular"', '"may"', '"mon"',
                    '32', '1', '999', '0', '"nonexistent"', '-1.8',
                    '93.1', '-46.2', '1.34', '5099.1', '"no"'))
    content = header + '\r\n' + row + '\r\n'
    inner_data = io.BytesIO()
    with ZipFile(inner_data, 'w') as inner:
        inner.writestr('bank-additional/bank-additional-full.csv', content)
    archive = tmp_path / 'bank+marketing.zip'
    with ZipFile(archive, 'w') as outer:
        outer.writestr('bank-additional.zip', inner_data.getvalue())
    output = tmp_path / 'bank-additional-full.csv'
    assert bank_csv(archive, output) == output
    assert pd.read_csv(output, sep=';').columns.tolist() == list(columns)
    # Cached file also validates without needing to reopen the archive.
    archive.unlink()
    assert bank_csv(archive, output) == output


def test_bank_direct_quoted_header_and_utf8_bom(tmp_path):
    archive = tmp_path / 'bank.zip'
    with ZipFile(archive, 'w') as outer:
        outer.writestr('bank-additional-full.csv',
                       b'\xef\xbb\xbf"age";"job";"duration";"y"\r\n22;admin.;32;no\r\n')
    output = tmp_path / 'data.csv'
    assert bank_csv(archive, output) == output
    assert pd.read_csv(output, sep=';').columns.tolist() == ['age', 'job', 'duration', 'y']


@pytest.mark.parametrize('header', [
    '"age";"job";"duration";"duration";"y"\n',
    '"age";"job";"duration";"outcome"\n',
    '"age";"job";"duration";"y";"y"\n',
])
def test_bank_duplicate_or_missing_columns_rejected(tmp_path, header):
    archive = tmp_path / 'bank.zip'
    with ZipFile(archive, 'w') as outer:
        outer.writestr('bank-additional-full.csv', header + '22;admin.;32;no\n')
    output = tmp_path / 'output.csv'
    with pytest.raises(ValueError, match='Unexpected Bank Marketing CSV header'):
        bank_csv(archive, output)
    assert not output.exists()


def test_bank_nested_archive_missing_csv_fails_closed(tmp_path):
    inner_buffer = io.BytesIO()
    with ZipFile(inner_buffer, 'w') as inner:
        inner.writestr('bank-additional/bank-additional.csv', 'wrong dataset')
    outer_path = tmp_path / 'bank.zip'
    with ZipFile(outer_path, 'w') as outer:
        outer.writestr('bank-additional.zip', inner_buffer.getvalue())
    output = tmp_path / 'raw' / 'bank-additional-full.csv'
    with pytest.raises(ValueError, match='Expected unique bank-additional-full.csv'):
        bank_csv(outer_path, output)
    assert not output.exists()


def test_bank_nested_archive_duplicate_csv_fails_closed(tmp_path):
    inner_buffer = io.BytesIO()
    with ZipFile(inner_buffer, 'w') as inner:
        inner.writestr('one/bank-additional-full.csv', 'age;job;duration;y\n22;a;5;yes\n')
        inner.writestr('two/bank-additional-full.csv', 'age;job;duration;y\n22;a;5;yes\n')
    outer_path = tmp_path / 'bank.zip'
    with ZipFile(outer_path, 'w') as outer:
        outer.writestr('bank-additional.zip', inner_buffer.getvalue())
    with pytest.raises(ValueError, match='Expected unique bank-additional-full.csv'):
        bank_csv(outer_path, tmp_path / 'raw' / 'bank-additional-full.csv')


def test_bank_nested_archive_rejects_invalid_csv_header(tmp_path):
    outer_path = tmp_path / 'bank.zip'
    with ZipFile(outer_path, 'w') as outer:
        outer.writestr('bank-additional-full.csv', 'not a csv')
    output = tmp_path / 'raw' / 'bank-additional-full.csv'
    with pytest.raises(ValueError, match='Unexpected Bank Marketing CSV header'):
        bank_csv(outer_path, output)
    assert not output.exists()

def create_har_train(tmp_path, *, nested_zip=False):
    zpath = tmp_path / 'har.zip'
    output = tmp_path / 'raw' / 'UCI HAR Dataset'
    def write_contents(z):
        files = {'train/X_train.txt': ' '.join(['0.1'] * 561) + '\n',
                 'train/y_train.txt': '1\n', 'train/subject_train.txt': '1\n',
                 'test/subject_test.txt': '4\n'}
        for signal in ('total_acc', 'body_acc', 'body_gyro'):
            for axis in 'xyz':
                files[f'train/Inertial Signals/{signal}_{axis}_train.txt'] = ' '.join(['0.2'] * 128) + '\n'
        for key, content in files.items():
            z.writestr('UCI HAR Dataset/' + key, content)
    if nested_zip:
        memory = io.BytesIO()
        with ZipFile(memory, 'w') as inner:
            write_contents(inner)
        with ZipFile(zpath, 'w') as outer:
            outer.writestr('UCI HAR Dataset.zip', memory.getvalue())
    else:
        with ZipFile(zpath, 'w') as z:
            write_contents(z)
    prepare_har(zpath, output)
    return output


@pytest.mark.parametrize('nested_zip', [False, True])
def test_har_zip_allowlist_and_shape(tmp_path, nested_zip):
    data = create_har_train(tmp_path, nested_zip=nested_zip)
    assert har_sequence_train_data(data).shape == (1, 128, 9)
    assert not (data / 'test/y_test.txt').exists()  # sealed final labels not extracted
    assert not (data / 'test/X_test.txt').exists()


def test_har_subject_split_contract_and_no_test_labels(tmp_path):
    path = tmp_path / 'har'
    (path / 'train').mkdir(parents=True)
    (path / 'test').mkdir()
    n = 10 * 12
    np.savetxt(path / 'train/X_train.txt', np.zeros((n, 561)), fmt='%d')
    np.savetxt(path / 'train/y_train.txt', np.tile(np.arange(1, 7), 20), fmt='%d')
    np.savetxt(path / 'train/subject_train.txt', np.repeat(np.arange(1, 11), 12), fmt='%d')
    np.savetxt(path / 'test/subject_test.txt', [11, 12], fmt='%d')
    split = load_har(path)
    assert not set(split.metadata['train_group_ids']) & set(split.metadata['val_group_ids'])
    assert split.n_sealed_test == 2
    assert split.x_train.shape[1] == 561
    # No test label file was ever created/read.


def test_har_rejects_subject_overlap_with_test(tmp_path):
    path = tmp_path / 'har'
    (path / 'train').mkdir(parents=True)
    (path / 'test').mkdir()
    n = 100
    np.savetxt(path / 'train/X_train.txt', np.zeros((n, 561)), fmt='%d')
    np.savetxt(path / 'train/y_train.txt', np.tile(np.arange(1, 6), 20), fmt='%d')
    np.savetxt(path / 'train/subject_train.txt', np.repeat(np.arange(1, 11), 10), fmt='%d')
    np.savetxt(path / 'test/subject_test.txt', np.arange(1, 11), fmt='%d')
    with pytest.raises(ValueError, match='overlap'):
        load_har(path)


def test_cora_public_masks_and_label_slicing_offline(tmp_path, monkeypatch):
    """Exercise Cora contract without downloading the network dataset."""
    import sys
    import types
    import torch
    from graph_ml.cp1.cora import load_cora

    raw = tmp_path / 'Cora' / 'raw'
    raw.mkdir(parents=True)
    for component in ('x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index'):
        (raw / f'ind.cora.{component}').write_bytes(b'official-fixture-placeholder')

    class FakeGraph:
        x = torch.eye(6, 3)
        y = torch.tensor([0, 1, 0, 1, 0, 1])
        edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]])
        train_mask = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)
        val_mask = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool)
        test_mask = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool)
        num_nodes = 6

    fake = types.ModuleType('torch_geometric.datasets')
    fake.Planetoid = lambda **kwargs: [FakeGraph()]
    monkeypatch.setitem(sys.modules, 'torch_geometric.datasets', fake)
    split = load_cora(tmp_path)
    assert split.n_sealed_test == 2
    assert split.y_train.tolist() == [0, 1]
    assert split.y_val.tolist() == [0, 1]
    assert split.metadata['transductive_warning']
    assert len(split.metadata['raw_file_sha256']) == 8
