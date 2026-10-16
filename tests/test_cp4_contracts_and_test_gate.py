"""CP4 offline gates; NEVER touch sealed test labels or network."""
import json
from pathlib import Path
import pytest
from graph_ml.cp4.contracts import build_lock, ensure_lock, required_checkpoint_path, MODELS
from graph_ml.cp4.render import render_evidence
from graph_ml.cp1.common import validation_metrics


def test_no_test_data_in_prelock_source():
    from graph_ml.cp4 import contracts
    source=Path(contracts.__file__).read_text(encoding='utf-8')
    assert 'graph.y[graph.test_mask]' not in source
    assert 'y_test.txt' not in source
    assert 'pd.read_csv' not in source  # byte-hashing the cached CSV is permitted; parsing labels is not
    assert 'Planetoid(' not in source


def test_locked_report_hashes_and_model_inventory(tmp_path):
    import hashlib
    root=Path(__file__).resolve().parents[1]
    cp1=(root/'outputs'/'cp1'/'cp1_validation_report.json')
    # Unit tests don't assume user-generated CP reports are shipped/committed.
    assert len(MODELS)==8 and len(set(MODELS))==8
    assert required_checkpoint_path(tmp_path,'bank_mlp')==tmp_path/'outputs'/'cp2'/'checkpoints'/'bank_mlp_best.pt'
    assert required_checkpoint_path(tmp_path,'cora_gcn')==tmp_path/'outputs'/'cp3'/'checkpoints'/'gcn_h64_d05_best.pt'
    if cp1.exists():
        assert json.loads(cp1.read_text(encoding='utf-8'))['checkpoint']=='CP1'


def test_atomic_lock_is_not_silent_rewrite(tmp_path,monkeypatch):
    import graph_ml.cp4.contracts as contract
    monkeypatch.setattr(contract,'build_lock',lambda r: {'version':1})
    _,h=ensure_lock(tmp_path)
    assert len(h)==64
    _,same=ensure_lock(tmp_path)
    assert h==same
    monkeypatch.setattr(contract,'build_lock',lambda r: {'version':2})
    with pytest.raises(ValueError,match='lock differs'):
        ensure_lock(tmp_path)


def test_metrics_sanity_and_test_label_count():
    import numpy as np
    from graph_ml.cp4.heldout import _score
    v=_score(np.array([0,1,1]),np.array([[.9,.1],[.3,.7],[.2,.8]]),2)
    assert v['n_test']==3 and 'n_val' not in v
    assert v['f1_macro']==1.0


def test_checkpoint_metadata_requires_hash(tmp_path):
    import torch
    from graph_ml.cp4.heldout import _checkpoint
    from graph_ml.models.mlp import MLP
    path=required_checkpoint_path(tmp_path,'bank_mlp')
    path.parent.mkdir(parents=True)
    m=MLP(4)
    torch.save({'state_dict':m.state_dict(),'selected_epoch':4,'seed':42},path)
    bad={'checkpoint_inventory':{'bank_mlp':{'checkpoint_sha256':'0'*64,'selected_epoch':4,
                                              'parameter_count':sum(p.numel() for p in m.parameters())}}}
    with pytest.raises(ValueError,match='Checkpoint changed'):_checkpoint(tmp_path,bad,'bank_mlp',m)


def test_evidence_render_is_report_only(tmp_path):
    from graph_ml.cp4.contracts import MODELS
    import graph_ml.cp4.render as renderer
    source=Path(renderer.__file__).read_text(encoding='utf-8')
    assert 'torch.load(' not in source and 'Planetoid(' not in source
    assert 'y_test.txt' not in source


def test_bank_final_test_synthetic_path_train_only(tmp_path,monkeypatch):
    """Exercise final test code on synthetic rows only; never download or use real sealed data."""
    import numpy as np
    import pandas as pd
    from graph_ml.cp4 import heldout
    from graph_ml.cp1.bank import bank_estimator,FORBIDDEN_COLUMNS
    root=tmp_path
    raw=root/'data'/'raw'/'bank_marketing'
    raw.mkdir(parents=True)
    n=40; train=28; validation=6; test=6
    df=pd.DataFrame({'age':np.arange(n)+20, 'month':['may','nov']*(n//2),
                     'job':['admin.','management']*(n//2), 'duration':np.arange(n)*10000,
                     'y':['yes' if i%3==0 else 'no' for i in range(n)]})
    df.to_csv(raw/'bank-additional-full.csv',index=False,sep=';')
    xtr=df.iloc[:train].drop(columns=list(FORBIDDEN_COLUMNS))
    t=bank_estimator(xtr).named_steps['preprocess'].fit(xtr)
    n_in=t.transform(xtr).shape[1]
    cp1={'tasks':{'bank':{'n_train':train,'n_validation':validation,'n_test_sealed':test}}}
    cp2={'results':{'bank_mlp':{'data_contract':{'input_dim':n_in},'training_duration_seconds':9.0}}}
    fake_lock={'checkpoint_inventory':{'bank_mlp':{'parameter_count':64*n_in+64+64+1}},
               'bank_mlp_operating_threshold_selected_on_validation':0.25}
    # No external checkpoint or test-tuning involved in this unit test.
    monkeypatch.setattr(heldout,'_checkpoint',lambda r,l,a,m:m.eval())
    results=heldout.evaluate_bank(root,fake_lock,cp1,cp2)
    assert set(results)=={'bank_logistic','bank_mlp'}
    assert all(item['metrics_test']['n_test']==test for item in results.values())
    assert all('n_val' not in item['metrics_test'] for item in results.values())
    assert results['bank_mlp']['validation_locked_operating_point_test']['threshold']==0.25


def test_har_test_sequence_layout_synthetic(tmp_path):
    import numpy as np
    from graph_ml.cp4.heldout import _read_har_sequence_test
    folder=tmp_path/'test'/'Inertial Signals';folder.mkdir(parents=True)
    for kind in ('total_acc','body_acc','body_gyro'):
        for axis in 'xyz':
            np.savetxt(folder/f'{kind}_{axis}_test.txt', np.ones((2,128),dtype=np.float32))
    assert _read_har_sequence_test(tmp_path).shape==(2,128,9)


def test_locked_har_test_extraction_only_known_members(tmp_path):
    """A ZIP with fake small test data gets extracted only through an explicit CP4 helper."""
    from zipfile import ZipFile
    from graph_ml.cp4.heldout import _materialize_official_har_test_files
    from pathlib import Path
    archive=tmp_path/'data'/'raw'/'har'/'har_uci.zip'
    archive.parent.mkdir(parents=True)
    required=['test/X_test.txt','test/y_test.txt']
    required += [f'test/Inertial Signals/{kind}_{axis}_test.txt'
                 for kind in ('total_acc','body_acc','body_gyro') for axis in 'xyz']
    with ZipFile(archive,'w') as z:
        for name in required:
            z.writestr('bundle/UCI HAR Dataset/'+name, b'1 2 3\n')
        z.writestr('bundle/UCI HAR Dataset/evil.txt', b'do not extract')
    out=_materialize_official_har_test_files(tmp_path)
    assert all((out/name).read_bytes()==b'1 2 3\n' for name in required)
    assert not (out/'evil.txt').exists()


def test_prelock_checks_cached_bank_csv_archive_identity(tmp_path):
    from zipfile import ZipFile
    from io import BytesIO
    from graph_ml.cp4.contracts import _verify_cached_bank_csv_matches_official_archive
    outer=tmp_path/'bank_marketing.zip'; raw=tmp_path/'bank-additional-full.csv'
    actual=b'"age";"job";"y"\n20;admin;no\n'
    nested=BytesIO()
    with ZipFile(nested,'w') as z:
        z.writestr('bank-additional/bank-additional-full.csv',actual)
    with ZipFile(outer,'w') as z:
        z.writestr('bank-additional.zip',nested.getvalue())
    raw.write_bytes(actual)
    _verify_cached_bank_csv_matches_official_archive(outer,raw)
    raw.write_bytes(actual+b'31;admin;yes\n')
    with pytest.raises(ValueError,match='differs'):
        _verify_cached_bank_csv_matches_official_archive(outer,raw)
