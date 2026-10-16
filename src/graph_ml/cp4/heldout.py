"""Final test-once comparisons. Imported only after verified selection lock and explicit user flag."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.pipeline import make_pipeline
from graph_ml.cp1.bank import bank_estimator, FORBIDDEN_COLUMNS
from graph_ml.cp1.har import load_har, har_estimator, har_sequence_train_data
from graph_ml.cp1.cora import cora_estimator
from graph_ml.cp1.common import validation_metrics, sha256_file
from graph_ml.cp2.data import subject_partition
from graph_ml.cp2.training import probability_matrix, set_seed
from graph_ml.cp4.contracts import CKPTS, required_checkpoint_path
from graph_ml.models.mlp import MLP
from graph_ml.models.lstm import LSTMClassifier
from graph_ml.models.transformer import TransformerClassifier


def _score(y, p, classes):
    res = validation_metrics(y, p, labels=list(range(classes)))
    res['n_test'] = res.pop('n_val')
    if classes==2:
        from sklearn.metrics import brier_score_loss
        res['brier_score_positive'] = float(brier_score_loss(y, np.asarray(p)[:,1]))
    return res

def _timed_predict(predict, y, classes, model, kind, task, parameter_count=None, train_time=None, **extra):
    start = time.perf_counter()
    prob = predict()
    elapsed = time.perf_counter()-start
    assert len(prob) == len(y)
    return {'task':task, 'model':model, 'family':kind,
            'metrics_test':_score(y, prob, classes),
            'test_predict_seconds':round(elapsed,6),
            'test_examples_per_second_observed':round(len(y)/max(elapsed,1e-9),2),
            'parameter_count':parameter_count,
            'train_duration_seconds':train_time,
            'measurement_note':'One CPU prediction pass including Python and relevant model overhead; machine-dependent, not statistical latency CI.',
            **extra}

def _checkpoint(root, lock, alias, model):
    path = required_checkpoint_path(root, alias)
    if sha256_file(path) != lock['checkpoint_inventory'][alias]['checkpoint_sha256']:
        raise ValueError(f'Checkpoint changed since CP4 lock: {alias}')
    state = torch.load(path, map_location='cpu', weights_only=True)
    item = lock['checkpoint_inventory'][alias]
    if (state.get('selected_epoch') != item['selected_epoch'] or state.get('seed') != 42):
        raise ValueError(f'Checkpoint selection metadata changed: {alias}')
    if sum(p.numel() for p in model.parameters()) != item['parameter_count']:
        raise ValueError(f'Checkpoint model size differs: {alias}')
    model.load_state_dict(state['state_dict'], strict=True)
    return model.eval().cpu()

def evaluate_bank(root, lock, cp1, cp2):
    csv = root/'data'/'raw'/'bank_marketing'/'bank-additional-full.csv'
    df = pd.read_csv(csv,sep=';')
    row = cp1['tasks']['bank']
    nt,nv,ns = (row[k] for k in ('n_train','n_validation','n_test_sealed'))
    if len(df)!=nt+nv+ns or not {'y','duration','job','month','age'}.issubset(df):
        raise ValueError('Bank final split/schema differs from locked report')
    tr,te = df.iloc[:nt], df.iloc[nt+nv:]
    if not tr['y'].isin(['yes','no']).all() or not te['y'].isin(['yes','no']).all():
        raise ValueError('Invalid Bank labels')
    xtr = tr.drop(columns=list(FORBIDDEN_COLUMNS))
    xte = te.drop(columns=list(FORBIDDEN_COLUMNS))
    if list(xtr.columns)!=list(xte.columns):
        raise ValueError('Bank test features differ from training columns')
    ytr = (tr.y=='yes').to_numpy(dtype=int)
    yte = (te.y=='yes').to_numpy(dtype=int)
    set_seed(42)
    lr = bank_estimator(xtr)
    tick=time.perf_counter(); lr.fit(xtr,ytr); train_s=time.perf_counter()-tick
    out = {'bank_logistic':_timed_predict(lambda:lr.predict_proba(xte),yte,2,'LogisticRegression','traditional_ml','bank',
                                          train_time=round(train_s,6), training_policy='TRAIN only, fixed CP1 baseline settings')}
    transformer = bank_estimator(xtr).named_steps['preprocess']
    transformer.fit(xtr,ytr)
    x = transformer.transform(xte)
    if sparse.issparse(x): x=x.toarray()
    x=np.asarray(x,dtype=np.float32)
    if x.shape != (ns,cp2['results']['bank_mlp']['data_contract']['input_dim']) or not np.isfinite(x).all():
        raise ValueError('Bank MLP test transformation mismatch')
    model=_checkpoint(root,lock,'bank_mlp',MLP(input_dim=x.shape[1],hidden_dim=64))
    probs_for_threshold=[]
    def mlp_predictions():
        matrix=probability_matrix(model,x,2,256,torch.device('cpu'))
        probs_for_threshold.append(matrix)
        return matrix
    out['bank_mlp']=_timed_predict(mlp_predictions,yte,2,
                                   'MLP','neural','bank',parameter_count=lock['checkpoint_inventory']['bank_mlp']['parameter_count'],
                                   train_time=cp2['results']['bank_mlp']['training_duration_seconds'],
                                   training_policy='CP2 validation-selected checkpoint; no final-test refitting')
    from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
    threshold=lock['bank_mlp_operating_threshold_selected_on_validation']
    decision=(probs_for_threshold[0][:,1]>=threshold).astype(int)
    out['bank_mlp']['validation_locked_operating_point_test']={
        'threshold':threshold,'test_precision_positive':float(precision_score(yte,decision,zero_division=0)),
        'test_recall_positive':float(recall_score(yte,decision,zero_division=0)),
        'test_f1_positive':float(f1_score(yte,decision,zero_division=0)),
        'test_confusion_matrix':confusion_matrix(yte,decision,labels=[0,1]).tolist(),
        'warning':'Threshold selected on CP2 validation only, then applied unchanged to final test.'}
    return out

def _materialize_official_har_test_files(root: Path):
    """Extract only known HAR final-test members, AFTER the explicit final-test unlock.

    CP1 deliberately extracts train and test-subject IDs only. CP4 must open the
    cached authenticated-source ZIP after final-test approval; it never fetches a
    new dataset or trusts an arbitrary directory/filename from a ZIP.
    """
    from io import BytesIO
    from zipfile import ZipFile
    from graph_ml.cp1.downloads import _extract_allowlist
    archive=root/'data'/'raw'/'har'/'har_uci.zip'
    output=root/'data'/'raw'/'har'/'UCI HAR Dataset'
    required=['test/X_test.txt','test/y_test.txt']
    required += [f'test/Inertial Signals/{kind}_{axis}_test.txt'
                 for kind in ('total_acc','body_acc','body_gyro') for axis in 'xyz']
    with ZipFile(archive) as outer:
        nested=[n for n in outer.namelist() if n.endswith('UCI HAR Dataset.zip')]
        if len(nested)>1:
            raise ValueError('Ambiguous nested HAR archive; abort sealed-test extraction')
        if nested:
            info=outer.getinfo(nested[0])
            if info.file_size>150*1024*1024:
                raise ValueError('Nested HAR archive is unexpectedly large')
            with ZipFile(BytesIO(outer.read(nested[0]))) as inner:
                _extract_allowlist(inner,required,output)
        else:
            _extract_allowlist(outer,required,output)
    return output


def _read_har_sequence_test(directory:Path):
    signals=[]
    for kind in ('total_acc','body_acc','body_gyro'):
        for axis in 'xyz':
            signal=np.atleast_2d(np.loadtxt(directory/'test'/'Inertial Signals'/f'{kind}_{axis}_test.txt', dtype=np.float32))
            signals.append(signal)
    x=np.stack(signals,axis=-1)
    if x.ndim!=3 or x.shape[1:]!=(128,9):
        raise ValueError('HAR final sequence shape mismatch')
    return x

def evaluate_har(root,lock,cp1,cp2):
    directory=_materialize_official_har_test_files(root)
    meta=cp1['tasks']['har']['data_contract']
    baseline_splits=load_har(directory)
    if (len(baseline_splits.y_train),len(baseline_splits.y_val))!=(cp1['tasks']['har']['n_train'],cp1['tasks']['har']['n_validation']):
        raise ValueError('HAR baseline partitions changed')
    ids=np.atleast_1d(np.loadtxt(directory/'test'/'subject_test.txt',dtype=int))
    y=np.atleast_1d(np.loadtxt(directory/'test'/'y_test.txt',dtype=int))-1
    x=np.atleast_2d(np.loadtxt(directory/'test'/'X_test.txt',dtype=np.float32))
    if (len(y)!=len(x) or len(ids)!=len(y) or len(y)!=cp1['tasks']['har']['n_test_sealed']
            or x.shape[1]!=561 or not np.isin(y,np.arange(6)).all()):
        raise ValueError('HAR final test label/features invalid')
    if set(ids)&(set(meta['train_group_ids'])|set(meta['val_group_ids'])):
        raise ValueError('HAR test subject contamination')
    set_seed(42)
    baseline=har_estimator()
    tick=time.perf_counter();baseline.fit(baseline_splits.x_train,baseline_splits.y_train);sec=time.perf_counter()-tick
    result={'har_logistic':_timed_predict(lambda:baseline.predict_proba(x),y,6,'LogisticRegression','traditional_ml','har',
                                         train_time=round(sec,6), representation='561 engineered features')}
    raw=har_sequence_train_data(directory)
    train_subjects=np.atleast_1d(np.loadtxt(directory/'train'/'subject_train.txt',dtype=int))
    tr,_=subject_partition(train_subjects,meta['train_group_ids'],meta['val_group_ids'])
    means=raw[tr].mean(axis=(0,1),dtype=np.float64).astype(np.float32)
    std=np.maximum(raw[tr].std(axis=(0,1),dtype=np.float64).astype(np.float32),1e-6)
    test_sequences=_read_har_sequence_test(directory)
    if len(test_sequences)!=len(y):raise ValueError('HAR raw test row count mismatch')
    prepared=((test_sequences-means)/std).astype(np.float32)
    if not np.isfinite(prepared).all():raise ValueError('HAR final sequences non-finite')
    config={'har_lstm':LSTMClassifier(input_dim=9,hidden_dim=32,num_layers=1,num_classes=6,dropout=.1),
            'har_transformer':TransformerClassifier(input_dim=9,model_dim=32,num_heads=4,num_layers=1,num_classes=6,max_length=128,dropout=.1)}
    for alias,m in config.items():
        model=_checkpoint(root,lock,alias,m)
        result[alias]=_timed_predict(lambda model=model:probability_matrix(model,prepared,6,256,torch.device('cpu')),
                                     y,6,alias,'neural','har',parameter_count=lock['checkpoint_inventory'][alias]['parameter_count'],
                                     train_time=cp2['results'][alias]['training_duration_seconds'],
                                     representation='nine raw sensor channels over completed 128-step windows')
    return result

def evaluate_cora(root,lock,cp1,cp3):
    from torch_geometric.datasets import Planetoid
    from graph_ml.cp3.runner import make_model
    from graph_ml.cp3.data import construct_graph_task
    folder=root/'data'/'raw'/'cora'
    graph=Planetoid(root=str(folder),name='Cora',split='public')[0]
    raw=folder/'Cora'/'raw'
    source={p.name:sha256_file(p) for p in sorted(raw.glob('ind.cora.*'))}
    task=construct_graph_task(graph,cp1['tasks']['cora']['data_contract'],source)
    ix=graph.test_mask.nonzero(as_tuple=False).flatten()
    y=graph.y[ix].detach().cpu().numpy()
    if len(y)!=1000 or not np.isin(y,np.arange(7)).all():raise ValueError('Invalid Cora test mask/classes')
    train_x=task.x[task.train_idx].numpy()
    test_x=task.x[ix].numpy()
    baseline=cora_estimator()
    set_seed(42)
    tick=time.perf_counter();baseline.fit(train_x,task.y_train.numpy());train_s=time.perf_counter()-tick
    result={'cora_logistic':_timed_predict(lambda:baseline.predict_proba(test_x),y,7,'LogisticRegression','traditional_ml','cora',
                                           train_time=round(train_s,6),representation='node features, no graph edges')}
    for alias in ('cora_gcn','cora_graphsage'):
        item=lock['checkpoint_inventory'][alias]
        kind='gcn' if alias=='cora_gcn' else 'graphsage'
        row=cp3['results'][item['name']]
        model=make_model(kind,row['config']['hidden_dim'],row['config']['dropout'])
        model=_checkpoint(root,lock,alias,model)
        def predict_graph(model=model):
            with torch.inference_mode():
                return torch.softmax(model(task.x,task.edge_index)[ix],dim=1).numpy()
        result[alias]=_timed_predict(predict_graph,y,7,item['name'],'gnn','cora',
                                     parameter_count=item['parameter_count'],train_time=row['training_duration_seconds'],
                                     graph_type='transductive: all features/topology visible, train labels only during fit')
    return result
