"""Presentation from recorded measurements only, without reading or re-scoring test data."""
from __future__ import annotations
import html
import json
from pathlib import Path
from graph_ml.cp4.contracts import MODELS


def render_evidence(root: Path) -> tuple[Path, Path]:
    output=root/'outputs'/'cp4'
    payload=json.loads((output/'cp4_final_test_report.json').read_text(encoding='utf-8'))
    if payload.get('checkpoint')!='CP4' or set(payload.get('results',{}))!=set(MODELS):
        raise ValueError('Refuse to render incomplete CP4 evidence')
    from graph_ml.cp1.common import sha256_file
    lock_file=output/'cp4_selection_lock.json'
    if not lock_file.is_file() or sha256_file(lock_file)!=payload.get('selection_lock_sha256'):
        raise ValueError('Selection lock changed; refuse to present final scores')
    for stage in ('cp1','cp2','cp3'):
        path=root/'outputs'/stage/(stage+'_validation_report.json')
        if not path.is_file() or sha256_file(path)!=payload.get('source_report_sha256',{}).get(stage):
            raise ValueError(f'{stage} original report changed; refuse to present mismatched results')
    cp1=json.loads((root/'outputs'/'cp1'/'cp1_validation_report.json').read_text(encoding='utf-8'))
    cp2=json.loads((root/'outputs'/'cp2'/'cp2_validation_report.json').read_text(encoding='utf-8'))
    cp3=json.loads((root/'outputs'/'cp3'/'cp3_validation_report.json').read_text(encoding='utf-8'))
    labels={'bank_logistic':'Logistic regression','bank_mlp':'MLP',
            'har_logistic':'Logistic regression','har_lstm':'LSTM','har_transformer':'Transformer',
            'cora_logistic':'Graph-free logistic regression','cora_gcn':'GCN (selected)',
            'cora_graphsage':'GraphSAGE (secondary)'}
    validation={
        'bank_logistic':cp1['tasks']['bank']['metrics_validation'],
        'bank_mlp':cp2['results']['bank_mlp']['metrics_validation'],
        'har_logistic':cp1['tasks']['har']['metrics_validation'],
        'har_lstm':cp2['results']['har_lstm']['metrics_validation'],
        'har_transformer':cp2['results']['har_transformer']['metrics_validation'],
        'cora_logistic':cp1['tasks']['cora']['metrics_validation'],
        'cora_gcn':cp3['results']['gcn_h64_d05']['metrics_validation'],
        'cora_graphsage':cp3['results']['graphsage_h64_d05']['metrics_validation']}
    sections=[]
    doc=['# Project 13 — Frozen Evaluation Evidence','',
         'Generated solely from saved CP1–CP4 reports; no additional training or test access.',
         'CP4 technical evaluation complete; screenshot handoff and GitHub publication are NOT complete.','']
    for task,primary in (('bank','pr_auc_average_precision'),('har','f1_macro'),('cora','f1_macro')):
        headings={'bank':'Bank Marketing (PR-AUC)','har':'HAR (Macro-F1)','cora':'Cora (Macro-F1)'}
        doc.extend(['## '+headings[task],'','| Model | Validation | Final test | Test Macro-F1 | Test predict seconds |',
                    '|---|---:|---:|---:|---:|'])
        rows=[]
        for name in MODELS:
            r=payload['results'][name]
            if r['task']!=task:continue
            validation_score=validation[name][primary]
            final_score=r['metrics_test'][primary]
            doc.append(f'| {labels[name]} | {validation_score:.4f} | {final_score:.4f} | {r["metrics_test"]["f1_macro"]:.4f} | {r["test_predict_seconds"]:.4f} |')
            rows.append(f'<tr><td>{html.escape(labels[name])}</td><td>{validation_score:.4f}</td><td>{final_score:.4f}</td><td>{r["metrics_test"]["f1_macro"]:.4f}</td><td>{r["test_predict_seconds"]:.4f}</td></tr>')
        sections.append(f'<section><h2>{headings[task]}</h2><p>Validation and final test values on the same task and frozen split.</p><table><thead><tr><th>Model</th><th>Validation</th><th>Final test</th><th>Test Macro-F1</th><th>Predict sec</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section>')
        doc.append('')
    doc.extend(['## Evaluation conditions','',
                '- Final test used once after locking selected model files and report SHA256 hashes.',
                '- Metrics are direct recorded outputs. No test-tuned model selection.',
                '- CP4 remains open until security audit, real screenshot Word handoff, GitHub checks and user confirmation.',
                '', '## Limitations',''])
    for v in payload['limitations']:doc.append('- '+v)
    md=output/'cp4_evidence_summary.md'
    md.write_text('\n'.join(doc)+'\n',encoding='utf-8')
    limits=''.join('<li>'+html.escape(x)+'</li>' for x in payload['limitations'])
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P13 — Verified Evaluation</title><style>
    :root{font-family:system-ui,Segoe UI,Arial,sans-serif;color:#19283b;background:#f0f4f8}main{max-width:1060px;margin:2rem auto;background:white;padding:2rem 3rem;border-radius:18px;box-shadow:0 5px 25px #d6dce5}header{border-bottom:3px solid #32496a;padding-bottom:1rem}h1{font-size:2rem}h2{margin-top:2rem;color:#244260}p,li{line-height:1.5}table{width:100%;border-collapse:collapse;margin-top:1rem}th{text-align:left;background:#263f61;color:white}th,td{padding:.85rem;border-bottom:1px solid #d6dce5}tr:nth-child(even){background:#f7fafd}.note{background:#fff6e3;padding:1rem;border-left:4px solid #e4b755}.minor{color:#62748a;font-size:.9rem}footer{margin-top:2rem;padding-top:1rem;border-top:1px solid #ccc}</style></head><body><main><header><p class="minor">AI Portfolio · Project 13 · CP4</p><h1>Deep Learning, Sequence & Graph ML</h1><p>Results copied directly from the validated CP1–CP4 JSON reports.</p></header><p class="note"><strong>Final test evidence, not publication approval.</strong> One-time locked test comparison. No test-based hyperparameter tuning. All displayed results are from real outputs, not generated placeholders.</p>'''
    page+=''.join(sections)
    page+='<h2>Limitations</h2><ul>'+limits+'</ul><footer><p class="minor">CP4 remains open until security audit, actual screenshot Word handoff, explicit screenshot confirmation, and verified GitHub CI. Source: local JSON evidence files.</p></footer></main></body></html>'
    h=output/'cp4_evidence_dashboard.html'
    h.write_text(page,encoding='utf-8')
    return md,h
