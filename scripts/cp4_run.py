"""CP4: deterministic preflight, explicitly authorized one-time sealed evaluation."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import platform,sys,subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from graph_ml.cp1.common import sha256_file
from graph_ml.cp4.contracts import ensure_lock,atomic_json


def main():
    p=argparse.ArgumentParser(description='Project 13 final evidence gate')
    p.add_argument('--final-test',action='store_true',help='Explicitly unlock sealed labels ONCE after preflight')
    p.add_argument('--render',action='store_true',help='Recreate HTML/text evidence from existing CP4 JSON; never reads test labels')
    p.add_argument('--skip-tests',action='store_true',help='Only when full pytest was run separately; not the default')
    args=p.parse_args()
    report=ROOT/'outputs'/'cp4'/'cp4_final_test_report.json'
    if args.render:
        from graph_ml.cp4.render import render_evidence
        if not report.exists():raise SystemExit('No CP4 final report exists yet')
        render_evidence(ROOT)
        print('Evidence HTML/text regenerated from existing JSON (no test access).')
        return
    if not args.skip_tests:
        print('CP4 step 1: full offline tests...',flush=True)
        subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,check=True)
        print('CP4 step 2: package requirements compatibility...',flush=True)
        subprocess.run([sys.executable,'-m','pip','check'],cwd=ROOT,check=True)
    print('CP4 step 3: verify lineage, source hashes and validation-selected checkpoints; no test labels...',flush=True)
    lock,lockhash=ensure_lock(ROOT)
    print('PRE-FINAL LOCK VERIFIED: '+str(ROOT/'outputs'/'cp4'/'cp4_selection_lock.json'))
    print('Lock SHA256: '+lockhash)
    if not args.final_test:
        print('PREFLIGHT PASSED. Test labels are still sealed.')
        print('Only after reviewing the frozen selection, run:')
        print(r'& .\.venv\Scripts\python.exe scripts\cp4_run.py --final-test')
        return
    if report.exists():
        raise SystemExit('Final report ALREADY EXISTS. Refusing to retest sealed data. Use --render instead.')
    print('CP4 step 4: EXPLICIT FINAL-TEST UNLOCK, one-time predeclared model inventory...',flush=True)
    import json
    from graph_ml.cp4.heldout import evaluate_bank,evaluate_har,evaluate_cora
    from graph_ml.cp4.render import render_evidence
    cp1=json.loads((ROOT/'outputs'/'cp1'/'cp1_validation_report.json').read_text(encoding='utf-8'))
    cp2=json.loads((ROOT/'outputs'/'cp2'/'cp2_validation_report.json').read_text(encoding='utf-8'))
    cp3=json.loads((ROOT/'outputs'/'cp3'/'cp3_validation_report.json').read_text(encoding='utf-8'))
    results={}
    for label,runner,args_ in (('bank',evaluate_bank,(ROOT,lock,cp1,cp2)),
                               ('har',evaluate_har,(ROOT,lock,cp1,cp2)),
                               ('cora',evaluate_cora,(ROOT,lock,cp1,cp3))):
        print('Final one-time test evaluation: '+label,flush=True)
        rows=runner(*args_)
        for name,v in rows.items():
            results[name]=v
            print(f'  {name}: test Macro-F1={v["metrics_test"]["f1_macro"]:.4f}; prediction={v["test_predict_seconds"]:.3f}s',flush=True)
    if set(results)!=set(lock['final_test_model_inventory']):
        raise ValueError('Incomplete final predeclared model inventory; refusing final artifact')
    payload={'project':'13_deep_learning_graph_ml','checkpoint':'CP4',
             'status':'final_test_evaluated_pending_audit_screenshots_publication',
             'timestamp_utc':datetime.now(timezone.utc).isoformat(),
             'platform':platform.platform(),'python':sys.version.split()[0],
             'selection_lock_sha256':lockhash,
             'source_report_sha256':{'cp1':lock['cp1_report_sha256'],'cp2':lock['cp2_report_sha256'],'cp3':lock['cp3_report_sha256']},
             'test_use':'One-time frozen checkpoint and baseline comparison; no test-based tuning, refitting or model selection.',
             'limitations':['Single seed. No inferential CI for benchmark differences.',
                            'Measured latency on one Windows CPU, not a controlled cross-machine benchmark.',
                            'HAR engineered vs raw representations; same subjects/windows, different inputs.',
                            'Cora public transductive graph: test-node features/edges are visible; test labels were withheld until final scoring.',
                            'Bank model threshold 0.5 is not optimized on held-out test; do not present accuracy without recall.'],
             'results':results}
    atomic_json(report,payload)
    print('FINAL TEST REPORT CREATED (not CP4 completion): '+str(report),flush=True)
    render_evidence(ROOT)
    print('Review JSON, HTML, tests and security; then provide screenshots for Word handoff BEFORE GitHub publication.',flush=True)

if __name__=='__main__':
    main()
