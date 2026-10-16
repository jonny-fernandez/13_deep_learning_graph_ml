"""Single entry point for CP3, with offline suite first, and no final test scoring."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description='CP3 Cora GCN/GraphSAGE; validation-only')
    p.add_argument('--epochs', type=int, default=45)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--mlflow', action='store_true', help='Optional: log to local MLflow if preinstalled')
    p.add_argument('--skip-tests', action='store_true', help='For repeat run after separately passing pytest only')
    a = p.parse_args()
    if not a.skip_tests:
        print('CP3 step 1: full offline regression suite...', flush=True)
        subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, check=True)
    print('CP3 step 2: sealed Cora graph experiments and validation-only evidence...', flush=True)
    from graph_ml.cp3.runner import run_all
    run_all(ROOT, epochs=a.epochs, patience=a.patience, device=a.device, mlflow_enabled=a.mlflow)


if __name__ == '__main__':
    main()
