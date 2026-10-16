"""CP1 consolidated validation entry point. Usage: python scripts/cp1_run.py"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description='Run CP1 offline unit tests then acquire public data and fit validation-only baselines')
    p.add_argument('--tasks', nargs='+', choices=['bank', 'har', 'cora'], default=['bank', 'har', 'cora'])
    p.add_argument('--skip-tests', action='store_true', help='Skip pytest for rerun after a separate successful pytest')
    args = p.parse_args()
    if not args.skip_tests:
        print('CP1 step 1: run all automated regression tests (no network)', flush=True)
        subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, check=True)
    print('CP1 step 2: download selected public datasets if absent, fit three baselines', flush=True)
    from graph_ml.cp1.runner import run_all
    run_all(ROOT, args.tasks)


if __name__ == '__main__':
    main()
