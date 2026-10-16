"""One-command CP2: offline tests, three validation-only neural models."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description='CP2 MLP + LSTM + Transformer; NO held-out test scoring')
    parser.add_argument('--epochs', type=int, default=6, help='Maximum training epochs per model (default 6)')
    parser.add_argument('--batch-size', type=int, default=256, help='Training/inference batch size (default 256)')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu', help='Use cpu by default')
    parser.add_argument('--models', nargs='+', choices=('bank_mlp', 'har_lstm', 'har_transformer'),
                        default=('bank_mlp', 'har_lstm', 'har_transformer'))
    parser.add_argument('--skip-tests', action='store_true', help='Only for rerunning after separate successful pytest')
    args = parser.parse_args()
    if not args.skip_tests:
        print('CP2 step 1: full offline pytest regression suite...', flush=True)
        subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, check=True)
    print('CP2 step 2: train selected neural models (CPU by default; no network)...', flush=True)
    from graph_ml.cp2.runner import run_all
    run_all(ROOT, epochs=args.epochs, batch_size=args.batch_size,
            device=args.device, models=tuple(args.models))


if __name__ == '__main__':
    main()
