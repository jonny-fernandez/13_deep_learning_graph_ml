"""Read-only CP1 inventory for Windows PowerShell; no dependency install or training."""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "torch": "torch",
    "torch-geometric": "torch_geometric",
    "torchmetrics": "torchmetrics",
    "scikit-learn": "sklearn",
    "pandas": "pandas",
    "numpy": "numpy",
    "optuna": "optuna",
    "mlflow": "mlflow",
    "captum": "captum",
    "shap": "shap",
    "PyYAML": "yaml",
    "pytest": "pytest",
    "pytest-cov": "pytest_cov",
}


def safe_package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"
    except Exception as exc:
        return f"VERSION CHECK FAILED: {type(exc).__name__}"


def main() -> int:
    print("PROJECT 13 — CP1 READ-ONLY LOCAL INVENTORY")
    print("No files, configurations, dependency packages or credentials are modified.")
    print(f"Project root: {ROOT}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"VIRTUAL_ENV active: {bool(os.environ.get('VIRTUAL_ENV'))}")
    print(f"Python >= 3.11: {sys.version_info >= (3, 11)}")

    print("\n[Project files]")
    for name in (
        "CHECKPOINTS.md", "PROJECT_PLAN.md", "PROJECT_RULES.md", "README.md",
        "requirements.txt", "pyproject.toml", "configs/base.yaml",
        "docs/EVALUATION_PLAN.md", "tests/test_models.py", "tests/test_sequence_models.py",
        "src/graph_ml/data.py", "src/graph_ml/train.py", "src/graph_ml/evaluate.py",
        "src/graph_ml/explain.py",
    ):
        print(f"{'FOUND  ' if (ROOT / name).is_file() else 'MISSING'} {name}")
    model_dir = ROOT / "src/graph_ml/models"
    print("Model source files:", ", ".join(p.name for p in sorted(model_dir.glob("*.py"))) if model_dir.exists() else "MISSING")
    test_dir = ROOT / "tests"
    print("Test files:", ", ".join(p.name for p in sorted(test_dir.glob("test_*.py"))) if test_dir.exists() else "MISSING")
    for folder in ("data/raw", "data/processed"):
        base = ROOT / folder
        count = sum(p.is_file() and p.name != ".gitkeep" for p in base.rglob("*")) if base.is_dir() else 0
        print(f"Data inventory {folder}: {count} file(s)")

    print("\n[Local security — content never printed]")
    env_exists = (ROOT / ".env").exists()
    print(f".env found locally: {env_exists}")
    gitignore = ROOT / ".gitignore"
    ignored = gitignore.is_file() and ".env" in {x.strip() for x in gitignore.read_text(encoding="utf-8", errors="replace").splitlines()}
    print(f".gitignore has exact '.env' rule: {ignored}")
    if env_exists:
        print("Do not share .env. Rotate any key included in a previously uploaded ZIP.")

    print("\n[Installed package distributions — no install]")
    for distribution, module in PACKAGES.items():
        print(f"{distribution:<19} {safe_package_version(distribution)}")

    print("\n[GPU/CUDA: best effort, read-only]")
    try:
        if importlib.util.find_spec("torch") is None:
            print("Torch import: unavailable")
        else:
            import torch
            print(f"torch.__version__: {torch.__version__}")
            print(f"CUDA build: {torch.version.cuda or 'CPU-only / none'}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    except Exception as exc:
        print(f"Torch/CUDA inspection failed (not fatal): {type(exc).__name__}: {exc}")
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            res = subprocess.run(
                [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=12, check=False,
            )
            print("nvidia-smi:", (res.stdout or res.stderr).strip() or f"returncode {res.returncode}")
        except Exception as exc:
            print(f"nvidia-smi could not be read: {type(exc).__name__}")
    else:
        print("nvidia-smi: not found (CPU execution is valid)")

    print("\nCP1 Gate A status: INSPECTION OUTPUT ONLY — review this with the project owner.")
    print("No dataset, traditional baseline, model results or checkpoint pass is implied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
