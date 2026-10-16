# CP3 on Windows — consolidated commands

Prerequisites: The same Python 3.12 `.venv`, public Cora dataset already cached at `data/raw/cora`, and existing CP1/CP2 JSON reports in `outputs/cp1` and `outputs/cp2`.

From the Project 13 folder in PowerShell:

```powershell
& .\.venv\Scripts\python.exe scripts\cp3_run.py
```

The single command first runs the **entire offline pytest suite**, then trains two GCN configurations and two GraphSAGE configurations on Cora's 140 labeled train-mask nodes, scoring only the 500 validation-mask labels. It chooses one configuration by validation macro-F1 (with parameter-count tie break) and retrains a **no-edge ablation** with the same architecture. It saves model checkpoints and hashes, metrics, training curves and a predicted-class node gradient explanation. Cora's 1,000 test labels stay sealed.

Output to upload after execution:

`outputs\cp3\cp3_validation_report.json`

Default bounded CPU experiments have at most 45 epochs each, early stopping patience 10. This is a five-fit graph run; it may take several minutes on CPU. No new dependency is required for the default run; PyTorch and PyTorch Geometric were already installed and verified during CP1.

Optional MLflow tracking: if already installed inside this `.venv`, you may add `--mlflow` to the command. The default run records JSON/hash lineage without requiring MLflow. Do not install MLflow merely to run CP3. Never commit `.env`, local datasets, artifacts, model weights, MLflow stores or JSON outputs.

**Do not run final sealed test evaluations**, upload checkpoints, or publish to GitHub yet. If the command fails, send the full traceback for a grouped correction.
