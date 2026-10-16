# Project 13 — CP2 single-command user handoff

**Prerequisites:** CP1 completed on user's Windows PC; `.venv` with the same working torch/sklearn/PyG versions; downloaded and extracted Bank and HAR training data in ignored `data/raw/`; `outputs/cp1/cp1_validation_report.json` must exist. The CP2 runner refuses missing or inconsistent CP1 evidence.

## Install/update files

1. Back up the current `13_deep_learning_graph_ml` folder, **not** by uploading `.env` or `.venv` to chat.
2. Extract this ZIP into parent `C:\Users\Jonny Fernandez\Documents\github files`, replacing the same-named Project 13 source/documentation files.
3. **Keep** `.venv`, `.env`, all `data/raw/` downloads, and `outputs/cp1/cp1_validation_report.json`. The ZIP contains none of those local contents, except `.gitkeep` placeholders.
4. Do **not** install any new package: CP2 uses PyTorch, NumPy, SciPy (installed through sklearn), pandas, sklearn, pytest already present in CP1 environment. No paid APIs or cloud services.

## Run from the project root (PowerShell)

```powershell
& .\.venv\Scripts\python.exe scripts\cp2_run.py
```

The command runs all offline automated regression tests first. If those pass, trains Bank MLP, HAR LSTM, HAR Transformer for up to **6 epochs each**, with early stopping, using **CPU** by default and an appropriate `DataLoader` with zero Windows workers. It records epoch histories, best validation checkpoints, per-class metrics, actual training/inference times, parameter counts, and original CP1 baseline comparisons. No official test labels/features are read. The code never calls an external AI API or downloads anything during CP2.

To rerun after explicitly recording the first output, pass `--skip-tests` (optional). To experiment with a justified larger budget, pass `--epochs 10`; this will **overwrite** selected model checkpoint/report files so retain earlier evidence if comparisons matter. A targeted model-only run (e.g., `--models har_lstm`) marks the report **partial**, not complete, and overwrites the consolidated report.

## Send back for review

Upload `outputs/cp2/cp2_validation_report.json` and send the final PowerShell output. The report is generated only from actual local training. Do not mark CP2 passed or claim wins until its report has been reviewed. Avoid staging/generated model `.pt` weights or private `.env` in Git. Do not run the final sealed test set yet.
