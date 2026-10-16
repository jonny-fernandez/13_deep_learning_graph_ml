# Deep Learning, Sequence & Graph ML Evaluation Platform

**Project 13 of the AI / Data Science Portfolio** · Python 3.12 · PyTorch · PyTorch Geometric · scikit-learn

**Status (October 16, 2026):** CP1–CP3 completed and independently reviewed; CP4 evaluation and screenshot handoff completed. **GitHub publishing and live GitHub Actions verification are pending.** The one-time held-out test evaluation is finished and must not be rerun to adjust models.

## Question

When do deep neural networks and graph neural networks improve on traditional machine-learning baselines enough to justify their added training, inference, and model complexity? This project compares each neural family to an appropriate simpler baseline **within the same task**. It retains negative as well as positive findings; it does not treat metrics across unrelated datasets as directly comparable.

## Datasets and protocols

| Dataset / task | Classical baseline | Neural/graph models | Split and leakage controls |
|---|---|---|---|
| UCI Bank Marketing — pre-call subscription classification | Logistic regression | Multilayer perceptron (MLP) | 70/15/15 source-ordered split; `duration` removed because it is post-call information; transformations fit on training only |
| UCI Human Activity Recognition (HAR) — six-class window classification | Logistic regression on 561 published engineered features | LSTM and Transformer on raw 9-channel, 128-step windows | Official subject-disjoint final test; separate held-out subjects for validation; train-only standardization |
| Planetoid Cora — seven-class paper-node classification | Graph-free logistic regression | GCN and GraphSAGE | Public transductive split: 140 train labels / 500 validation / 1,000 test labels; node features and citation topology visible throughout, but test labels withheld until final reporting |

The HAR comparison uses **different feature representations** (engineered vs. raw), even though subjects, windows and labels are matched. Cora is **transductive**, not an unseen-graph inductive benchmark. See [data sources, license notes and citations](docs/DATA_SOURCES_AND_LICENSES.md).

## Verified one-time final test results

Model configurations and a threshold for the Bank MLP were frozen on validation before final testing. Eight predeclared models were evaluated once, with no test-based model selection or refit.

### Bank Marketing — PR-AUC (Average Precision)

| Model | Validation AP | Final test AP | Test Macro-F1 | CPU test prediction (s) |
|---|---:|---:|---:|---:|
| Logistic regression | 0.2094 | 0.4466 | 0.4513 | 0.0470 |
| MLP | 0.2166 | **0.4756** | 0.3810 | 0.1010 |

**Interpretation:** The MLP slightly improved final-test average precision (+0.0291), but at the default 0.5 threshold it predicted **no positive cases**. The preselected 0.25 operating threshold recovered **20.71% subscriber recall**, at **58.36% precision**. The test positive prevalence (38.45%) is much higher than validation (10.65%), so AP values across these partitions cannot be interpreted as a clean improvement in model quality. Report both ranking and classification metrics.

### HAR — Macro-F1

| Model | Validation | Final test | CPU test prediction (s) |
|---|---:|---:|---:|
| **Logistic regression (engineered features)** | 0.9538 | **0.9511** | 0.0178 |
| LSTM (raw sequences) | 0.6976 | 0.6723 | 0.3292 |
| Transformer (raw sequences) | 0.9508 | 0.8610 | 2.0486 |

The simple engineered-feature baseline outperformed both raw-sequence neural models. The Transformer also declined from validation to final test. The observations are from the locked architectures and six-epoch CPU training budget, not a general claim that Transformers or LSTMs cannot perform better.

### Cora — Macro-F1

| Model | Validation | Final test | CPU test prediction (s) |
|---|---:|---:|---:|
| Graph-free logistic regression | 0.5048 | 0.5070 | 0.0097 |
| **GCN — selected by validation** | **0.7881** | **0.7984** | 0.0249 |
| GraphSAGE — prespecified comparator | 0.7748 | 0.7918 | 0.1139 |

The selected GCN exceeded the graph-free baseline by **+0.2914 absolute Macro-F1** on the final test (29.14 percentage points). In a validation-only ablation, retraining the selected GCN **without graph edges** lowered Macro-F1 from **0.7881 to 0.5910**. This is evidence of useful connectivity in this setup, not a causal estimate.

See [the complete results interpretation](docs/FINAL_EVALUATION_RESULTS.md) for per-task caveats and runtime context. All metric values are taken from actual local JSON reports, not synthetic illustrations.

## Reproduce and inspect

These examples assume PowerShell and an existing project-local Python 3.12 environment. On a fresh machine, create a virtual environment and install a **compatible CPU version of PyTorch**, then the dependencies listed in `requirements-cp1.txt`. See [Windows setup instructions](docs/CP1_WINDOWS_COMMANDS.md). Downloads require internet access; **no OpenAI API key or paid cloud service is needed**.

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe scripts\cp4_security_audit.py
```

Earlier checkpoint runs live in `scripts/cp1_run.py` through `scripts/cp3_run.py`. **Do not rerun the one-time final-test command** (`scripts/cp4_run.py --final-test`) on the original results. Local dataset caches, model checkpoints, selection-lock files, and final-test reports are intentionally untracked; use the committed evaluation summaries, code, and screenshot evidence to inspect the portfolio results. A fresh research replication requires its own controlled evaluation and honest record of new model selection and testing.

## Real screenshot evidence

Screenshots were captured from completed local runs and saved in `screenshots/`, with names confirmed October 16, 2026. These are actual screenshots, not mock CI or cloud deployments.

| Evidence | Screenshot |
|---|---|
| CP1 — Baseline tests, data collection and validation metrics | [01_cp1_traditional_baselines.png](screenshots/01_cp1_traditional_baselines.png) |
| CP2 — MLP, LSTM and Transformer training | [02_cp2_neural_sequence_training.png](screenshots/02_cp2_neural_sequence_training.png) |
| CP3 — GCN, GraphSAGE and no-edge ablation | [03_cp3_graph_training_and_ablation.png](screenshots/03_cp3_graph_training_and_ablation.png) |
| CP4 — Pre-final-test locking | [04_cp4_selection_lock_preflight.png](screenshots/04_cp4_selection_lock_preflight.png) |
| CP4 — One-time sealed-test output | [05_cp4_one_time_final_test.png](screenshots/05_cp4_one_time_final_test.png) |
| Dashboard — Bank Marketing and HAR | [06_final_dashboard_bank_har.png](screenshots/06_final_dashboard_bank_har.png) |
| Dashboard — Cora and limitations | [07_final_dashboard_cora_limitations.png](screenshots/07_final_dashboard_cora_limitations.png) |
| Source-tree security audit | [08_cp4_security_audit_pass.png](screenshots/08_cp4_security_audit_pass.png) |

The screenshot Word handoff was completed before publication. The screenshot audit covers local source-file names and Git ignore state only, **not** a comprehensive content, dependency-vulnerability, or credentials audit.

## Reproducibility and limitations

- CP1–CP3 preserved evaluation-contract hashes and selected-checkpoint hashes; CP4 locked the selections before one-time test scoring.
- CPU-only Windows 10 / Python 3.12.0, with four PyTorch CPU threads for neural training. Reported prediction durations are single observed full-test prediction passes, **not** repeated latency distributions or SLA guarantees.
- One random seed (42), no independent repeated-run confidence intervals or statistical significance test.
- Positive-class imbalance and dataset shift materially affect Bank Marketing; comparisons should not rely on accuracy alone.
- Cora topology and test-node **features**, but not test labels, were available for transductive message passing.
- Optional MLflow was not requested in the recorded CP3 run; JSON and SHA256 artifact lineage are the verified tracking mechanisms. Do not interpret optional integrations as executed experiments.
- Public datasets are downloaded into ignored local directories and are **not** redistributed in this repository.

## Security and publication status

`.env`, `.venv`, raw data, intermediate results, and model checkpoints are ignored by Git. The local filename audit passed after eight real screenshots were captured, but manual staged-content/secret inspection, data-source review, repository initialization and live CI remain separate publication requirements. **No successful GitHub Actions run is claimed prior to its actual execution.**

Read [the final publication checklist and PowerShell commands](docs/GITHUB_PUBLICATION_CHECKLIST.md), then confirm staged files before pushing. Do not commit keys, secrets, confidential employer records, or private datasets. An API key previously present in an early local ZIP should be revoked/rotated before publication.
