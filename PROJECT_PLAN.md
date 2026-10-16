# Project 13 — Deep Learning, Sequence & Graph ML Evaluation Platform

## Scientific question

When do MLPs, temporal models and graph neural networks provide enough **verified** predictive value over well-tuned, simpler baselines to justify complexity, training time, inference latency and memory requirements?

**Current state (2026-10-16):** CP1, CP2 and CP3 implemented and reviewed with user-run JSON/terminal evidence. CP4 selection lock, one-time test evaluation, final HTML dashboard, security audit, Word evidence handoff and eight saved screenshots are verified. Only Git stage/content review, publication and live CI verification remain. Final test must not be reopened for post-hoc model tuning. Detailed measured results: `docs/FINAL_EVALUATION_RESULTS.md`.

## Locked architectural scope, conditional on appropriate data

- Traditional ML: logistic regression and/or gradient boosting appropriate to the problem.
- Tabular: MLP compared against the **same tabular prediction task** and traditional baseline.
- Sequence: LSTM and Transformer encoder compared to a simple sequence-aware baseline on the **same temporal target** with chronological split and historical-only features.
- Graph: GCN and GraphSAGE compared with graph-free baseline(s) on **the same node classification task**, or a documented alternative if the selected data supports it. No link prediction without a valid link-split evaluation design.
- Optuna / MLflow / SHAP / Captum / anomaly detection: use only when justified and compatible. CPU execution must remain possible.

A single shared dataset may support several views; if different datasets/tasks are necessary, **present separate within-task comparisons** and only compare process/compute characteristics across tasks with clear qualifications. Never assert that a raw ROC-AUC from one task outperforms the ROC-AUC on an unrelated task.

## Approved datasets and evaluation contracts

Contracts are now specified for approved UCI Bank Marketing, UCI HAR and Cora in `docs/EVALUATION_PLAN.md`. Record observed dataset-specific values after downloads and validation. Contract fields: origin, license, size, target, unit, prediction time, leakage risks, missingness assumptions, chronological/group/node/edge split policy, train-only transformation policy, metrics, sealed test usage, baseline thresholds, reproducibility seeds and feasible hardware. Keep project-safe/public/synthetic material only. No employer confidential data or PII.

Split training vs validation vs test **before preprocessing**. Tune/early-stop on train+validation only. Use sealed test once after final selection, aside from documented unavoidable error corrections. Track all experiment datasets and splits with identifiers/hashes.

## Four consolidated deliveries

See `CHECKPOINTS.md` for detailed gates, exit criteria and the mapping from the **original 16** tasks:

| Group | Original checkpoints | Delivery focus |
|---|---|---|
| CP1 — Foundation & Baselines | 1–3 | Read-only audit, locked contracts, leak-free traditional baseline |
| CP2 — Deep Learning & Sequences | 4–7 | Shared training loop, MLP, LSTM and Transformer |
| CP3 — Graph ML & Experimentation | 8–11 and 13 | Graph data/GCN/GraphSAGE, tracking/tuning, interpretation |
| CP4 — Final Evaluation & Publication | 12, 14–16 | Diagnostics, ablations, compute benchmarks, audit, screenshots and GitHub |

One planned bulk replacement package per CP implementation. Group refactors and tests together. Allow corrective packages when necessary; **no arbitrary limit is evidence of success**.

## Model selection and outcome reporting

Report ROC-AUC/PR-AUC where meaningful, class-specific precision/recall/F1, confidence/uncertainty, calibration, selected thresholds and error analysis. Save observed training duration, inference latency, model size, memory/device, seeds and library versions. Ablate at least the most plausible causes of apparent gains and keep negative results. If the simpler baseline wins, report it without changing the target to make the network look better.

## Final evidence and GitHub restriction

Complete actual technical checks and real screenshots first; prepare a Word document containing embedded screenshots, exact PNG filenames and what they prove. Obtain confirmation of saved screenshots **before** README finalization / repository publication. Publish only after a secret scan, including `.env` exclusion. Verify real CI results; never assert a run that has not happened.

## Latest checkpoint status (2026-10-16)

CP1/CP2/CP3 reviewed and approved from real user-produced evidence. CP4 has an explicit final-test unlock; publication remains blocked until a real screenshot Word handoff and user confirmation. See `docs/CP4_WINDOWS_COMMANDS.md`.
