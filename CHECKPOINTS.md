# Project 13 — Four Consolidated Checkpoints

Status (October 16, 2026): **CP1–CP3 completed and reviewed; CP4 final evaluation, local security checks, and eight-image Word screenshot handoff completed. Screenshots saved/confirmed. Local Git staging review, repository publication, and real GitHub CI verification remain pending.**

Each checkpoint is a substantial, independently testable delivery. Target **one consolidated replacement ZIP per implementation checkpoint**, rather than separate edits for each model or test. A checkpoint passes only when the user has run the specified checks and shared evidence of the results; creating a ZIP is not proof of completion.

## [x] CP1 — Foundation, data contracts and traditional baseline (original CP1–CP3)

**Gate A — inspection completed on user PC (October 16, 2026):** inspect the original repository, dependency versions, Python interpreter/venv, CPU/CUDA support, potential secrets, starter tests, available datasets, configuration and missing functionality. Use `python scripts/cp1_readonly_audit.py`; share its output. Avoid running tests until compatible packages are confirmed.

**Gate B — implemented and validated on user PC:** datasets approved and downloaded (UCI Bank Marketing, UCI HAR, Cora); task-specific contracts frozen in `docs/EVALUATION_PLAN.md`. Training-only sklearn baselines, public-data loaders and offline tests are packaged. Run `& .\.venv\Scripts\python.exe scripts\cp1_run.py`, which first runs tests and then downloads datasets and fits only validation baselines. Keep final test sealed. Never mark complete on the strength of package creation alone.

**Exit evidence (confirmed):** environment audit reviewed; dataset/license decision recorded; deterministic split/leakage tests pass; baseline validation scores and run settings recorded (NOT invented); locked evaluation contract; validation command output. Verified user-run results and limitations are recorded in `docs/CP1_RESULTS_REVIEW.md`; the original CP1 JSON is retained locally in `outputs/cp1/`.

## [x] CP2 — MLP, LSTM and Transformer experiments (original CP4–CP7)

Implement one coherent PyTorch training/evaluation loop (Dataset/DataLoader, CPU fallback, optional GPU, optimizer/regularization/scheduler, early stopping, checkpoint lineage). Train MLP and fit-for-purpose LSTM/Transformer models. Construct chronological sequence features without future leakage, compare each sequence method with an appropriate simple baseline **on the same task and same compatible split**; decide whether anomaly detection is justified rather than forcing it.

**Exit evidence (confirmed user-run October 16, 2026):** actual validation metrics, training logs, seeds and hardware recorded; model and temporal-leakage tests pass; reproducible checkpoint/artifact outputs; comparison table for the relevant task(s). Final test stays sealed pending selection.

## [x] CP3 — GNNs, tuning and explainability (original CP8–CP11, CP13)

Define/validate graph nodes and edges, provenance, train/validation/test boundaries, permissible message-passing edge access, and transductive/inductive assumptions. Train GCN and GraphSAGE with suitable graph-naive baselines for the **same graph task**. Record controlled tuning (Optuna only if justified), experiment lineage (MLflow where available), appropriate Captum/SHAP/graph interpretation, and graph ablations as supported by the data.

**Exit evidence (confirmed user-run October 16, 2026):** graph leakage and split tests pass; observed baseline-vs-GNN results, model configs, experiment IDs and validation results saved; graph attribution examples clearly labeled with limitations. Do not compare raw unrelated task scores.

## [ ] CP4 — Final evaluation, audit, screenshots and GitHub (publication/CI still pending) (original CP12, CP14–CP16)

Finalize model selection on validation, then execute the **one-time sealed test evaluation** and record metrics, error analysis, threshold/calibration and robustness checks as appropriate. Run ablations, repeated-seed uncertainty where feasible, inference latency, training duration, parameters and CPU/GPU memory measures with hardware details. Compare accuracy vs complexity, including unfavorable results. Run the full regression suite, dependency/license/security/secret checks and repository hygiene audit.

**Evidence/publication gate:** assemble **real screenshots** with exact PNG names and explanations in a Word handoff; share for confirmation. After screenshot confirmation, finalize README, verify Git state/secret exclusion, publish to GitHub and verify actual CI results. These final actions all belong to CP4, but GitHub publishing is blocked until the screenshot gate is confirmed.

**Exit evidence:** verified test outputs, reproducibility and benchmark reports, final screenshot handoff confirmed, approved repository published, CI evidence collected. Provide resume bullets only from observed measurements.

## Change-control / delivery rule

- Consolidated **four checkpoint groups, not four guaranteed runs or a limit of four corrective patches**. The project is scientific work and an unexpected defect may require a correction.
- Bundle implementation + tests + config + docs + one clear validation entry point per checkpoint. Provide a manifest identifying changed files and exact PowerShell commands.
- Fix multiple findings together after inspecting a user's failed output; do not replace files one by one unless necessary.
- Do not add a library only for keywords, assume PyTorch/PyG compatibility, fabricate results, or change the frozen evaluation contract without explicitly documenting why.
- CP1 inspection and dependency bootstrap were evidenced on the user's Windows PC; the actual CP1 baseline evaluation and tests in this implementation package have not yet been evidenced.

## CP4 package delivery note

CP4 includes `scripts/cp4_run.py` for preflight and explicitly unlocked final test, `scripts/cp4_security_audit.py`, `docs/CP4_WINDOWS_COMMANDS.md`, `docs/CP4_FINAL_EVIDENCE_PLAN.md`, and a proposed GitHub Actions workflow. **No CP4 final evaluation has occurred in this package.** The screenshot Word handoff, user confirmation, final Git and CI verification remain pending.

## Latest CP4 evidence (verified October 16, 2026)

- All eight locked models evaluated **once** on final test after checksum preflight; final metrics documented in `docs/FINAL_EVALUATION_RESULTS.md`.
- Local pytest, pip check, and the corrected source-tree audit passed; the audit is not an exhaustive secret scan.
- Word screenshot handoff finished; the user showed eight numbered PNGs saved under `screenshots/` alongside the manifest.
- **Do not mark CP4 complete** until local staged-file/security review, actual GitHub publication, and a verified CI run.
