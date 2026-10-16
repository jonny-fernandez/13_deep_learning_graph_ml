# Project Rules — Project 13

1. **Four consolidated checkpoint groups** replace the original 16, with the full scope and evidence gates in `CHECKPOINTS.md`. This does not imply four guaranteed successful executions.
2. **Do not modify the original code during inspection-only CP1 Gate A.** Inspect local interpreter, dependency compatibility, starter tests and project assumptions first. Implement CP1 baseline only after the environment and data contract are reviewed.
3. Deliver each substantial checkpoint with **one complete ZIP**, including all necessary updated source files, tests, configs, docs, a changed-file manifest and exact PowerShell commands. Avoid a drip feed of individual file edits.
4. A ZIP or generated output alone does **not** establish checkpoint completion; require actual user-run validation evidence, diagnose failures and batch related corrections.
5. Do not manufacture experiments, scores, confidence intervals, compute savings, CI statuses or GitHub publication. AI may interpret only actual recorded measurements.
6. Prevent train/test leakage. Lock the target/splits/data availability; fit all preprocessing on train only; validate chronological sequence windows and graph edge access policies; do not select hyperparameters using sealed test data.
7. Compare each complex method with simpler baselines on **the same task** and compatible splits. Document different datasets/tasks separately. Retain negative results.
8. Use reproducible seeds, configs, versioned data splits, dependency records, tracked artifacts and meaningful automated tests. Preserve CPU fallback.
9. Only install necessary dependencies in the project-specific environment after checking installed versions; PyTorch/PyG wheels depend on Python, Torch, OS and CUDA versions. Do not reinstall unrelated tools.
10. Use licensed public/synthetic non-sensitive datasets; never include employer data, PII, credentials or secrets. Keep `.env`, raw datasets, caches, runs, model checkpoints and other non-source artifacts out of Git unless explicitly reviewed and safe.
11. Perform a dependency, source and secret audit before GitHub publishing. The starter ZIP included a `.env` file with an apparent key; it is intentionally **excluded** from the refreshed package. Revoke/rotate any exposed key.
12. After technical completion and **before** GitHub, provide a Word screenshot handoff with the genuine screenshots embedded, their numbered PNG filenames and an explanation of each. Wait for confirmation before final README and publication.
13. Only after actual repository/CI verification, provide resume bullets in the form **Action Verb + Technical Task + Verified Outcome**; never invent improvements.
