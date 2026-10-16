# Captured Screenshot Evidence — Prepublication Handoff

**Status:** Word handoff with eight real screenshots completed October 16, 2026, and user confirmed on-screen that all eight PNGs were saved under `screenshots/` using the exact names below. This document records evidence, not GitHub publication approval.

| Confirmed PNG filename | What the captured image documents |
|---|---|
| `01_cp1_traditional_baselines.png` | CP1 tests, public data acquisition, and three validation-only baseline results |
| `02_cp2_neural_sequence_training.png` | CP2 MLP, LSTM and Transformer training, validation comparisons and test-seal status |
| `03_cp3_graph_training_and_ablation.png` | Cora GCN/GraphSAGE validation and no-edge retraining |
| `04_cp4_selection_lock_preflight.png` | CP4 full regression suite, pip check and source/model selection lock |
| `05_cp4_one_time_final_test.png` | One-time final held-out results of eight locked models |
| `06_final_dashboard_bank_har.png` | Final Bank and HAR tables from local report |
| `07_final_dashboard_cora_limitations.png` | Final Cora comparisons and limitations from local report |
| `08_cp4_security_audit_pass.png` | Local filename/.gitignore security audit PASS after screenshot fixes |

The corresponding Word evidence document was prepared and reviewed before publication. The `.png` files and `SCREENSHOT_MANIFEST.txt` are in the project's `screenshots/` directory on the user's computer. The dashboard images show final scores from the user's actual JSON reports.

**Still pending:** manual staged-file secret/license review; repository initialization/publication; **real** GitHub Actions result. Add a ninth CI screenshot only **after** an actual successful GitHub Actions run—do not fabricate a green badge or CI evidence.
