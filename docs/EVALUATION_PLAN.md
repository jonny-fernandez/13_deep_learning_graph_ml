# Project 13 — Evaluation Contracts Frozen for CP1 (October 16, 2026)

**Status:** CP1 completed and actual user-run validation evidence reviewed. See `docs/CP1_RESULTS_REVIEW.md`. CP2 neural training completed and reviewed (`docs/CP2_RESULTS_REVIEW.md`); CP3 validation-only experiments completed on the user PC; see docs/CP3_RESULTS_REVIEW.md. Final tests sealed.

## Three distinct prediction tasks (do NOT compare scores across datasets)

| Task | Unit and prediction time | Classes and proposed baseline | Split and test seal | Selection metric |
|---|---|---|---|---|
| Bank Marketing, `bank-additional-full.csv` | Contact attempt, immediately **before** the current phone call | Subscription `y`, yes=1/no=0; encoded-feature logistic regression | Official full file in recorded date order: 70% train, 15% validation, last 15% sealed | Validation PR-AUC / average precision; also ROC-AUC, macro-F1, log loss, class metrics |
| UCI HAR, smartphone inertial windows | One 2.56-second, 128-reading completed sensor window; classification **after** window end (NOT future forecasting) | Six activities; standardized 561 engineered window features + logistic regression versus CP2 LSTM/Transformer on same window signals | Respect official group-separated train/test. Within official train, 80/20 **subject-group** train/validation; official test sealed | Validation macro-F1, also balanced accuracy and per-class recall |
| Cora citation graph | One publication node in the observed citation graph snapshot | Seven publication topics; graph-free logistic regression on node features versus CP3 GCN/GraphSAGE | PyG `Planetoid('Cora', split='public')` fixed train/validation/test masks; only train labels during fitting | Validation macro-F1, class-specific performance and compute |

## Sources, usage and datasets

- Bank Marketing (UCI dataset 222): https://archive.ics.uci.edu/dataset/222/bank+marketing ; `bank-additional-full.csv` contains 41,188 source-order observations (validate actual local size). Download only from https://archive.ics.uci.edu/static/public/222/bank+marketing.zip . Credit Moro, Cortez and Rita (2014); check current UCI reuse terms before redistribution.
- HAR (UCI dataset 240): https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones ; original data from Reyes-Ortiz et al. DOI 10.24432/C54S4K, **CC BY 4.0**. Download from https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip . Dataset's internal window signal samples are *preprocessed*, with 50% overlap; subject separation is therefore essential. The 561 engineered features are computed using complete windows and are eligible only after a window finishes.
- Cora: PyTorch Geometric `Planetoid`, public masks, raw from the Planetoid repository. Cite the original Cora/Planetoid research. Public research availability does not itself establish redistribution rights; **do not upload raw dataset** to GitHub until license is independently verified.

Downloaded archives and extracts remain only in `data/raw/` (Git ignored). Each run records available archive SHA-256, installed library versions, train/validation counts, train time and validation inference time. No confidential material, API credentials or paid access required.

## Leakage and access contract

1. Bank data are already ordered by date. **Never random-shuffle before splitting**. The field `duration` is outcome-proximate and unknown before the call; it must be excluded from *all* model features and ablations aimed at pre-call prediction. `y` is never a feature. Each contact may not be an independent client; without client identifiers we cannot guarantee group-disjoint train/test. Also the dataset lacks full dates per row: source-order split is the operational surrogate for chronological evaluation. These limitations must be reported.
2. HAR original UCI test users cannot be used for tuning, model selection, scaling or evaluation until CP4. The CP1 data preparation **does not extract official test labels/features**; it extracts only `test/subject_test.txt` for the disjointness check. Scalers fit on training subjects only. For CP2, reconstruct sequence splits from the persisted train/validation subject ID lists and training inertial windows. Do not interpret HAR window classification as forecasting unseen future frames.
3. Cora is a **transductive** benchmark. During node classification, features and links of unlabeled validation/test nodes are visible in the one fixed graph. The node-label loss/training targets must be restricted to the public train mask; validation labels only for model selection, final test labels sealed. This is not a prospective/new-node performance claim. Edge/node visibility is an explicit allowance, not label leakage. No link prediction claim is made.
4. Preprocessing including bank one-hot encodings and scaling are fit via `Pipeline.fit` using train only. Labels from the final test partition may not be used to tune, choose thresholds or compare models. Validation reports are explicitly labeled **validation-only**.
5. No synthetic or demo metrics may be presented as real dataset results. Test scoring happens once in CP4 after architecture/hyperparameter decisions are locked.

## Evaluation and scientific reproducibility

- Fixed seed: **42**. Bank source-order split 70/15/15. HAR official subject-disjoint test plus training subjects split by `GroupShuffleSplit(test_size=.2, random_state=42)`. Cora public masks.
- Binary: validation PR-AUC as primary; ROC-AUC, accuracy, macro-F1, precision/recall, confusion, log loss secondary. Default 0.5 threshold for baseline diagnostics (not optimized on test).
- Multiclass: macro-F1 primary; accuracy, balanced accuracy, per-class precision/recall/F1, confusion matrix, log loss secondary.
- Record measured train and validation inference duration, CPU model/environment, package versions, source hashes. For CP2/CP3 capture model checkpoint IDs, batch sizes, seeds, hardware, learning curves. CP4: observed compute comparisons and robust final test assessment.
- HAR derived 561-feature logistic regression versus neural models on raw 9-channel 128-step window sees the **same window information but different representations** (pre-engineered vs learned). Report that qualification explicitly rather than calling them identical feature inputs.
- CP1 verified results remain at `outputs/cp1/cp1_validation_report.json` on the user PC. A measured-score summary and limitations appear in `docs/CP1_RESULTS_REVIEW.md`. CP2 predictions and checkpoints stay local/ignored; test labels remain unopened.

## CP1 acceptance checks

- Offline pytest passes including bank-duration exclusion, chronological ordering, HAR subject-disjoint masks, graph-independent split checks, archive extraction allowlist and probability metrics.
- Source archives download from approved HTTPS URLs, file formats and schemas validate, all three actual baselines fit on train and report validation scores.
- Zero final-test labels evaluated. The report is reviewed and recorded in the checkpoint evidence before CP1 is marked complete.
