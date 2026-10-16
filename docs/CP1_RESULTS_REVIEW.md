# CP1 — verified local Windows baseline results (October 16, 2026)

Evidence supplied directly by project owner: `outputs/cp1/cp1_validation_report.json` (created 2026-10-16T19:47:52Z, seed=42, Windows 10, Python 3.12, Torch 2.14.1+cpu, PyG 2.8.1, sklearn 1.9.1). The source report remains in its ignored output directory on the owner's machine; it is **not** reconstructed or overwritten by this ZIP.

| Task | Train / val / sealed | Traditional model | Validation accuracy | Validation macro-F1 | Primary validation metric |
|---|---|---|---|---|---|
| UCI Bank Marketing | 28,831 / 6,178 / 6,179 | LogisticRegression | 0.893493 | 0.471876 | PR-AUC/AP **0.209382** |
| UCI HAR | 5,551 / 1,801 / 2,947 | LogisticRegression | 0.957801 | 0.953795 | Macro-F1 **0.953795** |
| Cora | 140 / 500 / 1,000 | LogisticRegression (graph-free) | 0.514000 | 0.504832 | Macro-F1 **0.504832** |

**Important limitations / interpretation**

- Bank Marketing default 0.5-threshold predictions are **all negative**: 5,520 TN, 658 FN, 0 TP/FP; minority recall 0 and balanced accuracy 0.5. Hence 89.35% raw accuracy is NOT evidence of good subscriber identification. Train positive rate 5.57%, val positive rate 10.65%, showing potential chronological distribution shift. AP 0.2094 and ROC-AUC 0.6650 are more informative threshold-free rankings. CP2 includes documented validation-only threshold diagnostics and PR-AUC comparisons. No threshold is chosen on final test data.
- HAR confusion matrix shows activity-class 2 recall 0.8140, the lowest of six. Some confusion is concentrated with class 1. The CP1 logistic baseline uses precomputed 561 engineered features; CP2 sequence models use raw 128x9 windows, so the representation differs while task/windows/groups stay aligned.
- Cora graph-free baseline is weak but valid as a comparator; GCN and GraphSAGE use graph structure under public **transductive** masking in CP3. Test node features/topology may be visible, but test/val labels must not enter model fitting.
- Both CP1 and CP2 results are validation-only, not unbiased final test estimates. The per-task numerical scores are **not comparable across datasets**.
- The reported sklearn probability-sum warning was induced by float32 rounding in log-loss; CP2 adds validated float64 normalization before sklearn's log-loss computation, preserving prediction ranking. It does not recompute or rewrite CP1 results.

**CP1 evidence gate:** Approved datasets, CPU-only virtualenv, successful user-run test suite, actual local baseline scores, leakage controls and sealed test partitions confirmed. CP1 marked complete after review of this JSON. The CP2 delivery is code only until it runs on the user's Windows installation.
