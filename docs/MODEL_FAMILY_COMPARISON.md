# Model Family Comparison — Final Verified Local Results

**Study design:** Within-task classical-vs-neural evaluation on distinct datasets. The absolute scores below must not be ranked across unrelated tasks.

| Task / primary metric | Traditional baseline final test | Complex models final test | Conclusion |
|---|---:|---|---|
| Bank / average precision (PR-AUC) | Logistic 0.4466 | MLP 0.4756 | Small AP advantage, but MLP predicts zero positives at 0.5. Validation-selected 0.25 threshold recovers 20.7% test recall. Severe class-rate shift across partitions. |
| HAR / Macro-F1 | Engineered-feature logistic 0.9511 | LSTM 0.6723; Transformer 0.8610 | Feature-engineered traditional model wins, with lower observed CPU prediction time. Comparison confounds feature representation and architecture. |
| Cora / Macro-F1 | Graph-free logistic 0.5070 | GCN 0.7984; GraphSAGE 0.7918 | Selected GCN wins under public **transductive** Cora protocol. Retrained no-edge GCN 0.5910 on validation vs connected 0.7881. |

Tests were evaluated once after frozen selection and no test-based redesign. Experiments used one seed, Windows CPU; observed inference durations are one complete test pass, not statistical latency distributions. Full interpretation: `FINAL_EVALUATION_RESULTS.md`.
