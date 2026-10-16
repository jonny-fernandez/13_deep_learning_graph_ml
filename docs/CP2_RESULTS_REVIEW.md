# CP2 verified user-run results — October 16, 2026

Source: user-provided `cp2_validation_report.json` and screenshot from the Windows CP2 run. This is **validation-only**, not an estimate on the final sealed test set.

| Experiment | Traditional CP1 baseline | CP2 neural model | Validation difference | CP2 train seconds |
|---|---:|---:|---:|---:|
| Bank Marketing MLP, PR-AUC (average precision) | .2093816778 | .2165911319 | +.0072094542 | 9.18621 |
| HAR LSTM, macro-F1 | .9537954431 | .6976319038 | -.2561635393 | 22.86061 |
| HAR Transformer, macro-F1 | .9537954431 | .9508291892 | -.0029662539 | 205.02328 |

All neural models selected validation checkpoints with seed 42 and CPU execution (4 torch threads); train and validation counts match CP1. CP2 model outputs were compared on the **same targets and splits** as the respective CP1 baselines, but HAR logistic regression uses 561 engineered features while LSTM and Transformer use complete raw nine-channel windows.

Important limitations:
- Bank MLP, like the CP1 logistic baseline, classified zero positive clients at the default threshold .5 (validation recall of positive class 0). Validation-only threshold choice .25 improved recall to .2173 and positive F1 to .2584, **not** a final generalization estimate.
- HAR LSTM validation macro-F1 increased through the sixth and final epoch; low score is an **observed result of a short, bounded training run**, not evidence LSTMs cannot solve HAR.
- HAR Transformer reached .9508 macro-F1 after ~205 training seconds, compared with the strong logistic baseline .9538. No unverified claim of speed or accuracy improvement.
- This report is a fixed record. Do not overwrite CP2 validation results by tuning on the final test set. Further ablations/fair compute comparisons belong in CP4.

CP2 was reviewed and marked complete; no test labels were evaluated. The original CP1 and CP2 JSON files remain in local `outputs/` and are ignored by Git.
