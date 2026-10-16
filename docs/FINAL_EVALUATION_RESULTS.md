# Final Evaluation — Verified Local Evidence (2026-10-16)

**Scope:** Final test evaluated **once** after predeclared validation selection and SHA256 lock. Technical evaluation and screenshots complete; publishing / GitHub CI verification pending. The authoritative JSON files remain local under ignored `outputs/cp1` … `outputs/cp4` and were uploaded separately for review.

## Results by dataset

| Task | Model | Val primary | Test primary | Test Macro-F1 | Test prediction seconds | Model parameter count |
|---|---|---:|---:|---:|---:|---:|
| Bank (PR-AUC) | LogisticRegression | 0.2094 | 0.4466 | 0.4513 | 0.0470 | n/a |
| Bank (PR-AUC) | MLP | 0.2166 | 0.4756 | 0.3810 | 0.1010 | 4,033 |
| HAR (Macro-F1) | LogisticRegression | 0.9538 | 0.9511 | 0.9511 | 0.0178 | n/a |
| HAR (Macro-F1) | LSTM | 0.6976 | 0.6723 | 0.6723 | 0.3292 | 5,702 |
| HAR (Macro-F1) | Transformer | 0.9508 | 0.8610 | 0.8610 | 2.0486 | 13,222 |
| Cora (Macro-F1) | LogisticRegression | 0.5048 | 0.5070 | 0.5070 | 0.0097 | n/a |
| Cora (Macro-F1) | GCN h64 d0.5 | 0.7881 | 0.7984 | 0.7984 | 0.0249 | 92,231 |
| Cora (Macro-F1) | GraphSAGE h64 d0.5 | 0.7748 | 0.7918 | 0.7918 | 0.1139 | 184,391 |

All durations are from one Windows CPU; sizes, architectures and representation differences matter. Do **not** rank scores from Bank, HAR, and Cora against one another; their tasks and primary metrics differ.

## Findings and honest qualifications

1. **Cora:** Selected GCN 0.7984 test Macro-F1 vs graph-free LR 0.5070, an absolute +0.2914. A validation-only retrained no-edge GCN scored 0.5910 vs full graph 0.7881, illustrating useful connectivity under transductive conditions, **not a causal effect**. GraphSAGE scored 0.7918 and took longer per measured test inference batch than GCN.
2. **HAR:** LR on 561 engineered features scored 0.9511 test Macro-F1. Transformer on raw windows scored 0.8610 and LSTM 0.6723. Because feature representations differ, the observed gap combines feature-engineering and model-family choices. Transformer validation was 0.9508; final test 0.8610.
3. **Bank:** MLP outscored LR in test AP (0.4756 vs 0.4466) but failed to identify positives with default threshold 0.5; MLP test Macro-F1 0.3810. Validation-selected threshold 0.25 obtained test precision 0.5836, recall 0.2071, and positive-class F1 0.3057. The overall positive rate changed from train **5.57%**, validation **10.65%** to test **38.45%**; cross-partition AP differences reflect changing prevalence as well as model rankings. LR had test positive recall 9.47% at 0.5 and test Macro-F1 0.4513. None of this demonstrates production-ready targeting.

## Integrity and limitations

- CP1/CP2/CP3 source-report SHA256 relationships and CP4 selection lock were verified on the user's PC before final testing. Hashes are in the local report files; no actual trained checkpoints are published.
- Cora test-node features and edges are visible transductively; only 140 training labels used to fit. Do not claim an inductive or future-edge result.
- Single seed, no multi-seed estimate or model-score confidence interval. Latency is one observed prediction pass, not controlled benchmarking across machines.
- Exact CPU and version data were recorded: Windows 10, Python 3.12.0, CPU-only PyTorch, 4 threads. No cloud execution or deployed inference claimed.
- The local automated tests and basic path security audit passed. Live GitHub Actions verification and manual source/dependency audit are **pending**.
