# CP3 Results Review — User-run evidence, October 16, 2026

**Status: APPROVED.** Evidence: user-supplied console output and `outputs/cp3/cp3_validation_report.json`. Training and model selection used only the public Cora training/validation labels. All 1,000 test-node labels remained sealed.

| Model | Cora validation Macro-F1 | Validation delta vs frozen logistic regression |
|---|---:|---:|
| Graph-free logistic regression | 0.504832 | — |
| GCN 32 units / 0.3 dropout | 0.783381 | +0.278549 |
| **GCN 64 units / 0.5 dropout** | **0.788072** | **+0.283241** |
| GraphSAGE 32 units / 0.3 dropout | 0.752784 | +0.247953 |
| GraphSAGE 64 units / 0.5 dropout | 0.774775 | +0.269943 |
| GCN 64 units without graph edges (retrained) | 0.591027 | +0.086195 |

- Selected checkpoint: `gcn_h64_d05`, selected at epoch 10 on validation Macro-F1; 92,231 parameters, 6.93 seconds observed CPU training.
- The retrained no-edge GCN was 0.197045 lower in validation Macro-F1. This is evidence for useful graph information under the fixed Cora experiment, **not** a causal estimate.
- 140 labeled train nodes, 500 validation nodes; 1,000 sealed test nodes; 2,708 nodes and 10,556 directed edge-index entries visible transductively. All node features and topology were visible, **not** their test labels.
- The CP3 report records checkpoint SHA256 hashes, validation learning curves, per-class metrics, and one validation-node gradient-times-input explanation. That explanation is not global or causal.
- MLflow option was not requested in the run; JSON experiment records and SHA256 lineage exist. Do not assert that an MLflow server was used.
- Report SHA256: `c041676d28a2bdfc90e2866b5047918e1659837bc616d3eb613cddbf44fc1ec1`

**Next:** CP4 preflight locks existing validation decisions and verifies the *local* saved checkpoints before any held-out test access. Running CP4 still requires user-executed validation and review; CP4 is not complete merely because its ZIP is generated.
