# Datasets — Sources, Attribution and Redistribution

No dataset files are included in this GitHub repository; local dataset files are downloaded only to ignored `data/raw/` and/or processed caches.

| Dataset | Official source / attribution | License information |
|---|---|---|
| **UCI Bank Marketing** | S. Moro, P. Rita, P. Cortez (2014), *Bank Marketing*, UCI Machine Learning Repository, [DOI 10.24432/C5K306](https://doi.org/10.24432/C5K306) · [UCI record](https://archive.ics.uci.edu/dataset/222/bank+marketing) | UCI record states **CC BY 4.0**; cite the dataset and authors |
| **UCI Human Activity Recognition Using Smartphones** | J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto, X. Parra (2013), *Human Activity Recognition Using Smartphones*, UCI, [DOI 10.24432/C54S4K](https://doi.org/10.24432/C54S4K) · [UCI record](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | UCI record states **CC BY 4.0**; cite the dataset and authors |
| **Cora (Planetoid prepared files)** | [Planetoid data and method](https://github.com/kimiyoung/planetoid) — Z. Yang, W. W. Cohen, R. Salakhutdinov (2016), *Revisiting Semi-Supervised Learning with Graph Embeddings*; original Cora citation-network dataset | The Planetoid repository publicly distributes prepared files and requests citation. The repository's **MIT code license should not automatically be interpreted as a standalone license for all upstream dataset rights**. We redistribute no graph data; confirm further upstream terms if planning to redistribute it later. |

### Use policy

- Always credit the datasets and related research; do not include original records or extracted data in the Git history.
- Bank uses the UCI date-ordered `bank-additional-full.csv` subset (41,188 records); the final 15% is evaluated once. Field `duration` is excluded from a pre-call prediction because it is observed after a call.
- HAR uses official subject-disjoint test windows. The 561 engineered-feature LR uses a different representation than the nine raw sensor-channel neural models.
- Cora uses the public **transductive** Planetoid split, so test-node attributes and graph edges are visible during training, but test labels are excluded from the loss/selection steps.
- Dataset/public availability does not automatically authorize committing third-party dataset bytes, and the lack of an explicit project code license should not be mistaken for permission to relicense upstream components.
