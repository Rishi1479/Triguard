# Text module results

> **WARNING: SYNTHETIC DATA.** These numbers come from generated postings, not EMSCAD. They only prove the pipeline runs; do not quote them as results.

Analyzer model in use: `text_tfidf_lr` (threshold 0.500, tuned on validation).

| Set | Variant | precision | recall | f1 | pr_auc | roc_auc | mcc | accuracy | ECE |
|---|---|---|---|---|---|---|---|---|---|
| EMSCAD test | model only | 1.000 | 0.978 | 0.989 | 0.999 | 1.000 | 0.987 | 0.997 | 0.001 |
| EMSCAD test | model + rule boost | 1.000 | 0.978 | 0.989 | 1.000 | 1.000 | 0.987 | 0.997 | 0.021 |
| Indian custom | model only | 0.544 | 1.000 | 0.705 | 1.000 | 1.000 | 0.000 | 0.544 | 0.455 |
| Indian custom | model + rule boost | 0.544 | 1.000 | 0.705 | 1.000 | 1.000 | 0.000 | 0.544 | 0.455 |

## Baselines (from reports/text_baselines.json, EMSCAD test)

| Model | precision | recall | f1 | pr_auc | roc_auc | mcc | accuracy | ECE |
|---|---|---|---|---|---|---|---|---|
| tfidf_lr | 1.000 | 0.978 | 0.989 | 0.999 | 1.000 | 0.987 | 0.997 | 0.001 |
| bilstm | 1.000 | 0.978 | 0.989 | 1.000 | 1.000 | 0.987 | 0.997 | 0.002 |

## Comparison with the base paper (Pillai 2023, Bi-LSTM, EMSCAD)

| Model | F1 | Recall |
|---|---|---|
| Pillai (reported) | 0.86 | 0.83 |
| ours: bilstm | 0.989 | 0.978 |
| ours: tfidf_lr | 0.989 | 0.978 |
| ours: DistilBERT | _pending Colab training_ | _pending_ |

Note: the base paper's AUROC appears to be computed from hard labels; all AUCs here use probabilities.
