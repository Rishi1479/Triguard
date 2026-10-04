# Text module results

> **WARNING: SYNTHETIC DATA.** These numbers come from generated postings, not EMSCAD. They only prove the pipeline runs; do not quote them as results.

Analyzer model in use: `text_hybrid_lgbm` (threshold 0.120, tuned on validation).

| Set | Variant | precision | recall | f1 | pr_auc | roc_auc | mcc | accuracy | ECE |
|---|---|---|---|---|---|---|---|---|---|
| EMSCAD test | model only | 0.957 | 0.978 | 0.967 | 0.997 | 1.000 | 0.962 | 0.991 | 0.033 |
| EMSCAD test | model + rule boost | 0.815 | 0.978 | 0.889 | 0.987 | 0.997 | 0.875 | 0.968 | 0.046 |
| Indian custom | model only | 0.500 | 1.000 | 0.667 | 1.000 | 1.000 | 0.000 | 0.500 | 0.137 |
| Indian custom | model + rule boost | 0.500 | 1.000 | 0.667 | 1.000 | 1.000 | 0.000 | 0.500 | 0.221 |

## Baselines (from reports/text_baselines.json, EMSCAD test)

| Model | precision | recall | f1 | pr_auc | roc_auc | mcc | accuracy | ECE |
|---|---|---|---|---|---|---|---|---|
| tfidf_lr | 1.000 | 0.956 | 0.977 | 1.000 | 1.000 | 0.974 | 0.994 | 0.010 |
| bilstm | 1.000 | 0.956 | 0.977 | 0.995 | 0.999 | 0.974 | 0.994 | 0.002 |

## Comparison with the base paper (Pillai 2023, Bi-LSTM, EMSCAD)

| Model | F1 | Recall |
|---|---|---|
| Pillai (reported) | 0.86 | 0.83 |
| ours: bilstm | 0.977 | 0.956 |
| ours: tfidf_lr | 0.977 | 0.956 |
| ours: DistilBERT | _pending Colab training_ | _pending_ |

Note: the base paper's AUROC appears to be computed from hard labels; all AUCs here use probabilities.
