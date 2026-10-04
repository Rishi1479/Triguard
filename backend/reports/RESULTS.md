# TriGuard – results

> **WARNING – SYNTHETIC DATA.** Sections marked 🧪 were computed on generated stand-in data (text: True, URL: True, aligned set: True). Do not quote them as scientific results; rerun `python -m reports.evaluate_all` after adding the real datasets (see PROGRESS.md → WAITING FOR USER).

## 1. 5-fold stratified cross-validation (mean ± std) 🧪 SYNTHETIC

| Model | precision | recall | f1 | pr_auc | roc_auc | mcc |
|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.997 ± 0.007 | 0.967 ± 0.024 | 0.981 ± 0.010 | 0.997 ± 0.004 | 0.999 ± 0.002 | 0.979 ± 0.011 |
| LightGBM (lexical) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

Thresholds and Platt calibration are fitted inside each fold on a 15% validation split; vectorizers/tokenizers fit on the training part only.

## 2. Comparison with the base paper (EMSCAD test split) 🧪 SYNTHETIC

| Model | F1 | Recall | ROC-AUC (probabilities) | ROC-AUC (hard labels) | PR-AUC |
|---|---|---|---|---|---|
| Pillai (2023) – reported Bi-LSTM | 0.860 | 0.830 | n/a | n/a | n/a |
| Ours: Bi-LSTM reproduction | 0.989 | 0.978 | 1.000 | 0.989 | 1.000 |
| Ours: text_hybrid_lgbm | 0.989 | 0.978 | 1.000 | 0.989 | 1.000 |
| Ours: DistilBERT | pending (Colab) | pending | pending | pending | pending |

The base paper reports AUROC apparently computed from hard (0/1) predictions; the two AUROC columns show how much that choice changes the number for our models.

## 3. Ablation on the aligned multimodal set (out-of-fold, n=240) 🧪 SYNTHETIC

| Configuration | Coverage | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| Text only | 92% | 1.000 | 0.902 | 0.949 | 0.936 | 0.906 |
| URL only | 73% | 1.000 | 0.720 | 0.837 | 0.815 | 0.757 |
| Image only | 65% | 1.000 | 0.598 | 0.748 | 0.735 | 0.664 |
| Text + URL | 99% | 0.792 | 0.744 | 0.767 | 0.888 | 0.917 |
| Text + Image | 96% | 0.975 | 0.951 | 0.963 | 0.968 | 0.952 |
| Image + URL | 91% | 0.696 | 0.866 | 0.772 | 0.811 | 0.813 |
| All three | 100% | 0.986 | 0.829 | 0.901 | 0.933 | 0.945 |
| All three + cross-modal checks | 100% | 0.988 | 0.976 | 0.982 | 0.986 | 0.983 |
| Meta-classifier (all three + cross-modal) | 100% | 1.000 | 0.951 | 0.975 | 0.999 | 1.000 |

Coverage = share of cases where the configuration has at least one input. Thresholds are tuned on each fold's training part. ![PR curves](figures/pr_ablation.png)

## 4. Generalisation: EMSCAD-trained text model → Indian set 🧪 SYNTHETIC

| Set | Variant | Precision | Recall | F1 | PR-AUC | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|
| EMSCAD test | model only | 1.000 | 0.978 | 0.989 | 1.000 | 1.000 | 0.987 |
| EMSCAD test | model + rules | 0.978 | 1.000 | 0.989 | 1.000 | 1.000 | 0.987 |
| Indian custom | model only | 0.544 | 1.000 | 0.705 | 0.685 | 0.629 | 0.000 |
| Indian custom | model + rules | 0.544 | 1.000 | 0.705 | 1.000 | 1.000 | 0.000 |

Threshold fixed from EMSCAD validation; a gap between ROC/PR-AUC (ranking) and F1 (threshold) indicates calibration shift rather than lost signal.

### 4b. Fixing the transfer: prior correction + target recalibration 🧪 SYNTHETIC

| Model | Transfer | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|---|
| tfidf_lr | naive | 0.500 | 0.559 | 1.000 | 0.717 | 1.000 | 1.000 | 0.000 |
| tfidf_lr | adapted | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid_lgbm | naive | 0.340 | 0.559 | 1.000 | 0.717 | 0.679 | 0.567 | 0.000 |
| hybrid_lgbm | adapted | 0.500 | 0.679 | 1.000 | 0.809 | 0.679 | 0.567 | 0.521 |
| bilstm | naive | 0.190 | 0.559 | 1.000 | 0.717 | 0.902 | 0.881 | 0.000 |
| bilstm | adapted | 0.475 | 0.760 | 1.000 | 0.864 | 0.902 | 0.881 | 0.675 |

Source scam prior 0.130 -> target 0.544 (logit shift +2.074); method `prior_plus_target_platt`, 34 target rows used for recalibration, metrics reported on the 34 held-out target rows.

`naive` carries the EMSCAD threshold over unchanged — the MCC of ~0 means the model labels *every* target row as scam, so ranking is perfect but the decision is worthless. `adapted` shifts the logit by the prior log-odds difference and re-fits Platt scaling on half the target rows, then scores the other half. See `ml/common/prior_shift.py`.

## 5. OCR comparison (character error rate)

> **Do not quote the backend ranking from this table.** The images are rendered from fonts, so OCR reads back the exact glyphs that were drawn; real offer letters arrive as recompressed phone photos with skew and glare, where CER is typically 10-30x worse and the EasyOCR/Tesseract ordering commonly inverts. Re-run on real annotated images before drawing any conclusion.

> **SYNTHETIC images.** Clean rendered letters; real phone photos will be much harder. Replace `data/images/ocr_gt.csv` (path,text) with real annotated images and re-run.

Images: 20

| Backend | Preprocessing | Mean CER | Median CER | s/image |
|---|---|---|---|---|
| easyocr | raw grayscale | 0.039 | 0.038 | 1.54 |
| easyocr | preprocessed | 0.042 | 0.048 | 2.35 |
| easyocr | preprocessed + binarised | 0.042 | 0.043 | 2.21 |
| tesseract | raw grayscale | 0.002 | 0.000 | 0.21 |
| tesseract | preprocessed | 0.002 | 0.000 | 0.31 |
| tesseract | preprocessed + binarised | 0.002 | 0.000 | 0.28 |


## 6. Calibration (ECE) 🧪 SYNTHETIC

| Model | ECE before calibration | ECE after Platt scaling |
|---|---|---|
| Text (TF-IDF+LR) | 0.014 | 0.001 |
| URL (LightGBM) | 0.000 | 0.000 |

![text](figures/reliability_text.png) ![url](figures/reliability_url.png) ![fusion](figures/reliability_fusion_meta.png)

## 7. Confusion matrices & PR curves

![cm_fusion.png](figures/cm_fusion.png) ![cm_text.png](figures/cm_text.png) ![cm_url.png](figures/cm_url.png) ![pr_ablation.png](figures/pr_ablation.png) ![pr_text.png](figures/pr_text.png) ![pr_url.png](figures/pr_url.png)

## 8. Error analysis

See [error_analysis.md](error_analysis.md) (10 examples).

## 9. Adversarial robustness 🧪 SYNTHETIC

**EMSCAD test**

| Attack | rule_recall | recall | precision | f1 | mcc |
|---|---|---|---|---|---|
| none (clean) | 0.978 | 1.000 | 0.978 | 0.989 | 0.987 |
| spacing | 0.889 | 0.978 | 0.978 | 0.978 | 0.974 |
| homoglyph | 0.956 | 0.978 | 0.978 | 0.978 | 0.974 |
| obfuscate_contact | 0.978 | 1.000 | 0.978 | 0.989 | 0.987 |
| spell_amounts | 0.978 | 1.000 | 0.978 | 0.989 | 0.987 |
| synonym_swap | 0.933 | 0.978 | 0.978 | 0.978 | 0.974 |
| all_combined | 0.911 | 0.978 | 0.978 | 0.978 | 0.974 |

**Indian custom**

| Attack | rule_recall | recall | precision | f1 | mcc |
|---|---|---|---|---|---|
| none (clean) | 1.000 | 1.000 | 0.544 | 0.705 | 0.000 |
| spacing | 0.324 | 1.000 | 0.544 | 0.705 | 0.000 |
| homoglyph | 0.459 | 1.000 | 0.544 | 0.705 | 0.000 |
| obfuscate_contact | 1.000 | 1.000 | 0.544 | 0.705 | 0.000 |
| spell_amounts | 0.946 | 1.000 | 0.544 | 0.705 | 0.000 |
| synonym_swap | 0.649 | 1.000 | 0.544 | 0.705 | 0.000 |
| all_combined | 0.270 | 1.000 | 0.544 | 0.705 | 0.000 |

Scam rows only are rewritten by label-preserving obfuscations (spacing, homoglyphs, contact obfuscation, spelled-out amounts, euphemisms). `rule_recall` isolates the regex layer: where it collapses, keyword matching alone would have missed the scam. Full table: [robustness.md](robustness.md).

## 10. Ablation with 95% bootstrap confidence intervals 🧪 SYNTHETIC

| Configuration | precision | recall | f1 | pr_auc | roc_auc | mcc |
|---|---|---|---|---|---|---|
| Text only | 1.000 [1.00–1.00] | 0.902 [0.84–0.96] | 0.949 [0.91–0.98] | 0.936 [0.90–0.98] | 0.906 [0.84–0.96] | 0.927 [0.88–0.97] |
| URL only | 1.000 [1.00–1.00] | 0.720 [0.62–0.82] | 0.837 [0.77–0.90] | 0.815 [0.75–0.88] | 0.757 [0.67–0.84] | 0.793 [0.72–0.86] |
| Image only | 1.000 [1.00–1.00] | 0.598 [0.49–0.71] | 0.748 [0.66–0.83] | 0.735 [0.66–0.81] | 0.664 [0.57–0.76] | 0.703 [0.62–0.78] |
| Text + URL | 1.000 [1.00–1.00] | 0.695 [0.60–0.79] | 0.820 [0.75–0.88] | 0.888 [0.84–0.93] | 0.917 [0.88–0.95] | 0.775 [0.70–0.85] |
| Text + Image | 1.000 [1.00–1.00] | 0.951 [0.90–0.99] | 0.975 [0.95–0.99] | 0.968 [0.94–0.99] | 0.952 [0.90–0.99] | 0.963 [0.93–0.99] |
| Image + URL | 0.696 [0.63–0.77] | 0.866 [0.79–0.94] | 0.772 [0.71–0.83] | 0.811 [0.75–0.87] | 0.813 [0.74–0.88] | 0.642 [0.55–0.74] |
| All three | 0.986 [0.95–1.00] | 0.829 [0.74–0.90] | 0.901 [0.85–0.95] | 0.933 [0.90–0.97] | 0.945 [0.91–0.97] | 0.862 [0.79–0.93] |
| All three + cross-modal checks | 1.000 [1.00–1.00] | 0.976 [0.94–1.00] | 0.988 [0.97–1.00] | 0.986 [0.96–1.00] | 0.983 [0.96–1.00] | 0.982 [0.95–1.00] |
| Meta-classifier (all three + cross-modal) | 1.000 [1.00–1.00] | 0.988 [0.96–1.00] | 0.994 [0.98–1.00] | 0.999 [1.00–1.00] | 1.000 [1.00–1.00] | 0.991 [0.97–1.00] |

Stratified percentile bootstrap, 2000 resamples. With n=240 the intervals are wide enough that most configurations in section 3 are not separable; quote intervals, not point estimates.
