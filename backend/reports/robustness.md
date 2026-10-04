# Adversarial robustness (text)

> **WARNING: SYNTHETIC DATA.** Re-run after adding the real datasets.

Scam rows are rewritten by label-preserving obfuscations (the kind real scammers use to
dodge keyword filters); genuine rows are untouched. `rule_recall` is the share of scam rows
where at least one red flag still fires, so it isolates the regex layer from the model.

## EMSCAD test (n=345, scam=45)

| Attack | rule_recall | recall | precision | f1 | mcc | mean_prob_drop |
|---|---|---|---|---|---|---|
| none (clean) | 0.978 | 0.978 | 0.815 | 0.889 | 0.875 | 0.000 |
| spacing | 0.956 | 0.978 | 0.815 | 0.889 | 0.875 | 0.023 |
| homoglyph | 0.956 | 0.978 | 0.815 | 0.889 | 0.875 | 0.027 |
| obfuscate_contact | 0.978 | 0.978 | 0.815 | 0.889 | 0.875 | 0.017 |
| spell_amounts | 0.978 | 0.978 | 0.815 | 0.889 | 0.875 | 0.000 |
| synonym_swap | 0.978 | 0.978 | 0.815 | 0.889 | 0.875 | 0.004 |
| all_combined | 0.978 | 0.978 | 0.815 | 0.889 | 0.875 | 0.071 |

Weakest point: `spacing` leaves the rule layer firing on only 95.6% of scam rows (clean baseline 97.8%).

## Indian custom (n=74, scam=37)

| Attack | rule_recall | recall | precision | f1 | mcc | mean_prob_drop |
|---|---|---|---|---|---|---|
| none (clean) | 1.000 | 1.000 | 0.500 | 0.667 | 0.000 | 0.000 |
| spacing | 0.324 | 1.000 | 0.500 | 0.667 | 0.000 | 0.021 |
| homoglyph | 0.486 | 1.000 | 0.500 | 0.667 | 0.000 | 0.003 |
| obfuscate_contact | 1.000 | 1.000 | 0.500 | 0.667 | 0.000 | -0.000 |
| spell_amounts | 0.946 | 1.000 | 0.500 | 0.667 | 0.000 | 0.001 |
| synonym_swap | 0.649 | 1.000 | 0.500 | 0.667 | 0.000 | 0.001 |
| all_combined | 0.270 | 1.000 | 0.500 | 0.667 | 0.000 | 0.013 |

Weakest point: `all_combined` leaves the rule layer firing on only 27.0% of scam rows (clean baseline 100.0%).

