# Adversarial robustness (text)

> **WARNING: SYNTHETIC DATA.** Re-run after adding the real datasets.

Scam rows are rewritten by label-preserving obfuscations (the kind real scammers use to
dodge keyword filters); genuine rows are untouched. `rule_recall` is the share of scam rows
where at least one red flag still fires, so it isolates the regex layer from the model.

## EMSCAD test (n=345, scam=45)

| Attack | rule_recall | recall | precision | f1 | mcc | mean_prob_drop |
|---|---|---|---|---|---|---|
| none (clean) | 0.978 | 1.000 | 0.978 | 0.989 | 0.987 | 0.000 |
| spacing | 0.889 | 0.978 | 0.978 | 0.978 | 0.974 | 0.002 |
| homoglyph | 0.956 | 0.978 | 0.978 | 0.978 | 0.974 | 0.002 |
| obfuscate_contact | 0.978 | 1.000 | 0.978 | 0.989 | 0.987 | 0.000 |
| spell_amounts | 0.978 | 1.000 | 0.978 | 0.989 | 0.987 | 0.002 |
| synonym_swap | 0.933 | 0.978 | 0.978 | 0.978 | 0.974 | 0.002 |
| all_combined | 0.911 | 0.978 | 0.978 | 0.978 | 0.974 | 0.002 |

Weakest point: `spacing` leaves the rule layer firing on only 88.9% of scam rows (clean baseline 97.8%).

## Indian custom (n=68, scam=37)

| Attack | rule_recall | recall | precision | f1 | mcc | mean_prob_drop |
|---|---|---|---|---|---|---|
| none (clean) | 1.000 | 1.000 | 0.544 | 0.705 | 0.000 | 0.000 |
| spacing | 0.324 | 1.000 | 0.544 | 0.705 | 0.000 | 0.000 |
| homoglyph | 0.459 | 1.000 | 0.544 | 0.705 | 0.000 | 0.000 |
| obfuscate_contact | 1.000 | 1.000 | 0.544 | 0.705 | 0.000 | 0.000 |
| spell_amounts | 0.946 | 1.000 | 0.544 | 0.705 | 0.000 | 0.000 |
| synonym_swap | 0.649 | 1.000 | 0.544 | 0.705 | 0.000 | 0.000 |
| all_combined | 0.270 | 1.000 | 0.544 | 0.705 | 0.000 | -0.000 |

Weakest point: `all_combined` leaves the rule layer firing on only 27.0% of scam rows (clean baseline 100.0%).

