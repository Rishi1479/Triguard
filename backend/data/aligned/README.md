# Aligned multimodal set

One row per real-world case, where text, image and URL describe the SAME posting.

| column | meaning |
|---|---|
| case_id | unique integer |
| text | posting text (anonymise phones/emails/IDs before committing) |
| image_path | path to screenshot/offer letter (blank if none). Images must be gitignored. |
| url | application link (blank if none) |
| label | 1 = scam, 0 = genuine |
| notes | free text: source, why it is a scam, etc. |

Any of text / image_path / url may be blank, but at least one must be present. Save your
real file as `data/aligned/aligned.csv`; the fuser trains its meta-classifier on it and the
ablation study uses it. Until it exists, the app fuses with a fixed weighted average.
