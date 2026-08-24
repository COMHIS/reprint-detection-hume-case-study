# Data

This directory is intentionally empty in the initial release.

The derived data accompanying the paper — the pair-level feature table, the
labeled anchor splits, and the workflow prediction outputs — will be added
here after publication. They are omitted for now only because of their size.

What will be released:

- `pair_features.jsonl` — pair-level evidence features for the ECCO–ECCO
  candidate universe (22,735 pairs, 369 of them labeled);
- `gt_splits/` — the `discovery_set`, `main_eval_set`, and `hard_eval_set`
  anchor splits (111 positives in total: 60 in `main`, 22 in `hard`);
- `predictions/` — prediction files for the rule stages and the decision-tree
  baseline;
- `newspaper_audit.jsonl` — the manual audit of the ECCO–Newspaper predictions
  (176 predicted positives, 49 audited predicted negatives), with
  non-textual annotation fields only.

What will never be released here: raw ECCO OCR, raw Burney Newspapers
Collection OCR, and any extended text fragments from those corpora. Both are
subscription-controlled; see `../DATASETS.md`.

Field schemas for every file above are documented in `../DATASETS.md`.
