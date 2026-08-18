# Pair-Level Essay-Scale Reuse Detection

This repository contains the core code used for the CIKM paper on pair-level
essay-scale republication and reuse detection in eighteenth-century books and
newspapers.

The code is intentionally limited to the public, reusable parts of the
workflow:

- rule-based pair classification over precomputed pair-level evidence features;
- a lightweight decision-tree baseline;
- evaluation of labeled pair predictions by split;
- newspaper audit summarization from non-textual annotation fields;
- dataset schemas and release notes.

Raw ECCO and Burney newspaper OCR text, subscription-corpus content,
machine-specific run scripts, local paths, and private annotation working files
are not included.

## Repository Layout

```text
reprint-detection-hume-case-study/
  README.md
  DATASETS.md
  requirements.txt
  src/
    __init__.py
    io_utils.py
    metrics.py
    workflow_rules.py
    decision_tree_baseline.py
    newspaper_audit.py
  scripts/
    run_workflow.py
    evaluate_predictions.py
    run_decision_tree.py
    summarize_newspaper_audit.py
  examples/
    pair_features_example.jsonl
    newspaper_audit_example.jsonl
  data/
    (empty for now; derived data added after publication)
  LICENSE
```

## Quick Start

Install the minimal Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the final workflow on a pair-feature table:

```bash
python scripts/run_workflow.py \
  --input examples/pair_features_example.jsonl \
  --output outputs/workflow_predictions.jsonl \
  --summary outputs/workflow_summary.json
```

Evaluate predictions with gold labels:

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/workflow_predictions.jsonl \
  --metrics-json outputs/workflow_metrics.json \
  --metrics-csv outputs/workflow_metrics.csv
```

Train and apply the decision-tree baseline:

```bash
python scripts/run_decision_tree.py \
  --input examples/pair_features_example.jsonl \
  --output outputs/decision_tree_predictions.jsonl \
  --summary outputs/decision_tree_summary.json
```

Summarize a newspaper audit file:

```bash
python scripts/summarize_newspaper_audit.py \
  --annotations examples/newspaper_audit_example.jsonl \
  --summary outputs/newspaper_audit_summary.json
```

## Input Assumptions

The workflow starts after fragment-level text reuse retrieval. Each row in the
input table represents a source-target document pair with aggregated evidence
features such as coverage, bundle span, section concentration, quotation cues,
paratext cues, and title/heading cues. See `DATASETS.md` for the field schema.

The final rule cascade is implemented in `src/workflow_rules.py` as
`rule_final_workflow`.

## Rule Stages

`--method` takes the same four stage names the paper uses:

- `naive_heuristic` — a single shallow span signal;
- `structural_only` — coverage and bundle statistics;
- `context_aware` — adds title, heading, quotation, and paratext cues;
- `final_workflow` — adds hard-case rescue and suppression (default).

The decision-tree baseline is `scripts/run_decision_tree.py`. Run against the
released pair-feature table, these reproduce the rule-based rows of Tables 3
and 4 exactly, including the deployment counts of 961, 710, 1,265, and 771
predicted positives.

The four gates of the final cascade in Table 2 — Structural, Coverage,
Context, Rescue — appear in `src/workflow_rules.py` as `structural_gate`,
`structural_or_coverage_gate`, `cue_or_broad_evidence_gate`, and `rescue_gate`.

The two direct LLM baselines and the automated rule adaptation family reported
in the paper are **not** included here: both depend on a local vLLM deployment
of Qwen3-30B-A3B-Instruct-2507 and on cluster-specific job scripts.

## Notes on Reproducibility

The paper uses copyrighted or subscription-controlled historical OCR data.
For public release, this code expects derived pair-level features rather than
raw document text. If the original corpora are not available, the scripts can
still be used to inspect the rule logic, evaluate released derived features,
or apply the workflow to a compatible feature table.

## Citation

```bibtex
@inproceedings{shu2026pairlevel,
  author    = {Shu, Ke and Hinderks, Kira and M\"{a}kel\"{a}, Eetu and Tolonen, Mikko},
  title     = {Pair-Level Essay-Scale Republication and Reuse from Fragmented Historical Text Reuse: A Workflow Study on Eighteenth-Century Books and Newspapers},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  doi       = {10.1145/3799682.3839901}
}
```

## License

The code in this repository is released under the MIT License; see `LICENSE`.

The paper itself is published open access under CC BY 4.0. The derived data to
be added under `data/` will be released under CC BY 4.0 as well. Neither
licence extends to the underlying ECCO or Burney Newspapers Collection
material, which is subscription-controlled and is not redistributed here.
