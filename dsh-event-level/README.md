# Event-Level Evidence Construction from Text Reuse

Code for the DSH article *From matches to events: the iterative construction of
text-reuse evidence*. It covers the layer above pair-level consolidation: how
fragments with offsets become local passages, how passages become event
candidates a scholar can review, how three kinds of human label are kept apart,
and the five tests the resulting layer has to pass.

The pair-level stage this builds on is in [`../cikm-pair-level`](../cikm-pair-level).

## What is here and what is not

Included: the reusable construction and testing logic, the schemas it expects,
and one small example file per schema.

Not included: ECCO and Gale OCR text, page images, subscription-corpus content,
human review working files, and any dataset path from our own environment.
Every script takes its paths on the command line.

## Layout

```text
dsh-event-level/
  src/
    io_utils.py             read/write, checksums, per-stage provenance
    passage_links.py        adjacency edge features, link model, components
    newspaper_units.py      article-level units and destination-balanced weights
    label_registry.py       the three label kinds and the rules between them
    event_consolidation.py  destination-side overlap components and overlays
    metrics.py              grouped cross-validation, Wilson, weighted counts
    evidence_tests/
      null_model.py         structural null for a distributional claim
      compiled_units.py     offset geometry against the work-level grouping
      source_ablation.py    copy-level and work-level, conditioned on content
      coverage_audit.py     pooled probability audits and the recall bound
      cross_media.py        shared identity-free core and the transfer matrix
  scripts/                  command-line entry points, one per stage
  examples/                 one small file per schema
  data/                     empty skeleton; see data/README.md
```

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

Each stage writes a manifest recording input checksums, output checksums and
parameters. No stage overwrites an earlier one.

```bash
# 1. fragments -> adjacency edges -> local passages
python scripts/build_passage_candidates.py \
  --fragments examples/fragments_example.jsonl \
  --out-edges data/processed/edges.jsonl \
  --out-passages data/processed/passages.jsonl \
  --out-manifest data/processed/passages.manifest.json

# 2. fit the link model against the fixed-gap rule it replaced
python scripts/run_passage_link_model.py \
  --edges data/processed/edges.jsonl \
  --edge-labels data/annotation/edge_labels.jsonl \
  --out-decisions data/processed/edge_decisions.jsonl \
  --out-report data/processed/link_model_report.json

# 3. newspapers: inherit the layout-segmented article as the unit
python scripts/build_newspaper_article_units.py \
  --fragments examples/newspaper_fragments_example.jsonl \
  --out-candidates data/processed/article_candidates.jsonl \
  --out-manifest data/processed/article_candidates.manifest.json

# 4. passages -> event candidates for human review
python scripts/consolidate_events.py \
  --passages data/processed/passages.jsonl \
  --out-candidates data/processed/event_candidates.jsonl \
  --out-manifest data/processed/event_candidates.manifest.json

# 5. the evidence tests
python scripts/run_evidence_tests.py \
  --events data/processed/events.jsonl \
  --passages data/processed/passages.jsonl \
  --audits examples/coverage_audits_example.json \
  --frame-destinations 2190 \
  --out-report data/processed/evidence_tests.json
```

## What the models are for

A model probability orders a review queue and divides unlabelled candidates
into a full-review and a reduced-review pool. It is never written into a human
label field and never becomes a historical event. `label_registry.py` enforces
this: sealed evaluation labels must carry their sampling probability, training
and evaluation must be disjoint by destination, and a model-assigned label
cannot appear in a final-review record.

Passage reconstruction is the one stage where a supervised model materially
outperforms a rule, which is why `passage_links.legacy_gap_rule` is kept — so
the comparison can be reproduced, not because the rule is recommended.

## License

MIT; see the repository `LICENSE`. Copyright (c) 2026 Ke Shu and Mikko Tolonen.
