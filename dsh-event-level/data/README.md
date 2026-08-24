# Data layout

This directory is intentionally empty. It documents the layout the scripts
expect so that a licensed environment can be reconstructed; no ECCO or Gale
material is distributed here.

```text
data/
  raw/          immutable extraction: fragments with half-open offsets on both
                sides, never pre-merged on ingest
  processed/    rebuildable derivatives: adjacency edges, passage candidates,
                newspaper article candidates, event candidates
  annotation/   human labels, each carrying its label_kind at acquisition
  evaluation/   sealed evaluation sets, each with its sampling design, stratum
                and per-record sampling probability
```

Two rules hold across all four. No stage overwrites an earlier one: corrections
are appended as overlays carrying the original record, the new decision and the
reason. And half-open character offsets are preserved on both sides at every
stage, at a storage cost of two integers per fragment.

`examples/` at the repository root contains one small file per schema.
