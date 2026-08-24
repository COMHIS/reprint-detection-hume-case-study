# Dataset and File Schemas

This code release separates reusable workflow logic from restricted source
materials. Raw ECCO and Burney newspaper text is not distributed here.

## Pair-Feature Table

Format: JSONL, one source-target pair per line.

Required identifiers:

- `pair_id`: stable pair identifier, preferably `src_doc_id::dst_doc_id`;
- `src_doc_id`: source document identifier;
- `dst_doc_id`: target document identifier.

Required numeric features for the final workflow:

- `max_bundle_span`: maximum span of a consolidated evidence bundle;
- `max_coverage`: maximum source-section coverage;
- `sum_coverage`: cumulative source-section coverage;
- `section_count`: number of source sections represented by the pair evidence;
- `dst_strong_section_fanout`: number of strong source-section links for the target;
- `max_chain_score`: score of the strongest locally chained evidence bundle;
- `max_bundle_fragment_count`: number of fragments in the strongest bundle;
- `top_section_gap`: separation between the strongest and next strongest section evidence.

Required boolean cues:

- `title_hit`;
- `title_prefix_hit`;
- `heading_genre_cue`;
- `heading_match_cue`;
- `quotation_cue`;
- `paratext_cue`.

Optional labels and split metadata:

- `gold_label`: `reprint`, `non_reprint`, or null;
- `split_tags`: list containing zero or more of `discovery_set`,
  `main_eval_set`, and `hard_eval_set`.

Optional deployment-ranking features:

- `dst_score_rank`;
- `dst_score_margin_to_next`.

## Prediction File

Format: JSONL, one prediction per line.

Required fields:

- `pair_id`;
- `src_doc_id`;
- `dst_doc_id`;
- `pred_label`: `reprint` or `non_reprint`;
- `method_name`.

Evaluation additionally requires:

- `gold_label`;
- `split_tags`.

## Newspaper Audit File

Format: JSONL, one audited pair per line.

Required fields:

- `pair_id`;
- `pred_label`: workflow prediction;
- `annotation_label`: manual audit label, normally `reprint`,
  `non_reprint`, or `uncertain`.

Optional fields:

- `mentions_hume`: boolean or yes/no string;
- `annotation_reason`: short reason code or explanation;
- `audit_subset`: source of the audited case, for example
  `predicted_positive`, `near_boundary_negative`, or `disagreement_negative`.

## Data Not Included

The following materials are excluded from this public code package:

- raw ECCO OCR;
- raw Burney Newspapers Collection OCR;
- long source or destination text fragments copied from restricted corpora;
- local annotation workspaces containing copyrighted passages;
- machine-specific paths and private run logs;
- local cluster job scripts.
