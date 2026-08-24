# Schemas

Raw ECCO and Gale newspaper text is not distributed. These are the schemas the
code expects, so that the pipeline can be rebuilt inside a licensed
environment. One example file per schema is in `examples/`.

## Fragment

JSONL, one local alignment per line. A fragment is never a pre-merged bundle.
Half-open character offsets are kept on both sides at every stage, so any later
passage or event can return to the original match.

- `src_doc_id`, `src_section_id`, `src_trs_id`, `src_trs_start`, `src_trs_end`
- `dst_doc_id`, `dst_trs_id`, `dst_trs_start`, `dst_trs_end`
- `src_piece_length`, `dst_piece_length` — used only to derive a local scale
- `section_candidate_id` — `src_doc_id::src_section_id::dst_doc_id`
- `src_section_length` — denominator for source coverage
- `fragment_id` — assigned by the pipeline if absent

## Adjacency edge

One per adjacent fragment pair inside a section candidate, ordered by
destination offset, so k fragments give k−1 edges.

- `edge_id`, `section_candidate_id`, `dst_doc_id`
- `left_fragment_id`, `right_fragment_id`
- `destination_intervals_overlap`, `source_order_is_forward`
- `src_positive_gap`, `dst_positive_gap`, `gap_drift`, `source_backward_chars`
- `local_piece_scale` and the four `*_relative` ratios derived from it

Every ratio is divided by a scale derived from the two fragments themselves, so
no absolute character threshold is imposed on texts whose OCR density and page
layout differ.

## Edge label

From a model-blind link audit. Only two values.

- `edge_id`, `edge_label` ∈ {`same_passage`, `split_passage`}
- `audit`, `annotator_confidence`

## Passage candidate

A local component inside one section candidate. Books only: newspapers keep the
layout-segmented article as the unit.

- `passage_candidate_id`, `section_candidate_id`
- `src_doc_id`, `src_section_id`, `dst_doc_id`
- `dst_start`, `dst_end`, `fragment_ids`, `fragment_count`

## Newspaper article candidate

One relation between a source essay section and one layout-segmented article.
Features carry no author, edition or newspaper identity.

- `candidate_id` — `src_doc_id::src_section_id::dst_doc_id`
- the fifteen fields in `newspaper_units.ARTICLE_FEATURES`

## Annotation record

Every record states the kind of label it is, at acquisition. The three kinds
cannot substitute for one another and the difference is not recoverable
afterwards.

- `record_id`, `label_kind` ∈ {`adaptive_training`, `sealed_evaluation`,
  `final_event_review`}, `label_source`
- `annotation_label` ∈ {`reprint`, `non_reprint`} — records that cannot be
  decided in the first pass go to expert adjudication rather than a third class
- `reprint_scope` — single-choice description of what most supported a positive
- `non_reprint_reason` — single-choice description of what most supported a
  negative; not a multi-label coding, so it cannot be read as an incidence rate
- `confidence`, `all_fragments_reviewed`
- sealed evaluation additionally requires `sampling_stratum` and
  `sampling_probability`, without which weighted metrics cannot reconstruct
  the frame
- final review additionally carries `decision` and `evidence_seen`

## Event candidate

A review container built from overlapping destination evidence. Not a
historical event.

- `event_candidate_id`, `dst_doc_id`, `dst_start`, `dst_end`
- `passage_candidate_ids`, `passage_count`, `fragment_count`
- `source_manifestations`, `source_sections`
- `conflicting_negative_ids` — existing human non-republication evidence that
  overlaps the component, attached rather than dropped
- `grouping_policy`, `review_status`

A split child carries `parent_event_candidate_id`, `correction_reason` and
`original_span`. Corrections append; the parent record is preserved.

## Confirmed event

An event candidate, or a split child, that a reader confirmed as `reprint` from
page images, OCR, all fragments and bibliographical context. For the evidence
tests it also carries `year`, `work` and `src_section_id`.

## Probability audit

Input to the pooled recall bound. Two audits combine only if they draw from the
same frame at recorded probabilities and their destinations are disjoint.

- `name`, `draws`, `positives`, `destinations`, `design`
