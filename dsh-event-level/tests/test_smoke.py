#!/usr/bin/env python3
"""End-to-end smoke test over the example files.

Run from the repository root:

    python3 tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src import event_consolidation, newspaper_units, passage_links  # noqa: E402
from src.evidence_tests import (  # noqa: E402
    compiled_units, coverage_audit, cross_media, null_model, source_ablation,
)
from src.io_utils import read_json, read_jsonl  # noqa: E402
from src.label_registry import (  # noqa: E402
    LabelKind, LabelPolicyError, assert_disjoint, evaluation_rows,
    normalise_label, training_rows,
)
from src.metrics import weighted_confusion, wilson_interval  # noqa: E402

EXAMPLES = REPO_ROOT / "examples"
checks = 0


def check(condition: bool, message: str) -> None:
    global checks
    if not condition:
        raise AssertionError(message)
    checks += 1
    print(f"  ok  {message}")


def main() -> None:
    print("passage reconstruction")
    fragments = read_jsonl(EXAMPLES / "fragments_example.jsonl")
    for row in fragments:
        row.setdefault("fragment_id", passage_links.fragment_id(row, "example_v1"))
    edges = passage_links.build_edges(fragments)
    check(len(edges) == len(fragments) - len({f["section_candidate_id"] for f in fragments}),
          "k fragments per candidate give k-1 edges")
    decisions = passage_links.link_edges(edges, lambda rows: np.zeros(len(rows)))
    passages = passage_links.passage_components(fragments, decisions)
    check(sum(p["fragment_count"] for p in passages) == len(fragments),
          "no fragment is dropped by passage reconstruction")

    print("newspaper units")
    news = read_jsonl(EXAMPLES / "newspaper_fragments_example.jsonl")
    candidates = newspaper_units.build_article_candidates(news)
    weights = newspaper_units.destination_balanced_weights(candidates)
    check(abs(weights.sum() - len({c["dst_doc_id"] for c in candidates})) < 1e-9,
          "destination-balanced weights sum to the article count")

    print("label registry")
    records = read_jsonl(EXAMPLES / "annotation_example.jsonl")
    check(normalise_label("Attrbuted Reprint") == "reprint",
          "the historical annotation typo maps to reprint")
    check(len(training_rows(records)) == 1, "one adaptive training record in the example")
    check(len(evaluation_rows(records)) == 1, "sealed evaluation rows carry a probability")
    try:
        evaluation_rows([{"label_kind": LabelKind.SEALED_EVALUATION}])
    except LabelPolicyError:
        check(True, "a sealed row without sampling_probability is refused")
    try:
        assert_disjoint([{"dst_doc_id": "X"}], [{"dst_doc_id": "X"}])
    except LabelPolicyError:
        check(True, "a destination shared by training and evaluation is refused")

    print("event consolidation")
    events = event_consolidation.consolidate(passages)
    check(all(e["review_status"] == "awaiting_human_review" for e in events),
          "consolidation produces review containers, not events")

    print("evidence tests")
    synthetic = [{"year": 1750 + i % 40, "work": f"w{i % 12}"} for i in range(240)]
    result = null_model.run(synthetic, iterations=200, random_seed=0)
    check("null_percentile_of_observed" in result,
          "the null model returns a percentile for the observed statistic")

    by_destination: dict[str, list] = {}
    for row in passages:
        by_destination.setdefault(str(row["dst_doc_id"]), []).append(row)
    blind = compiled_units.geometry_blind_destinations(by_destination)
    check(isinstance(blind, list), "the geometric instrument reports its own blind set")
    shape = compiled_units.grouping_shape({"a": "970-essays", "b": "X-elegant extracts"})
    check(abs(shape["title_shaped_share"] - 0.5) < 1e-9,
          "title-shaped grouping values are counted, not explained")

    ablation = source_ablation.recovery_by_manifestation(
        [{"src_section_id": "3", "source_manifestations": ["m1"]}],
        ["m1", "m2"], {"m1": {"3"}, "m2": set()},
    )
    check(ablation[0]["conditional_recovery"] == 1.0,
          "conditional recovery separates absent content from lost digitisation")

    audits = read_json(EXAMPLES / "coverage_audits_example.json")
    pooled = coverage_audit.pool_audits(audits, frame_destinations=2190)
    check(pooled["draws"] == 80 and pooled["positives"] == 1,
          "destination-disjoint audits pool to 80 draws and 1 positive")
    low, high = (round(v) for v in pooled["estimated_missed_95"])
    check((low, high) == (5, 148),
          "the recall bound reproduces the 5-148 interval reported in the article")
    try:
        coverage_audit.pool_audits(
            [{"name": "a", "draws": 1, "positives": 0, "destinations": ["D"]},
             {"name": "b", "draws": 1, "positives": 0, "destinations": ["D"]}], 10)
    except ValueError:
        check(True, "audits sharing a destination are refused")

    low, high = wilson_interval(1, 80)
    check(low < 0.0125 < high, "the Wilson interval brackets the observed rate")
    confusion = weighted_confusion([1, 0], [0.9, 0.9], [1.0, 100.0])
    check(confusion["estimated_frame_size"] == 101.0,
          "inverse-probability weights reconstruct the frame size")
    check(len(cross_media.SHARED_CORE_FEATURES) == 15,
          "the shared core carries fifteen identity-free features")

    print(f"\n{checks} checks passed")


if __name__ == "__main__":
    main()
