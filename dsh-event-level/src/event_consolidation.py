"""Destination-side event consolidation.

Consolidation compresses evidence that repeatedly supports one destination
location into a container a scholar can review. Book passages inside one
destination are joined into transitive components over intersecting or touching
destination intervals; newspapers organise overlapping evidence by position
inside the layout article.

The process never merges non-overlapping positions on the basis of source
title, edition identity or a permitted gap. Where several source manifestations
hit one position they are attached as witnesses of a single candidate. Where
review finds that one container holds several independent positions, split
children are generated that point back to the parent. Only a record finally
reviewed as `reprint` is a historical republication event.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

GROUPING_POLICY = (
    "destination intervals intersect or touch; "
    "no positive gap threshold; no title, edition or family merge"
)


def overlap_components(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Transitive components over intersecting or touching destination intervals."""
    ordered = sorted(rows, key=lambda r: (int(r["dst_start"]), int(r["dst_end"])))
    components: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_end: int | None = None
    for row in ordered:
        if current_end is None or int(row["dst_start"]) > current_end:
            if current:
                components.append(current)
            current = [row]
            current_end = int(row["dst_end"])
        else:
            current.append(row)
            current_end = max(current_end, int(row["dst_end"]))
    if current:
        components.append(current)
    return components


def consolidate(
    passages: Iterable[dict[str, Any]],
    negative_evidence: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Build event candidates per destination.

    Existing human *non*-republication evidence overlapping a component is
    attached as a conflict rather than used as a seed or silently dropped.
    """
    by_destination: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in passages:
        by_destination[str(row["dst_doc_id"])].append(row)

    negatives_by_destination: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in negative_evidence:
        negatives_by_destination[str(row["dst_doc_id"])].append(row)

    candidates: list[dict[str, Any]] = []
    for destination, rows in sorted(by_destination.items()):
        for index, component in enumerate(overlap_components(rows), start=1):
            start = min(int(r["dst_start"]) for r in component)
            end = max(int(r["dst_end"]) for r in component)
            conflicts = [
                n for n in negatives_by_destination.get(destination, [])
                if int(n["dst_start"]) < end and int(n["dst_end"]) > start
            ]
            candidates.append({
                "event_candidate_id": f"{destination}::overlap::{index:04d}::{start}-{end}",
                "dst_doc_id": destination,
                "dst_start": start,
                "dst_end": end,
                "passage_candidate_ids": [r["passage_candidate_id"] for r in component],
                "passage_count": len(component),
                "fragment_count": sum(int(r.get("fragment_count", 0)) for r in component),
                "source_manifestations": sorted({r["src_doc_id"] for r in component}),
                "source_sections": sorted(
                    {str(r.get("src_section_id")) for r in component if r.get("src_section_id")}
                ),
                "conflicting_negative_ids": [n["passage_candidate_id"] for n in conflicts],
                "grouping_policy": GROUPING_POLICY,
                "review_status": "awaiting_human_review",
            })
    return candidates


def split_child(parent: dict[str, Any], start: int, end: int, reason: str) -> dict[str, Any]:
    """An overlay record: corrections append, they never overwrite the parent."""
    return {
        "event_candidate_id": f"{parent['event_candidate_id']}::child::{start}-{end}",
        "parent_event_candidate_id": parent["event_candidate_id"],
        "dst_doc_id": parent["dst_doc_id"],
        "dst_start": start,
        "dst_end": end,
        "correction_reason": reason,
        "original_span": [parent["dst_start"], parent["dst_end"]],
    }


def confirmed_events(
    candidates: Iterable[dict[str, Any]], reviews: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Only records a reader confirmed as `reprint` from page images become events."""
    decision = {r["event_candidate_id"]: r for r in reviews}
    events = []
    for candidate in candidates:
        review = decision.get(candidate["event_candidate_id"])
        if review is not None and review.get("decision") == "reprint":
            events.append({**candidate, "review_status": "confirmed_reprint"})
    return events
