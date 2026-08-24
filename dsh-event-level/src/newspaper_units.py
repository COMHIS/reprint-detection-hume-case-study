"""Newspaper article units.

Newspapers do not receive passage reconstruction. Upstream layout analysis has
already delimited each destination identifier as an article, and model-blind
link audits showed that subdividing it further only reproduced the segmentation
already present in the data. The classification unit is therefore fixed as the
relation between one source essay section and one layout-segmented article.

Because one article can generate several rows through different source
manifestations, each record is weighted by the inverse of its destination group
size, so every article contributes equally to the loss. Cross-validation is
grouped by article throughout.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

#: Article-level features. None carries author, edition or newspaper identity.
ARTICLE_FEATURES: tuple[str, ...] = (
    "fragment_count",
    "matched_char_total",
    "matched_char_max",
    "source_section_coverage",
    "source_section_coverage_max",
    "destination_span_chars",
    "destination_coverage_ratio",
    "fragment_density",
    "longest_fragment_chars",
    "mean_fragment_chars",
    "source_order_forward_ratio",
    "distinct_source_manifestations",
    "gap_drift_median_relative",
    "destination_interval_union_chars",
    "source_interval_union_chars",
)


def union_length(intervals: Iterable[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def build_article_candidates(
    fragments: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One candidate per (source essay section, layout-segmented article) pair."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fragments:
        key = (str(row["src_doc_id"]), str(row["src_section_id"]), str(row["dst_doc_id"]))
        grouped[key].append(row)

    candidates: list[dict[str, Any]] = []
    for (src_doc, src_section, dst_doc), rows in grouped.items():
        src_intervals = [(int(r["src_trs_start"]), int(r["src_trs_end"])) for r in rows]
        dst_intervals = [(int(r["dst_trs_start"]), int(r["dst_trs_end"])) for r in rows]
        dst_span = max(e for _, e in dst_intervals) - min(s for s, _ in dst_intervals)
        lengths = [e - s for s, e in dst_intervals]
        section_length = max(int(rows[0].get("src_section_length", 0)), 1)
        forward = sum(
            1
            for a, b in zip(rows, rows[1:])
            if int(b["src_trs_start"]) >= int(a["src_trs_start"])
        )
        candidates.append({
            "candidate_id": f"{src_doc}::{src_section}::{dst_doc}",
            "src_doc_id": src_doc,
            "src_section_id": src_section,
            "dst_doc_id": dst_doc,
            "fragment_count": len(rows),
            "matched_char_total": sum(lengths),
            "matched_char_max": max(lengths),
            "longest_fragment_chars": max(lengths),
            "mean_fragment_chars": sum(lengths) / len(lengths),
            "destination_span_chars": dst_span,
            "destination_interval_union_chars": union_length(dst_intervals),
            "source_interval_union_chars": union_length(src_intervals),
            "destination_coverage_ratio": union_length(dst_intervals) / max(dst_span, 1),
            "fragment_density": len(rows) / max(dst_span, 1),
            "source_section_coverage": union_length(src_intervals) / section_length,
            "source_section_coverage_max": max(lengths) / section_length,
            "source_order_forward_ratio": forward / max(len(rows) - 1, 1),
            "distinct_source_manifestations": len({r["src_doc_id"] for r in rows}),
            "gap_drift_median_relative": float(
                np.median([abs(float(r.get("gap_drift_relative", 0.0))) for r in rows])
            ),
        })
    return sorted(candidates, key=lambda c: (c["dst_doc_id"], c["candidate_id"]))


def destination_balanced_weights(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Inverse destination-group-size weights, so each article counts once."""
    counts = Counter(row["dst_doc_id"] for row in rows)
    return np.asarray([1.0 / counts[row["dst_doc_id"]] for row in rows], dtype=float)


def feature_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row.get(name, 0.0) or 0.0) for name in ARTICLE_FEATURES] for row in rows],
        dtype=float,
    )
