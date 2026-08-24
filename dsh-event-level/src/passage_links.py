"""Passage reconstruction inside book destinations.

An essay-to-document candidate may span a whole volume while the historical
judgement applies to one or a few local positions inside it. Rather than
merging fragments with a fixed character gap, which mistakes differences of
scale caused by OCR, typography and text length for a stable boundary, every
raw fragment is retained and only *adjacent* fragments are judged.

Fragments inside a candidate are sorted by destination offset, so a candidate
with k fragments yields k-1 adjacency edges. Edges whose destination intervals
overlap are linked deterministically; the rest are decided by a supervised
model. No fragment is ever deleted -- the model changes component membership
only.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .io_utils import stable_id

#: Nine edge features. Each ratio is divided by a piece scale derived from the
#: two fragments themselves, so no absolute character threshold is imposed on
#: texts whose OCR density and page layout differ.
EDGE_FEATURES: tuple[str, ...] = (
    "destination_intervals_overlap",
    "source_order_is_forward",
    "src_gap_relative",
    "dst_gap_relative",
    "gap_drift_relative",
    "source_backward_relative",
    "log_src_positive_gap",
    "log_dst_positive_gap",
    "log_local_piece_scale",
)


def fragment_id(row: dict[str, Any], corpus_version: str) -> str:
    return stable_id(
        corpus_version,
        row["src_doc_id"], row["src_trs_id"], row["src_trs_start"], row["src_trs_end"],
        row["dst_doc_id"], row["dst_trs_id"], row["dst_trs_start"], row["dst_trs_end"],
        prefix="fr_",
    )


def edge_features(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Features for one adjacency edge between two destination-ordered fragments."""
    src_signed_gap = int(right["src_trs_start"]) - int(left["src_trs_end"])
    dst_signed_gap = int(right["dst_trs_start"]) - int(left["dst_trs_end"])
    src_gap = max(src_signed_gap, 0)
    dst_gap = max(dst_signed_gap, 0)
    source_backward = max(int(left["src_trs_start"]) - int(right["src_trs_start"]), 0)
    local_scale = max(
        (
            min(int(left["src_piece_length"]), int(left["dst_piece_length"]))
            + min(int(right["src_piece_length"]), int(right["dst_piece_length"]))
        ) / 2,
        1.0,
    )
    gap_drift = abs(dst_gap - src_gap)
    return {
        "src_signed_gap": src_signed_gap,
        "dst_signed_gap": dst_signed_gap,
        "src_positive_gap": src_gap,
        "dst_positive_gap": dst_gap,
        "gap_drift": gap_drift,
        "source_backward_chars": source_backward,
        "local_piece_scale": local_scale,
        "src_gap_relative": src_gap / local_scale,
        "dst_gap_relative": dst_gap / local_scale,
        "gap_drift_relative": gap_drift / local_scale,
        "source_backward_relative": source_backward / local_scale,
        "destination_intervals_overlap": dst_signed_gap <= 0,
        "source_order_is_forward": int(right["src_trs_start"]) >= int(left["src_trs_start"]),
    }


def build_edges(fragments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One edge per adjacent pair within each section candidate.

    Sorting by destination offset means k fragments give k-1 edges rather than
    an all-to-all comparison.
    """
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fragments:
        by_candidate[str(row["section_candidate_id"])].append(row)

    edges: list[dict[str, Any]] = []
    for candidate_id, rows in by_candidate.items():
        ordered = sorted(
            rows, key=lambda r: (int(r["dst_trs_start"]), int(r["dst_trs_end"]), r["fragment_id"])
        )
        for left, right in zip(ordered, ordered[1:]):
            features = edge_features(left, right)
            edges.append({
                "edge_id": stable_id(left["fragment_id"], right["fragment_id"], prefix="ed_"),
                "section_candidate_id": candidate_id,
                "dst_doc_id": left["dst_doc_id"],
                "left_fragment_id": left["fragment_id"],
                "right_fragment_id": right["fragment_id"],
                **features,
            })
    return edges


def _derived(row: dict[str, Any], name: str) -> float:
    if name == "log_src_positive_gap":
        return math.log1p(max(int(row["src_positive_gap"]), 0))
    if name == "log_dst_positive_gap":
        return math.log1p(max(int(row["dst_positive_gap"]), 0))
    if name == "log_local_piece_scale":
        return math.log1p(max(float(row["local_piece_scale"]), 0.0))
    value = row.get(name)
    if isinstance(value, bool):
        return float(value)
    if value is None:
        return float("nan")
    return float(value)


def feature_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[_derived(row, name) for name in EDGE_FEATURES] for row in rows], dtype=float
    )


def model_builders(random_seed: int) -> dict[str, Callable[[], Pipeline]]:
    """The three candidates compared under destination-grouped cross-validation.

    The comparison that matters is with the rule these replaced: a fixed
    maximum gap with drift and bridge parameters.
    """
    return {
        "link_logistic": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=3000, random_state=random_seed,
            )),
        ]),
        "link_tree": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", DecisionTreeClassifier(
                class_weight="balanced", max_depth=3, min_samples_leaf=5,
                random_state=random_seed,
            )),
        ]),
        "link_random_forest": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", max_depth=5,
                min_samples_leaf=3, random_state=random_seed, n_jobs=1,
            )),
        ]),
    }


def legacy_gap_rule(
    edge: dict[str, Any],
    max_gap_chars: int = 2000,
    max_drift_chars: int = 1500,
) -> bool:
    """The fixed-threshold rule the supervised model replaced.

    Kept so the comparison in the article can be reproduced, not because it is
    recommended.
    """
    if bool(edge["destination_intervals_overlap"]):
        return True
    return (
        int(edge["dst_positive_gap"]) <= max_gap_chars
        and int(edge["gap_drift"]) <= max_drift_chars
        and bool(edge["source_order_is_forward"])
    )


def link_edges(
    edges: Sequence[dict[str, Any]],
    predict_proba: Callable[[Sequence[dict[str, Any]]], np.ndarray],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Overlapping destination intervals link deterministically; the rest are modelled."""
    decisions: list[dict[str, Any]] = []
    modelled = [e for e in edges if not bool(e["destination_intervals_overlap"])]
    scores = predict_proba(modelled) if modelled else np.empty(0)
    score_by_edge = {e["edge_id"]: float(s) for e, s in zip(modelled, scores)}
    for edge in edges:
        overlapping = bool(edge["destination_intervals_overlap"])
        score = None if overlapping else score_by_edge[edge["edge_id"]]
        decisions.append({
            "edge_id": edge["edge_id"],
            "section_candidate_id": edge["section_candidate_id"],
            "dst_doc_id": edge["dst_doc_id"],
            "left_fragment_id": edge["left_fragment_id"],
            "right_fragment_id": edge["right_fragment_id"],
            "decision_source": "destination_overlap" if overlapping else "supervised_model",
            "link_probability": score,
            "same_passage": True if overlapping else score >= threshold,
        })
    return decisions


def passage_components(
    fragments: Sequence[dict[str, Any]], decisions: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Group fragments into local passages by following linked adjacency edges.

    Membership only: every fragment reaches exactly one passage, and none is
    dropped.
    """
    parent: dict[str, str] = {row["fragment_id"]: row["fragment_id"] for row in fragments}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for decision in decisions:
        if decision["same_passage"]:
            union(decision["left_fragment_id"], decision["right_fragment_id"])

    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fragments:
        members[find(row["fragment_id"])].append(row)

    passages: list[dict[str, Any]] = []
    for root, rows in members.items():
        ordered = sorted(rows, key=lambda r: int(r["dst_trs_start"]))
        passages.append({
            "passage_candidate_id": stable_id(root, prefix="pc_"),
            "section_candidate_id": ordered[0]["section_candidate_id"],
            "src_doc_id": ordered[0]["src_doc_id"],
            "src_section_id": ordered[0].get("src_section_id"),
            "dst_doc_id": ordered[0]["dst_doc_id"],
            "dst_start": min(int(r["dst_trs_start"]) for r in ordered),
            "dst_end": max(int(r["dst_trs_end"]) for r in ordered),
            "fragment_ids": [r["fragment_id"] for r in ordered],
            "fragment_count": len(ordered),
        })
    return sorted(passages, key=lambda p: (p["dst_doc_id"], p["dst_start"]))
