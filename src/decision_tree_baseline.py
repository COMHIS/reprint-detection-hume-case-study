from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .workflow_rules import number


BASIC_FEATURES = [
    "hit_count",
    "max_src_piece_length",
    "max_dst_piece_length",
    "max_combined_piece_length",
    "sum_combined_piece_length",
    "mean_combined_piece_length",
    "src_reuse_span",
    "dst_reuse_span",
    "src_fragment_dispersion",
    "dst_fragment_dispersion",
    "segment_count",
    "section_count",
    "sum_coverage",
    "max_coverage",
    "mean_coverage",
    "max_bundle_span",
    "sum_top_bundle_spans",
    "top_section_coverage",
    "second_section_coverage",
    "top_section_gap",
    "max_chain_score",
    "max_bundle_fragment_count",
    "multi_fragment_bundle_count",
    "mean_bundle_fragment_count",
    "max_total_internal_gap",
    "dst_strong_section_fanout",
]

EXTENDED_FEATURES = BASIC_FEATURES + [
    "strongest_fragment_strength",
    "quotation_cue",
    "paratext_cue",
    "by_hume_cue",
    "heading_genre_cue",
    "heading_match_cue",
    "structural_title_overlap",
    "content_section_start_cue",
    "title_hit",
    "title_prefix_hit",
    "src_section_is_content_like",
    "dst_global_source_count",
    "dst_global_pair_count",
    "dst_global_hit_count",
    "dst_score_rank",
    "dst_score_margin_to_next",
    "dst_score_gap_to_top",
]


@dataclass
class TreeNode:
    positive_count: int
    negative_count: int
    probability: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.left is not None:
            payload["left"] = self.left.to_dict()
        if self.right is not None:
            payload["right"] = self.right.to_dict()
        return payload


def feature_matrix(rows: Sequence[dict[str, Any]], feature_names: Sequence[str]) -> list[list[float]]:
    return [[number(row, feature) for feature in feature_names] for row in rows]


def label_vector(rows: Sequence[dict[str, Any]]) -> list[int]:
    return [1 if row["gold_label"] == "reprint" else 0 for row in rows]


def gini(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    positives = sum(labels)
    p_pos = positives / len(labels)
    p_neg = 1.0 - p_pos
    return 1.0 - p_pos * p_pos - p_neg * p_neg


def candidate_thresholds(values: Sequence[float]) -> list[float]:
    unique_values = sorted(set(values))
    return [
        (unique_values[index] + unique_values[index + 1]) / 2.0
        for index in range(len(unique_values) - 1)
    ]


def best_split(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    min_leaf_size: int,
) -> tuple[int | None, float | None, float]:
    if not rows:
        return None, None, 0.0
    parent_impurity = gini(labels)
    best_feature = None
    best_threshold = None
    best_gain = 0.0

    for feature_index in range(len(rows[0])):
        values = [row[feature_index] for row in rows]
        for threshold in candidate_thresholds(values):
            left = [label for row, label in zip(rows, labels) if row[feature_index] <= threshold]
            right = [label for row, label in zip(rows, labels) if row[feature_index] > threshold]
            if len(left) < min_leaf_size or len(right) < min_leaf_size:
                continue
            weighted = len(left) / len(labels) * gini(left) + len(right) / len(labels) * gini(right)
            gain = parent_impurity - weighted
            if gain > best_gain:
                best_feature = feature_index
                best_threshold = threshold
                best_gain = gain
    return best_feature, best_threshold, best_gain


def train_tree(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    depth: int = 0,
    max_depth: int = 4,
    min_leaf_size: int = 4,
    min_gain: float = 1e-4,
) -> TreeNode:
    positives = sum(labels)
    negatives = len(labels) - positives
    probability = positives / len(labels) if labels else 0.0
    node = TreeNode(positives, negatives, probability)

    if (
        depth >= max_depth
        or len(labels) < 2 * min_leaf_size
        or positives == 0
        or negatives == 0
    ):
        return node

    feature_index, threshold, gain = best_split(rows, labels, min_leaf_size)
    if feature_index is None or threshold is None or gain < min_gain:
        return node

    left_rows: list[Sequence[float]] = []
    left_labels: list[int] = []
    right_rows: list[Sequence[float]] = []
    right_labels: list[int] = []
    for row, label in zip(rows, labels):
        if row[feature_index] <= threshold:
            left_rows.append(row)
            left_labels.append(label)
        else:
            right_rows.append(row)
            right_labels.append(label)

    node.feature_index = feature_index
    node.threshold = threshold
    node.left = train_tree(left_rows, left_labels, depth + 1, max_depth, min_leaf_size, min_gain)
    node.right = train_tree(right_rows, right_labels, depth + 1, max_depth, min_leaf_size, min_gain)
    return node


def predict_probability(node: TreeNode, row: Sequence[float]) -> float:
    current = node
    while not current.is_leaf:
        if current.feature_index is None or current.threshold is None:
            break
        if row[current.feature_index] <= current.threshold:
            current = current.left or current
        else:
            current = current.right or current
    return current.probability

