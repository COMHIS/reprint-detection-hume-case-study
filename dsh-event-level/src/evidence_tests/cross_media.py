"""Transfer between archives.

Both media are given a shared representation carrying no author, edition,
newspaper or collection identity, so a boundary fitted in one archive can be
evaluated in the other. The result reported in the article is that a
book-fitted boundary recovers far less in the newspaper frame than a boundary
fitted there, which is read as a property of the archives rather than a
deficiency of one model.

The evaluation frame is a permanent holdout whose stratum sampling
probabilities are recorded, so weighted metrics reconstruct the frame.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..metrics import weighted_confusion

#: Identity-free features shared by book passages and newspaper articles.
SHARED_CORE_FEATURES: tuple[str, ...] = (
    "fragment_count",
    "matched_char_total",
    "matched_char_max",
    "longest_fragment_chars",
    "mean_fragment_chars",
    "source_section_coverage",
    "source_section_coverage_max",
    "destination_span_chars",
    "destination_interval_union_chars",
    "source_interval_union_chars",
    "destination_coverage_ratio",
    "fragment_density",
    "source_order_forward_ratio",
    "distinct_source_manifestations",
    "gap_drift_median_relative",
)


def shared_core_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(r.get(name, 0.0) or 0.0) for name in SHARED_CORE_FEATURES] for r in rows],
        dtype=float,
    )


def transfer_matrix(
    fitted: dict[str, Callable[[np.ndarray], np.ndarray]],
    holdout: Sequence[dict[str, Any]],
    labels: Sequence[int],
    weights: Sequence[float],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Evaluate every fitted boundary on one weighted permanent holdout."""
    features = shared_core_matrix(holdout)
    rows: list[dict[str, Any]] = []
    for name, predict in fitted.items():
        scores = predict(features)
        rows.append({
            "fitted_on": name,
            **weighted_confusion(labels, scores, weights, threshold=threshold),
        })
    return rows


def gate(
    result: dict[str, Any],
    min_accuracy: float = 0.85,
    min_precision: float = 0.85,
    min_recall: float = 0.85,
) -> dict[str, Any]:
    """A pass mark set before the holdout is opened, not searched for afterwards."""
    precision = result.get("weighted_precision", float("nan"))
    recall = result.get("weighted_recall", float("nan"))
    tp, fp, fn, tn = (
        result["weighted_tp"], result["weighted_fp"], result["weighted_fn"], result["weighted_tn"],
    )
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    passed = (
        accuracy >= min_accuracy and precision >= min_precision and recall >= min_recall
    )
    return {
        "weighted_accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "gate_passed": bool(passed),
        "note": (
            "Searching thresholds on the same holdout after the fact would not be "
            "independent evidence, so the gate is evaluated once."
        ),
    }
