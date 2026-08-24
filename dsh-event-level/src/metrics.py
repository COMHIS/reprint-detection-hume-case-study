"""Evaluation under grouped splits and known sampling probabilities.

Two rules run through this module. Cross-validation is grouped by destination,
so near-duplicate witnesses of one passage cannot fall on both sides of a
split. And any set drawn by model-directed or stratified sampling is summarised
with inverse-probability weights, because unweighted statistics on such a set
describe the queried region rather than a population.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


def grouped_cross_validation(
    build_model: Callable[[], Any],
    features: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[Any],
    n_splits: int = 5,
    random_seed: int = 0,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Out-of-fold metrics with each destination confined to one fold."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof = np.full(len(labels), np.nan)
    folds: list[dict[str, float]] = []
    for index, (train_idx, test_idx) in enumerate(
        splitter.split(features, labels, groups=groups), start=1
    ):
        model = build_model()
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["model__sample_weight"] = sample_weight[train_idx]
        model.fit(features[train_idx], labels[train_idx], **fit_kwargs)
        scores = model.predict_proba(features[test_idx])[:, 1]
        oof[test_idx] = scores
        folds.append({
            "fold": index,
            "pr_auc": float(average_precision_score(labels[test_idx], scores)),
            "roc_auc": float(roc_auc_score(labels[test_idx], scores)),
            "f1_at_0_5": float(f1_score(labels[test_idx], scores >= 0.5, zero_division=0)),
        })
    predicted = oof >= 0.5
    return {
        "folds": folds,
        "out_of_fold": {
            "pr_auc": float(average_precision_score(labels, oof)),
            "roc_auc": float(roc_auc_score(labels, oof)),
            "balanced_accuracy_at_0_5": float(balanced_accuracy_score(labels, predicted)),
            "precision_at_0_5": float(precision_score(labels, predicted, zero_division=0)),
            "recall_at_0_5": float(recall_score(labels, predicted, zero_division=0)),
            "f1_at_0_5": float(f1_score(labels, predicted, zero_division=0)),
        },
        "note": (
            "These metrics describe the labelled distribution. Where the labels were "
            "acquired by model-directed sampling they are not an unbiased estimate of "
            "open-world performance."
        ),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, used for small audit samples."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = p + z * z / (2 * trials)
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return ((centre - spread) / denominator, (centre + spread) / denominator)


def weighted_confusion(
    labels: Sequence[int], scores: Sequence[float], weights: Sequence[float], threshold: float = 0.5
) -> dict[str, float]:
    """Inverse-probability weighted confusion counts, reconstructing the frame."""
    labels = np.asarray(labels)
    predicted = np.asarray(scores) >= threshold
    weights = np.asarray(weights, dtype=float)
    tp = float(weights[(predicted) & (labels == 1)].sum())
    fp = float(weights[(predicted) & (labels == 0)].sum())
    fn = float(weights[(~predicted) & (labels == 1)].sum())
    tn = float(weights[(~predicted) & (labels == 0)].sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "weighted_tp": tp, "weighted_fp": fp, "weighted_fn": fn, "weighted_tn": tn,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "estimated_frame_size": float(weights.sum()),
    }
