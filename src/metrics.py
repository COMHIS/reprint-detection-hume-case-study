from __future__ import annotations

from typing import Any


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def normalize_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered == "reprint":
        return "reprint"
    if lowered in {"non_reprint", "non-reprint", "non reprint"}:
        return "non_reprint"
    return None


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    positive_gold = negative_gold = 0

    for row in rows:
        gold = normalize_label(row.get("gold_label"))
        pred = normalize_label(row.get("pred_label") or row.get("parsed_label"))
        if gold is None:
            continue
        if gold == "reprint":
            positive_gold += 1
            if pred == "reprint":
                tp += 1
            else:
                fn += 1
        else:
            negative_gold += 1
            if pred == "reprint":
                fp += 1
            else:
                tn += 1

    total = positive_gold + negative_gold
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, total)
    negative_precision = safe_divide(tn, tn + fn)
    negative_recall = safe_divide(tn, tn + fp)
    negative_f1 = safe_divide(
        2 * negative_precision * negative_recall,
        negative_precision + negative_recall,
    )

    return {
        "n_examples": total,
        "n_positive_gold": positive_gold,
        "n_negative_gold": negative_gold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "macro_f1": (f1 + negative_f1) / 2 if total else 0.0,
    }


def group_by_split(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "discovery_set": [],
        "main_eval_set": [],
        "hard_eval_set": [],
        "all_labeled": [],
    }
    for row in rows:
        if normalize_label(row.get("gold_label")) is None:
            continue
        grouped["all_labeled"].append(row)
        for tag in row.get("split_tags", []):
            if tag in grouped:
                grouped[tag].append(row)
    return grouped

