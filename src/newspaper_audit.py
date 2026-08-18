from __future__ import annotations

from typing import Any

from .metrics import safe_divide


def normalized_audit_label(value: Any) -> str:
    if not isinstance(value, str):
        return "uncertain"
    lowered = value.strip().lower()
    if lowered in {"reprint", "reuse", "true_positive", "positive"}:
        return "reprint"
    if lowered in {"non_reprint", "non-reprint", "non reprint", "negative"}:
        return "non_reprint"
    return "uncertain"


def normalized_prediction(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered == "reprint":
        return "reprint"
    if lowered in {"non_reprint", "non-reprint", "non reprint"}:
        return "non_reprint"
    return None


def mentions_hume(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"yes", "y", "true", "1"}:
            return True
        if lowered in {"no", "n", "false", "0"}:
            return False
    return None


def summarize_newspaper_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicted_positive = 0
    predicted_negative = 0
    resolved_positive = 0
    resolved_negative = 0
    uncertain = 0
    positive_mentions_hume = 0
    positive_no_hume_mention = 0

    tp = fp = tn = fn = 0
    for row in rows:
        pred = normalized_prediction(row.get("pred_label"))
        audit = normalized_audit_label(row.get("annotation_label"))

        if pred == "reprint":
            predicted_positive += 1
        elif pred == "non_reprint":
            predicted_negative += 1

        if audit == "uncertain":
            uncertain += 1
            continue
        if audit == "reprint":
            resolved_positive += 1
        else:
            resolved_negative += 1

        if pred == "reprint" and audit == "reprint":
            tp += 1
        elif pred == "reprint" and audit == "non_reprint":
            fp += 1
        elif pred == "non_reprint" and audit == "non_reprint":
            tn += 1
        elif pred == "non_reprint" and audit == "reprint":
            fn += 1

        if pred == "reprint" and audit == "reprint":
            mention = mentions_hume(row.get("mentions_hume"))
            if mention is True:
                positive_mentions_hume += 1
            elif mention is False:
                positive_no_hume_mention += 1

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "n_rows": len(rows),
        "predicted_positive": predicted_positive,
        "predicted_negative": predicted_negative,
        "resolved_positive": resolved_positive,
        "resolved_negative": resolved_negative,
        "uncertain": uncertain,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_on_resolved": precision,
        "recall_on_resolved": recall,
        "f1_on_resolved": f1,
        "confirmed_positive_mentions_hume": positive_mentions_hume,
        "confirmed_positive_without_hume_mention": positive_no_hume_mention,
    }

