"""Three kinds of human label that cannot substitute for one another.

Adaptive training labels may be fed back into the current model. Sealed
evaluation labels are drawn after a model is frozen, with sampling stratum and
probability recorded. Final event review is a historical decision and is not a
proxy for model accuracy.

The difference is invisible once the three are in the same table, and
reconstructing it afterwards is guesswork, so the kind is recorded at
acquisition and enforced here.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Sequence


class LabelKind(str, Enum):
    ADAPTIVE_TRAINING = "adaptive_training"
    SEALED_EVALUATION = "sealed_evaluation"
    FINAL_EVENT_REVIEW = "final_event_review"


class LabelPolicyError(RuntimeError):
    """Raised when labels are used for something their acquisition does not license."""


#: Only two core labels. Records that cannot be decided in the first pass go to
#: expert adjudication rather than into a third class.
BINARY_LABELS = ("reprint", "non_reprint")


def normalise_label(raw: str) -> str:
    value = str(raw).strip().lower().replace(" ", "_")
    mapping = {
        "attributed_reprint": "reprint",
        "unattributed_reprint": "reprint",
        "attrbuted_reprint": "reprint",  # historical typo in the annotation tool
        "reprint": "reprint",
    }
    if value in mapping:
        return "reprint"
    return "non_reprint"


def training_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only adaptive training labels may be fitted on."""
    return [r for r in rows if r.get("label_kind") == LabelKind.ADAPTIVE_TRAINING]


def evaluation_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sealed evaluation labels, which must carry their sampling probability."""
    selected = [r for r in rows if r.get("label_kind") == LabelKind.SEALED_EVALUATION]
    missing = [r for r in selected if r.get("sampling_probability") is None]
    if missing:
        raise LabelPolicyError(
            f"{len(missing)} sealed evaluation rows have no sampling_probability; "
            "weighted metrics cannot reconstruct the frame without it"
        )
    return selected


def assert_disjoint(
    training: Sequence[dict[str, Any]],
    evaluation: Sequence[dict[str, Any]],
    key: str = "dst_doc_id",
) -> None:
    """Near-duplicate witnesses of one passage must not fall on both sides."""
    shared = {r[key] for r in training} & {r[key] for r in evaluation}
    if shared:
        raise LabelPolicyError(
            f"{len(shared)} values of {key} appear in both training and evaluation: "
            f"{sorted(shared)[:5]}"
        )


def assert_not_ground_truth(rows: Iterable[dict[str, Any]]) -> None:
    """A model probability must never be written into a human label field."""
    offenders = [
        r for r in rows
        if r.get("label_source") == "model" and r.get("label_kind") == LabelKind.FINAL_EVENT_REVIEW
    ]
    if offenders:
        raise LabelPolicyError(
            f"{len(offenders)} final-review rows carry a model-assigned label"
        )
