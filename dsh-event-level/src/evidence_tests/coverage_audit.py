"""Test 5: re-test outside the layer, and bound what the record misses.

A pattern that holds among reviewed events may hold there because review put it
there. The complementary question is what the reviewed record leaves out, and
it is answerable when the unreviewed pool has been sampled at known
probabilities.

Two audits of the low-scoring pool can be pooled when they draw from the same
frame at recorded probabilities and their destinations are disjoint. Pooled,
they give a binomial bound on the number of unreviewed destinations still
carrying a republication -- which turns the confirmed record from an
unqualified count into a lower bound with a stated gap.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..metrics import wilson_interval


def pool_audits(
    audits: Sequence[dict[str, Any]], frame_destinations: int
) -> dict[str, Any]:
    """Combine destination-disjoint probability audits of one unreviewed pool.

    Each audit is ``{"name", "draws", "positives", "destinations"}`` where
    ``draws`` counts destination-level draws and ``positives`` the number that
    turned out to be republications.
    """
    seen: set[str] = set()
    for audit in audits:
        destinations = set(audit.get("destinations", []))
        overlap = seen & destinations
        if overlap:
            raise ValueError(
                f"audits are not destination-disjoint; {len(overlap)} shared destinations"
            )
        seen |= destinations

    draws = sum(int(a["draws"]) for a in audits)
    positives = sum(int(a["positives"]) for a in audits)
    rate = positives / draws if draws else float("nan")
    low, high = wilson_interval(positives, draws)
    return {
        "audits": [a["name"] for a in audits],
        "draws": draws,
        "positives": positives,
        "rate": rate,
        "wilson_95": [low, high],
        "frame_destinations": frame_destinations,
        "estimated_missed_destinations": rate * frame_destinations,
        "estimated_missed_95": [low * frame_destinations, high * frame_destinations],
        "note": (
            "Conservative in a stated direction where one audit over-sampled the "
            "destinations most likely to be positive and found none there."
        ),
    }


def negative_predictive_value(reviewed: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How safe it is to leave the model-negative pool unread.

    Licenses reduced review; licenses nothing about precision.
    """
    total = len(reviewed)
    negatives = sum(1 for r in reviewed if r["decision"] == "non_reprint")
    low, high = wilson_interval(negatives, total)
    return {
        "reviewed": total,
        "confirmed_negative": negatives,
        "npv": negatives / total if total else float("nan"),
        "wilson_95": [low, high],
        "licenses": "reduced review of the low-scoring pool, with continuing random recall audits",
        "does_not_license": "any statement about precision",
    }
