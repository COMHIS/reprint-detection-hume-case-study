"""Test 4: source-side ablation at more than one bibliographical level.

Asking how much of the confirmed record a single digitised manifestation
recovers gives one answer. Asking the same question conditional on whether the
essay is present in that volume at all gives another. The two answer different
questions, and reporting only the copy-level result invites a bibliographical
reading of what is a digitisation effect.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence


def recovery_by_manifestation(
    events: Sequence[dict[str, Any]],
    manifestations: Iterable[str],
    contents: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Unconditional and conditional recovery for each source manifestation.

    ``contents`` maps a manifestation to the set of essays it actually
    contains; conditioning on it separates "this volume lacks the essay" from
    "this volume has the essay but the digitisation lost it".
    """
    total = len(events)
    rows: list[dict[str, Any]] = []
    for manifestation in manifestations:
        present = contents.get(manifestation, set())
        eligible = [e for e in events if e["src_section_id"] in present]
        recovered = [e for e in eligible if manifestation in e["source_manifestations"]]
        rows.append({
            "manifestation": manifestation,
            "events_total": total,
            "events_recovered": len(recovered),
            "unconditional_recovery": len(recovered) / total if total else float("nan"),
            "events_eligible": len(eligible),
            "conditional_recovery": len(recovered) / len(eligible) if eligible else float("nan"),
        })
    return sorted(rows, key=lambda r: -r["unconditional_recovery"])


def recovery_by_work(
    events: Sequence[dict[str, Any]], manifestation_to_work: dict[str, str]
) -> list[dict[str, Any]]:
    """The same question asked of the work rather than the digitised copy."""
    total = len(events)
    by_work: dict[str, set[int]] = defaultdict(set)
    for index, event in enumerate(events):
        for manifestation in event["source_manifestations"]:
            work = manifestation_to_work.get(manifestation)
            if work is not None:
                by_work[work].add(index)
    return sorted(
        (
            {
                "work": work,
                "events_recovered": len(indices),
                "recovery": len(indices) / total if total else float("nan"),
            }
            for work, indices in by_work.items()
        ),
        key=lambda r: -r["recovery"],
    )


def decompose_work_level_gain(
    events: Sequence[dict[str, Any]],
    best_manifestation: str,
    contents: dict[str, set[str]],
    manifestation_to_work: dict[str, str],
) -> dict[str, Any]:
    """How much of the work-level gain is content the best single volume lacks."""
    present = contents.get(best_manifestation, set())
    work = manifestation_to_work.get(best_manifestation)
    siblings = {m for m, w in manifestation_to_work.items() if w == work}

    best_recovered = {
        i for i, e in enumerate(events) if best_manifestation in e["source_manifestations"]
    }
    work_recovered = {
        i for i, e in enumerate(events) if siblings & set(e["source_manifestations"])
    }
    gain = work_recovered - best_recovered
    absent = {i for i in gain if events[i]["src_section_id"] not in present}
    return {
        "best_manifestation": best_manifestation,
        "best_recovery": len(best_recovered) / len(events) if events else float("nan"),
        "work_recovery": len(work_recovered) / len(events) if events else float("nan"),
        "gain_events": len(gain),
        "gain_from_absent_content": len(absent),
        "share_of_gain_from_absent_content": len(absent) / len(gain) if gain else float("nan"),
        "note": (
            "Additional manifestations of one work buy protection against OCR and "
            "segmentation loss. They do not show that different editions carry "
            "different text, which is a bibliographical claim needing bibliographical "
            "evidence."
        ),
    }
