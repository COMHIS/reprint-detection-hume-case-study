"""Test 3: two instruments for bibliographical identity.

Where destinations are reissues or successive editions of one book, the same
act of compilation is counted more than once, so the layer needs some way of
asking whether two destinations are the same publication. Two instruments are
available and neither can be used alone.

The catalogue-side instrument is the work-level grouping supplied with the
bibliographic data. The object-side instrument is offset geometry: if two
volumes carry the same source essays at the same relative offsets in the same
order, they are unlikely to be independent compilations.

Their domains differ and can be measured. Offset geometry needs two or more
source positions in a volume and is silent otherwise. The work-level grouping
is title-sensitive, so it divides where a title varies. Disagreement between
them is the trigger for a title-page check, not a verdict.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence


def essay_offset_signature(
    passages: Sequence[dict[str, Any]], round_to: int = 500
) -> tuple[tuple[str, int], ...]:
    """Ordered (essay, quantised offset) pairs for one destination volume."""
    ordered = sorted(passages, key=lambda p: int(p["dst_start"]))
    return tuple(
        (str(p["src_section_id"]), int(p["dst_start"]) // round_to) for p in ordered
    )


def detect_compiled_units(
    passages_by_destination: dict[str, Sequence[dict[str, Any]]],
    min_positions: int = 2,
    round_to: int = 500,
) -> dict[str, list[str]]:
    """Group destinations that share an essay sequence and offset geometry.

    Destinations with fewer than ``min_positions`` confirmed positions carry no
    internal geometry and are returned in no group: this is the instrument's
    stated domain, not a failure to detect.
    """
    signatures: dict[tuple, list[str]] = defaultdict(list)
    for destination, passages in passages_by_destination.items():
        if len({p["src_section_id"] for p in passages}) < min_positions:
            continue
        signatures[essay_offset_signature(passages, round_to)].append(destination)
    return {
        f"unit_{index:02d}": sorted(members)
        for index, (_, members) in enumerate(
            sorted((k, v) for k, v in signatures.items() if len(v) > 1), start=1
        )
    }


def geometry_blind_destinations(
    passages_by_destination: dict[str, Sequence[dict[str, Any]]], min_positions: int = 2
) -> list[str]:
    """Destinations the geometric instrument cannot characterise at all."""
    return sorted(
        destination
        for destination, passages in passages_by_destination.items()
        if len({p["src_section_id"] for p in passages}) < min_positions
    )


def compare_instruments(
    units: dict[str, list[str]], work_grouping: dict[str, str]
) -> list[dict[str, Any]]:
    """Where geometry and the work-level grouping disagree, inspect the object."""
    report: list[dict[str, Any]] = []
    for unit, destinations in units.items():
        groups = {work_grouping.get(d) for d in destinations}
        report.append({
            "unit": unit,
            "destinations": destinations,
            "volume_count": len(destinations),
            "work_grouping_values": sorted(g for g in groups if g is not None),
            "work_grouping_count": len(groups),
            "instruments_agree": len(groups) == 1,
            "action": "agree" if len(groups) == 1 else "inspect title pages before any verdict",
        })
    return report


def grouping_shape(work_grouping: dict[str, str]) -> dict[str, Any]:
    """Report the observable behaviour of the supplied grouping on this corpus.

    Values that take the form of a normalised title rather than an identifier
    indicate that, for those records, grouping is grouping by title.
    """
    values = list(work_grouping.values())
    title_shaped = [v for v in values if not str(v).split("-", 1)[0].isdigit()]
    return {
        "destinations": len(values),
        "distinct_groups": len(set(values)),
        "title_shaped_values": len(title_shaped),
        "title_shaped_share": len(title_shaped) / len(values) if values else float("nan"),
        "note": (
            "Measured on this corpus only. Report the behaviour observed, not an "
            "account of how the grouping was produced."
        ),
    }
