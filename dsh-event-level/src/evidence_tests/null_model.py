"""Test 2: a null that preserves the structure of the derivation.

Events per year, events per destination and relations per essay are all shaped
by how the unit was built. Before any of them supports a historical statement,
it has to be compared with a null in which the derivation is preserved and only
the historical assignment is randomised.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np


def annual_concentration(events: Sequence[dict[str, Any]]) -> float:
    """Mean share of a year's events contributed by its largest single work."""
    by_year: dict[Any, Counter] = defaultdict(Counter)
    for event in events:
        by_year[event["year"]][event["work"]] += 1
    shares = [max(counts.values()) / sum(counts.values()) for counts in by_year.values() if counts]
    return float(np.mean(shares)) if shares else float("nan")


def permute_works_across_years(
    events: Sequence[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Reassign each work's events to years at random, preserving work sizes.

    Work sizes and the number of events per year are properties of the
    derivation, so the null holds them fixed and randomises only which year a
    work's events land in.
    """
    years = [event["year"] for event in events]
    permuted_years = rng.permutation(years)
    return [
        {**event, "year": year} for event, year in zip(events, permuted_years)
    ]


def run(
    events: Sequence[dict[str, Any]],
    iterations: int = 10_000,
    random_seed: int = 0,
    statistic=annual_concentration,
) -> dict[str, Any]:
    """Compare the observed statistic against its null distribution."""
    rng = np.random.default_rng(random_seed)
    observed = statistic(events)
    null = np.asarray(
        [statistic(permute_works_across_years(events, rng)) for _ in range(iterations)]
    )
    percentile = float((null < observed).mean() * 100)
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_percentile_of_observed": percentile,
        "iterations": iterations,
        "supports_claim": bool(percentile >= 95.0),
        "note": (
            "A statistic that does not exceed its structural null is a property of "
            "the derivation, not of the past, and cannot carry a historical claim."
        ),
    }
