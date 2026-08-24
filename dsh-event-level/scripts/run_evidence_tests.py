#!/usr/bin/env python3
"""Run the null model, the two-instrument comparison and the pooled recall bound.

Each test reads only released derived files. None of them needs the models.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from src.evidence_tests import compiled_units, coverage_audit, null_model
from src.io_utils import read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True,
                        help="confirmed events with year, work, dst_doc_id, src_section_id")
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--work-grouping", type=Path,
                        help="JSON mapping dst_doc_id to its work-level group value")
    parser.add_argument("--audits", type=Path,
                        help="JSON list of probability audits of the unreviewed pool")
    parser.add_argument("--frame-destinations", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    events = read_jsonl(args.events)
    passages = read_jsonl(args.passages)
    by_destination: dict[str, list] = defaultdict(list)
    for row in passages:
        by_destination[str(row["dst_doc_id"])].append(row)

    report: dict[str, object] = {}

    if all("year" in e and "work" in e for e in events):
        report["null_model"] = null_model.run(
            events, iterations=args.iterations, random_seed=args.random_seed
        )

    units = compiled_units.detect_compiled_units(by_destination)
    report["compiled_units"] = {
        "units": units,
        "geometry_blind_destinations": len(
            compiled_units.geometry_blind_destinations(by_destination)
        ),
        "destinations_total": len(by_destination),
    }
    if args.work_grouping and args.work_grouping.exists():
        grouping = read_json(args.work_grouping)
        report["compiled_units"]["instrument_comparison"] = compiled_units.compare_instruments(
            units, grouping
        )
        report["compiled_units"]["grouping_shape"] = compiled_units.grouping_shape(grouping)

    if args.audits and args.audits.exists() and args.frame_destinations:
        report["coverage_audit"] = coverage_audit.pool_audits(
            read_json(args.audits), frame_destinations=args.frame_destinations
        )

    write_json(args.out_report, report)
    print(f"wrote {args.out_report}")


if __name__ == "__main__":
    main()
