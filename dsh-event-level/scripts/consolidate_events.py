#!/usr/bin/env python3
"""Consolidate passages into destination-side event candidates for human review.

An event candidate is a review container, not a historical event. Only a record
a reader confirms as `reprint` from page images becomes one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from src.io_utils import manifest, read_jsonl, write_json, write_jsonl
from src.event_consolidation import confirmed_events, consolidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--negative-evidence", type=Path,
                        help="passages already reviewed as non_reprint; attached as conflicts")
    parser.add_argument("--reviews", type=Path,
                        help="human decisions; omit to stop at candidates")
    parser.add_argument("--out-candidates", type=Path, required=True)
    parser.add_argument("--out-events", type=Path)
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args()

    passages = read_jsonl(args.passages)
    negatives = read_jsonl(args.negative_evidence) if args.negative_evidence else []
    candidates = consolidate(passages, negatives)
    write_jsonl(args.out_candidates, candidates)

    events = []
    if args.reviews and args.out_events:
        events = confirmed_events(candidates, read_jsonl(args.reviews))
        write_jsonl(args.out_events, events)

    write_json(args.out_manifest, manifest(
        stage="consolidate_events",
        inputs={"passages": args.passages},
        outputs={"candidates": args.out_candidates},
        parameters={
            "passages": len(passages),
            "event_candidates": len(candidates),
            "destinations": len({c["dst_doc_id"] for c in candidates}),
            "candidates_with_conflicting_negative_evidence": sum(
                1 for c in candidates if c["conflicting_negative_ids"]
            ),
            "confirmed_events": len(events),
        },
    ))
    print(f"{len(passages)} passages -> {len(candidates)} event candidates"
          + (f" -> {len(events)} confirmed events" if events else ""))


if __name__ == "__main__":
    main()
