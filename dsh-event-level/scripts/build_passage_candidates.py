#!/usr/bin/env python3
"""Turn raw fragments into adjacency edges and local passage candidates.

Every path is given on the command line. No fragment is deleted at any point;
the link decisions change component membership only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from src.io_utils import manifest, read_jsonl, write_json, write_jsonl
from src.passage_links import build_edges, fragment_id, link_edges, passage_components


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragments", type=Path, required=True,
                        help="JSONL of raw fragments with half-open offsets on both sides")
    parser.add_argument("--edge-decisions", type=Path,
                        help="JSONL of link decisions; omit to link on destination overlap only")
    parser.add_argument("--out-edges", type=Path, required=True)
    parser.add_argument("--out-passages", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--corpus-version", default="unspecified")
    args = parser.parse_args()

    fragments = read_jsonl(args.fragments)
    for row in fragments:
        row.setdefault("fragment_id", fragment_id(row, args.corpus_version))

    edges = build_edges(fragments)
    write_jsonl(args.out_edges, edges)

    if args.edge_decisions and args.edge_decisions.exists():
        decisions = read_jsonl(args.edge_decisions)
    else:
        decisions = link_edges(edges, predict_proba=lambda rows: [0.0] * len(rows))

    passages = passage_components(fragments, decisions)
    write_jsonl(args.out_passages, passages)
    write_json(args.out_manifest, manifest(
        stage="build_passage_candidates",
        inputs={"fragments": args.fragments},
        outputs={"edges": args.out_edges, "passages": args.out_passages},
        parameters={
            "corpus_version": args.corpus_version,
            "fragments": len(fragments),
            "edges": len(edges),
            "passages": len(passages),
            "linked_on_destination_overlap": sum(
                1 for d in decisions if d["decision_source"] == "destination_overlap"
            ),
        },
    ))
    print(f"{len(fragments)} fragments -> {len(edges)} edges -> {len(passages)} passages")


if __name__ == "__main__":
    main()
