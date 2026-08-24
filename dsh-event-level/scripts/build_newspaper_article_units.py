#!/usr/bin/env python3
"""Build essay-section to newspaper-article candidates.

The article boundary comes from upstream layout analysis and is adopted as
given: no passage reconstruction is attempted below it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from src.io_utils import manifest, read_jsonl, write_json, write_jsonl
from src.newspaper_units import build_article_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--out-candidates", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args()

    fragments = read_jsonl(args.fragments)
    candidates = build_article_candidates(fragments)
    write_jsonl(args.out_candidates, candidates)
    write_json(args.out_manifest, manifest(
        stage="build_newspaper_article_units",
        inputs={"fragments": args.fragments},
        outputs={"candidates": args.out_candidates},
        parameters={
            "fragments": len(fragments),
            "candidates": len(candidates),
            "articles": len({c["dst_doc_id"] for c in candidates}),
            "unit": "one source essay section to one layout-segmented article",
        },
    ))
    print(f"{len(fragments)} fragments -> {len(candidates)} candidates "
          f"over {len({c['dst_doc_id'] for c in candidates})} articles")


if __name__ == "__main__":
    main()
