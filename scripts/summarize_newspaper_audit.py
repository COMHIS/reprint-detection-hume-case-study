from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import read_jsonl, write_json
from src.newspaper_audit import summarize_newspaper_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize resolved newspaper audit annotations.")
    parser.add_argument("--annotations", type=Path, required=True, help="Newspaper audit JSONL file.")
    parser.add_argument("--summary", type=Path, required=True, help="Output summary JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.annotations)
    write_json(args.summary, summarize_newspaper_audit(rows))


if __name__ == "__main__":
    main()
