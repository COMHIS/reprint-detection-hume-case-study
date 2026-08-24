from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import read_jsonl, write_json, write_jsonl
from src.workflow_rules import WORKFLOW_RULES, predict_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a rule workflow to pair-level feature rows.")
    parser.add_argument("--input", type=Path, required=True, help="Input pair-feature JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output prediction JSONL file.")
    parser.add_argument("--summary", type=Path, required=True, help="Output summary JSON file.")
    parser.add_argument(
        "--method",
        choices=sorted(WORKFLOW_RULES),
        default="final_workflow",
        help="Workflow rule to apply.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    predictions = [predict_row(row, args.method) for row in rows]
    write_jsonl(args.output, predictions)
    write_json(
        args.summary,
        {
            "input": str(args.input),
            "output": str(args.output),
            "method": args.method,
            "n_pairs": len(predictions),
            "n_positive_predictions": sum(row["pred_label"] == "reprint" for row in predictions),
        },
    )


if __name__ == "__main__":
    main()
