from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import read_jsonl, write_csv, write_json
from src.metrics import compute_metrics, group_by_split


FIELDS = [
    "split",
    "n_examples",
    "n_positive_gold",
    "n_negative_gold",
    "tp",
    "fp",
    "tn",
    "fn",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "macro_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pair-level predictions by labeled split.")
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction JSONL file.")
    parser.add_argument("--metrics-json", type=Path, required=True, help="Output metrics JSON file.")
    parser.add_argument("--metrics-csv", type=Path, required=True, help="Output metrics CSV file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    split_metrics = {
        split: compute_metrics(split_rows)
        for split, split_rows in group_by_split(rows).items()
        if split_rows
    }
    write_json(args.metrics_json, {"prediction_file": str(args.predictions), "splits": split_metrics})
    write_csv(
        args.metrics_csv,
        [{"split": split, **metrics} for split, metrics in split_metrics.items()],
        FIELDS,
    )


if __name__ == "__main__":
    main()
