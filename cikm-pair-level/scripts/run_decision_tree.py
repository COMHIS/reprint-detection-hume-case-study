from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision_tree_baseline import (
    EXTENDED_FEATURES,
    feature_matrix,
    label_vector,
    predict_probability,
    train_tree,
)
from src.io_utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small decision-tree baseline and predict all rows.")
    parser.add_argument("--input", type=Path, required=True, help="Input pair-feature JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output prediction JSONL file.")
    parser.add_argument("--summary", type=Path, required=True, help="Output summary JSON file.")
    parser.add_argument("--train-split", default="discovery_set", help="Split tag used for training.")
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum tree depth.")
    parser.add_argument("--min-leaf-size", type=int, default=4, help="Minimum examples per leaf.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive probability threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    train_rows = [
        row
        for row in rows
        if row.get("gold_label") in {"reprint", "non_reprint"}
        and args.train_split in row.get("split_tags", [])
    ]
    if not train_rows:
        raise SystemExit(f"No labeled training rows found for split: {args.train_split}")

    tree = train_tree(
        feature_matrix(train_rows, EXTENDED_FEATURES),
        label_vector(train_rows),
        max_depth=args.max_depth,
        min_leaf_size=args.min_leaf_size,
    )

    predictions = []
    for row, values in zip(rows, feature_matrix(rows, EXTENDED_FEATURES)):
        probability = predict_probability(tree, values)
        predictions.append(
            {
                "pair_id": row["pair_id"],
                "src_doc_id": row["src_doc_id"],
                "dst_doc_id": row["dst_doc_id"],
                "gold_label": row.get("gold_label"),
                "split_tags": row.get("split_tags", []),
                "pred_label": "reprint" if probability >= args.threshold else "non_reprint",
                "score": probability,
                "method_name": "decision_tree_extended",
            }
        )

    write_jsonl(args.output, predictions)
    write_json(
        args.summary,
        {
            "input": str(args.input),
            "output": str(args.output),
            "train_split": args.train_split,
            "n_train_rows": len(train_rows),
            "features": EXTENDED_FEATURES,
            "tree": tree.to_dict(),
        },
    )


if __name__ == "__main__":
    main()
