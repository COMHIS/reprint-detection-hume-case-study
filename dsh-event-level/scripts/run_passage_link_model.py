#!/usr/bin/env python3
"""Fit and compare the adjacency link models against the rule they replaced.

Edges whose destination intervals overlap are linked deterministically and are
excluded from fitting. Cross-validation is grouped by destination document.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from src.io_utils import manifest, read_jsonl, write_json, write_jsonl
from src.metrics import grouped_cross_validation
from src.passage_links import (
    feature_matrix, legacy_gap_rule, link_edges, model_builders,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--edge-labels", type=Path, required=True,
                        help="JSONL with edge_id and edge_label in {same_passage, split_passage}")
    parser.add_argument("--out-decisions", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--select", default="link_random_forest")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    edges = read_jsonl(args.edges)
    labels_by_edge = {
        row["edge_id"]: int(row["edge_label"] == "same_passage")
        for row in read_jsonl(args.edge_labels)
    }
    labelled = [e for e in edges if e["edge_id"] in labels_by_edge]
    if not labelled:
        raise SystemExit("no labelled edges found; check --edge-labels")

    features = feature_matrix(labelled)
    labels = np.asarray([labels_by_edge[e["edge_id"]] for e in labelled])
    groups = [e["dst_doc_id"] for e in labelled]

    report = {"labelled_edges": len(labelled), "models": {}}
    for name, build in model_builders(args.random_seed).items():
        report["models"][name] = grouped_cross_validation(
            build, features, labels, groups, n_splits=args.folds, random_seed=args.random_seed,
        )
    rule_predictions = np.asarray([int(legacy_gap_rule(e)) for e in labelled])
    report["legacy_gap_rule"] = {
        "accuracy": float((rule_predictions == labels).mean()),
        "note": "The fixed maximum-gap rule the supervised model replaced.",
    }

    selected = model_builders(args.random_seed)[args.select]()
    selected.fit(features, labels)
    decisions = link_edges(
        edges,
        predict_proba=lambda rows: selected.predict_proba(feature_matrix(rows))[:, 1],
        threshold=args.threshold,
    )
    write_jsonl(args.out_decisions, decisions)
    report["selected_model"] = args.select
    report["threshold"] = args.threshold
    write_json(args.out_report, report)
    write_json(args.out_report.with_suffix(".manifest.json"), manifest(
        stage="run_passage_link_model",
        inputs={"edges": args.edges, "edge_labels": args.edge_labels},
        outputs={"decisions": args.out_decisions, "report": args.out_report},
        parameters={"select": args.select, "threshold": args.threshold,
                    "random_seed": args.random_seed},
    ))
    print(f"{len(labelled)} labelled edges; selected {args.select}")


if __name__ == "__main__":
    main()
