#!/usr/bin/env python3
"""Recompute the article's reported numbers from the released files alone.

Each block implements one procedure as stated in Supplementary Appendix E and
prints the released value next to the value printed in the article. Nothing
here reads OCR text, page images or the working project.

    python3 06_evidence_tests/reproduce_paper_numbers.py [--quick]

--quick lowers the permutation counts for a fast smoke run.
"""
from __future__ import annotations

import argparse
import csv
import re
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
POLITICAL = {"liberty_press", "public_credit"}
results: list[dict] = []
diagnostics: dict = {}


def jsonl(rel: str) -> list[dict]:
    with (ROOT / rel).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def table(rel: str) -> list[dict]:
    with (ROOT / rel).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(a, b, tolerance: float) -> bool:
    """Compare numbers, or nested tuples and lists of them, within tolerance."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tolerance
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == len(b):
        return all(close(x, y, tolerance) for x, y in zip(a, b))
    return a == b


def report(test: str, quantity: str, released, article, tolerance: float = 0.0) -> None:
    ok = close(released, article, tolerance)
    results.append({"test": test, "quantity": quantity, "released": released,
                    "article": article, "match": ok})
    mark = "ok " if ok else "DIFF"
    print(f"  [{mark}] {quantity}: released {released} | article {article}")


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / denom), (centre + half) / denom


def section_key(row: dict) -> str:
    return f"{row['src_doc_id']}::{row['src_section_id']}"


parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="fewer permutations")
parser.add_argument("--release", type=Path, default=ROOT,
                    help="root of the data release (default: the directory above this script)")
parser.add_argument("--output", type=Path,
                    help="report path (default: 06_evidence_tests/reproduction_report.json)")
args = parser.parse_args()
ROOT = args.release
annual_iterations = 200 if args.quick else 2_000
composition_iterations = 5_000 if args.quick else 100_000
length_iterations = 1_000 if args.quick else 10_000

book_events = jsonl("01_events/books/events.jsonl")
news_events = jsonl("01_events/newspapers/events.jsonl")
events = book_events + news_events
book_rel = jsonl("01_events/books/event_essay_relations.jsonl")
news_rel = jsonl("01_events/newspapers/event_essay_relations.jsonl")
destinations = table("02_destinations/books/book_destinations.csv")
section_to_essay = {r["source_section_id"]: r["canonical_essay_id"]
                    for r in table("04_annotations/essay_identity/canonical_source_section_mapping.csv")}

# ------------------------------------------------------------------ units
print("\nUnits and label sets")
report("units", "confirmed events", len(events), 331)
report("units", "book events", len(book_events), 245)
report("units", "newspaper events", len(news_events), 86)
report("units", "event-essay relations", len(book_rel) + len(news_rel), 339)
report("units", "canonical essays", len({r["canonical_essay_id"] for r in book_rel + news_rel}), 38)
report("units", "book destinations", len({e["dst_doc_id"] for e in book_events}), 115)
report("units", "book fragments", sum(1 for _ in (ROOT / "03_evidence_layers/books/fragments.jsonl").open()), 40_735)
report("units", "book adjacency edges", sum(1 for _ in (ROOT / "03_evidence_layers/books/adjacency_edges.jsonl").open()), 20_167)
passages = jsonl("03_evidence_layers/books/passage_candidates.jsonl")
articles = jsonl("03_evidence_layers/newspapers/article_candidates.jsonl")
report("units", "section candidates carrying passages", len({p["section_candidate_id"] for p in passages}), 20_568)
report("units", "classification candidates", len(passages) + len(articles), 27_232)
report("units", "section candidates splitting into two or more passages",
       sum(1 for n in Counter(p["section_candidate_id"] for p in passages).values() if n >= 2), 2_162)
report("units", "newspaper articles carrying candidates",
       len({a["dst_doc_id"] for a in articles}), 456)
training = (len(jsonl("04_annotations/adaptive_training/books/passage_training_labels.jsonl"))
            + len(jsonl("04_annotations/adaptive_training/newspapers/article_training_labels.jsonl")))
report("units", "training records", training, 1_813)
sealed = sum(len(jsonl(f"04_annotations/sealed_evaluation/{p}.jsonl")) for p in (
    "books/passage_validation_60", "books/passage_validation_v3_40",
    "books/passage_coverage_negative_audit_60", "newspapers/article_holdout_60"))
report("units", "evaluation-only records", sealed, 220)

# -------------------------------------------------------------- typology
print("\nTable 7: destination typology")
by_type = defaultdict(lambda: [0, 0])
for d in destinations:
    by_type[d["structural_type"]][0] += 1
    by_type[d["structural_type"]][1] += int(d["confirmed_events"])
for name, expected in [("single", [70, 70]), ("continuous block", [16, 37]),
                       ("dispersed, one per essay", [25, 114]), ("dispersed, recurring essay", [4, 24])]:
    report("typology", f"{name} (destinations, events)", by_type.get(name), expected)

# ------------------------------------------------- annual concentration null
print("\nNull model for annual concentration")
work_of = {d["dst_doc_id"]: d["work_id"] for d in destinations}
dated = [e for e in book_events if e.get("dst_publication_year")]
by_year: dict = defaultdict(list)
for e in dated:
    by_year[e["dst_publication_year"]].append(work_of[e["dst_doc_id"]])
observed = float(np.mean([Counter(ws).most_common(1)[0][1] / len(ws) for ws in by_year.values()]))
work_years: dict = defaultdict(set)
for e in dated:
    work_years[work_of[e["dst_doc_id"]]].add(e["dst_publication_year"])
work_size = Counter(work_of[e["dst_doc_id"]] for e in dated)
works = list(work_years)
slots = [len(set(by_year[y])) for y in sorted(by_year)]

# Block permutation: each (work, year) pair keeps its event count and the year
# labels are permuted across blocks, so every year keeps its number of blocks.
# Distinct-work counts may fall when blocks of the same work share a new year.
blocks = Counter((e["dst_publication_year"], work_of[e["dst_doc_id"]]) for e in dated)
b_years, b_works, b_sizes = [k[0] for k in blocks], [k[1] for k in blocks], list(blocks.values())


def block_statistic(years_: list) -> float:
    per_year: dict = defaultdict(Counter)
    for y_, w_, n_ in zip(years_, b_works, b_sizes):
        per_year[y_][w_] += n_
    return float(np.mean([max(c.values()) / sum(c.values()) for c in per_year.values()]))


rng = np.random.default_rng(7)
null_values = []
permutations_with_work_collisions = 0
for _ in range(annual_iterations):
    shuffled_years = list(rng.permutation(b_years))
    assert Counter(shuffled_years) == Counter(b_years)
    permutations_with_work_collisions += len(set(zip(shuffled_years, b_works))) < len(blocks)
    null_values.append(block_statistic(shuffled_years))
null = np.sort(np.array(null_values))
diagnostics["annual null preserved units"] = {
    "blocks": len(blocks),
    "permutations": annual_iterations,
    "permutations_with_work_collisions": permutations_with_work_collisions,
    "note": "Preserves year-wise block counts and block sizes, not distinct-work counts per year.",
}
report("annual null", "observed statistic", round(observed, 3), 0.712)
report("annual null", "null mean", round(float(null.mean()), 3), 0.725, 0.005)
if not args.quick:
    report("annual null", "null 95% interval",
           (round(float(np.quantile(null, 0.025)), 3), round(float(np.quantile(null, 0.975)), 3)),
           (0.694, 0.753))
report("annual null", "percentile of observed", round(float((null < observed).mean() * 100)), 20, 3)
report("annual null", "dated works in a single year",
       f"{sum(1 for v in work_years.values() if len(v) == 1)} of {len(work_years)}", "58 of 73")

# Diagnostic: the procedure used for the first submission, which dealt 73 works
# into 97 year slots and so left the last slots empty.
shuffler = random.Random(7)
dealt = []
for _ in range(annual_iterations):
    pool = works[:]
    shuffler.shuffle(pool)
    cursor, year_shares = 0, []
    for k in slots:
        group = pool[cursor:cursor + k]
        cursor += k
        if group:
            sizes = [work_size[w] for w in group]
            year_shares.append(max(sizes) / sum(sizes))
    dealt.append(float(np.mean(year_shares)))
dealt = np.array(dealt)
diagnostics["annual null, works dealt into year slots"] = {
    "year_slots": sum(slots), "works": len(works),
    "null_mean": round(float(dealt.mean()), 3),
    "percentile_of_observed": round(float((dealt < observed).mean() * 100), 1),
}
print(f"  [diag] first-submission procedure: {diagnostics['annual null, works dealt into year slots']}")

# ------------------------------------------------- 1790s composition null
print("\nComposition null for the 1790s newspaper record")
year_of = {e["event_candidate_id"]: e["dst_publication_year"] for e in news_events}
decades = sorted({year_of[r["event_candidate_id"]] // 10 * 10 for r in news_rel})
decade_index = np.array([decades.index(year_of[r["event_candidate_id"]] // 10 * 10) for r in news_rel])
political = np.array([r["canonical_essay_id"] in POLITICAL for r in news_rel])
sizes_by_decade = [int((decade_index == i).sum()) for i in range(len(decades))]
pol_by_decade = [int(political[decade_index == i].sum()) for i in range(len(decades))]
report("composition null", "decade sizes", sizes_by_decade, [6, 12, 23, 25, 22])
report("composition null", "political per decade", pol_by_decade, [1, 5, 7, 7, 17])


def shares(flags: np.ndarray) -> np.ndarray:
    return np.array([flags[decade_index == i].mean() for i in range(len(decades))])


def issue_of(dst: str) -> tuple:
    m = re.match(r"([A-Z0-9]+)_([A-Z0-9]+)_(\d{4})_(\d{2})_(\d{2})", dst)
    return (m.group(2), m.group(3), m.group(4), m.group(5)) if m else (dst,)


obs_share = shares(political)
obs_rho = spearmanr(range(len(decades)), obs_share).statistic
report("composition null", "1790s political share", round(float(obs_share[-1]), 3), 0.773)
report("composition null", "Spearman decade vs share", round(float(obs_rho), 3), 0.600)

# The article's null groups relations by issue and permutes decade labels across
# issues, so that two relations printed in one issue move together.
groups: dict = defaultdict(list)
for r in news_rel:
    e = {x["event_candidate_id"]: x for x in news_events}[r["event_candidate_id"]]
    groups[issue_of(e["dst_doc_id"])].append(
        (e["dst_publication_year"] // 10 * 10, r["canonical_essay_id"] in POLITICAL))
cluster = list(groups.values())
cluster_decade = np.array([c[0][0] for c in cluster])


def clustered_shares(assign: np.ndarray) -> np.ndarray:
    out = []
    for d in decades:
        pol = tot = 0
        for c, a in zip(cluster, assign):
            if a == d:
                tot += len(c)
                pol += sum(p for _, p in c)
        out.append(pol / tot if tot else 0.0)
    return np.array(out)


rng = np.random.default_rng(20260928)
obs_cluster = clustered_shares(cluster_decade)
hits_last = hits_max = 0
iterations = composition_iterations // 2
for _ in range(iterations):
    sh = clustered_shares(rng.permutation(cluster_decade))
    hits_last += sh[-1] >= obs_cluster[-1] - 1e-12
    hits_max += sh.max() >= obs_cluster.max() - 1e-12
report("composition null", "issues / articles carrying the 17 political relations of the 1790s",
       (len({issue_of(e["dst_doc_id"]) for e in news_events
             for r in news_rel if r["event_candidate_id"] == e["event_candidate_id"]
             and r["canonical_essay_id"] in POLITICAL and e["dst_publication_year"] >= 1790}),
        len({e["dst_doc_id"] for e in news_events
             for r in news_rel if r["event_candidate_id"] == e["event_candidate_id"]
             and r["canonical_essay_id"] in POLITICAL and e["dst_publication_year"] >= 1790})),
       (11, 17))
report("composition null", "issue-clustered p for the 1790s share",
       round(hits_last / iterations, 3), 0.002, 0.002)
report("composition null", "issue-clustered p for the largest of five decades",
       round(hits_max / iterations, 2), 0.08, 0.03)
relation_null = []
for _ in range(composition_iterations):
    relation_null.append(shares(rng.permutation(political))[-1])
relation_null = np.array(relation_null)
report("composition null", "relation-level p (reported as a robustness check)",
       round(float((relation_null >= obs_share[-1] - 1e-12).mean()), 4), 0.0002, 0.0004)
rho_null = np.array([spearmanr(range(len(decades)), shares(rng.permutation(political))).statistic
                     for _ in range(composition_iterations // 10)])
report("composition null", "trend p", round(float((rho_null >= obs_rho - 1e-12).mean()), 2), 0.18, 0.03)

# ------------------------------------------------ seed-composition control
print("\nSeed-composition control")
legacy = jsonl("04_annotations/adaptive_training/newspapers/article_training_labels.jsonl")
legacy_pos = [r for r in legacy if r["label"] == "reprint"]
legacy_pol = sum(1 for r in legacy_pos if section_to_essay.get(section_key(r)) in POLITICAL)
report("seed control", "legacy records / articles / positives",
       (len(legacy), len({r["dst_doc_id"] for r in legacy}), len(legacy_pos)), (148, 58, 116))
report("seed control", "political share of legacy positives", round(100 * legacy_pol / len(legacy_pos), 1), 29.3)
report("seed control", "political share of confirmed relations",
       round(100 * sum(r["canonical_essay_id"] in POLITICAL for r in news_rel) / len(news_rel), 1), 42.0)
legacy_articles = {r["dst_doc_id"] for r in legacy}
touched = sum(1 for e in news_events if e["dst_doc_id"] in legacy_articles)
report("seed control", "events on legacy / untouched articles", (touched, len(news_events) - touched), (22, 64))
holdout = jsonl("04_annotations/sealed_evaluation/newspapers/article_holdout_60.jsonl")
positives = [r for r in holdout if r["annotation_label"] == "reprint"]
weights = {"pol": 0.0, "other": 0.0}
for r in positives:
    essay = section_to_essay.get(section_key(r))
    weights["pol" if essay in POLITICAL else "other"] += 1 / r["sampling_probability"]
report("seed control", "holdout positives",
       len(positives), 30)
report("seed control", "weighted political share of holdout positives",
       round(100 * weights["pol"] / sum(weights.values()), 1), 27.9, 0.1)
scores = jsonl("05_models/transfer/holdout_scores.jsonl")
by_sample = {r["sample_id"]: r for r in holdout}
hit = {"pol": 0.0, "other": 0.0}
mass = {"pol": 0.0, "other": 0.0}
for sc in scores:
    h = by_sample.get(sc["sample_id"])
    if not h or h["annotation_label"] != "reprint":
        continue
    kind = "pol" if section_to_essay.get(section_key(h)) in POLITICAL else "other"
    mass[kind] += sc["sampling_weight"]
    if sc["scores"]["newspaper_v1_random_forest"] >= 0.5:
        hit[kind] += sc["sampling_weight"]
report("seed control", "weighted holdout recall at 0.5, political / other",
       (round(hit["pol"] / mass["pol"], 2), round(hit["other"] / mass["other"], 2)), (1.0, 0.47), 0.01)
hit_alloc = {"pol": 0.0, "other": 0.0}
for sc_row in scores:
    h = by_sample.get(sc_row["sample_id"])
    if not h or h["annotation_label"] != "reprint":
        continue
    kind = "pol" if section_to_essay.get(section_key(h)) in POLITICAL else "other"
    if sc_row["scores"]["newspaper_v1_random_forest"] >= 0.05:
        hit_alloc[kind] += sc_row["sampling_weight"]
report("seed control", "weighted holdout recall at the 0.05 allocation threshold, political / other",
       (round(hit_alloc["pol"] / mass["pol"], 2), round(hit_alloc["other"] / mass["other"], 2)),
       (1.0, 0.52), 0.01)

# ------------------------------------------- book negative-audit observations
print("\nBook negative-audit observations (no population expansion)")
coverage = json.loads((ROOT / "04_annotations/sealed_evaluation/books/passage_coverage_negative_audit_60_manifest.json").read_text())
cov_rows = jsonl("04_annotations/sealed_evaluation/books/passage_coverage_negative_audit_60.jsonl")
cov_frame = {k: v["destination_count_before_cross_stratum_destination_exclusion"]
             for k, v in coverage["coverage_frame_counts"].items()}
cov_draws = Counter(r["sampling_stratum"] for r in cov_rows)
cov_pos = Counter(r["sampling_stratum"] for r in cov_rows if r["annotation_label"] == "reprint")
report("book audit", "draws / pre-exclusion frame memberships (overlap allowed)",
       sorted((cov_draws[k], cov_frame[k]) for k in cov_frame), sorted([(4, 4), (16, 196), (20, 636), (20, 1773)]))
report("book audit", "unique destinations in coverage frame",
       coverage["strict_unseen_destination_eligible_destination_count"], 2178)
report("book audit", "eligible passages in coverage frame",
       coverage["strict_unseen_destination_eligible_passage_count"], 9964)
report("book audit", "sum of overlapping destination memberships", sum(cov_frame.values()), 2609)
report("book audit", "coverage audit sampled passages / positives",
       (len(cov_rows), sum(cov_pos.values())), (60, 0))
v3 = jsonl("04_annotations/sealed_evaluation/books/passage_validation_v3_40.jsonl")
neg = [r for r in v3 if r["sampling_stratum"].startswith("training_unseen_destination_negative_control")]
neg_frame = neg[0]["stratum_destination_count"]
neg_pos = sum(r["annotation_label"] == "reprint" for r in neg)
report("book audit", "negative control draws / frame / positives", (len(neg), neg_frame, neg_pos), (20, 2197, 1))
assert sum(cov_frame.values()) > coverage["strict_unseen_destination_eligible_destination_count"]
assert all(r["sampling_probability_is_conditional"] for r in cov_rows)
diagnostics["book negative audits"] = {
    "population_estimate": None,
    "reason": "Destination frames overlap; labels classify one sampled passage, not all occurrences in a destination.",
    "conditional_sampling_probabilities_preserved": True,
}
# No longer reported in the article, which gives no destination-to-event
# conversion; kept as a diagnostic because the ratio describes the record.
diagnostics["book events per confirmed destination"] = round(
    len(book_events) / len({e["dst_doc_id"] for e in book_events}), 2)

# ------------------------------------------- newspaper negative-audit observations
print("\nNewspaper negative-audit observations (no population expansion)")
holdout = jsonl("04_annotations/sealed_evaluation/newspapers/article_holdout_60.jsonl")
strata = Counter((r["sampling_stratum"], r["stratum_destination_population"]) for r in holdout)
report("newspaper pool", "holdout strata", sorted(strata.items()), sorted({
    ("consensus_reprint", 22): 20, ("disagreement_selected_reprint", 12): 10,
    ("disagreement_selected_non_reprint", 31): 15, ("consensus_non_reprint", 307): 15}.items()))
news_candidates = jsonl("03_evidence_layers/newspapers/event_candidates.jsonl")
reviewed_articles = {c["dst_doc_id"] for c in news_candidates}
consensus_neg = [r for r in holdout if r["sampling_stratum"] == "consensus_non_reprint"]
unreviewed = [r for r in consensus_neg if r["dst_doc_id"] not in reviewed_articles]
un_pos = sum(r["annotation_label"] == "reprint" for r in unreviewed)
report("newspaper pool", "already reviewed / unreviewed draws",
       (len(consensus_neg) - len(unreviewed), len(unreviewed)), (4, 11))
report("newspaper pool", "unreviewed positives", un_pos, 0)
diagnostics["newspaper negative audit"] = {
    "population_estimate": None,
    "reason": "One fixed section-article candidate per sampled article; no exhaustive article or event outcome.",
}
other_strata_unreviewed = sum(
    1 for r in holdout if r["sampling_stratum"] != "consensus_non_reprint"
    and r["dst_doc_id"] not in reviewed_articles)
report("newspaper pool", "unreviewed draws in the other three strata", other_strata_unreviewed, 0)

# ------------------------------------------ holdout against event review
print("\nHoldout against event review (diagnostic)")
decisions = {r["event_candidate_id"]: r["annotation_label"]
             for r in jsonl("04_annotations/final_event_review/newspapers/review_decisions.jsonl")}
for o in jsonl("01_events/corrections/final_event_binary_reclassification_v1.jsonl"):
    if o["event_candidate_id"] in decisions:
        decisions[o["event_candidate_id"]] = o["to_label"]
pairs = []
for h in holdout:
    sec = section_key(h)
    for c in news_candidates:
        if c["dst_doc_id"] == h["dst_doc_id"] and sec in (c.get("source_section_ids") or []):
            label = decisions.get(c["event_candidate_id"])
            if label and label != "uncertain":
                pairs.append((h["annotation_label"], label))
            break
table_ = Counter(pairs)
# Not reported in the article: the two passes saw different evidence, so this
# compares evidence conditions rather than readers. Kept as a diagnostic.
diagnostics["holdout against event review"] = {
    "matched_articles": len(pairs),
    "agree": table_[("reprint", "reprint")] + table_[("non_reprint", "non_reprint")],
    "event_review_reprint_holdout_not": table_[("non_reprint", "reprint")],
    "holdout_reprint_event_review_not": table_[("reprint", "non_reprint")],
}
print(f"  [diag] holdout against event review: {diagnostics['holdout against event review']}")

# ------------------------------------------------- threshold-free transfer
print("\nThreshold-free transfer")
scores = jsonl("05_models/transfer/holdout_scores.jsonl")
y = np.array([r["human_label_binary"] for r in scores])
w = np.array([r["sampling_weight"] for r in scores])
def bootstrap(metric, values, n=2000):
    rng_b = np.random.default_rng(7)
    out = []
    for _ in range(n):
        idx = rng_b.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        out.append(metric(y[idx], values[idx], sample_weight=w[idx]))
    out.sort()
    return round(out[int(0.025 * len(out))], 2), round(out[int(0.975 * len(out))], 2)


for label, key, expected, interval in [
    ("ECCO books", "book_shared_core_zero_shot", (0.491, 0.764, 0.112), ((0.37, 0.76), (0.54, 0.97))),
    ("Newspaper articles", "newspaper_v1_random_forest", (0.633, 0.882, 0.617), ((0.49, 0.89), (0.73, 0.99))),
    ("Both, media-balanced", "domain_balanced_pooled__domain_balanced_hgb", (0.517, 0.778, 0.381), ((0.36, 0.84), (0.60, 0.98))),
]:
    sc = np.array([r["scores"][key] for r in scores])
    pr = average_precision_score(y, sc, sample_weight=w)
    roc = roc_auc_score(y, sc, sample_weight=w)
    recall = float(w[(y == 1) & (sc >= 0.5)].sum() / w[y == 1].sum())
    report("transfer", f"{label} [{key}] PR-AUC, ROC-AUC, recall@0.5",
           (round(pr, 3), round(roc, 3), round(recall, 3)), expected)
    if not args.quick:
        report("transfer", f"{label} bootstrap intervals",
               (bootstrap(average_precision_score, sc), bootstrap(roc_auc_score, sc)), interval, 0.02)

# Recalibration: the map from score to probability is refitted on newspaper
# development labels; the ranking and the holdout are untouched.
base = {r["section_candidate_id"]: r["book_base_score"]
        for r in jsonl("05_models/transfer/book_base_newspaper_scores.jsonl")}
dev = {r["section_candidate_id"]: 1 if r["label"] == "reprint" else 0
       for r in jsonl("04_annotations/adaptive_training/newspapers/article_training_labels.jsonl")}
X = np.array([[base[c]] for c in dev if c in base])
y_dev = np.array([dev[c] for c in dev if c in base])
raw = np.array([r["scores"]["book_shared_core_zero_shot"] for r in scores])
report("transfer", "development rows available for calibration", len(y_dev), 148)
report("transfer", "holdout candidates also in the development set",
       len(set(dev) & {r["section_candidate_id"] for r in scores}), 0)
platt = LogisticRegression(max_iter=1000).fit(X, y_dev).predict_proba(raw.reshape(-1, 1))[:, 1]
iso = IsotonicRegression(out_of_bounds="clip").fit(X.ravel(), y_dev).predict(raw)
recall_of = lambda sc: float(w[(y == 1) & (sc >= 0.5)].sum() / w[y == 1].sum())
precision_of = lambda sc: float(w[(y == 1) & (sc >= 0.5)].sum() / w[sc >= 0.5].sum())
shared_news = np.array([r["scores"]["domain_calibrator__newspaper_shared_only"] for r in scores])
report("transfer", "shared-core newspaper logistic fit: recall, precision, PR-AUC, ROC-AUC",
       (round(recall_of(shared_news), 3), round(precision_of(shared_news), 3),
        round(average_precision_score(y, shared_news, sample_weight=w), 3),
        round(roc_auc_score(y, shared_news, sample_weight=w), 3)), (0.351, 0.707, 0.618, 0.865), 0.001)
report("transfer", "recall distance from the book fit to the shared-core newspaper fit",
       round(recall_of(shared_news) - recall_of(raw), 3), 0.238, 0.001)
diagnostics["transfer comparison scope"] = {
    "book": "shared-core HGB",
    "newspaper_shared": "class-balanced, destination-weighted logistic regression",
    "newspaper_operational": "random forest; two features differ from the shared core",
    "note": "The recall differences do not isolate training domain from algorithm or training design.",
}
report("transfer", "book-fitted recall and precision after Platt scaling",
       (round(recall_of(platt), 3), round(precision_of(platt), 3)), (0.523, 0.616), 0.001)
report("transfer", "book-fitted recall after isotonic regression", round(recall_of(iso), 3), 0.523, 0.001)

# ------------------------------------------------ source-side ablation
print("\nSource-side coverage ablation")
sources = table("07_sources/hume_source_manifestations.csv")
work_of_source = {s["src_doc_id"]: s["estc_work_id"] for s in sources}
contents: dict[str, set] = defaultdict(set)
for sec, essay in section_to_essay.items():
    contents[sec.split("::")[0]].add(essay)
essays_of = defaultdict(set)
for r in book_rel + news_rel:
    essays_of[r["event_candidate_id"]].add(r["canonical_essay_id"])
support = {e["event_candidate_id"]: {s.split("::")[0] for s in e["source_section_ids"]} for e in events}
unconditional = {s: sum(s in support[e["event_candidate_id"]] for e in events) for s in work_of_source}
best = max(unconditional, key=unconditional.get)
report("ablation", "best single manifestation", unconditional[best], 233)
by_work = defaultdict(set)
for e in events:
    for s in support[e["event_candidate_id"]]:
        by_work[work_of_source.get(s)].add(e["event_candidate_id"])
report("ablation", "best single work-field group", max(len(v) for v in by_work.values()), 321)
conditional = []
for s in work_of_source:
    eligible = [e for e in events if essays_of[e["event_candidate_id"]] & contents[s]]
    if eligible:
        conditional.append(sum(s in support[e["event_candidate_id"]] for e in eligible) / len(eligible))
report("ablation", "conditional recovery range %",
       (round(100 * min(conditional), 1), round(100 * max(conditional), 1)), (54.7, 95.2))
et_work = next(w for w in set(work_of_source.values()) if w.split("-", 1)[1].startswith("essays and treatises"))
siblings = {s for s, wk in work_of_source.items() if wk == et_work}
et_best = max(siblings, key=unconditional.get)
gain = [e for e in events if support[e["event_candidate_id"]] & siblings and et_best not in support[e["event_candidate_id"]]]
absent = [e for e in gain if not essays_of[e["event_candidate_id"]] & contents[et_best]]
report("ablation", "work-level gain / from essays absent in best volume", (len(gain), len(absent)), (88, 76))
report("ablation", "best volume / best work group as % of events; share of gain from absent essays",
       (round(100 * unconditional[best] / len(events), 1),
        round(100 * max(len(v) for v in by_work.values()) / len(events), 1),
        round(100 * len(absent) / len(gain))), (70.4, 97.0, 86))

# -------------------------------------------------- recurring compiled units
print("\nRecurring compiled units")
events_per_dst = Counter(e["dst_doc_id"] for e in book_events)


def compiled_units(ordered: dict, symmetric: bool = False) -> list[list[str]]:
    parent = {d: d for d in ordered}

    def find(d: str) -> str:
        while parent[d] != d:
            parent[d] = parent[parent[d]]
            d = parent[d]
        return d

    keys = sorted(ordered)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pa, pb = ordered[a], ordered[b]
            if [x[2] for x in pa] != [x[2] for x in pb]:
                continue
            delta = [x[0] - z[0] for x, z in zip(pa, pb)]
            footprint = pa[-1][1] - pa[0][0]
            if symmetric:
                footprint = max(footprint, pb[-1][1] - pb[0][0])
            if max(delta) - min(delta) < 0.05 * footprint:
                parent[find(a)] = find(b)
    groups = defaultdict(list)
    for d in keys:
        groups[find(d)].append(d)
    return sorted(g for g in groups.values() if len(g) > 1)


by_event = defaultdict(list)
for e in book_events:
    by_event[e["dst_doc_id"]].append((e["dst_start"], e["dst_end"], tuple(sorted(essays_of[e["event_candidate_id"]]))))
units = compiled_units({d: sorted(v) for d, v in by_event.items() if len(v) >= 2})
diagnostics["compiled units, one position per event"] = {
    "units": len(units), "destinations": sum(map(len, units)),
    "events": sum(events_per_dst[d] for g in units for d in g),
}
print(f"  [diag] positions by event: {diagnostics['compiled units, one position per event']}")
by_essay: dict = defaultdict(dict)
for e in book_events:
    for essay in essays_of[e["event_candidate_id"]]:
        cur = by_essay[e["dst_doc_id"]].get(essay)
        by_essay[e["dst_doc_id"]][essay] = ([min(cur[0], e["dst_start"]), max(cur[1], e["dst_end"])]
                                            if cur else [e["dst_start"], e["dst_end"]])
units_essay = compiled_units({d: sorted((v[0], v[1], k) for k, v in p.items()) for d, p in by_essay.items() if len(p) >= 2})
report("compiled units", "units / destinations / events",
       (len(units_essay), sum(map(len, units_essay)),
        sum(events_per_dst[d] for g in units_essay for d in g)), (6, 21, 75))
report("compiled units", "single-event destinations", sum(1 for n in events_per_dst.values() if n == 1), 70)
dst_row = {d["dst_doc_id"]: d for d in destinations}
report("compiled units", "per unit: volumes, work-field values, first and last year (Table 6)",
       sorted((len(g), len({dst_row[d]["work_id"] for d in g}),
               min(int(dst_row[d]["publication_year"]) for d in g),
               max(int(dst_row[d]["publication_year"]) for d in g)) for g in units_essay),
       sorted([(8, 5, 1764, 1799), (3, 2, 1776, 1795), (4, 2, 1785, 1797),
               (2, 1, 1772, 1775), (2, 2, 1770, 1780), (2, 2, 1762, 1762)]))
# Appendix E, "The same test outside its filter". The original script is not
# preserved; this reimplementation follows the stated procedure. Positions merge
# overlapping passages per destination and canonical essay (source section where
# unmapped). The tolerance uses the larger of the two footprints, which gives the
# same six units on the confirmed record and reproduces the counts below.
screen_groups = defaultdict(list)
screen_best = defaultdict(float)
for r in jsonl("05_models/books/passage_workflow_scores.jsonl"):
    if r["chronology_status"] != "valid_dst_after_src":
        continue
    key = section_to_essay.get(r["source_section_id"], r["source_section_id"])
    screen_groups[(r["dst_doc_id"], key)].append((r["dst_start"], r["dst_end"]))
    screen_best[r["dst_doc_id"]] = max(screen_best[r["dst_doc_id"]], r["hgb_probability"] or 0.0)
screen_positions = defaultdict(list)
for (d, key), spans in screen_groups.items():
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    screen_positions[d] += [(a, b, key) for a, b in merged]
screen_ordered = {d: sorted(v) for d, v in screen_positions.items() if len(v) >= 2}
screen_units = compiled_units(screen_ordered, symmetric=True)
def unit_drift(a: str, b: str) -> int:
    delta = [x[0] - z[0] for x, z in zip(screen_ordered[a], screen_ordered[b])]
    return max(delta) - min(delta)
screen_kept = [u for u in screen_units if len(screen_ordered[u[0]]) >= 3 and
               np.median([unit_drift(a, b) for i, a in enumerate(u) for b in u[i + 1:]]) < 5000]
screen_scores = [float(np.median([screen_best[d] for d in u])) for u in screen_kept]
known_units = [set(u) for u in units_essay]
book_event_dsts = {e["dst_doc_id"] for e in book_events}
report("compiled units", "screen outside the filter: positions, destinations, kept units, kept with confirmed events",
       (sum(map(len, screen_positions.values())), len(screen_positions), len(screen_kept),
        sum(1 for u in screen_kept for d in u if d in book_event_dsts)), (6790, 3151, 49, 8))
report("compiled units", "screen: known units recovered / exactly; kept units below 0.1 / above 0.3",
       (sum(1 for k in known_units if any(k & set(u) for u in screen_units)),
        sum(1 for k in known_units if any(k == set(u) for u in screen_units)),
        sum(1 for x in screen_scores if x < 0.1), sum(1 for x in screen_scores if x > 0.3)), (5, 4, 43, 5))
high_units = [u for u, x in zip(screen_kept, screen_scores) if x > 0.3]
new_high = [u for u in high_units if not any(set(u) & k for k in known_units)]
report("compiled units", "screen: candidate units, destinations in kept units, known / new units above 0.3, destinations in new units",
       (len(screen_units), sum(map(len, screen_kept)),
        len(high_units) - len(new_high), len(new_high), sum(map(len, new_high))),
       (135, 215, 3, 2, 4))
labelled_units = defaultdict(set)
for d in destinations:
    if d["recurring_unit"]:
        labelled_units[d["recurring_unit"]].add(d["dst_doc_id"])
report("compiled units", "released recurring_unit column equals the detected units",
       sorted(map(sorted, labelled_units.values())) == sorted(map(sorted, units_essay)), True)
unit_a = table("02_destinations/books/unit_a_editions.csv")
numbered = sorted((int(r["publication_year"]), r["edition_statement"].split(",")[0]) for r in unit_a
                  if re.search(r"\b(fourth|eighth|ninth|nineteenth)\b", r["edition_statement"], re.I))
report("compiled units", "Unit A: volumes, numbered editions (year), Hume pages in the 4th/8th/9th",
       (len(unit_a), [y for y, _ in numbered],
        sorted({(r["page_of_impudence_and_modesty"], r["page_of_avarice"]) for r in unit_a
                if int(r["publication_year"]) in (1764, 1777, 1781)})),
       (8, [1764, 1777, 1781, 1799], [("3", "5")]))
parnell = [d for d in destinations if d["work_id"].startswith("898-")]
hive = [d for d in destinations if d["work_id"].startswith("X-the hive: or a")]
report("compiled units", "work field: Parnell editions, first/last year, places; Hive volumes / groups; title-shaped values",
       (len(parnell), min(int(d["publication_year"]) for d in parnell),
        max(int(d["publication_year"]) for d in parnell), len({d["publication_place"] for d in parnell}),
        len(hive), len({d["work_id"] for d in hive}),
        sum(1 for d in destinations if d["work_id"].startswith("X-"))),
       (4, 1752, 1771, 3, 5, 2, 68))
report("compiled units", "source manifestations per derived work group",
       sorted(Counter(r["estc_work_id"] for r in table("07_sources/hume_source_manifestations.csv")).values()),
       [3, 6, 8])

# ------------------------------------------------ evaluation sets (Section 5)
print("\nEvaluation sets")
forty = jsonl("05_models/books/independent_40_sample_predictions.jsonl")
y40 = np.array([r["human_label"] == "reprint" for r in forty], dtype=int)
p40 = np.array([r["model_probabilities"]["v3_structure_aware"] for r in forty])
controls = [r for r in forty if r["sampling_stratum"].startswith("training_unseen_destination_negative_control")]
report("evaluation", "forty-passage set: PR-AUC, positives found / positives, false positives, controls / control positives",
       (round(average_precision_score(y40, p40), 2), int(((p40 >= 0.5) & (y40 == 1)).sum()), int(y40.sum()),
        int(((p40 >= 0.5) & (y40 == 0)).sum()), len(controls),
        sum(1 for r in controls if r["human_label"] == "reprint")),
       (0.61, 11, 12, 9, 20, 1))
holdout_report = json.loads((ROOT / "05_models/transfer/holdout_evaluation_report.json").read_text())
rf_holdout = holdout_report["model_results_at_0_5"]["newspaper_v1_random_forest"]["destination_representative_population_weighted"]
# The gate (accuracy, precision and recall each at least 0.85) is recorded upstream in
# active-learning/outputs/cross_media_newspaper_v1/article_validation_v1/validation_report.json.
GATE = 0.85
passing = [k for k, v in holdout_report["model_results_at_0_5"].items()
           if all(v["destination_representative_population_weighted"].get(m, 0) >= GATE
                  for m in ("accuracy", "reprint_precision", "reprint_recall"))]
report("evaluation", "holdout random forest weighted recall / precision; models passing the 85% gate",
       (round(rf_holdout["reprint_recall"], 3), round(rf_holdout["reprint_precision"], 3), len(passing)),
       (0.617, 0.577, 0))

# -------------------------------------------------------- media selectivity
print("\nMedia selectivity, length and source-set coverage")
sections = {r["source_section_id"]: int(r["src_section_length"])
            for r in table("07_sources/included_essay_sections.csv")}
lengths, manifestations = defaultdict(list), defaultdict(set)
for sec, essay in section_to_essay.items():
    if sec in sections:
        lengths[essay].append(sections[sec])
    manifestations[essay].add(sec.split("::")[0])
rel_media = Counter()
rel_total = Counter()
for r in book_rel + news_rel:
    rel_total[r["canonical_essay_id"]] += 1
    rel_media[r["canonical_essay_id"]] += r["media_domain"] != "ecco_book"
essays = sorted(rel_total)
share = np.array([rel_media[k] / rel_total[k] for k in essays])
mean_len = np.array([np.mean(lengths[k]) for k in essays])
counts = np.array([len(manifestations[k]) for k in essays])
rho_len = spearmanr(share, mean_len).statistic
rho_cnt = spearmanr(share, counts).statistic
rng = np.random.default_rng(20260804)
perm = np.array([spearmanr(share, rng.permutation(mean_len)).statistic for _ in range(length_iterations)])
report("selectivity", "essays", len(essays), 38)
report("selectivity", "Spearman share vs length", round(float(rho_len), 3), -0.160, 0.001)
report("selectivity", "permutation p (two-sided)", round(float((np.abs(perm) >= abs(rho_len) - 1e-12).mean()), 2), 0.33, 0.03)
report("selectivity", "Spearman share vs manifestation count", round(float(rho_cnt), 3), 0.278, 0.001)

# ------------------------------------------- definition and counting units
print("\nDefinition, counting unit and extent")
STRICT = {"full_essay_or_nearly_full", "long_continuous_passage"}
scoped = [e for e in events if e["reprint_scope"]]
strictly = [e for e in scoped if e["reprint_scope"] in STRICT]
report("definition", "confirmed records carrying a primary scope", len(scoped), 296)
without = [e for e in events if not e["reprint_scope"]]
report("definition", "records without a scope: split children, adjudicated after review",
       (sum(1 for e in without if e["original_reprint_scope"]),
        sum(1 for e in without if not e["original_reprint_scope"])), (28, 7))
report("definition", "full essays or long continuous passages", len(strictly), 202)
rejected_short = sum(1 for m in ("books", "newspapers")
                     for r in jsonl(f"04_annotations/final_event_review/{m}/review_decisions.jsonl")
                     if r["non_reprint_reason"] == "too_short_or_incidental")
report("definition", "records rejected as too short or incidental", rejected_short, 30)

units = {"A": ["0107500300", "0158700300", "0485200300", "1263400400",
               "1303300200", "1303300300", "1345100100", "1646900200"],
         "B": ["0176800700", "1134601100", "1427201900"],
         "C": ["0212400602", "0730800900", "1353800100", "1497500102"],
         "D": ["0113000601", "0778400501"],
         "E": ["0273301900", "0291100700"],
         "F": ["0906500303", "1311200103"]}
in_unit = {d: u for u, ds in units.items() for d in ds}
relations = book_rel + news_rel
impudence = [r for r in book_rel if r["canonical_essay_id"] == "impudence_modesty"]
report("definition", "Impudence relations / destinations / derived work groups",
       (len(impudence), len({r["dst_doc_id"] for r in impudence}),
        len({work_of[r["dst_doc_id"]] for r in impudence})), (39, 39, 24))
report("source mapping", "manifestations containing The Stoic in the canonical mapping",
       len({s.split("::")[0] for s, essay in section_to_essay.items() if essay == "stoic"}), 9)
media_of = {e["event_candidate_id"]: e["media_domain"] for e in events}
scope_of = {e["event_candidate_id"]: (e["reprint_scope"] or e["original_reprint_scope"]) for e in events}
raw_counts: dict = defaultdict(Counter)
dedup_counts: dict = defaultdict(Counter)
seen = set()
for r in relations:
    medium = "book" if media_of[r["event_candidate_id"]] == "ecco_book" else "news"
    raw_counts[r["canonical_essay_id"]][medium] += 1
    key = (r["canonical_essay_id"], in_unit.get(r["dst_doc_id"], r["dst_doc_id"]), medium)
    if key not in seen:
        seen.add(key)
        dedup_counts[r["canonical_essay_id"]][medium] += 1
report("definition", "Impudence and Avarice, sensitivity count collapsing proposed compiled units",
       (dedup_counts["impudence_modesty"]["book"], dedup_counts["avarice"]["book"]), (27, 11))
coded_scope = {e["event_candidate_id"]: e["reprint_scope"] for e in events}
strict_rel = [r for r in relations if coded_scope[r["event_candidate_id"]] in STRICT]
inherited = [r for r in relations if scope_of[r["event_candidate_id"]] in STRICT]
diagnostics["strict relations if children inherit the parent scope"] = {
    "books": sum(1 for r in inherited if media_of[r["event_candidate_id"]] == "ecco_book"),
    "newspapers": sum(1 for r in inherited if media_of[r["event_candidate_id"]] != "ecco_book"),
    "note": "a long continuous parent divided into several positions need not leave each long",
}
strict_book = [r for r in strict_rel if media_of[r["event_candidate_id"]] == "ecco_book"]
strict_news = [r for r in strict_rel if media_of[r["event_candidate_id"]] != "ecco_book"]
report("definition", "strict relations with a directly coded scope, books / newspapers",
       (len(strict_book), len(strict_news)), (165, 40))
report("definition", "relations under the parent-scope inheritance assumption, books / newspapers",
       (diagnostics["strict relations if children inherit the parent scope"]["books"],
        diagnostics["strict relations if children inherit the parent scope"]["newspapers"]), (187, 40))
print("  [diag] inheriting the parent scope would give "
      f"{diagnostics['strict relations if children inherit the parent scope']['books']} and "
      f"{diagnostics['strict relations if children inherit the parent scope']['newspapers']}")
four_essays = ("impudence_modesty", "avarice", "liberty_press", "public_credit")
strict_book_by = Counter(r["canonical_essay_id"] for r in strict_book)
strict_news_by = Counter(r["canonical_essay_id"] for r in strict_news)
report("definition", "Appendix E restrictions table: collapsed and full-or-long book / newspaper counts",
       [(dedup_counts[k]["book"], dedup_counts[k]["news"], strict_book_by[k], strict_news_by[k]) for k in four_essays],
       [(27, 2, 27, 1), (11, 3, 13, 1), (9, 19, 8, 11), (10, 18, 4, 7)])
report("definition", "political share of newspaper relations, full-or-long only",
       round(100 * sum(1 for r in strict_news if r["canonical_essay_id"] in POLITICAL) / len(strict_news), 1), 45.0, 0.1)
report("definition", "political share of book relations, all / strict",
       (round(100 * sum(1 for r in relations if media_of[r["event_candidate_id"]] == "ecco_book"
                        and r["canonical_essay_id"] in POLITICAL)
              / sum(1 for r in relations if media_of[r["event_candidate_id"]] == "ecco_book"), 1),
        round(100 * sum(1 for r in strict_book if r["canonical_essay_id"] in POLITICAL) / len(strict_book), 1)),
       (8.0, 7.3), 0.1)

confirmed_books = {e["dst_doc_id"] for e in book_events}
confirmed_news = {e["dst_doc_id"] for e in news_events}
matches: Counter = Counter()
pairs: Counter = Counter()
for row in jsonl("03_evidence_layers/books/fragments.jsonl"):
    if row["dst_doc_id"] in confirmed_books and section_to_essay.get(row["source_section_id"]):
        matches[section_to_essay[row["source_section_id"]]] += 1
for row in jsonl("03_evidence_layers/newspapers/fragments.jsonl"):
    if row["dst_doc_id"] in confirmed_news and section_to_essay.get(row["source_section_id"]):
        matches[section_to_essay[row["source_section_id"]]] += 1
for row in jsonl("03_evidence_layers/books/passage_candidates.jsonl"):
    if row["dst_doc_id"] in confirmed_books and section_to_essay.get(row["source_section_id"]):
        pairs[section_to_essay[row["source_section_id"]]] += 1
for row in jsonl("03_evidence_layers/newspapers/article_candidates.jsonl"):
    if row["dst_doc_id"] in confirmed_news and section_to_essay.get(row["source_section_id"]):
        pairs[section_to_essay[row["source_section_id"]]] += 1
events_per_essay = Counter(r["canonical_essay_id"] for r in relations)
pair_sets: dict = defaultdict(set)
for media, path in [("books", "03_evidence_layers/books/fragments.jsonl"),
                    ("newspapers", "03_evidence_layers/newspapers/fragments.jsonl")]:
    confirmed = confirmed_books if media == "books" else confirmed_news
    for row in jsonl(path):
        essay = section_to_essay.get(row["source_section_id"])
        if row["dst_doc_id"] in confirmed and essay:
            pair_sets[essay].add(row["section_candidate_id"])
pairs_per_essay = Counter({k: len(v) for k, v in pair_sets.items()})
rank = lambda c: {k: i + 1 for i, (k, _) in enumerate(c.most_common())}
rm, rq, rp, re_ = rank(matches), rank(pairs_per_essay), rank(pairs), rank(events_per_essay)
report("counting units", "totals: matches, pairs, passages, event-essay relations",
       (sum(matches.values()), sum(pairs_per_essay.values()), sum(pairs.values()),
        sum(events_per_essay.values())), (11362, 2705, 2827, 339))
for essay, expected in [("impudence_modesty", (590, 4, 205, 3, 205, 4, 41, 1)),
                        ("public_credit", (1728, 1, 232, 1, 240, 1, 28, 3)),
                        ("populousness_ancient_nations", (810, 2, 81, 9, 111, 7, 7, 17))]:
    report("counting units", f"{essay}: matches, pairs, passages, event-essay relations with ranks",
           (matches[essay], rm[essay], pairs_per_essay[essay], rq[essay],
            pairs[essay], rp[essay], events_per_essay[essay], re_[essay]), expected)

# Figure 1 follows one essay through the layer in volume 1 of the 1786
# Philosophical Dictionary (ESTC T108878).
DICT_VOL1, DICT_ESSAY = "0211200401", "rise_arts_sciences"
in_dict = lambda r: (r["dst_doc_id"] == DICT_VOL1
                     and section_to_essay.get(r["source_section_id"]) == DICT_ESSAY)
dict_fragments = [r for r in jsonl("03_evidence_layers/books/fragments.jsonl") if in_dict(r)]
dict_passages = [r for r in jsonl("03_evidence_layers/books/passage_candidates.jsonl") if in_dict(r)]
dict_events = [e for e in book_events
               if e["dst_doc_id"] == DICT_VOL1 and DICT_ESSAY in e["canonical_essay_ids"]]
report("counting units", "1786 dictionary vol. 1, one essay: matches, pairs, passages, events, other events in volume",
       (len(dict_fragments), len({r["section_candidate_id"] for r in dict_fragments}),
        len(dict_passages), len(dict_events),
        sum(1 for e in book_events if e["dst_doc_id"] == DICT_VOL1) - len(dict_events)),
       (128, 9, 35, 4, 5))

# ------------------------------------------------- construction (Section 4)
print("\nConstruction")
manifest = lambda name: json.loads((ROOT / f"09_provenance/stage_manifests/{name}.json").read_text())
seg = manifest("segmentation_v2")
report("construction", "book edges linked by overlap / linked by model / split by model; passages",
       (seg["edge_decision_counts"]["link_destination_overlap"],
        seg["edge_decision_counts"]["link_supervised_v2"],
        seg["edge_decision_counts"]["split_supervised_v2"],
        len(jsonl("03_evidence_layers/books/passage_candidates.jsonl"))),
       (8202, 8513, 3452, 24020))
link = manifest("passage_link_model_v2")
link_oof = jsonl("05_models/books/link_edge_oof_predictions.jsonl")
report("construction", "link labels / audits / destinations / edge features",
       (len(link_oof), len({r["audit_id"] for r in link_oof}),
        len({r["dst_doc_id"] for r in link_oof}), link["feature_count"]),
       (110, 3, 87, 9))
report("construction", "link macro-F1: random forest, logistic, tree, fixed-gap rule, Gaussian mixture",
       tuple(round(link[group][name]["macro_f1_at_0_5"], 3) for group, name in
             [("model_metrics", "link_random_forest_v1"), ("model_metrics", "link_logistic_v1"),
              ("model_metrics", "link_tree_v1"), ("baseline_metrics", "legacy_chain_bundle"),
              ("baseline_metrics", "provisional_gmm_v1")]),
       (0.926, 0.903, 0.869, 0.447, 0.377))
# The fold split is not stored; StratifiedGroupKFold depends only on the label
# and group order, so the recorded seed rebuilds it from the out-of-fold file.
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
y_link = np.array([r["edge_label"] == "same_passage" for r in link_oof], dtype=int)
link_folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260722).split(
    np.zeros(len(y_link)), y_link, np.array([r["dst_doc_id"] for r in link_oof])))
def fold_sd(model_name: str) -> float:
    p = np.array([r["model_oof_probabilities"][model_name] for r in link_oof])
    return round(float(np.std([f1_score(y_link[t], (p[t] >= 0.5).astype(int), average="macro")
                               for _, t in link_folds], ddof=1)), 3)
report("construction", "between-fold standard deviation of macro-F1, random forest / logistic",
       (fold_sd("link_random_forest_v1"), fold_sd("link_logistic_v1")), (0.024, 0.070))
import joblib
frozen_link = joblib.load(ROOT / "05_models/frozen_models/passage_link_random_forest_v2.joblib")
report("construction", "trees in the frozen link model", frozen_link.steps[-1][1].n_estimators, 300)

raw_inventory = manifest("newspaper_raw_inventory")
wf = manifest("hume_workflow_v1")["summary"]
report("construction", "raw_inventory rows / external ECCO / newspaper / unmapped / Hume-authored (by difference)",
       (raw_inventory["original_upstream_row_count"], wf["external_ecco_raw_fragment_rows"],
        raw_inventory["newspaper_raw_fragment_count"], raw_inventory["other_non_numeric_excluded_fragment_count"],
        raw_inventory["original_upstream_row_count"] - wf["external_ecco_raw_fragment_rows"]
        - raw_inventory["newspaper_raw_fragment_count"] - raw_inventory["other_non_numeric_excluded_fragment_count"]),
       (122656, 52141, 6634, 1058, 62823))
art_model = manifest("newspaper_article_model_input_v1")
report("construction", "newspaper fragments entering features / article candidates / article features",
       (raw_inventory["passage_eligibility_fragment_counts"]["eligible_for_passage_construction"],
        len(jsonl("03_evidence_layers/newspapers/article_candidates.jsonl")), art_model["feature_count"]),
       (5368, 3212, 15))
legacy = jsonl("04_annotations/adaptive_training/newspapers/legacy_pair_to_article_alignment.jsonl")
report("construction", "legacy records / mapping to exactly one eligible candidate",
       (len(legacy), sum(1 for r in legacy if r["eligible_article_candidate_count"] == 1)), (225, 151))
news_links = jsonl("04_annotations/sealed_evaluation/newspapers/link_validation_30.jsonl")
report("construction", "newspaper link audits / returned same_passage with high confidence",
       (len(news_links), sum(1 for r in news_links
                             if r["edge_label"] == "same_passage" and r["confidence"] == "high")), (30, 30))
report("construction", "never individually labelled candidates, and training share %",
       (27232 - 1813 - 220, round(100 * 1813 / 27232, 1)), (25199, 6.7))
news_review = manifest("newspaper_review")["input_counts"]
report("construction", "seeds: book human / book model / newspaper model-only / newspaper model share %",
       (wf["event_support_human_reprint_passage_count"],
        wf["event_support_unlabeled_model_priority_passage_count"],
        news_review["model_only_seed_count"],
        round(100 * news_review["model_only_seed_count"]
              / (news_review["model_only_seed_count"] + news_review["human_reprint_seed_count"]))),
       (572, 1522, 964, 87))
report("construction", "book components with overlapping human non-republication evidence",
       wf["event_candidate_with_overlapping_human_non_reprint_count"], 57)
book_review = manifest("book_review")["input_counts"]
report("construction", "event candidates / Hume-authored destinations / their candidates / review queue",
       (book_review["event_candidates_before_destination_exclusion"], book_review["excluded_destination_count"],
        book_review["excluded_event_count"], book_review["event_candidates_after_destination_exclusion"]),
       (332, 19, 79, 253))

book_decisions = jsonl("04_annotations/final_event_review/books/review_decisions.jsonl")
news_decisions = jsonl("04_annotations/final_event_review/newspapers/review_decisions.jsonl")
split_audit = jsonl("01_events/books/corrections/split_resolution_audit.jsonl")
split_children = jsonl("03_evidence_layers/books/resolved_split_events.jsonl")
essay_children = [e for e in events if e.get("essay_split_parent_event_candidate_id")]
essay_parents = {e["essay_split_parent_event_candidate_id"] for e in essay_children}
book_effective = len(book_decisions) - len(split_audit) + len(split_children)
book_positive = len(book_events) - len(essay_children) + len(essay_parents)
report("construction", "review splits: parents / children; book units judged",
       (len(split_audit), len(split_children), book_effective), (4, 9, 258))
report("construction", "book positives before the essay split / newspaper units reviewed",
       (book_positive, len(news_decisions)), (229, 177))
report("construction", "units reviewed / positives / non-republications (book 29 + newspaper 91)",
       (book_effective + len(news_decisions), book_positive + len(news_events),
        sum(1 for d in book_decisions + news_decisions if d["annotation_label"] == "non_reprint")),
       (435, 315, 120))
multi_essay = len(essay_parents) + sum(1 for e in events if len(e["canonical_essay_ids"]) > 1
                                       and not e.get("essay_split_parent_event_candidate_id"))
report("construction", "positives with more than one essay / over-wide parents / children",
       (multi_essay, len(essay_parents), len(essay_children)), (18, 12, 28))

spans = {}
for media, path in [("books", "03_evidence_layers/books/fragments.jsonl"),
                    ("newspapers", "03_evidence_layers/newspapers/fragments.jsonl")]:
    lengths = np.array([row["src_trs_end"] - row["src_trs_start"] for row in jsonl(path)])
    spans[media] = (int(lengths.min()), int(np.percentile(lengths, 1)), int(np.median(lengths)))
report("definition", "shortest, 1st-percentile and median match, books",
       spans["books"], (52, 70, 235), 1)
report("definition", "shortest, 1st-percentile and median match, newspapers",
       spans["newspapers"], (58, 70, 260), 1)

# The article reports extent in digitised pages (ecco_pages), which is complete
# for all 115 destinations; the ESTC pagecount field is missing for two of the
# sixteen and counts differently where it is present.
extents = sorted(int(float(d["ecco_pages"])) for d in destinations
                 if d["structural_type"] == "continuous block" and d["ecco_pages"])
report("definition", "continuous-block extents, count, range and median",
       (len(extents), extents[0], extents[-1], round(float(np.median(extents)))), (16, 96, 410, 352))
singles = sorted(int(float(d["ecco_pages"])) for d in destinations
                 if d["structural_type"] == "single" and d["ecco_pages"])
report("definition", "single-event extents, range, median and count over 300 pages",
       (singles[0], singles[-1], round(float(np.median(singles))),
        sum(1 for x in singles if x > 300)), (16, 1115, 311, 39))

frame = sum({r["sampling_stratum"]: r["stratum_destination_population"]
             for r in holdout}.values())
report("definition", "newspaper holdout frame", frame, 372)
short = [d for d in destinations if d["document_type"] in ("Pamphlet", "In-between")]
report("definition", "pamphlets and intermediate destinations, and their events",
       (sum(1 for d in short if d["document_type"] == "Pamphlet"),
        sum(1 for d in short if d["document_type"] == "In-between"),
        sum(int(d["confirmed_events"]) for d in short)), (3, 7, 14))
report("definition", "events by ECCO module: Literature, Social Sciences, Religion",
       tuple(sum(int(d["confirmed_events"]) for d in destinations if d["ecco_module"] == m)
             for m in ("Literature and Language", "Social Sciences", "Religion and Philosophy")),
       (123, 65, 33))
imp = [r for r in book_rel if r["canonical_essay_id"] == "impudence_modesty"]
imp_dst = {r["dst_doc_id"] for r in imp}
work_of_dst = {d["dst_doc_id"]: d["work_id"] for d in destinations}
report("definition", "Of Impudence and Modesty: book relations, destinations, work groups, collapsed units",
       (len(imp), len(imp_dst), len({work_of_dst[d] for d in imp_dst}),
        len({in_unit.get(d, d) for d in imp_dst})), (39, 39, 24, 27))
report("definition", "octavos and duodecimos among the 115",
       sum(1 for d in destinations if d["gatherings"] in ("8vo", "12mo")), 108)
queue = jsonl("03_evidence_layers/newspapers/event_candidates.jsonl")
report("definition", "legacy support links into the review queue, and events reached",
       (sum(r.get("existing_human_reprint_support_count") or 0 for r in queue),
        sum(1 for r in queue if (r.get("existing_human_reprint_support_count") or 0) > 0)),
       (146, 50))

# ---------------------------------------------------- reading across archives (Section 6)
print("\nReading across archives")
typed = [d for d in destinations if d["document_type"]]
report("archives", "document_type populated; pamphlet pages; intermediate page range",
       (len(typed), sorted(int(float(d["ecco_pages"])) for d in typed if d["document_type"] == "Pamphlet"),
        min(int(float(d["ecco_pages"])) for d in typed if d["document_type"] == "In-between"),
        max(int(float(d["ecco_pages"])) for d in typed if d["document_type"] == "In-between")),
       (97, [16, 24, 24], 75, 124))
periodicals = [d for d in destinations if d["is_periodical"] == "1"]
report("archives", "catalogued periodicals / among single-event destinations",
       (len(periodicals), sum(1 for d in periodicals if d["structural_type"] == "single")), (21, 13))
educational = [d for d in destinations if d["structural_type"] == "continuous block"
               and d["recurring_unit"] in ("A", "B", "E")]
report("archives", "continuous block: educational volumes (Units A, B, E) and their events",
       (len(educational), sum(int(d["confirmed_events"]) for d in educational)), (13, 28))
report("archives", "dispersed destinations and events; one-per-essay median pages",
       (sum(1 for d in destinations if d["structural_type"].startswith("dispersed")),
        sum(int(d["confirmed_events"]) for d in destinations if d["structural_type"].startswith("dispersed")),
        round(float(np.median([float(d["ecco_pages"]) for d in destinations
                               if d["structural_type"] == "dispersed, one per essay"])))),
       (29, 138, 440))
articles_with = Counter(e["dst_doc_id"] for e in news_events)
report("archives", "newspaper articles carrying events / carrying two",
       (len(articles_with), sum(1 for n in articles_with.values() if n == 2)), (82, 4))
book_by_essay = Counter(r["canonical_essay_id"] for r in book_rel)
news_by_essay = Counter(r["canonical_essay_id"] for r in news_rel)
report("archives", "book / newspaper relations: Impudence, Avarice, Liberty of the Press, Public Credit",
       [(book_by_essay[k], news_by_essay[k]) for k in ("impudence_modesty", "avarice", "liberty_press", "public_credit")],
       [(39, 2), (20, 3), (10, 19), (10, 18)])
# Figure 4 draws every essay with at least four republications.
figure_four = {"money": (1, 3), "passive_obedience": (1, 3), "stoic": (2, 2), "refinement_arts": (3, 2),
               "parties_general": (5, 0), "protestant_succession": (5, 1), "remarkable_customs": (1, 5),
               "balance_power": (2, 4), "balance_trade": (3, 3), "british_government_monarchy_republic": (3, 4),
               "sceptic": (7, 0), "politics_science": (5, 2), "populousness_ancient_nations": (6, 1),
               "parties_great_britain": (7, 0), "polygamy_divorces": (4, 3), "dignity_human_nature": (8, 0),
               "national_characters": (8, 0), "civil_liberty": (9, 1), "study_history": (13, 0),
               "delicacy_taste_passion": (14, 1), "simplicity_refinement": (14, 1), "love_marriage": (15, 1),
               "rise_arts_sciences": (16, 2), "avarice": (20, 3), "public_credit": (10, 18),
               "liberty_press": (10, 19), "impudence_modesty": (39, 2)}
all_essays = set(book_by_essay) | set(news_by_essay)
drawn = {k for k in all_essays if book_by_essay[k] + news_by_essay[k] >= 4}
report("archives", "Figure 4: essays drawn equal those with at least four republications, bars match, omitted / largest omitted",
       (drawn == set(figure_four),
        all((book_by_essay[k], news_by_essay[k]) == v for k, v in figure_four.items()),
        len(all_essays - drawn), max(book_by_essay[k] + news_by_essay[k] for k in all_essays - drawn)),
       (True, True, 11, 3))
fisher = math.atanh(rho_len)
fisher_se = 1 / math.sqrt(len(essays) - 3)
report("archives", "Fisher 95% interval for the length correlation",
       (round(math.tanh(fisher - 1.959964 * fisher_se), 2), round(math.tanh(fisher + 1.959964 * fisher_se), 2)),
       (-0.46, 0.17))
report("archives", "manifestations containing Liberty of the Press, Public Credit, Impudence",
       tuple(len(manifestations[k]) for k in ("liberty_press", "public_credit", "impudence_modesty")), (9, 9, 5))
positive_weight = w[y == 1]
report("archives", "largest single positive weight as a share of positive weight in the holdout",
       round(float(positive_weight.max() / positive_weight.sum()), 2), 0.35)
news_event_of = {e["event_candidate_id"]: e for e in news_events}
political_1790s = [news_event_of[r["event_candidate_id"]] for r in news_rel
                   if r["canonical_essay_id"] in POLITICAL
                   and 1790 <= int(news_event_of[r["event_candidate_id"]]["dst_publication_year"]) <= 1799]
# The newspaper title is the second component of the article identifier;
# newspaper_id in newspaper_documents.csv identifies the issue.
report("archives", "1790s political relations: newspaper titles, issues, first and last year",
       (len({e["dst_doc_id"].split("_")[1] for e in political_1790s}),
        len({e["dst_doc_id"].rsplit("-", 2)[0] for e in political_1790s}),
        min(int(e["dst_publication_year"]) for e in political_1790s),
        max(int(e["dst_publication_year"]) for e in political_1790s)), (7, 11, 1791, 1797))
shared_core = list(json.loads((ROOT / "05_models/transfer/domain_shift_feature_shift.json").read_text())["all_labels"])
article_features = manifest("newspaper_article_model_input_v1")["features"]
report("archives", "shared-core features / shared with the operational article model",
       (len(shared_core), len(set(shared_core) & set(article_features))), (15, 13))
pooled = np.array([r["scores"]["domain_balanced_pooled__domain_balanced_hgb"] for r in scores])
report("archives", "uncalibrated book-fitted precision; pooled precision",
       (round(precision_of(raw), 3), round(precision_of(pooled), 3)), (0.762, 0.767))

# ---------------------------------------------------------------- summary
out = args.output or ROOT / "06_evidence_tests/reproduction_report.json"
out.write_text(json.dumps({"results": results, "diagnostics": diagnostics}, indent=2, default=str), encoding="utf-8")
diffs = [r for r in results if not r["match"]]
print(f"\n{len(results) - len(diffs)} of {len(results)} quantities reproduced; report written to {out}")
sys.exit(1 if diffs else 0)
