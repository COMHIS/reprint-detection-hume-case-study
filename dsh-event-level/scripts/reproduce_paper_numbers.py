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
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
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


def report(test: str, quantity: str, released, article, tolerance: float = 0.0) -> None:
    if isinstance(article, (int, float)) and isinstance(released, (int, float)):
        ok = abs(released - article) <= tolerance
    else:
        ok = released == article
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
for name, expected in [("single", [70, 70]), ("contiguous run", [16, 37]),
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
# labels are permuted across blocks, so every year keeps its number of works.
blocks = Counter((e["dst_publication_year"], work_of[e["dst_doc_id"]]) for e in dated)
b_years, b_works, b_sizes = [k[0] for k in blocks], [k[1] for k in blocks], list(blocks.values())


def block_statistic(years_: list) -> float:
    per_year: dict = defaultdict(Counter)
    for y_, w_, n_ in zip(years_, b_works, b_sizes):
        per_year[y_][w_] += n_
    return float(np.mean([max(c.values()) / sum(c.values()) for c in per_year.values()]))


rng = np.random.default_rng(7)
null = np.sort(np.array([block_statistic(list(rng.permutation(b_years))) for _ in range(annual_iterations)]))
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


obs_share = shares(political)
obs_rho = spearmanr(range(len(decades)), obs_share).statistic
rng = np.random.default_rng(20260804)
last_null, rho_null = [], []
for _ in range(composition_iterations):
    perm = rng.permutation(political)
    s = shares(perm)
    last_null.append(s[-1])
    rho_null.append(spearmanr(range(len(decades)), s).statistic)
last_null, rho_null = np.array(last_null), np.array(rho_null)
exceed = int((last_null >= obs_share[-1] - 1e-12).sum())
report("composition null", "1790s political share", round(float(obs_share[-1]), 3), 0.773)
report("composition null", "null mean of 1790s share", round(float(last_null.mean()), 3), 0.421, 0.005)
results.append({"test": "composition null",
                "quantity": f"permutations at or above the observed 1790s share (of {composition_iterations:,})",
                "released": exceed, "article": "< 0.0002 as a share of permutations",
                "match": exceed / composition_iterations < 0.0002})
print(f"  [{'ok ' if results[-1]['match'] else 'DIFF'}] permutations at or above observed: "
      f"released {exceed} of {composition_iterations:,} (p = {exceed / composition_iterations:.5f}) "
      f"| article p < 0.0002")
report("composition null", "Spearman decade vs share", round(float(obs_rho), 3), 0.600)
report("composition null", "trend p", round(float((rho_null >= obs_rho - 1e-12).mean()), 2), 0.18, 0.02)

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

# --------------------------------------------------- unreviewed book pool
print("\nUnreviewed book-side pool")
coverage = json.loads((ROOT / "04_annotations/sealed_evaluation/books/passage_coverage_negative_audit_60_manifest.json").read_text())
cov_rows = jsonl("04_annotations/sealed_evaluation/books/passage_coverage_negative_audit_60.jsonl")
cov_frame = {k: v["destination_count_before_cross_stratum_destination_exclusion"]
             for k, v in coverage["coverage_frame_counts"].items()}
cov_draws = Counter(r["sampling_stratum"] for r in cov_rows)
cov_pos = Counter(r["sampling_stratum"] for r in cov_rows if r["annotation_label"] == "reprint")
report("book pool", "coverage audit draws of frame",
       sorted((cov_draws[k], cov_frame[k]) for k in cov_frame), sorted([(4, 4), (16, 196), (20, 636), (20, 1773)]))
report("book pool", "coverage audit positives", sum(cov_pos.values()), 0)
v3 = jsonl("04_annotations/sealed_evaluation/books/passage_validation_v3_40.jsonl")
neg = [r for r in v3 if r["sampling_stratum"].startswith("training_unseen_destination_negative_control")]
neg_frame = neg[0]["stratum_destination_count"]
neg_pos = sum(r["annotation_label"] == "reprint" for r in neg)
report("book pool", "negative control draws / frame / positives", (len(neg), neg_frame, neg_pos), (20, 2197, 1))
ht = sum(cov_pos[k] * cov_frame[k] / cov_draws[k] for k in cov_frame) + neg_pos * neg_frame / len(neg)
lo, hi = wilson(neg_pos, len(neg))
report("book pool", "Horvitz-Thompson total", round(ht), 110)
report("book pool", "scaled Wilson interval", (round(lo * neg_frame), round(hi * neg_frame)), (20, 519))

# ----------------------------------------------- unreviewed newspaper pool
print("\nUnreviewed newspaper-side pool")
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
lo, hi = wilson(un_pos, len(unreviewed))
report("newspaper pool", "already reviewed / unreviewed draws",
       (len(consensus_neg) - len(unreviewed), len(unreviewed)), (4, 11))
report("newspaper pool", "unreviewed positives", un_pos, 0)
report("newspaper pool", "scaled Wilson interval", (round(lo * 307), round(hi * 307)), (0, 79))

# ------------------------------------------ holdout against event review
print("\nWhat consolidation and page images add")
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
report("two passes", "matched holdout articles", len(pairs), 49)
report("two passes", "agree", table_[("reprint", "reprint")] + table_[("non_reprint", "non_reprint")], 28)
report("two passes", "event review reprint, holdout not", table_[("non_reprint", "reprint")], 15)
report("two passes", "holdout reprint, event review not", table_[("reprint", "non_reprint")], 6)

# ------------------------------------------------- threshold-free transfer
print("\nThreshold-free transfer")
scores = jsonl("05_models/transfer/holdout_scores.jsonl")
y = np.array([r["human_label_binary"] for r in scores])
w = np.array([r["sampling_weight"] for r in scores])
for label, key, expected in [
    ("ECCO books", "book_shared_core_zero_shot", (0.491, 0.764, 0.112)),
    ("Newspaper articles", "newspaper_v1_random_forest", (0.633, 0.882, 0.617)),
    ("Both, media-balanced", "domain_balanced_pooled__domain_balanced_hgb", (0.517, 0.778, 0.381)),
]:
    s = np.array([r["scores"][key] for r in scores])
    pr = average_precision_score(y, s, sample_weight=w)
    roc = roc_auc_score(y, s, sample_weight=w)
    recall = float(w[(y == 1) & (s >= 0.5)].sum() / w[y == 1].sum())
    report("transfer", f"{label} [{key}] PR-AUC, ROC-AUC, recall@0.5",
           (round(pr, 3), round(roc, 3), round(recall, 3)), expected)

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

# -------------------------------------------------- recurring compiled units
print("\nRecurring compiled units")
events_per_dst = Counter(e["dst_doc_id"] for e in book_events)


def compiled_units(ordered: dict) -> list[list[str]]:
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
            if max(delta) - min(delta) < 0.05 * (pa[-1][1] - pa[0][0]):
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

# -------------------------------------------------------- media selectivity
print("\nMedia selectivity controlled for length and availability")
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

# ---------------------------------------------------------------- summary
out = ROOT / "06_evidence_tests/reproduction_report.json"
out.write_text(json.dumps({"results": results, "diagnostics": diagnostics}, indent=2, default=str), encoding="utf-8")
diffs = [r for r in results if not r["match"]]
print(f"\n{len(results) - len(diffs)} of {len(results)} reported quantities reproduced; report written to {out.relative_to(ROOT)}")
sys.exit(0)
