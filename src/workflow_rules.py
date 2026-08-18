from __future__ import annotations

from typing import Any, Callable

Row = dict[str, Any]
Rule = Callable[[Row], bool]


def number(row: Row, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def flag(row: Row, name: str) -> bool:
    return bool(row.get(name, False))


# ---------------------------------------------------------------------------
# Gates of the final rule cascade (Table 2 of the paper).
# These are building blocks, not stages; the reportable stages are further down.
# ---------------------------------------------------------------------------


def structural_gate(row: Row) -> bool:
    """Table 2, "Structural": compact chains, or denser bundles."""
    compact_chain = (
        number(row, "max_bundle_span") >= 900
        and number(row, "max_coverage") >= 0.09
        and number(row, "sum_coverage") >= 0.10
        and number(row, "section_count") <= 8
        and number(row, "dst_strong_section_fanout") <= 40
        and number(row, "max_chain_score") >= 1200
    )
    dense_bundle = (
        number(row, "max_bundle_span") >= 650
        and number(row, "max_coverage") >= 0.15
        and number(row, "sum_coverage") >= 0.15
        and number(row, "section_count") <= 4
        and number(row, "dst_strong_section_fanout") <= 8
        and number(row, "max_bundle_fragment_count") >= 2
    )
    return compact_chain or dense_bundle


def structural_or_coverage_gate(row: Row) -> bool:
    """Table 2, "Structural" or "Coverage"."""
    structural_branch = structural_gate(row) and number(row, "max_coverage") >= 0.10
    coverage_branch = (
        number(row, "max_coverage") >= 0.30
        and number(row, "dst_strong_section_fanout") <= 40
        and number(row, "section_count") <= 8
    )
    return structural_branch or coverage_branch


def title_or_heading_cue(row: Row) -> bool:
    return (
        flag(row, "title_hit")
        or flag(row, "title_prefix_hit")
        or flag(row, "heading_genre_cue")
        or flag(row, "heading_match_cue")
    )


def cue_or_broad_evidence_gate(row: Row) -> bool:
    """Table 2, "Context", first half: a title/heading cue, or broad evidence
    that is neither quotation- nor paratext-dominated."""
    cued_branch = title_or_heading_cue(row) and not flag(row, "paratext_cue")
    broad_non_cued_branch = (
        not flag(row, "quotation_cue")
        and not flag(row, "paratext_cue")
        and (
            (
                number(row, "max_coverage") >= 0.20
                and number(row, "max_bundle_span") >= 1200
                and number(row, "sum_coverage") >= 0.28
            )
            or (
                number(row, "sum_coverage") >= 0.28
                and number(row, "section_count") >= 2
                and number(row, "max_bundle_span") >= 1200
            )
            or (
                number(row, "max_bundle_fragment_count") >= 2
                and number(row, "sum_coverage") >= 0.28
                and number(row, "max_chain_score") >= 1800
            )
        )
    )
    return cued_branch or broad_non_cued_branch


def rescue_gate(row: Row) -> bool:
    """Table 2, "Rescue": short near misses with no quotation or paratext cue
    and zero destination fanout."""
    if flag(row, "quotation_cue") or flag(row, "paratext_cue"):
        return False
    if number(row, "dst_strong_section_fanout") != 0:
        return False
    compact_single = (
        number(row, "max_coverage") >= 0.10
        and 800 <= number(row, "max_bundle_span") <= 870
        and number(row, "section_count") <= 2
        and number(row, "top_section_gap") >= 0.08
    )
    compact_multi = (
        number(row, "sum_coverage") >= 0.085
        and 450 <= number(row, "max_bundle_span") <= 870
        and 2 <= number(row, "section_count") <= 4
        and number(row, "top_section_gap") <= 0.09
    )
    return compact_single or compact_multi


# ---------------------------------------------------------------------------
# The four rule stages reported in the paper.
# ---------------------------------------------------------------------------


def rule_naive_heuristic(row: Row) -> bool:
    """Stage 1: a single shallow span signal."""
    return number(row, "max_bundle_span") >= 1500


def rule_structural_only(row: Row) -> bool:
    """Stage 2: coverage and bundle statistics only."""
    return (
        number(row, "max_coverage") >= 0.25
        and number(row, "sum_coverage") >= 0.25
        and number(row, "max_bundle_span") >= 400
    )


def rule_context_aware(row: Row) -> bool:
    """Stage 3: adds title, heading, quotation, and paratext cues."""
    if not cue_or_broad_evidence_gate(row):
        return False
    if title_or_heading_cue(row):
        return True
    return not (
        number(row, "section_count") >= 6
        and number(row, "max_bundle_fragment_count") >= 6
    )


def rule_final_workflow(row: Row) -> bool:
    """Stage 4: the retained cascade, adding hard-case rescue and suppression."""
    if structural_or_coverage_gate(row) and rule_context_aware(row):
        return True
    return rescue_gate(row)


WORKFLOW_RULES: dict[str, Rule] = {
    "naive_heuristic": rule_naive_heuristic,
    "structural_only": rule_structural_only,
    "context_aware": rule_context_aware,
    "final_workflow": rule_final_workflow,
}


def pseudo_score(row: Row) -> float:
    return max(
        number(row, "max_coverage") * 3.0,
        number(row, "sum_coverage"),
        number(row, "max_bundle_span") / 1000.0,
        number(row, "max_chain_score") / 1000.0,
    )


def predict_row(row: Row, method_name: str) -> dict[str, Any]:
    rule = WORKFLOW_RULES[method_name]
    positive = rule(row)
    return {
        "pair_id": row["pair_id"],
        "src_doc_id": row["src_doc_id"],
        "dst_doc_id": row["dst_doc_id"],
        "gold_label": row.get("gold_label"),
        "split_tags": row.get("split_tags", []),
        "pred_label": "reprint" if positive else "non_reprint",
        "score": pseudo_score(row),
        "method_name": method_name,
    }
