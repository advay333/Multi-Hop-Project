"""
Metric suite (report Part D, Table 3). Gold-aware: used by the rollout's gold-side scoring and by
evaluate.py, never by the environment, featuriser or reader.

  per question (headline)   em_norm (outcome reward R_ans), set_f1, join-path match
                            (sufficiency.path_match), sub-table answer coverage (row lineage:
                            sufficiency.sufficient; values: answer_in_subtable)
  per question (diagnostic) join-edge and filter precision/recall (item_counts), per-hop entity
                            recall/precision vs E* (sufficiency.entity_overlap), reader tokens
  aggregation               summarise: overall and by source, 2-/3-table, category, monotone
  statistics                paired_bootstrap over questions (10,000 resamples, 95% interval)

Answer text format (the reader prompt asks for it): one answer row per line, cells of a row
separated by " | ". Every cell goes through `norm_value`, the one normaliser.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from mhtable.data.norm import norm_value
from mhtable.oracle import edge_key


EMPTY = '""'  # how an empty cell is written, so it survives the text format


def _cell_text(c) -> str:
    """A cell as written in an answer line: no line breaks, and "|" (the separator) as "/"."""
    s = " ".join(str(c).replace("|", "/").split())
    return s if s else EMPTY


def _cell(c) -> str:
    s = _cell_text(c)
    return "" if s == EMPTY else norm_value(s)


def format_answer(rows: list[list]) -> str:
    return "\n".join(" | ".join(_cell_text(c) for c in r) for r in rows)


def parse_answer(text: str | None) -> list[tuple[str, ...]]:
    rows = []
    for line in (text or "").splitlines():
        if line.strip():
            rows.append(tuple(_cell(c) for c in line.split("|")))
    return rows


def _gold_rows(gold_answer: list[list]) -> set[tuple[str, ...]]:
    return {tuple(_cell(c) for c in r) for r in gold_answer}


def em_norm(pred: str | None, gold_answer: list[list]) -> float:
    """1.0 if the predicted rows equal the gold rows as sets (row order is free). A prediction
    that lists the same values without the row structure (e.g. one value per line) also counts."""
    gold, rows = _gold_rows(gold_answer), set(parse_answer(pred))
    if not gold or not rows:
        return 0.0
    if rows == gold:
        return 1.0
    return float({c for r in rows for c in r} == {c for r in gold for c in r})


def set_f1(pred: str | None, gold_answer: list[list]) -> float:
    """F1 over the sets of answer values: partial credit for List / Count / Numerical answers."""
    gold = {c for r in _gold_rows(gold_answer) for c in r}
    mine = {c for r in parse_answer(pred) for c in r}
    hit = len(gold & mine)
    if not hit:
        return 0.0
    p, r = hit / len(mine), hit / len(gold)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------- sub-table answer coverage
def subtable_cells(subtable: str) -> set[str]:
    """Normalised cell values of a materialised sub-table (header and separator skipped)."""
    out = set()
    for line in subtable.splitlines()[2:]:
        if line.startswith("|"):
            out |= {norm_value(c.strip()) for c in line.strip().strip("|").split("|") if c.strip()}
    return out


def answer_in_subtable(subtable: str, gold_answer: list[list]) -> float:
    """Fraction of the gold answer's values that appear as cells of the sub-table the reader saw.
    1.0 means the reader could copy the answer; below 1 on Select/List questions points at the
    projection (columns not shown) or the budget, not the reader. Count/Numerical answers are
    usually computed, not copied, so read this metric by category."""
    gold = {c for r in _gold_rows(gold_answer) for c in r if c}
    if not gold:
        return 0.0
    return len(gold & subtable_cells(subtable)) / len(gold)


# ---------------------------------------------------------------- join-edge / filter P/R
def item_counts(joins, anchors, gold_edges, gold_anchors) -> dict:
    """Matched / predicted / gold counts for join edges and filters (micro-averaged later)."""
    pe, ge = {edge_key(e) for e in joins}, {edge_key(e) for e in gold_edges}
    pa, ga = {a.key() for a in anchors}, {a.key() for a in gold_anchors}
    return {"edge_hit": len(pe & ge), "edge_pred": len(pe), "edge_gold": len(ge),
            "anchor_hit": len(pa & ga), "anchor_pred": len(pa), "anchor_gold": len(ga)}


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


# ---------------------------------------------------------------- aggregation
HEADLINE = ("em", "set_f1", "path_match", "sufficient", "answer_in_subtable")
EFFICIENCY = ("tokens", "steps", "overflow")
SLICES = ("source", "tables", "category", "monotone")


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for k in HEADLINE + EFFICIENCY:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = round(float(np.mean(vals)), 4)
    shown = [r["em"] for r in rows if r.get("answer_in_subtable") == 1.0]
    out["em_when_answer_shown"] = round(float(np.mean(shown)), 4) if shown else None  # reader quality alone
    out["answer_shown_n"] = len(shown)
    on_path = [r["em"] for r in rows if r.get("path_match") == 1.0]
    out["em_on_path"] = round(float(np.mean(on_path)), 4) if on_path else None  # oracle: excludes unreachable questions
    out["on_path_n"] = len(on_path)
    tot = defaultdict(float)
    for r in rows:
        for k in ("edge_hit", "edge_pred", "edge_gold", "anchor_hit", "anchor_pred", "anchor_gold"):
            tot[k] += r.get(k, 0)
    out["edge_precision"], out["edge_recall"] = _ratio(tot["edge_hit"], tot["edge_pred"]), _ratio(tot["edge_hit"], tot["edge_gold"])
    out["anchor_precision"] = _ratio(tot["anchor_hit"], tot["anchor_pred"])
    out["anchor_recall"] = _ratio(tot["anchor_hit"], tot["anchor_gold"])
    hops = defaultdict(lambda: {"recall": [], "precision": []})
    for r in rows:
        for h in r.get("hops") or []:
            for k in ("recall", "precision"):
                if h.get(k) is not None:
                    hops[h["hop"]][k].append(h[k])
    out["per_hop"] = {f"hop{h}": {k: (round(float(np.mean(v)), 4) if v else None) for k, v in d.items()}
                      for h, d in sorted(hops.items())}
    return out


def summarise(rows: list[dict], slices=SLICES) -> dict:
    """Overall numbers plus the same numbers per slice value (report Part D reporting slices)."""
    out = _aggregate(rows)
    for field in slices:
        groups = defaultdict(list)
        for r in rows:
            groups[str(r.get(field))].append(r)
        out[f"by_{field}"] = {g: _aggregate(rs) for g, rs in sorted(groups.items())}
    return out


# ---------------------------------------------------------------- statistics
def paired_bootstrap(a, b, n: int = 10_000, seed: int = 0) -> dict:
    """Paired bootstrap over questions for mean(a) - mean(b) (same questions, same order):
    the difference, its 95% interval, and a two-sided p-value for "no difference"."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("paired_bootstrap needs two equal-length, non-empty 1-d score lists")
    d = a - b
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p": round(float(min(p, 1.0)), 4), "n": int(len(d)), "resamples": n}
