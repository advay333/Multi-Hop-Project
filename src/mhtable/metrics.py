"""
Answer scoring (report Part D). P5 needs only the outcome reward R_ans = EM_norm; the full
metric suite (join-path accuracy, coverage, edge P/R, slices, bootstrap) is added in P4/P6.

Answer text format (the reader prompt asks for it): one answer row per line, cells of a row
separated by " | ". Every cell goes through `norm_value`, the one normaliser.
"""

from __future__ import annotations

from mhtable.data.norm import norm_value


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
