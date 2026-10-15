"""
Lexical text similarity used by the environment to rank candidates (columns, tables, cell values)
against the question.

The report ranks candidates with frozen-encoder embeddings (P4). Until the encoder cache exists,
`LexicalSimilarity` stands in: token overlap plus character-trigram overlap. The environment only
depends on the `Similarity` interface, so the embedding scorer replaces it without other changes.
These scores only choose WHICH candidates are offered and in what order; the policy itself learns
from features, not from this ranking.
"""

from __future__ import annotations

import re
from functools import lru_cache

from mhtable.data.norm import norm_text

STOPWORDS = frozenset("""a an the of in on at to for from by with and or is are was were be been being what which
who whom whose when where how many much that this these those do does did has have had it its as than
then there their they them all any each list show give find name names number count""".split())

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPLIT = re.compile(r"[^0-9a-z]+")
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?!\w|\.\d)")  # a sentence-final "." is fine


def _stem(t: str) -> str:
    """Light plural folding so "customers" matches "customer" ("class" is left alone)."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


@lru_cache(maxsize=200_000)
def tokens(s: str) -> tuple[str, ...]:
    """Content tokens: camelCase and snake_case split, lowercased, stopwords dropped, plurals folded."""
    s = _CAMEL.sub(" ", s)
    return tuple(_stem(t) for t in _SPLIT.split(norm_text(s)) if t and t not in STOPWORDS)


@lru_cache(maxsize=200_000)
def plain(s: str) -> str:
    """Lowercased words separated by single spaces, punctuation removed (stopwords kept)."""
    return " ".join(t for t in _SPLIT.split(norm_text(s)) if t)


DATE_LIT = re.compile(r"\b(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)\b")
LETTER = re.compile(r"\bletters?\s+['\"]?([a-zA-Z])['\"]?(?![\w'])")


def dates(text: str) -> list[str]:
    """ISO-style date literals mentioned in a text ("1975-01-01", "2008-04-27 10:53")."""
    return list(dict.fromkeys(DATE_LIT.findall(text)))


QUOTED = re.compile(r"""['"‘’“”]([^'"‘’“”]{2,60})['"‘’“”]""")

# Wording cues for comparison operators, checked against plain(question).
OP_CUES = {
    ">": ("more than", "greater than", "larger than", "bigger than", "higher than", "over", "above", "exceed",
          "after", "older than", "longer than", "later than", "bigger", "larger", "higher"),
    "<": ("less than", "fewer than", "smaller than", "lower than", "under", "below", "before", "younger than",
          "shorter than", "earlier than", "smaller", "lower", "younger"),
    ">=": ("at least", "or more", "no less than", "minimum of", "not less than", "or older", "or greater",
           "or higher", "or above", "and above", "or later", "or after", "greater than or equal", "of least"),
    "<=": ("at most", "or less", "no more than", "up to", "maximum of", "not more than", "or younger",
           "or fewer", "or below", "and below", "or earlier", "or before", "less than or equal"),
    "!=": ("not", "n t", "other than", "except", "excluding", "without", "never", "different from"),
}


def op_cues(text: str) -> set[str]:
    p = f" {plain(text)} "
    return {op for op, cues in OP_CUES.items() if any(f" {c} " in p for c in cues)}


@lru_cache(maxsize=200_000)
def trigrams(s: str) -> frozenset[str]:
    s = " " + " ".join(tokens(s)) + " "
    return frozenset(s[i:i + 3] for i in range(len(s) - 2))


_WORD_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_WORD_NUMBERS.update({w: i + 1 for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth".split())})
_WORD_NUMBERS.update({"thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000})


def numbers(text: str) -> list[float | int]:
    """Numbers mentioned in a text ("1,234" -> 1234, "37.4" -> 37.4, "third" -> 3), in order,
    deduplicated."""
    out = []
    for m in _NUM.findall(text):
        v = m.replace(",", "")
        x = float(v) if "." in v else int(v)
        if x not in out:
            out.append(x)
    for w in plain(text).split():
        if w in _WORD_NUMBERS and _WORD_NUMBERS[w] not in out:
            out.append(_WORD_NUMBERS[w])
    return out


class Similarity:
    """Scores texts against a query; higher is more similar, roughly in [0, 1]."""

    def score(self, query: str, text: str) -> float:
        raise NotImplementedError


class LexicalSimilarity(Similarity):
    def score(self, query: str, text: str) -> float:
        qt, tt = set(tokens(query)), set(tokens(text))
        if not qt or not tt:
            return 0.0
        overlap = len(qt & tt) / len(tt)  # fraction of the text's tokens mentioned in the query
        qg, tg = trigrams(query), trigrams(text)
        dice = 2 * len(qg & tg) / (len(qg) + len(tg)) if qg and tg else 0.0
        return max(overlap, dice)


def name_similarity(a: str, b: str) -> float:
    """Symmetric similarity of two identifiers (column/table names)."""
    ta, tb = set(tokens(a)), set(tokens(b))
    jac = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    ga, gb = trigrams(a), trigrams(b)
    dice = 2 * len(ga & gb) / (len(ga) + len(gb)) if ga and gb else 0.0
    return max(jac, dice)
