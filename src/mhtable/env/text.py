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
_GLUED_ID = re.compile(r"(?i)\b(?:id|no|#)\.?(\d+)(?!\w|\.\d)")  # "ID12882", "no.14"


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


_MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
_MONTH_YEAR = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s*,?\s*(\d{4})\b", re.IGNORECASE)


def date_prefixes(text: str) -> list[str]:
    """ISO prefixes a date column could be matched against: "in 2006" -> "2006",
    "February 2017" -> "2017-02"."""
    out = [f"{y}-{_MONTHS[m.lower()]:02d}" for m, y in _MONTH_YEAR.findall(text)]
    out += re.findall(r"(?<![\d.-])((?:19|20)\d{2})(?![\d.])", text)
    return list(dict.fromkeys(out))


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
    for m in _NUM.findall(text) + _GLUED_ID.findall(text):
        v = m.replace(",", "")
        x = float(v) if "." in v else int(v)
        if x not in out:
            out.append(x)
    for w in plain(text).split():
        if w in _WORD_NUMBERS and _WORD_NUMBERS[w] not in out:
            out.append(_WORD_NUMBERS[w])
    return out


# Explicit filters written out in the text, as BIRD's evidence does: "WasCompiled = 0",
# "IndepYear BETWEEN 1980 AND 1995", "courseLevel = 'Level_400'", "course.course_id < 10",
# "Name LIKE '%Ale%'", "Region IN ('East', 'West')". Identifiers may be `quoted`, "quoted" or [bracketed].
_IDENT = r"(?:`[^`]{1,60}`|\[[^\]]{1,60}\]|\"[^\"]{1,60}\"|[A-Za-z_][\w]*(?:\.(?:`[^`]{1,60}`|[A-Za-z_]\w*))?)"
_VAL = r"(?:'(?:[^']|'')*'|-?\d+(?:\.\d+)?)"
_FILTER = re.compile(
    rf"(?P<col>{_IDENT})\s*(?:"
    rf"(?P<op><>|!=|<=|>=|==|=|<|>)\s*(?P<v>{_VAL})"
    rf"|(?P<not>NOT\s+)?(?P<kw>BETWEEN)\s+(?P<lo>{_VAL})\s+AND\s+(?P<hi>{_VAL})"
    rf"|(?P<lnot>NOT\s+)?(?P<like>LIKE)\s+(?P<pat>'(?:[^']|'')*')"
    rf"|(?P<inot>NOT\s+)?(?P<in>IN)\s*\((?P<vals>\s*{_VAL}(?:\s*,\s*{_VAL})*)\s*\)"
    rf"|IS\s+(?P<isnot>NOT\s+)?NULL\b)",
    re.IGNORECASE)
_KEYWORDS = {"and", "or", "not", "where", "refers", "to", "is", "in", "between", "like", "the", "a", "an"}


def _lit(tok: str):
    if tok.startswith("'"):
        return tok[1:-1].replace("''", "'")
    return float(tok) if "." in tok else int(tok)


def explicit_filters(text: str) -> list[tuple[str, str, tuple]]:
    """[(identifier, op, values)] for filters written as SQL-like conditions in `text`. The
    identifier is lowercased and unquoted, and may carry a table prefix ("course.course_id")."""
    out = []
    for m in _FILTER.finditer(text):
        segs = re.findall(r"`[^`]+`|\[[^\]]+\]|\"[^\"]+\"|[A-Za-z_]\w*", m["col"])
        ident = ".".join(s.strip("`[]\"") for s in segs).lower()
        if ident in _KEYWORDS or ident.isdigit():
            continue
        if m["op"]:
            op = {"<>": "!=", "==": "="}.get(m["op"], m["op"])
            out.append((ident, op, (_lit(m["v"]),)))
        elif m["kw"]:
            if not m["not"]:
                out.append((ident, "between", (_lit(m["lo"]), _lit(m["hi"]))))
        elif m["like"]:
            out.append((ident, "not_like" if m["lnot"] else "like", (_lit(m["pat"]),)))
        elif m["in"]:
            vals = tuple(_lit(v) for v in re.findall(_VAL, m["vals"]))
            out.append((ident, "not_in" if m["inot"] else "in", vals))
        else:
            out.append((ident, "not_is" if m["isnot"] else "is", ()))
    return out


_CONTEXT_FILLER = STOPWORDS | {"between", "more", "less", "than", "over", "under", "above", "below", "least",
                                  "most", "equal", "about", "around", "refer", "refers", "not", "no", "per"}


def _context_words(s: str) -> list[str]:
    return [w for w in _SPLIT.split(norm_text(_CAMEL.sub(" ", s))) if w and w not in _CONTEXT_FILLER and not w.isdigit()]


def number_contexts(text: str, before: int = 3, after: int = 2) -> dict:
    """Number -> the content words around it ("game ID 178" -> "game id", "254 likes" -> "likes"),
    which usually name the column the number belongs to. Words are cut at clause punctuation."""
    out: dict = {}
    for m in list(_NUM.finditer(text)) + list(_GLUED_ID.finditer(text)):
        glued = m.re is _GLUED_ID
        raw = (m.group(1) if glued else m.group(0)).replace(",", "")
        x = float(raw) if "." in raw else int(raw)
        left = re.split(r"[,;:?!()]|\.\s", text[max(0, m.start() - 80):m.start()])[-1]
        if glued:
            left += " " + text[m.start():m.start(1)]
        right = re.split(r"[,;:?!()]|\.\s|\d", text[m.end():m.end() + 60])[0]
        words = _context_words(left)[-before:] + _context_words(right)[:after]
        if words and x not in out:
            out[x] = " ".join(words)
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
