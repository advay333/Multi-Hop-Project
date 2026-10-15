"""
The one value normaliser (PLAN.md §6 rule 3). Gold answers, gold entity sets, anchor literals,
reader answers and cell-index values all pass through `norm_value`, so comparisons never fail
on formatting: "7" == "7.0" == " 7 " == 7, "Alabama " == "alabama", NULL == "<null>".

Rules, in order:
  None                      -> "<null>"
  bool                      -> "1" / "0"   (SQLite stores booleans as integers)
  int / float / Decimal / numeric string
                            -> canonical decimal: no exponent, no trailing zeros, no "+",
                               "-0" -> "0", floats rounded to FLOAT_DIGITS significant digits
                               (so 0.1 + 0.2 == 0.3), thousands separators removed ("1,234")
  date / datetime string    -> ISO-8601: "2012/1/5" -> "2012-01-05",
                               "2012-01-05 00:00:00" -> "2012-01-05",
                               "2012-01-05 13:30:00" -> "2012-01-05t13:30:00" (casefolded)
  other strings             -> NFKC, strip, casefold, collapse internal whitespace
"""

from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation, localcontext

NULL = "<null>"
FLOAT_DIGITS = 10

_NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_DATE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[ tT](\d{1,2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?)?$")
_WS = re.compile(r"\s+")


def _canon_decimal(d: Decimal) -> str:
    if not d.is_finite():
        return str(d).lower()
    with localcontext() as ctx:
        ctx.prec = 400  # some cells hold >28-digit numbers; never round while canonicalising
        if d == d.to_integral_value():
            s = format(d.to_integral_value(), "f")
        else:
            s = format(d.normalize(), "f")
    return "0" if s in ("-0", "+0") else s


def _canon_float(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return str(x).lower()
    return _canon_decimal(Decimal(format(x, f".{FLOAT_DIGITS}g")))


def norm_number(s: str) -> str | None:
    """Canonical form of a numeric string, or None if it is not a number."""
    t = s.strip()
    if _THOUSANDS.match(t):
        t = t.replace(",", "")
    if not _NUMERIC.match(t):
        return None
    try:
        d = Decimal(t)
    except InvalidOperation:
        return None
    # Route through float rounding when there are more digits than a float can carry,
    # so a string "0.30000000000000004" matches the float 0.1 + 0.2.
    if len(d.as_tuple().digits) > FLOAT_DIGITS:
        return _canon_float(float(d))
    return _canon_decimal(d)


def norm_date(s: str) -> str | None:
    m = _DATE.match(s.strip())
    if not m:
        return None
    y, mo, d, hh, mm, ss = m.groups()
    if not (1 <= int(mo) <= 12 and 1 <= int(d) <= 31):
        return None
    date = f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    if hh is None or (int(hh) == 0 and int(mm) == 0 and int(ss or 0) == 0):
        return date
    return f"{date}t{int(hh):02d}:{int(mm):02d}:{int(ss or 0):02d}"


def norm_text(s: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", s)).strip().casefold()


def norm_value(v) -> str:
    if v is None:
        return NULL
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return _canon_float(v)
    if isinstance(v, Decimal):
        return norm_number(str(v)) or _canon_decimal(v)
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v).decode("utf-8", "replace")
    s = str(v)
    if s.strip() == "":
        return ""
    return norm_number(s) or norm_date(s) or norm_text(s)


def norm_row(row) -> list[str]:
    return [norm_value(v) for v in row]
