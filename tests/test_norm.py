from decimal import Decimal

import pytest

from mhtable.data.norm import NULL, norm_row, norm_value


@pytest.mark.parametrize("a,b", [
    ("7", "7.0"), (" 7 ", 7), (7, 7.0), (Decimal("30266.0"), 30266), ("1,234", 1234),
    ("-0", 0), ("+5", 5), ("1e3", 1000), (0.1 + 0.2, 0.3), ("0.30000000000000004", "0.3"),
    ("Alabama ", "alabama"), ("New   York", "new york"), ("ＡＢＣ", "abc"),
    ("2012/1/5", "2012-01-05"), ("2012-01-05 00:00:00", "2012-01-05"), (True, 1),
])
def test_equal_after_normalisation(a, b):
    assert norm_value(a) == norm_value(b)


@pytest.mark.parametrize("a,b", [
    ("7", "7.5"), ("1789", "1790"), ("2012-01-05 13:30:00", "2012-01-05"), ("abc", "abd"),
    (None, "null"), ("", NULL), ("1,23", "123"),
])
def test_distinct_after_normalisation(a, b):
    assert norm_value(a) != norm_value(b)


def test_canonical_forms():
    assert norm_value(None) == NULL
    assert norm_value(9.96) == "9.96"
    assert norm_value(Decimal("9.960")) == "9.96"
    assert norm_value(115897.0) == "115897"
    assert norm_value("2012-01-05 13:30:00") == "2012-01-05t13:30:00"
    assert norm_value("2012-13-05") == "2012-13-05"  # not a valid date: plain text
    assert norm_value(b"Abc") == "abc"
    assert norm_row([1, "X", None]) == ["1", "x", NULL]
    big = "1" + "0" * 40
    assert norm_value(big) == norm_value(big + ".0") == big  # beyond Decimal's default 28-digit precision
    assert norm_value("1E+30") == "1" + "0" * 30
