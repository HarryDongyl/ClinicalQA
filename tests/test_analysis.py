import pytest

from clinqa.analysis.checks import Context, _EQ_PATTERNS, _eq_ok, q10_numeric_arithmetic
from clinqa.analysis.features import compute_features
from clinqa.config import load_yaml
from clinqa.data_io import load_split


@pytest.mark.parametrize("text,expected", [
    ("198.9 / 40 = 4.97x", ("198.9", "40", "4.97")),
    ("150.7 / 56 = 2.69x)", ("150.7", "56", "2.69")),
    ("17.6 / 4.20 ≈ 4.19×", ("17.6", "4.20", "4.19")),
    ("10 / 4 = 2.5.", ("10", "4", "2.5")),
    ("−10 / 4 = −2.5", ("−10", "4", "−2.5")),
])
def test_ratio_uses_complete_numeric_tokens(text, expected):
    pattern = dict(_EQ_PATTERNS)["ratio"]
    assert pattern.search(text).groups() == expected
    assert _eq_ok("ratio", list(expected))[0]


@pytest.mark.parametrize("text", ["10 / 4 = 2.5%", "10 / 4 = 2.5 × 100", "10 / 4 = 2.5x100"])
def test_ratio_does_not_truncate_to_escape_suffix(text):
    assert dict(_EQ_PATTERNS)["ratio"].search(text) is None


def test_wrong_equation_still_fails_and_neighbouring_formulas_are_preserved():
    matches = list(dict(_EQ_PATTERNS)["ratio"].finditer("10 / 4 = 3.5x; 20 / 4 = 5.0×"))
    assert len(matches) == 2
    assert not _eq_ok("ratio", list(matches[0].groups()))[0]
    assert _eq_ok("ratio", list(matches[1].groups()))[0]


def test_percentage_and_difference_equations():
    for kind, text in [("pct", "5 / 20 × 100 = 25%"),
                       ("pct_change", "(20 - 10) / 10 × 100 = 100%"),
                       ("diff", "4 - 10 = -6")]:
        match = dict(_EQ_PATTERNS)[kind].search(text)
        assert match is not None
        assert _eq_ok(kind, list(match.groups()))[0]


def test_known_q10_false_positive_train_record_is_retained():
    record = next(r for r in load_split("train") if r["id"] == "train_1890")
    ctx = Context(data={"train": [record]},
                  feats={"train": [compute_features("train", record)]}, reference=[],
                  cfg=load_yaml("configs/analysis.yaml"))
    result = q10_numeric_arithmetic(ctx)
    assert result.flags == []
    assert result.metrics["equations_checked"] == 2
