"""calculate_egfr: CKD-EPI 2021 against the NKF equation, cross-implementation agreement, and the table-eGFR finding."""

import importlib.util
import json
import sys
from decimal import ROUND_HALF_UP, Decimal

import pytest

from clinqa.config import PROJECT_ROOT
from clinqa.parsing import extract_age_sex, parse_float
from clinqa.tools import calculate_egfr

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen1, gen2, score = _load("stretch_a_data"), _load("stretch_a_data_v2"), _load("stretch_a_score")


def nkf_2021(cr: float, age: int, sex: str) -> float:
    """NKF CKD-EPI 2021: 142 * min(Scr/k,1)^a * max(Scr/k,1)^-1.200 * 0.9938^age * 1.012 [female]."""
    k, a = (0.7, -0.241) if sex == "female" else (0.9, -0.302)
    return 142 * min(cr / k, 1) ** a * max(cr / k, 1) ** -1.200 * 0.9938 ** age * (1.012 if sex == "female" else 1.0)


def half_up(x: float) -> int:
    return int(Decimal(repr(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@pytest.mark.parametrize("cr,age,sex,raw", [(1.0, 50, "male", 91.69), (1.0, 50, "female", 68.63),
                                            (0.9, 40, "male", 110.73), (0.7, 40, "female", 112.05),
                                            (2.5, 58, "female", 21.75)])
def test_matches_the_nkf_equation(cr, age, sex, raw):
    assert nkf_2021(cr, age, sex) == pytest.approx(raw, abs=0.01)
    assert calculate_egfr(cr, age, sex) == half_up(nkf_2021(cr, age, sex))


def test_four_implementations_agree_on_a_dense_grid():
    n = 0
    for cr in [x / 100 for x in range(20, 1201)]:  # includes both kappa kinks (0.7, 0.9)
        for age in (18, 19, 30, 45, 58, 65, 80, 99, 120):
            for sex in ("male", "female"):
                values = {calculate_egfr(cr, age, sex), gen1.ckd_epi_2021(cr, age, sex),
                          gen2.independent_egfr(cr, age, sex), half_up(score.raw_egfr(cr, age, sex))}
                assert len(values) == 1, (cr, age, sex, values)
                n += 1
    assert n == 21258


def test_input_validation_returns_errors_not_exceptions():
    assert calculate_egfr(1.0, 50, "M") == calculate_egfr(1.0, 50, "male")
    for bad in [(0, 50, "male"), (1.0, 17, "male"), (1.0, 50.5, "male"), (1.0, 50, "x")]:
        assert str(calculate_egfr(*bad)).startswith("Error:")


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for k, i in enumerate(order):
        r[i] = k
    return r


def test_table_egfr_is_independent_of_creatinine():
    """The dataset's table eGFR is synthetic: why Stretch A uses only records without one (docs/STRETCH_A.md)."""
    cr, egfr = [], []
    for split in ("train", "val", "test"):
        for line in (PROJECT_ROOT / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows = {x[0]: x for x in r["table"]["rows"]}
            if "Creatinine" in rows and "eGFR" in rows and rows["Creatinine"][2] == "mg/dL":
                age, sex = extract_age_sex(r["note"])
                c, e = parse_float(rows["Creatinine"][1]), parse_float(rows["eGFR"][1])
                if None not in (age, sex, c, e):
                    cr.append(c)
                    egfr.append(e)
    assert len(cr) == 258
    ra, rb = _rank(cr), _rank(egfr)
    m = (len(cr) - 1) / 2
    rho = sum((a - m) * (b - m) for a, b in zip(ra, rb)) / sum((a - m) ** 2 for a in ra)
    assert abs(rho) < 0.1  # 0.008 measured; a creatinine-derived eGFR would be strongly negative
