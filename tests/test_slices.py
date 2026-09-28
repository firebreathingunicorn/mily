"""Fairness slice-reporting tests (plan §Principles 4)."""
from eval.slices import report_slices, worst_slice

ROWS = [
    {"id": "1", "burst": "b1", "person": "p0", "checksPassed": "1", "level": "A", "fitzpatrick": "III", "glasses": "no", "low_light": "no"},
    {"id": "2", "burst": "b1", "person": "p1", "checksPassed": "0", "level": "none", "fitzpatrick": "VI", "glasses": "no", "low_light": "yes"},
    {"id": "3", "burst": "b2", "person": "p0", "checksPassed": "1", "level": "A", "fitzpatrick": "III", "glasses": "yes", "low_light": "no"},
    {"id": "4", "burst": "b2", "person": "p1", "checksPassed": "0", "level": "none", "fitzpatrick": "VI", "glasses": "no", "low_light": "yes"},
    {"id": "5", "burst": "b3", "person": "p0", "checksPassed": "1", "level": "B", "fitzpatrick": "III", "glasses": "", "low_light": "no"},
]


def test_overall_rates():
    r = report_slices(ROWS)
    assert r["n"] == 5
    assert r["checksPassed"] == 3
    assert r["checksPassedRate"] == 0.6


def test_slice_split_with_worst_identified():
    r = report_slices(ROWS)
    fitz = r["slices"]["fitzpatrick"]
    assert fitz["III"]["checksPassedRate"] == 1.0 and fitz["III"]["n"] == 3
    assert fitz["VI"]["checksPassedRate"] == 0.0 and fitz["VI"]["n"] == 2
    worst = worst_slice(r)
    assert worst is not None and worst[0] == "fitzpatrick" and worst[1] == "VI" and worst[2] == 0.0


def test_unspecified_rows_kept_visible():
    """Row 5 has no glasses value: it must land in an explicit '(unspecified)'
    bucket, never be silently dropped."""
    r = report_slices(ROWS)
    glasses = r["slices"]["glasses"]
    assert glasses["(unspecified)"]["n"] == 1
    assert glasses["yes"]["n"] == 1
    assert glasses["no"]["n"] == 3
    total = sum(v["n"] for v in glasses.values())
    assert total == 5


def test_explicit_slice_selection():
    r = report_slices(ROWS, slices=["low_light"])
    assert set(r["slices"].keys()) == {"low_light"}
    assert r["slices"]["low_light"]["yes"]["checksPassedRate"] == 0.0
