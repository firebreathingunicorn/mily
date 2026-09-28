"""Fairness slice reporting (plan §Principles 4, §Evaluation).

Results are measured separately across skin tones, glasses, beards, head
coverings, children, and low light — face tools most often fail on exactly
these groups. This module turns flat result rows (CSV from `besttake eval`,
or any per-sample manifest) into per-slice metrics, so a model that regresses
any slice is visible instead of averaged away.

Usage:
    python3 eval/slices.py results.csv --slices fitzpatrick,glasses,low_light
    python3 eval/slices.py results.csv                  # overall only

Row schema (columns beyond `id` are all optional):
    id, burst, person, checksPassed, swapped, level, ...
    plus any slice columns present (fitzpatrick, glasses, beard,
    head_covering, child, low_light, ...).

Unknown slice columns are ignored; rows missing a slice value fall into the
"unspecified" bucket and are reported separately, never silently dropped.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

DEFAULT_SLICES = ["fitzpatrick", "glasses", "beard", "head_covering", "child", "low_light"]
UNSPECIFIED = "(unspecified)"


def _bucket(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        return UNSPECIFIED
    return str(value).strip()


def report_slices(rows: list[dict], slices: list[str] | None = None) -> dict:
    """Per-slice metrics over result rows.

    Metric: `checks` pass rate (the artifact/identity gate outcome), plus swap
    rate and level distribution — the gates' behavior per demographic slice is
    exactly what the fairness audit needs.
    """
    slices = slices if slices is not None else [
        c for c in DEFAULT_SLICES if rows and any(c in r for r in rows)
    ]
    out: dict = {"n": len(rows), "slices": {}}
    passed = [r for r in rows if str(r.get("checksPassed", "")).strip() in ("1", "true", "True")]
    out["checksPassed"] = len(passed)
    out["checksPassedRate"] = round(len(passed) / len(rows), 4) if rows else None

    for col in slices:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[_bucket(r.get(col))].append(r)
        out["slices"][col] = {}
        for value, grp in sorted(groups.items()):
            ok = sum(1 for r in grp if str(r.get("checksPassed", "")).strip() in ("1", "true", "True"))
            levels = defaultdict(int)
            for r in grp:
                levels[str(r.get("level", "?"))] += 1
            out["slices"][col][value] = {
                "n": len(grp),
                "checksPassed": ok,
                "checksPassedRate": round(ok / len(grp), 4),
                "levels": dict(levels),
            }
    return out


def worst_slice(report: dict, min_n: int = 1) -> tuple[str, str, float] | None:
    """The (slice, value, pass-rate) with the lowest rate — the audit target."""
    worst = None
    for col, values in report.get("slices", {}).items():
        for value, stats in values.items():
            if stats["n"] < min_n:
                continue
            rate = stats["checksPassedRate"]
            if worst is None or rate < worst[2]:
                worst = (col, value, rate)
    return worst


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = [a for a in argv if a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    path = args[0]
    slices = None
    for f in flags:
        if f.startswith("--slices="):
            slices = [s.strip() for s in f.split("=", 1)[1].split(",") if s.strip()]

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    report = report_slices(rows, slices)
    print(json.dumps(report, indent=2, sort_keys=True))
    worst = worst_slice(report)
    if worst:
        print(f"\nworst slice: {worst[0]}={worst[1]} pass rate {worst[2]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
