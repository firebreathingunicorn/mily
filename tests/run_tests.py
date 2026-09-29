"""Minimal test runner (no pytest dependency): discovers test_* functions.

Usage:
  python3 tests/run_tests.py           # full suite (slow: full-size renders)
  python3 tests/run_tests.py --fast    # same tests, renders clamped to 160px
"""
import importlib
import os
import sys
import traceback
import unittest

MODULES = [
    "test_geometry",
    "test_rendering",
    "test_face_model",
    "test_harmonize",
    "test_checks",
    "test_fitter",
    "test_reproject",
    "test_pipeline",
    "test_level_a",
    "test_orchestrator",
    "test_slices",
]


def main() -> int:
    fast = "--fast" in sys.argv
    if fast:
        # Seconds instead of minutes: renders clamped to 160px AND the
        # size-sensitive modules skipped (their thresholds are calibrated
        # on 288px renders — run the full suite before committing).
        os.environ["BESTTAKE_MAX_RENDER"] = "160"
        sys.argv = [sys.argv[0]]
    skipped_modules = {"test_fitter", "test_reproject",
                       "test_level_a", "test_pipeline"} if fast else set()
    failures = []
    skipped = 0
    total = 0
    for mod_name in MODULES:
        if mod_name in skipped_modules:
            n = sum(1 for name in dir(importlib.import_module(mod_name))
                    if name.startswith("test_") and callable(getattr(importlib.import_module(mod_name), name)))
            skipped += n
            print(f"SKIP {mod_name} ({n} tests — size-sensitive, full run only)")
            continue
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            if name.startswith("test_") and callable(getattr(mod, name)):
                total += 1
                try:
                    getattr(mod, name)()
                    print(f"PASS {mod_name}.{name}")
                except unittest.SkipTest as why:
                    skipped += 1
                    print(f"SKIP {mod_name}.{name}: {why}")
                except Exception:
                    failures.append((mod_name, name, traceback.format_exc()))
                    print(f"FAIL {mod_name}.{name}")
    for mod_name, name, tb in failures:
        print(f"\n--- {mod_name}.{name} ---\n{tb}")
    tail = f", {skipped} skipped (fast mode)" if skipped else ""
    print(f"\n{total - len(failures)}/{total} passed{tail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
