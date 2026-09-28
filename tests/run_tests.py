"""Minimal test runner (no pytest dependency): discovers test_* functions."""
import importlib
import sys
import traceback

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
    failures = []
    total = 0
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            if name.startswith("test_") and callable(getattr(mod, name)):
                total += 1
                try:
                    getattr(mod, name)()
                    print(f"PASS {mod_name}.{name}")
                except Exception:
                    failures.append((mod_name, name, traceback.format_exc()))
                    print(f"FAIL {mod_name}.{name}")
    for mod_name, name, tb in failures:
        print(f"\n--- {mod_name}.{name} ---\n{tb}")
    print(f"\n{total - len(failures)}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
