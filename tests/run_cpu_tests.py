"""Run standard unittest discovery and save a machine-readable CPU report."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "cpu-results.json")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    versions = {}
    for package in ("torch", "numpy", "scipy", "biopython", "dm-tree"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "duration_seconds": round(time.monotonic() - started, 3),
        "python": sys.version, "platform": platform.platform(), "packages": versions,
        "failures": [{"test": test.id(), "traceback": error} for test, error in result.failures],
        "errors": [{"test": test.id(), "traceback": error} for test, error in result.errors],
        "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CPU report: {args.report.resolve()}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
