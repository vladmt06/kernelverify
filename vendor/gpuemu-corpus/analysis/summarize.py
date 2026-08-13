#!/usr/bin/env python3
"""Summarize a run's results.jsonl: pass/fail counts and failure-kind breakdown.

Usage:
    python3 analysis/summarize.py <run_id> [--data-dir ../data]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(records: list[dict]) -> dict:
    by_kernel: dict[str, Counter] = defaultdict(Counter)
    failure_kinds: Counter = Counter()
    for r in records:
        k = r.get("kernel", "?")
        by_kernel[k]["total"] += 1
        if r.get("passed"):
            by_kernel[k]["passed"] += 1
        else:
            by_kernel[k]["failed"] += 1
            failure_kinds[r.get("failure_kind") or "unknown"] += 1
    return {
        "total": len(records),
        "passed": sum(1 for r in records if r.get("passed")),
        "failed": sum(1 for r in records if not r.get("passed")),
        "by_kernel": {k: dict(v) for k, v in by_kernel.items()},
        "failure_kinds": dict(failure_kinds),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    args = ap.parse_args()

    results_path = Path(args.data_dir) / args.run_id / "results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"no results at {results_path}")
    print(json.dumps(summarize(load_results(results_path)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
