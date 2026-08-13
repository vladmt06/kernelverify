#!/usr/bin/env python3
"""P1 headline table: benchmark verdict vs gpuemu verdict, per kernel.

The "correctness illusion" is the set of kernels the benchmark calls correct
("pass") that gpuemu's fuzzing shows to be wrong. Regenerable from raw
results.jsonl — this is P1's science gate.

Usage: python3 analysis/p1_table.py <run_id> [--data-dir ../data]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> list:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def build_table(records: list) -> dict:
    kernels: dict = {}
    for r in records:
        k = kernels.setdefault(r["kernel"], {
            "source": r.get("source", "?"),
            "benchmark_verdict": r.get("benchmark_verdict", "?"),
            "total": 0, "failed": 0, "kinds": Counter(),
        })
        k["total"] += 1
        if not r["passed"]:
            k["failed"] += 1
            k["kinds"][r.get("failure_kind") or "unknown"] += 1
    return kernels


def render(kernels: dict) -> str:
    rows = ["| kernel | source | bench | gpuemu | fail/total | failure kinds | illusion |",
            "|---|---|---|---|---|---|---|"]
    illusions = 0
    for name, k in sorted(kernels.items()):
        gp = "fail" if k["failed"] else "pass"
        illusion = k["benchmark_verdict"] == "pass" and k["failed"] > 0
        illusions += illusion
        kinds = ", ".join(f"{kk}:{vv}" for kk, vv in k["kinds"].most_common()) or "-"
        rows.append(
            f"| {name} | {k['source']} | {k['benchmark_verdict']} | {gp} | "
            f"{k['failed']}/{k['total']} | {kinds} | {'YES' if illusion else ''} |"
        )
    rows.append("")
    rows.append(f"**Correctness illusions (benchmark=pass, gpuemu=fail): "
                f"{illusions}/{len(kernels)} kernels.**")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    args = ap.parse_args()
    path = Path(args.data_dir) / args.run_id / "results.jsonl"
    if not path.exists():
        raise SystemExit(f"no results at {path}")
    print(render(build_table(load(path))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
