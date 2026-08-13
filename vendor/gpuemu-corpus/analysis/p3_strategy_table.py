#!/usr/bin/env python3
"""P3 strategy ablation table.

Reads a p3_strategies run and produces two paper-grade tables:
  1. Bug-recall matrix: strategy (rows) × buggy kernel (columns).
  2. Time-to-first-failure (s) per (strategy, buggy kernel) — efficiency story.

A correct kernel's row in the recall table indicates *false-positive surface*
(should be near 0 across strategies).

Usage: python3 analysis/p3_strategy_table.py <run_id> [--data-dir ../data]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load(run_dir: Path) -> tuple[list, dict]:
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    summary = json.loads((run_dir / "summary.json").read_text())
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    args = ap.parse_args()

    rows, summary = _load(Path(args.data_dir) / args.run_id)
    strategies = summary["strategies"]
    kernels = summary["kernels"]
    buggy = [k for k in kernels if "buggy" in k]
    correct = [k for k in kernels if "buggy" not in k]

    # Recall = fail / total per (strategy, kernel).
    pair: dict = defaultdict(lambda: {"total": 0, "fail": 0, "ttff": None})
    for r in rows:
        k = (r["strategy"], r["kernel"])
        pair[k]["total"] += 1
        if not r.get("passed"):
            pair[k]["fail"] += 1
        ttff = r.get("ttff_s")
        if ttff is not None:
            cur = pair[k]["ttff"]
            pair[k]["ttff"] = ttff if cur is None else min(cur, ttff)

    def cell(s, k):
        v = pair[(s, k)]
        if v["total"] == 0:
            return "—"
        pct = v["fail"] / v["total"] * 100.0
        return f"{v['fail']}/{v['total']} ({pct:.0f}%)"

    def ttff_cell(s, k):
        v = pair[(s, k)]
        if v["ttff"] is None or v["total"] == 0:
            return "—"
        return f"{v['ttff']:.2f}"

    print(f"# P3 strategy ablation (run {args.run_id})\n")
    print(f"_Run: {summary['total_rows']} rows; {len(strategies)} strategies × "
          f"{len(kernels)} kernels._\n")

    # --- Table 1: bug recall on buggy kernels ---
    print("## Bug recall — strategy vs LLM-style buggy kernel\n")
    header = "| strategy | " + " | ".join(buggy) + " | overall |"
    sep = "|---|" + "|".join("---:" for _ in buggy) + "|---:|"
    print(header); print(sep)
    for s in strategies:
        row_fail = sum(pair[(s, k)]["fail"] for k in buggy)
        row_total = sum(pair[(s, k)]["total"] for k in buggy)
        overall = f"{row_fail}/{row_total} ({row_fail/row_total*100:.0f}%)" if row_total else "—"
        print(f"| {s} | " + " | ".join(cell(s, k) for k in buggy) + f" | {overall} |")

    # --- Table 2: time-to-first-failure on buggy kernels ---
    print("\n## Time-to-first-failure (s) — lower = faster bug detection\n")
    header = "| strategy | " + " | ".join(buggy) + " |"
    sep = "|---|" + "|".join("---:" for _ in buggy) + "|"
    print(header); print(sep)
    for s in strategies:
        print(f"| {s} | " + " | ".join(ttff_cell(s, k) for k in buggy) + " |")

    # --- Table 3: false-positive surface on correct kernels ---
    print("\n## False positives on correct kernels (should be 0/total across the board)\n")
    header = "| strategy | " + " | ".join(correct) + " |"
    sep = "|---|" + "|".join("---:" for _ in correct) + "|"
    print(header); print(sep)
    for s in strategies:
        print(f"| {s} | " + " | ".join(cell(s, k) for k in correct) + " |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
