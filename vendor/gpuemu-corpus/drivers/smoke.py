#!/usr/bin/env python3
"""Smoke driver: exercises the harness lifecycle end-to-end with no GPU or gpuemu.

It emits a `results.jsonl` in the same schema the real drivers use, so the
stage -> run -> collect -> teardown pipeline (and the analysis code that reads
results) can be validated offline. Replace with `p1_llm_kernels.py` etc. for science.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--paper", default="p1")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--kernels", default="smoke_op")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    kernels = [k for k in args.kernels.split(",") if k]

    records = []
    for kernel in kernels:
        for i in range(args.iters):
            seed = rng.getrandbits(64)
            # Synthetic: ~5% "failures" to exercise downstream analysis.
            passed = rng.random() > 0.05
            records.append({
                "run_id": args.run_id,
                "paper": args.paper,
                "kernel": kernel,
                "iteration": i,
                "seed": seed,
                "dtype": rng.choice(["float32", "float16", "bfloat16"]),
                "layout": rng.choice(["contiguous", "strided", "transposed"]),
                "shape": [rng.choice([1, 7, 64, 257]), rng.choice([64, 128])],
                "passed": passed,
                "max_abs_err": 0.0 if passed else 10 ** rng.uniform(-2, 1),
                "failure_kind": None if passed else rng.choice(
                    ["tolerance", "nan", "shape", "layout"]),
            })

    with (out / "results.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = {
        "run_id": args.run_id,
        "driver": "smoke",
        "total": len(records),
        "passed": sum(r["passed"] for r in records),
        "failed": sum(not r["passed"] for r in records),
        "kernels": kernels,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
