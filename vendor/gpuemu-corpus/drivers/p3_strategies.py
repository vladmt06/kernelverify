#!/usr/bin/env python3
"""P3 driver: test-generation strategy ablation.

Runs the corpus through several *strategies* — schema/dtype modifiers that
control what the fuzzer explores — and records per-(strategy, kernel) bug
recall + time-to-first-failure. The P3 paper compares which strategies most
efficiently expose which classes of LLM-style bugs.

Strategies implemented (each is a small function that mutates an op's
op_schema/dtypes before submitting to the daemon):
  default               — native schema (mixed boundary + regular candidates)
  boundary              — H candidates restricted to small/odd sizes {1,3,7}
  regular               — H candidates restricted to power-of-two-ish {128,256,512}
  single_dtype_f32      — native shapes, only float32
  single_dtype_f16      — native shapes, only float16

Per-iteration output schema (results.jsonl) is identical to the P1 driver
plus a `strategy` field so analysis can group cleanly.

Common CLI: --run-id --paper --iters --kernels --seed --out [--corpus]
Driver-specific: --strategies "default,boundary,regular"
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _p1lib as lib  # noqa: E402

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "corpus"
BOUNDARY = [1, 3, 7]
REGULAR = [128, 256, 512]


def _set_H_candidates(schema: dict, candidates: list[int]) -> dict:
    s = copy.deepcopy(schema or {})
    for d in s.get("dims", []):
        if d.get("name") == "H":
            d["candidates"] = candidates
    return s


def _build_strategies(op: lib.CorpusOp):
    """Return {strategy_name: (op_schema, dtypes, value_distribution)}."""
    base_schema = op.op_schema
    native_dtypes = op.dtypes
    return {
        "default":          (base_schema, native_dtypes, None),
        "boundary":         (_set_H_candidates(base_schema, BOUNDARY), native_dtypes, None),
        "regular":          (_set_H_candidates(base_schema, REGULAR), native_dtypes, None),
        "single_dtype_f32": (base_schema, ["float32"], None),
        "single_dtype_f16": (base_schema, ["float16"], None),
        "nan_injected":     (base_schema, native_dtypes, "nan_injected"),
        "adversarial":      (base_schema, native_dtypes, "adversarial"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--paper", default="p3")
    ap.add_argument("--iters", type=int, default=10, help="iterations per (strategy, kernel)")
    ap.add_argument("--kernels", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--strategies",
                    default="default,boundary,regular,single_dtype_f32,single_dtype_f16,nan_injected,adversarial")
    args = ap.parse_args()

    names = [k for k in args.kernels.split(",") if k] or None
    ops = lib.load_corpus(Path(args.corpus), names)
    if not ops:
        print(f"no corpus ops under {args.corpus}", file=sys.stderr)
        return 2
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    records = []
    with lib.DaemonManager(args.corpus) as daemon:
        with daemon.client() as client:
            for strategy in strategies:
                for op in ops:
                    strat_map = _build_strategies(op)
                    if strategy not in strat_map:
                        continue
                    schema, dtypes, vd = strat_map[strategy]
                    # Per-strategy seed: deterministic + non-overlapping across strategies.
                    seed = args.seed + sum(ord(c) for c in strategy)
                    try:
                        cases = client.get_test_batch(
                            op.name, count=args.iters, seed=seed,
                            op_schema=schema, dtypes=dtypes,
                            value_distribution=vd)
                    except Exception as e:
                        records.append({
                            "run_id": args.run_id, "paper": args.paper,
                            "strategy": strategy, "kernel": op.name,
                            "passed": False, "failure_kind": "get_test_batch_failed",
                            "error": str(e)[:300],
                        })
                        continue
                    ttff = None
                    t0 = time.time()
                    for i, case in enumerate(cases):
                        base = {
                            "run_id": args.run_id, "paper": args.paper,
                            "strategy": strategy, "kernel": op.name,
                            "source": op.source,
                            "benchmark_verdict": op.benchmark_verdict,
                            "iteration": i, "seed": case["seed"],
                            "dtype": case["dtype"], "layout": case["layout"],
                            "shape": case["shape"],
                        }
                        try:
                            output = op.run(case["inputs"])
                            res = client.submit_output(op.name, case["inputs"], output, case["seed"])
                        except Exception as e:
                            records.append({**base, "passed": False,
                                            "failure_kind": "exception",
                                            "error": str(e)[:300]})
                            continue
                        kind = res.failures[0].get("kind") if res.failures else None
                        if not res.passed and ttff is None:
                            ttff = time.time() - t0
                        records.append({**base,
                            "passed": res.passed, "failure_kind": kind,
                            "max_abs_err": res.max_diff,
                            "max_rel_err": res.max_rel_diff,
                            "error_stats": res.error_stats,
                            "ttff_s": ttff,
                        })

    # Roll up to summary.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary: dict = {
        "run_id": args.run_id, "driver": "p3_strategies",
        "strategies": strategies, "kernels": [op.name for op in ops],
        "total_rows": len(records),
        "by_strategy": {},
    }
    for s in strategies:
        rs = [r for r in records if r.get("strategy") == s]
        per_kernel = {}
        for op in ops:
            kr = [r for r in rs if r.get("kernel") == op.name]
            per_kernel[op.name] = {
                "total": len(kr),
                "fail": sum(1 for r in kr if not r.get("passed", False)),
                "ttff_s": min((r.get("ttff_s") for r in kr if r.get("ttff_s") is not None), default=None),
            }
        summary["by_strategy"][s] = {
            "total": len(rs),
            "fail": sum(1 for r in rs if not r.get("passed", False)),
            "per_kernel": per_kernel,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"run_id": args.run_id, "strategies": strategies,
                      "total_rows": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
