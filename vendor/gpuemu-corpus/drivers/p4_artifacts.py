#!/usr/bin/env python3
"""P4 driver: pair static GPU-artifact metrics with measured runtime.

For every GPU-capable corpus kernel:
  1. Clear the Triton cache, run kernel.run() once on a representative shape
     (forces a fresh compile and populates the cache).
  2. Read every .ptx the run produced; send each to the daemon's lint_kernel
     for static metrics (registers, spills, local mem, instruction count).
  3. Wrap kernel.run() in _capture.time_kernel() for warmup + CUDA-event
     timing.
  4. Emit one results row per (kernel, PTX) pair tagging static metrics +
     measured perf. The P4 analysis correlates static deltas with measured
     deltas across kernel variants (e.g. softmax_triton vs softmax_triton_buggy).

Common CLI: --run-id --paper --iters --kernels --seed --out [--corpus]
Off-GPU hosts: skips Triton kernels gracefully and records `available=false`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _capture  # noqa: E402
import _p1lib as lib  # noqa: E402

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "corpus"


def _pick_shape(op: lib.CorpusOp) -> dict:
    """A "representative" shape: middle candidate of each dim."""
    schema = op.op_schema or {"dims": [], "inputs": [{"name": n, "dims": []} for n in op.input_names]}
    dim_pick = {d["name"]: d["candidates"][len(d["candidates"]) // 2] for d in schema.get("dims", [])}
    shapes = {}
    for ts in schema.get("inputs", []):
        shapes[ts["name"]] = [dim_pick[n] for n in ts["dims"]]
    # fallback for ops without a schema
    for name in op.input_names:
        shapes.setdefault(name, [128, 128])
    return shapes


def _make_inputs(shapes: dict, dtype: str = "float32") -> dict:
    rng = np.random.default_rng(0)
    npdt = {"float32": np.float32, "float16": np.float16}[dtype]
    return {n: (rng.standard_normal(shp).astype(npdt)) for n, shp in shapes.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--paper", default="p4")
    ap.add_argument("--iters", type=int, default=50)         # perf timing iters
    ap.add_argument("--kernels", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [k for k in args.kernels.split(",") if k] or None
    ops = lib.load_corpus(Path(args.corpus), names)
    if not ops:
        print(f"no corpus ops under {args.corpus}", file=sys.stderr)
        return 2

    triton_cache = Path(os.environ.get("TRITON_CACHE_DIR") or (Path.home() / ".triton" / "cache"))
    device = _capture.device_info()

    records = []
    with lib.DaemonManager(args.corpus) as daemon:
        with daemon.client() as client:
            for op in ops:
                rec_base = {
                    "run_id": args.run_id, "paper": args.paper,
                    "kernel": op.name, "source": op.source,
                    "benchmark_verdict": op.benchmark_verdict,
                    "dtype": args.dtype,
                    "device": device,
                }
                if not op.meta.get("requires_gpu"):
                    records.append({**rec_base, "skipped": "non-gpu kernel"})
                    continue

                shapes = _pick_shape(op)
                inputs = _make_inputs(shapes, args.dtype)
                rec_base["input_shapes"] = shapes

                # Clear Triton cache so we attribute the produced PTX to *this* kernel.
                if triton_cache.exists():
                    shutil.rmtree(triton_cache, ignore_errors=True)

                # First run: compiles + emits PTX into the cache. Also a correctness
                # sanity (we catch import / compile errors here).
                try:
                    _ = op.run(inputs)
                except Exception as e:
                    records.append({**rec_base, "error": f"first-run: {e!s}"[:400]})
                    continue

                ptxs = _capture.ptx_from_triton_cache(str(triton_cache))
                if not ptxs:
                    records.append({**rec_base, "error": "no .ptx emitted into Triton cache"})
                    continue

                # Static metrics via the daemon (real artifact analyzer).
                kernel_lints = []
                for p in ptxs:
                    lr = client.lint_kernel(p["ptx"])
                    if lr:
                        m = lr[0].get("metrics", {})
                        kernel_lints.append({
                            "ptx_name": p["name"],
                            "kernel_name": lr[0].get("kernel_name"),
                            "register_count": m.get("register_count"),
                            "spill_count": m.get("spill_count"),
                            "local_memory_bytes": m.get("local_memory_bytes"),
                            "instruction_count": m.get("instruction_count"),
                            "violations": [v.get("kind") for v in lr[0].get("violations", [])],
                        })

                # Measured perf: CUDA-event timing with warmup.
                perf = _capture.time_kernel(lambda: op.run(inputs), warmup=5, iters=args.iters)

                records.append({**rec_base, "ptx": kernel_lints, "perf": perf})

    with (out / "results.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = {
        "run_id": args.run_id, "driver": "p4_artifacts",
        "kernels_total": len(ops),
        "kernels_with_perf": sum(1 for r in records if (r.get("perf") or {}).get("available")),
        "kernels_skipped": sum(1 for r in records if r.get("skipped")),
        "kernels_errored": sum(1 for r in records if r.get("error")),
        "device": device,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
