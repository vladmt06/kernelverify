#!/usr/bin/env python3
"""P1 flagship driver: fuzz corpus kernels and contrast gpuemu's verdict with
each kernel's benchmark 'passing' verdict.

Writes results.jsonl (one row per fuzz iteration) and summary.json (per-kernel
roll-up incl. the minimal failing case). See drivers/README.md for the schema.

Oracle:
  --oracle local   (default offline) numpy mini-fuzzer + fp64 reference scripts
  --oracle daemon  gpuemu-py + a running daemon (canonical; used on GPU image)
  --oracle auto    daemon if importable & reachable, else local
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _p1lib as lib  # noqa: E402

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "corpus"


def _emit(out_dir: Path, records: list, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _run_local(ops, iters, seed, dtypes_override):
    records = []
    for op in ops:
        rng = np.random.default_rng(seed)
        dtypes = dtypes_override or op.dtypes
        for i in range(iters):
            dtype = dtypes[i % len(dtypes)]
            case = lib.sample_case(op, dtype, rng)
            inputs = lib.make_inputs(op, case["shapes"], dtype, rng)
            try:
                output = op.run(inputs)
                cmp = lib.compare(output, lib.run_reference(op.ref_script, inputs), op.tol(dtype))
            except Exception as e:  # kernel or ref crash counts as a failure
                cmp = {"passed": False, "failure_kind": "exception",
                       "max_abs_err": float("inf"), "max_rel_err": float("inf"),
                       "max_ulp": None, "error_stats": None, "error": str(e)}
            records.append({
                "kernel": op.name, "source": op.source,
                "benchmark_verdict": op.benchmark_verdict,
                "iteration": i, "dtype": dtype, "layout": "contiguous",
                "shape": case["repr_shape"], "input_shapes": case["shapes"],
                "oracle": "local", **cmp,
            })
    return records


def _run_daemon(ops, iters, seed, dtypes_override, corpus_dir):
    """Canonical oracle: a real (isolated) gpuemu daemon generates inputs and
    validates submitted outputs. CPU-only — no GPU needed."""
    records = []
    with lib.DaemonManager(corpus_dir) as daemon:
        with daemon.client() as client:
            for op in ops:
                cases = client.get_test_batch(
                    op.name, count=iters, seed=seed,
                    op_schema=op.op_schema, dtypes=dtypes_override or op.dtypes)
                for i, case in enumerate(cases):
                    base = {
                        "kernel": op.name, "source": op.source,
                        "benchmark_verdict": op.benchmark_verdict,
                        "iteration": i, "seed": case["seed"],
                        "dtype": case["dtype"], "layout": case["layout"],
                        "shape": case["shape"], "oracle": "daemon",
                    }
                    try:
                        output = op.run(case["inputs"])
                        res = client.submit_output(op.name, case["inputs"], output, case["seed"])
                    except Exception as e:  # GPU absent, Triton compile fail, OOM, ...
                        records.append({**base, "passed": False, "failure_kind": "exception",
                                        "max_abs_err": float("inf"),
                                        "max_rel_err": float("inf"),
                                        "error_stats": None, "error": str(e)[:200]})
                        continue
                    kind = res.failures[0].get("kind") if res.failures else None
                    records.append({**base,
                        "passed": res.passed, "failure_kind": kind,
                        "max_abs_err": res.max_diff, "max_rel_err": res.max_rel_diff,
                        "error_stats": res.error_stats,
                    })
    return records


def summarize(records: list, run_id: str) -> dict:
    kernels: dict = {}
    for r in records:
        k = kernels.setdefault(r["kernel"], {
            "kernel": r["kernel"], "source": r["source"],
            "benchmark_verdict": r["benchmark_verdict"],
            "total": 0, "passed": 0, "failed": 0,
            "failure_kinds": {}, "min_failing_numel": None, "min_failing_shape": None,
        })
        k["total"] += 1
        if r["passed"]:
            k["passed"] += 1
        else:
            k["failed"] += 1
            fk = r.get("failure_kind") or "unknown"
            k["failure_kinds"][fk] = k["failure_kinds"].get(fk, 0) + 1
            numel = int(np.prod(r["shape"])) if r["shape"] else 1
            if k["min_failing_numel"] is None or numel < k["min_failing_numel"]:
                k["min_failing_numel"] = numel
                k["min_failing_shape"] = r["shape"]
    # gpuemu finds a bug the benchmark oracle missed when benchmark said pass.
    for k in kernels.values():
        k["gpuemu_verdict"] = "fail" if k["failed"] > 0 else "pass"
        k["illusion"] = (k["benchmark_verdict"] == "pass" and k["failed"] > 0)
    return {
        "run_id": run_id, "driver": "p1_llm_kernels",
        "kernels": list(kernels.values()),
        "extra_bugs_found": sum(1 for k in kernels.values() if k["illusion"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--paper", default="p1")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--kernels", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--oracle", choices=["auto", "local", "daemon"], default="local")
    ap.add_argument("--dtypes", default="", help="comma list to override per-op dtypes")
    args = ap.parse_args()

    names = [k for k in args.kernels.split(",") if k] or None
    ops = lib.load_corpus(Path(args.corpus), names)
    if not ops:
        print(f"no corpus ops found under {args.corpus}", file=sys.stderr)
        return 2
    dtypes_override = [d for d in args.dtypes.split(",") if d] or None

    oracle = args.oracle
    if oracle == "auto":
        try:
            import gpuemu_py.client  # noqa: F401
            oracle = "daemon"
        except Exception:
            oracle = "local"

    if oracle == "daemon":
        records = _run_daemon(ops, args.iters, args.seed, dtypes_override, Path(args.corpus))
    else:
        records = _run_local(ops, args.iters, args.seed, dtypes_override)

    for r in records:
        r["run_id"] = args.run_id
        r["paper"] = args.paper

    summary = summarize(records, args.run_id)
    summary["oracle"] = oracle
    _emit(Path(args.out), records, summary)
    print(json.dumps({"run_id": args.run_id, "oracle": oracle,
                      "ops": [op.name for op in ops],
                      "extra_bugs_found": summary["extra_bugs_found"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
