#!/usr/bin/env python3
"""P4 correlation: pair correct/buggy kernel variants and compare static vs perf deltas.

Reads a p4_artifacts results.jsonl and produces:
  - Per-kernel table (registers, spills, local, instrs, perf_ms).
  - Pairwise correct-vs-buggy diffs (Δregs, Δinstrs, Δperf%).
  - (--gpu-runs) Cross-architecture Δperf% table per pair.

The "*_buggy" suffix convention identifies pairs automatically.

Usage:
  python3 analysis/p4_correlation.py <run_id>
  python3 analysis/p4_correlation.py --gpu-runs "H100_NVL=run-...,A100_SXM4=run-...,..."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(run_dir: Path) -> list:
    with (run_dir / "results.jsonl").open() as f:
        return [json.loads(l) for l in f if l.strip()]


def _summary(rec: dict) -> dict:
    ptxs = rec.get("ptx") or []
    perf = rec.get("perf") or {}
    if not ptxs or not perf.get("available"):
        return {"skipped": True, "reason": rec.get("skipped") or rec.get("error") or "no data"}
    p = ptxs[0]
    return {
        "skipped": False,
        "regs": p["register_count"],
        "spills": p["spill_count"],
        "local": p["local_memory_bytes"],
        "instrs": p["instruction_count"],
        "ms_median": perf["ms_median"],
        "violations": p.get("violations") or [],
    }


def _emit_cross_gpu_table(gpu_runs: dict, data_dir: Path) -> None:
    """Print one row per (correct, buggy) pair, with a Δperf% column per GPU."""
    gpus = list(gpu_runs.keys())
    per_gpu: dict[str, dict[str, dict]] = {}
    for g, rid in gpu_runs.items():
        per_gpu[g] = {r["kernel"]: _summary(r) for r in _load(data_dir / rid)}

    # Pair set: any *_buggy that has a sibling on at least one GPU and both have perf.
    pair_set: set[str] = set()
    for g in gpus:
        for n in per_gpu[g]:
            if not n.endswith("_buggy"):
                continue
            correct = n[:-len("_buggy")]
            if correct in per_gpu[g]:
                pair_set.add(n)
    pairs = sorted(pair_set)
    if not pairs:
        print("no (correct, buggy) pairs found across the supplied runs")
        return

    print("## Cross-architecture Δperf% per pair")
    print()
    header = "| pair | " + " | ".join(gpus) + " |"
    sep = "|---|" + "".join(["---:|"] * len(gpus))
    print(header)
    print(sep)
    for buggy in pairs:
        correct = buggy[:-len("_buggy")]
        cells: list[str] = []
        for g in gpus:
            c = per_gpu[g].get(correct)
            b = per_gpu[g].get(buggy)
            if c is None or b is None or c["skipped"] or b["skipped"]:
                cells.append("—")
                continue
            dp = (b["ms_median"] - c["ms_median"]) / c["ms_median"] * 100.0
            cells.append(f"{dp:+.1f}%")
        print(f"| {correct} → {buggy} | " + " | ".join(cells) + " |")

    # Mean Δregs across GPUs (registers are architecture-independent for the same
    # PTX; this is a consistency check, not a per-arch observation).
    print()
    print("## Δregs / Δinstrs (architecture-independent; reported as cross-GPU mean)")
    print()
    print("| pair | Δregs | Δinstrs |")
    print("|---|---:|---:|")
    for buggy in pairs:
        correct = buggy[:-len("_buggy")]
        drs, dis = [], []
        for g in gpus:
            c = per_gpu[g].get(correct)
            b = per_gpu[g].get(buggy)
            if c is None or b is None or c["skipped"] or b["skipped"]:
                continue
            drs.append(b["regs"] - c["regs"])
            dis.append(b["instrs"] - c["instrs"])
        if not drs:
            print(f"| {correct} → {buggy} | — | — |")
            continue
        dr = sum(drs) / len(drs)
        di = sum(dis) / len(dis)
        print(f"| {correct} → {buggy} | {dr:+.1f} | {di:+.1f} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?", help="single run id (omit when using --gpu-runs)")
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--gpu-runs", default="",
                    help="comma-separated GPU=run_id pairs for cross-architecture table")
    args = ap.parse_args()

    if args.gpu_runs:
        pairs = dict(p.split("=") for p in args.gpu_runs.split(",") if "=" in p)
        _emit_cross_gpu_table(pairs, Path(args.data_dir))
        return 0

    if not args.run_id:
        ap.error("either run_id or --gpu-runs is required")

    rows = _load(Path(args.data_dir) / args.run_id)
    by_name = {r["kernel"]: r for r in rows}
    summaries = {k: _summary(r) for k, r in by_name.items()}

    print(f"## Per-kernel static + measured perf  ({by_name[next(iter(by_name))].get('device', {}).get('name', '?')})")
    print()
    print("| kernel | regs | spills | local B | instrs | ms_median | violations |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for name in sorted(summaries):
        s = summaries[name]
        if s["skipped"]:
            print(f"| {name} | _skipped: {s['reason']}_ |  |  |  |  |  |")
            continue
        print(f"| {name} | {s['regs']} | {s['spills']} | {s['local']} | {s['instrs']} | "
              f"{s['ms_median']:.4f} | {', '.join(s['violations']) or '-'} |")

    # Pair correct vs *_buggy.
    pairs = [(n[:-len("_buggy")], n) for n in summaries if n.endswith("_buggy")
             and n[:-len("_buggy")] in summaries]
    if pairs:
        print()
        print("## Pairwise diffs (buggy − correct)")
        print()
        print("| pair | Δregs | Δinstrs | Δms_median | Δperf% |")
        print("|---|---:|---:|---:|---:|")
        for correct, buggy in pairs:
            c, b = summaries[correct], summaries[buggy]
            if c["skipped"] or b["skipped"]:
                continue
            dr = b["regs"] - c["regs"]
            di = b["instrs"] - c["instrs"]
            dms = b["ms_median"] - c["ms_median"]
            dp = (b["ms_median"] - c["ms_median"]) / c["ms_median"] * 100.0
            print(f"| {correct} → {buggy} | {dr:+d} | {di:+d} | {dms:+.4f} | {dp:+.1f}% |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
