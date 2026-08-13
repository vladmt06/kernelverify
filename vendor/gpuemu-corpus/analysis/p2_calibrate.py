#!/usr/bin/env python3
"""P2 tolerance calibration — calibrate from controls, evaluate on buggy.

Aggregates ErrorStats from accumulated P1 runs and reports:
  1. Per (op-family, dtype): proposed atol = p95(max_abs from the *correct*
     kernel's passing cases) × safety. Buggy variants inherit the proposal
     of their correct counterpart (the buggy kernel's own data is the wrong
     reference — calibration must come from observed-correct behaviour).
  2. Bug-detection table: for each buggy kernel, recall under the current
     per-op tol vs the calibrated tol. Tighter calibration should preserve
     (or improve) recall while shrinking the false-positive surface on the
     correct kernels.
  3. Headline flips: total false-positive recovery (correct kernels that
     previously failed and now pass) and recall preserved on buggy kernels.

The buggy-to-correct map is hard-coded by naming convention for now (the
corpus is small); future versions can read a `calibrate_against` field
from each kernel's meta.json.

Usage: python3 analysis/p2_calibrate.py [--data-dir ../data] [--corpus ../corpus]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

SAFETY = 1.5  # multiplier on p95 to leave headroom against worst-case rounding

# Buggy kernel -> correct kernel whose passing distribution we calibrate from.
BUGGY_TO_CORRECT = {
    "softmax_llm_buggy": "softmax_triton",     # GPU control for the family
    "softmax_triton_buggy": "softmax_triton",
    "gelu_triton_buggy": "gelu_triton",
    "silu_triton_buggy": "silu_triton",
    "rmsnorm_triton_buggy": "rmsnorm_triton",
    "leaky_relu_triton_buggy": "leaky_relu_triton",
    "l2norm_triton_buggy": "l2norm_triton",
}


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return math.nan
    xs = sorted(xs)
    idx = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
    return xs[idx]


def _load_runs(data_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for results in sorted(data_dir.glob("run-*/results.jsonl")):
        for line in results.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            es = r.get("error_stats")
            if r.get("failure_kind") == "exception" or es is None:
                continue
            ma = es.get("max_abs")
            if ma is None:
                continue
            rows.append({
                "kernel": r.get("kernel"),
                "dtype": r.get("dtype"),
                "passed": bool(r.get("passed")),
                "max_abs": float(ma),
                "run_id": results.parent.name,
            })
    return rows


def _corpus_tols(corpus: Path) -> dict:
    tols: dict[tuple[str, str], float] = {}
    for meta in sorted(corpus.glob("*/meta.json")):
        m = json.loads(meta.read_text())
        for dt, tol in (m.get("tolerances") or {}).items():
            tols[(m["name"], dt)] = float(tol)
    return tols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parents[1] / "corpus"))
    args = ap.parse_args()

    rows = _load_runs(Path(args.data_dir))
    if not rows:
        print("no rows with error_stats found under data/")
        return 2
    tols_now = _corpus_tols(Path(args.corpus))

    by_kd: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_kd[(r["kernel"], r["dtype"])].append(r)

    correct_kernels = {k for k, _ in by_kd if k not in BUGGY_TO_CORRECT}

    def calibrate_for(name: str, dt: str) -> float:
        """p95 max_abs over the *correct* counterpart's passing cases × SAFETY."""
        src_name = BUGGY_TO_CORRECT.get(name, name)
        rs = by_kd.get((src_name, dt), [])
        pass_xs = [r["max_abs"] for r in rs if r["passed"]]
        if not pass_xs:
            return math.nan
        return _pct(pass_xs, 0.95) * SAFETY

    # Table 1: calibrated tolerance per (correct kernel, dtype).
    print(f"# P2 tolerance calibration\n")
    print(f"_Sourced from {len(rows)} rows across "
          f"{len(list(Path(args.data_dir).glob('run-*/results.jsonl')))} runs._\n")
    print("## Calibrated atol per (correct kernel, dtype)\n")
    print("| kernel | dtype | n_pass | tol_now | p50_abs | p95_abs | proposed_tol | tightening |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for (k, dt) in sorted(by_kd):
        if k not in correct_kernels:
            continue
        rs = by_kd[(k, dt)]
        pass_xs = [r["max_abs"] for r in rs if r["passed"]]
        tol = tols_now.get((k, dt))
        proposed = calibrate_for(k, dt)
        p50 = _pct(pass_xs, 0.50)
        p95 = _pct(pass_xs, 0.95)
        tighten = f"{tol / proposed:.1f}x" if (tol and proposed and not math.isnan(proposed)) else "—"
        def f(x):
            return f"{x:.2e}" if isinstance(x, float) and not math.isnan(x) else "—"
        print(f"| {k} | {dt} | {len(pass_xs)} | {f(tol) if tol else '—'} | "
              f"{f(p50)} | {f(p95)} | {f(proposed)} | {tighten} |")

    # Table 2: bug-detection recall under current tol vs calibrated tol.
    print("\n## Bug recall (fraction of buggy cases flagged), current vs calibrated\n")
    print("| buggy kernel | dtype | tol_now | recall_now | calibrated_tol | recall_calibrated |")
    print("|---|---|---:|---:|---:|---:|")
    total_flagged_now = 0
    total_flagged_cal = 0
    total_buggy_rows = 0
    for (k, dt) in sorted(by_kd):
        if k not in BUGGY_TO_CORRECT:
            continue
        rs = by_kd[(k, dt)]
        tol = tols_now.get((k, dt))
        proposed = calibrate_for(k, dt)
        n_now = sum(1 for r in rs if tol is not None and r["max_abs"] > tol)
        n_cal = sum(1 for r in rs if not math.isnan(proposed) and r["max_abs"] > proposed)
        total_flagged_now += n_now
        total_flagged_cal += n_cal
        total_buggy_rows += len(rs)
        def f(x):
            return f"{x:.2e}" if isinstance(x, float) and not math.isnan(x) else "—"
        rnow = f"{n_now}/{len(rs)} ({n_now/len(rs)*100:.0f}%)" if rs else "—"
        rcal = f"{n_cal}/{len(rs)} ({n_cal/len(rs)*100:.0f}%)" if rs else "—"
        print(f"| {k} | {dt} | {f(tol) if tol else '—'} | {rnow} | {f(proposed)} | {rcal} |")

    # Table 3: false-positive recovery on the correct controls.
    print("\n## False positives recovered on correct kernels\n")
    print("| kernel | dtype | tol_now | n_fp_now | calibrated_tol | n_fp_calibrated | recovered |")
    print("|---|---|---:|---:|---:|---:|---:|")
    recovered_total = 0
    for (k, dt) in sorted(by_kd):
        if k not in correct_kernels:
            continue
        rs = by_kd[(k, dt)]
        tol = tols_now.get((k, dt))
        proposed = calibrate_for(k, dt)
        # "false positive" = the kernel is CORRECT yet a row was flagged.
        fp_now = sum(1 for r in rs if (tol is not None) and r["max_abs"] > tol)
        fp_cal = sum(1 for r in rs if (not math.isnan(proposed)) and r["max_abs"] > proposed)
        recovered = max(0, fp_now - fp_cal)
        recovered_total += recovered
        if fp_now == 0 and fp_cal == 0:
            continue
        def f(x):
            return f"{x:.2e}" if isinstance(x, float) and not math.isnan(x) else "—"
        print(f"| {k} | {dt} | {f(tol) if tol else '—'} | {fp_now} | "
              f"{f(proposed)} | {fp_cal} | {recovered} |")

    # Headline.
    print()
    print(f"**Headline:** Calibrated atol (p95 × {SAFETY} from correct kernels) "
          f"recovers **{recovered_total}** false positives on correct kernels while "
          f"flagging **{total_flagged_cal} / {total_buggy_rows}** buggy cases "
          f"(vs {total_flagged_now} / {total_buggy_rows} under the current per-op tol).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
