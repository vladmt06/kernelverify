#!/usr/bin/env python3
"""Per-kernel baseline: what upstream's Metal kernels reach on this machine.

llama-bench gives one tokens/s number per model, which is the right headline
and the wrong target: an optimiser replaces individual kernels, so it needs
per-kernel ceilings and per-kernel headroom. `test-backend-ops perf` runs
upstream's own kernel set on realistic shapes, and this script converts each
case into its position against the measured roofline.

For a mat-vec (n=1) the flop count is a misleading figure of merit, because
the kernel is streaming weights and doing two flops per weight; the honest
figure is the bandwidth it achieves, so for quantized types that is derived
from the shape and the bits-per-weight of the format.

    .venv/bin/python bench/baseline_kernels.py

Requires bench/results/roofline.json. Writes bench/results/kernel_baseline.json.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
TEST_BACKEND_OPS = Path("/Users/vlad/llama.cpp/build/bin/test-backend-ops")
ROOFLINE = ROOT / "bench" / "results" / "roofline.json"
OUT = ROOT / "bench" / "results" / "kernel_baseline.json"

BACKEND = "MTL0"

# The kernels an LLM inference path actually spends its time in. MUL_MAT_ID is
# left out because it only fires for mixture-of-experts models, which are not
# in the baseline model set.
OPS = ["MUL_MAT", "FLASH_ATTN_EXT", "SOFT_MAX", "RMS_NORM", "ROPE", "CPY", "ADD"]

# Bytes per weight for the storage formats that appear in the baseline models,
# taken from the block layouts in ggml-common.h.
BYTES_PER_WEIGHT = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q4_0": 18 / 32,
    "q4_1": 20 / 32,
    "q5_0": 22 / 32,
    "q5_1": 24 / 32,
    "q4_K": 144 / 256,
    "q5_K": 176 / 256,
    "q6_K": 210 / 256,
    "q2_K": 84 / 256,
    "q3_K": 110 / 256,
}

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(
    r"^\s*(?P<op>[A-Z_0-9]+)\((?P<params>.*)\):\s+(?P<runs>\d+) runs\s+-\s+"
    r"(?P<us>[\d.]+) us/run\s+-\s+(?P<work>[\d.]+) (?P<work_unit>\S+)/run\s+-\s+"
    r"(?P<rate>[\d.]+) (?P<rate_unit>\S+)"
)

# test-backend-ops picks its own unit prefix per case, so a case reported in
# TFLOPS and one reported in GFLOPS cannot be compared until both are scaled.
RATE_SCALE = {
    "MFLOPS": (1e-3, "GFLOPS"), "GFLOPS": (1.0, "GFLOPS"), "TFLOPS": (1e3, "GFLOPS"),
    "MB/s": (1e-3, "GB/s"), "GB/s": (1.0, "GB/s"), "TB/s": (1e3, "GB/s"),
}

SHAPE = re.compile(
    r"type_a=(?P<type_a>\w+),type_b=(?P<type_b>\w+),m=(?P<m>\d+),n=(?P<n>\d+),"
    r"k=(?P<k>\d+),bs=\[(?P<bs0>\d+),(?P<bs1>\d+)\],nr=\[(?P<nr0>\d+),(?P<nr1>\d+)\]"
)


def run_op(op: str) -> list[dict]:
    proc = subprocess.run(
        [str(TEST_BACKEND_OPS), "perf", "-o", op, "-b", BACKEND],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"test-backend-ops {op} failed:\n{proc.stderr[-2000:]}")

    rows = []
    for raw in ANSI.sub("", proc.stdout).splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        if m["rate_unit"] not in RATE_SCALE:
            raise ValueError(f"unhandled rate unit {m['rate_unit']} on {raw.strip()}")
        scale, unit = RATE_SCALE[m["rate_unit"]]
        rows.append(
            {
                "op": m["op"],
                "params": m["params"],
                "us_per_run": float(m["us"]),
                "rate": round(float(m["rate"]) * scale, 3),
                "rate_unit": unit,
            }
        )
    return rows


def annotate(row: dict, roof: dict) -> dict:
    """Score a case against the roofline at its own arithmetic intensity.

    A flat "percent of peak flops" is the wrong denominator for most of these
    kernels: at n=1 a matmul is streaming weights and could not reach the flop
    ceiling at any efficiency. The ceiling that applies to a case is
    min(peak flops, intensity * peak bandwidth), which is the roofline itself.
    """
    peak_bw = roof["measured"]["peak_bandwidth_gbs"]
    peak_flops = roof["measured"]["peak_fp16_gflops"]

    if row["rate_unit"] == "GB/s":
        row["pct_of_peak_bandwidth"] = round(100.0 * row["rate"] / peak_bw, 1)
        return row

    shape = SHAPE.search(row["params"])
    if row["op"] != "MUL_MAT" or not shape:
        # No byte model for this case, so no honest percentage: the rate is
        # reported and left unscored.
        return row

    type_a = shape["type_a"]
    m, n, k = int(shape["m"]), int(shape["n"]), int(shape["k"])
    batch = int(shape["bs0"]) * int(shape["bs1"]) * int(shape["nr0"]) * int(shape["nr1"])
    row |= {"type_a": type_a, "m": m, "n": n, "k": k, "batch": batch}

    bpw = BYTES_PER_WEIGHT.get(type_a)
    if bpw is None or batch != 1:
        # Broadcast and batched cases re-read the same operand a shape-
        # dependent number of times; that is a different traffic model and
        # guessing it would be worse than leaving these rows unscored.
        row["scored"] = False
        return row

    bytes_moved = m * k * bpw + k * n * 4 + m * n * 4
    flops = 2.0 * m * n * k
    ai = flops / bytes_moved
    ceiling = min(peak_flops, ai * peak_bw)

    row |= {
        "scored": True,
        "arithmetic_intensity_flop_per_byte": round(ai, 2),
        "weight_gbs": round(m * k * bpw / (row["us_per_run"] * 1e-6) / 1e9, 1),
        "achieved_gbs": round(bytes_moved / (row["us_per_run"] * 1e-6) / 1e9, 1),
        "roofline_ceiling_gflops": round(ceiling, 1),
        "pct_of_roofline": round(100.0 * row["rate"] / ceiling, 1),
        "bound_by": "memory" if ai < peak_flops / peak_bw else "compute",
    }
    return row


def summarise(rows: list[dict], roof: dict) -> str:
    ridge = roof["measured"]["ridge_fp16_flop_per_byte"]
    out = [
        "| op | cases | metric | best | median | scored |",
        "|---|---|---|---|---|---|",
    ]
    for op in OPS:
        sel = [r for r in rows if r["op"] == op]
        if not sel:
            continue
        units = {r["rate_unit"] for r in sel}
        rates = [r["rate"] for r in sel]
        scored = sum(1 for r in sel if r.get("scored") or "pct_of_peak_bandwidth" in r)
        out.append(
            f"| {op} | {len(sel)} | {'/'.join(sorted(units))} | {max(rates):.1f} |"
            f" {median(rates):.1f} | {scored} |"
        )

    scored = [r for r in rows if r.get("scored")]

    matvec = [r for r in scored if r["n"] == 1]
    if matvec:
        out += [
            "",
            f"mat-vec (n=1), the generation-time kernel. The ridge point is {ridge}"
            " flop/byte, so every one of these is bandwidth-bound and the ceiling"
            " that applies is bandwidth, not flops.",
            "",
            "| type_a | cases | best GB/s | median GB/s | median % of roofline |",
            "|---|---|---|---|---|",
        ]
        for t in sorted({r["type_a"] for r in matvec}):
            sel = [r for r in matvec if r["type_a"] == t]
            out.append(
                f"| {t} | {len(sel)} | {max(r['achieved_gbs'] for r in sel):.1f} |"
                f" {median([r['achieved_gbs'] for r in sel]):.1f} |"
                f" {median([r['pct_of_roofline'] for r in sel]):.1f}% |"
            )

    batched = [r for r in scored if r["n"] > 1]
    if batched:
        out += [
            "",
            "batched MUL_MAT (n>1), each case against the roofline at its own"
            " intensity rather than against the flop ceiling",
            "",
            "| n | cases | median flop/byte | median GFLOPS | median % of roofline | bound by |",
            "|---|---|---|---|---|---|",
        ]
        for n in sorted({r["n"] for r in batched}):
            sel = [r for r in batched if r["n"] == n]
            med_ai = median([r["arithmetic_intensity_flop_per_byte"] for r in sel])
            out.append(
                f"| {n} | {len(sel)} | {med_ai:.1f} |"
                f" {median([r['rate'] for r in sel]):.1f} |"
                f" {median([r['pct_of_roofline'] for r in sel]):.1f}% |"
                f" {'memory' if med_ai < ridge else 'compute'} |"
            )

    unscored = [r for r in rows if r.get("scored") is False]
    if unscored:
        out += [
            "",
            f"{len(unscored)} MUL_MAT cases are left unscored: they broadcast or batch"
            " an operand, which changes the traffic model, and an invented model would"
            " be worse than an absent one.",
        ]
    return "\n".join(out)


def main() -> int:
    if "--show" in sys.argv:
        stored = json.loads(OUT.read_text())
        print(summarise(stored["rows"], {"measured": stored["roofline"]}))
        return 0
    if not TEST_BACKEND_OPS.exists():
        print(f"not built: {TEST_BACKEND_OPS}", file=sys.stderr)
        return 1
    if not ROOFLINE.exists():
        print("run bench/roofline.py first", file=sys.stderr)
        return 1
    roof = json.loads(ROOFLINE.read_text())

    rows: list[dict] = []
    for op in OPS:
        print(f"[perf] {op}", flush=True)
        rows += [annotate(r, roof) for r in run_op(op)]

    result = {
        "roofline": roof["measured"],
        "machine": roof["machine"],
        "backend": BACKEND,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("\n" + summarise(rows, roof))
    print(f"\n{len(rows)} cases written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
