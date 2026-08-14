#!/usr/bin/env python3
"""One-off probes that resolve the three soft claims left open by ADR 0007.

ADR 0007 shipped one finding as measured and three as inferred. This closes
the gap on all three, each with an experiment that could have falsified it:

  quant   is the q3_K mat-vec really 1.75x off its peers, or was that one
          shape in upstream's perf set? Rebuilds the same 0.5B at six pure
          single-type quantizations and reads achieved bandwidth end to end,
          on entirely different shapes from the ones that raised the flag.

  valley  does the batch 3 to 16 hole move with model size, or is it a
          property of one operator shape? Sweeps prompt width on a 0.5B and
          a 7B and compares where each one falls in.

  overhead is small-model generation slow because of fixed per-token dispatch
          cost? Fits time per token against bytes per token across five
          models; a real fixed cost shows up as a non-zero intercept, and the
          slope has to land on the machine's measured bandwidth or the model
          is wrong.

    .venv/bin/python bench/probe_baseline_gaps.py

Writes bench/results/baseline_probes.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import gguf_info

ROOT = Path(__file__).resolve().parent.parent
LLAMA_BENCH = Path("/Users/vlad/llama.cpp/build/bin/llama-bench")
MODEL_DIR = Path("/Users/vlad/models/gguf")
PURE_DIR = MODEL_DIR / "pure"
ROOFLINE = ROOT / "bench" / "results" / "roofline.json"
BASELINE = ROOT / "bench" / "results" / "llamacpp_baseline.json"
OUT = ROOT / "bench" / "results" / "baseline_probes.json"

REPS = 5

# Pure single-type quantizations of one model: same shapes, same layer count,
# only the mat-vec kernel changes.
PURE_TYPES = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]

LLAMA_QUANTIZE = Path("/Users/vlad/llama.cpp/build/bin/llama-quantize")

# The same comparison at a large reduction dimension. These two are built by
# requantizing an already-quantized file, so they are numerically junk and
# exist only to put the right tensor types at the right shapes for a timing
# test. They are deleted after the probe; nothing may use them for quality.
LARGE_K_SOURCE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
LARGE_K_TYPES = ["Q3_K", "Q4_K"]

VALLEY_PROMPTS = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 512]
VALLEY_MODELS = [
    ("qwen2.5-0.5b q4_k_m", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ("qwen2.5-7b q4_k_m", "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
]


def bench(model: Path, extra: list[str]) -> list[dict]:
    cmd = [str(LLAMA_BENCH), "-m", str(model), "-r", str(REPS), "-o", "json"] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed on {model.name}\n{proc.stderr[-1500:]}")
    return json.loads(proc.stdout)


def probe_quant_kernels(peak_bw_gbs: float) -> list[dict]:
    """Generation bandwidth per quantization type, one model, pure types."""
    rows = []
    for t in PURE_TYPES:
        path = PURE_DIR / f"qwen0.5b-pure-{t}.gguf"
        if not path.exists():
            print(f"  missing {path.name}, skipping", file=sys.stderr)
            continue
        cost = gguf_info.cost_model(gguf_info.read(path))
        row = bench(path, ["-p", "0", "-n", "128"])[0]
        ts = row["avg_ts"]
        gbs = (gguf_info.gen_bytes(cost) + gguf_info.kv_bytes(cost, 64)) * ts / 1e9
        rows.append(
            {
                "type": t,
                "tokens_per_s": round(ts, 2),
                "stddev_pct": round(100 * row["stddev_ts"] / ts, 2),
                "weight_bytes": gguf_info.gen_bytes(cost),
                "achieved_gbs": round(gbs, 1),
                "pct_of_peak_bandwidth": round(100 * gbs / peak_bw_gbs, 1),
            }
        )
        print(f"  {t:5} {ts:8.2f} tok/s  {gbs:6.1f} GB/s"
              f"  {100 * gbs / peak_bw_gbs:5.1f}% of peak", flush=True)
    return rows


def probe_large_k_quant(peak_bw_gbs: float, keep: bool = False) -> list[dict]:
    """The same quantization comparison where the reduction dimension is large.

    The 0.5B's reductions are too short to expose the q3_K kernel, so this
    repeats it on the 7B, whose FFN reduction is 18944. Byte counts differ
    between the two types, which is the point: if the smaller file is slower in
    absolute tokens per second, no traffic model explains it and the kernel is
    the only thing left.
    """
    PURE_DIR.mkdir(parents=True, exist_ok=True)
    rows, built = [], []
    for t in LARGE_K_TYPES:
        path = PURE_DIR / f"qwen7b-perfonly-{t}.gguf"
        if not path.exists():
            print(f"  building {path.name} (timing artifact, junk numerics)", flush=True)
            subprocess.run(
                [str(LLAMA_QUANTIZE), "--allow-requantize", "--pure",
                 str(MODEL_DIR / LARGE_K_SOURCE), str(path), t, "8"],
                capture_output=True, text=True, check=True,
            )
            built.append(path)
        cost = gguf_info.cost_model(gguf_info.read(path))
        row = bench(path, ["-p", "0", "-n", "64"])[0]
        ts = row["avg_ts"]
        b = gguf_info.gen_bytes(cost) + gguf_info.kv_bytes(cost, 32)
        gbs = b * ts / 1e9
        rows.append(
            {
                "type": t,
                "weight_bytes": gguf_info.gen_bytes(cost),
                "tokens_per_s": round(ts, 2),
                "achieved_gbs": round(gbs, 1),
                "pct_of_peak_bandwidth": round(100 * gbs / peak_bw_gbs, 1),
            }
        )
        print(f"  {t:5} {ts:8.2f} tok/s  {gbs:6.1f} GB/s"
              f"  {100 * gbs / peak_bw_gbs:5.1f}% of peak", flush=True)

    if not keep:
        for path in built:
            path.unlink()
            print(f"  removed {path.name}", flush=True)
    return rows


def probe_valley(roof: dict) -> dict:
    """Where the batch hole sits, on two model sizes."""
    peak_bw = roof["measured"]["peak_bandwidth_gbs"]
    peak_flops = roof["measured"]["peak_fp16_gflops"]
    out = {}
    for label, filename in VALLEY_MODELS:
        path = MODEL_DIR / filename
        cost = gguf_info.cost_model(gguf_info.read(path))
        rows = bench(path, ["-p", ",".join(map(str, VALLEY_PROMPTS)), "-n", "0"])
        points = []
        for row in rows:
            n = row["n_prompt"]
            ts = row["avg_ts"]
            flops = gguf_info.prompt_flops(cost, n)
            bytes_moved = gguf_info.gen_bytes(cost) + gguf_info.kv_bytes(cost, n)
            ai = flops / bytes_moved
            ceiling = min(peak_flops, ai * peak_bw)
            achieved = flops * ts / n / 1e9
            points.append(
                {
                    "n": n,
                    "tokens_per_s": round(ts, 2),
                    "arithmetic_intensity_flop_per_byte": round(ai, 2),
                    "achieved_gflops": round(achieved, 1),
                    "pct_of_roofline": round(100 * achieved / ceiling, 1),
                }
            )
            print(f"  {label:22} pp{n:<4} {100 * achieved / ceiling:5.1f}% of roofline",
                  flush=True)
        out[label] = points
    return out


def probe_overhead(baseline: dict, peak_bw_gbs: float) -> dict:
    """Fit time per token against bytes per token over the baseline's models.

    If generation were purely bandwidth-bound with no fixed cost, time per
    token would be bytes / bandwidth with a zero intercept. A non-zero
    intercept is a fixed per-token cost that no amount of model shrinking
    removes, and it is the thing that makes small models look inefficient.
    """
    tg = [p for p in baseline["runs"] if p["kind"] == "tg" and p["backend"] == "metal"]
    gb = np.array([p["bytes_per_pass"] / 1e9 for p in tg])
    ms = np.array([1000.0 / p["tokens_per_s"] for p in tg])

    slope, intercept = np.polyfit(gb, ms, 1)
    resid = ms - (slope * gb + intercept)
    implied_bw = 1000.0 / slope  # ms per GB -> GB/s

    return {
        "models": [p["model"] for p in tg],
        "bytes_per_token_gb": [round(float(x), 3) for x in gb],
        "ms_per_token": [round(float(x), 3) for x in ms],
        "fixed_cost_ms_per_token": round(float(intercept), 3),
        "implied_bandwidth_gbs": round(float(implied_bw), 1),
        "measured_peak_bandwidth_gbs": peak_bw_gbs,
        "implied_over_measured": round(float(implied_bw) / peak_bw_gbs, 3),
        "max_residual_ms": round(float(np.abs(resid).max()), 3),
        "fixed_cost_share_pct": [
            round(float(100 * intercept / t), 1) for t in ms
        ],
    }


def main() -> int:
    roof = json.loads(ROOFLINE.read_text())
    baseline = json.loads(BASELINE.read_text())
    peak_bw = roof["measured"]["peak_bandwidth_gbs"]

    print("[quant] pure single-type quantizations, generation bandwidth")
    quant = probe_quant_kernels(peak_bw)

    print("[large-k] the same comparison where the reduction is long")
    large_k = probe_large_k_quant(peak_bw, keep="--keep-artifacts" in sys.argv)

    print("[valley] prompt width sweep on two model sizes")
    valley = probe_valley(roof)

    print("[overhead] fixed cost fit over the baseline's models")
    overhead = probe_overhead(baseline, peak_bw)

    result = {
        "roofline": roof["measured"],
        "quant_kernels": quant,
        "quant_kernels_large_k": large_k,
        "valley": valley,
        "overhead": overhead,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("\n" + render(result))
    print(f"\nwritten to {OUT.relative_to(ROOT)}")
    return 0


def render(r: dict) -> str:
    out = [
        "generation bandwidth by pure quantization type, qwen2.5-0.5b, same shapes",
        "",
        "| type | tok/s | GB/s | % of peak |",
        "|---|---|---|---|",
    ]
    for q in r["quant_kernels"]:
        out.append(
            f"| {q['type']} | {q['tokens_per_s']} | {q['achieved_gbs']} |"
            f" {q['pct_of_peak_bandwidth']}% |"
        )

    out += [
        "",
        "the same comparison at a long reduction (qwen2.5-7b, k=18944)",
        "",
        "| type | weight GB | tok/s | GB/s | % of peak |",
        "|---|---|---|---|---|",
    ]
    for q in r["quant_kernels_large_k"]:
        out.append(
            f"| {q['type']} | {q['weight_bytes'] / 1e9:.2f} | {q['tokens_per_s']} |"
            f" {q['achieved_gbs']} | {q['pct_of_peak_bandwidth']}% |"
        )

    out += ["", "prompt width against the roofline, two model sizes", "",
            "| n | 0.5b % of roofline | 7b % of roofline |", "|---|---|---|"]
    small = {p["n"]: p for p in r["valley"]["qwen2.5-0.5b q4_k_m"]}
    big = {p["n"]: p for p in r["valley"]["qwen2.5-7b q4_k_m"]}
    for n in sorted(set(small) | set(big)):
        out.append(
            f"| {n} | {small.get(n, {}).get('pct_of_roofline', '-')}% |"
            f" {big.get(n, {}).get('pct_of_roofline', '-')}% |"
        )

    o = r["overhead"]
    out += [
        "",
        "fixed per-token cost fit",
        "",
        f"  fixed cost      {o['fixed_cost_ms_per_token']} ms/token",
        f"  implied bandwidth {o['implied_bandwidth_gbs']} GB/s"
        f"  ({o['implied_over_measured']:.0%} of the measured ceiling)",
        f"  worst residual  {o['max_residual_ms']} ms",
        "",
        "| model | ms/token | fixed cost share |",
        "|---|---|---|",
    ]
    for m, ms, share in zip(o["models"], o["ms_per_token"], o["fixed_cost_share_pct"]):
        out.append(f"| {m} | {ms} | {share}% |")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
