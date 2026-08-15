#!/usr/bin/env python3
"""llama.cpp baseline on this machine, placed against the measured roofline.

This is the number any generated kernel has to beat. It runs upstream
llama.cpp at a pinned commit, out of the box, with no flags that a normal user
would not get by default, and then converts each result into the two
quantities the roofline is drawn in:

  token generation   weights are streamed once per token, so tokens/s times
                     model bytes is an achieved bandwidth, scored against the
                     measured streaming ceiling.
  prompt processing  a prompt of N tokens is a GEMM with N columns, so
                     2 * params * tokens/s is an achieved flop rate, scored
                     against the measured FMA ceiling.

Sweeping prompt length from 1 to 512 walks arithmetic intensity across the
ridge point, which is what turns two isolated numbers into a roofline.

    .venv/bin/python bench/baseline_llamacpp.py           # full baseline
    .venv/bin/python bench/baseline_llamacpp.py --quick   # smoke test

Requires bench/results/roofline.json (run bench/roofline.py first).
Writes bench/results/llamacpp_baseline.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gguf_info
from external import (
    CMAKE_FLAGS,
    GGUF_DIR as MODEL_DIR,
    LLAMA_BENCH,
    LLAMA_CPP,
    run_llama_bench,
)

ROOT = Path(__file__).resolve().parent.parent
ROOFLINE = ROOT / "bench" / "results" / "roofline.json"
OUT = ROOT / "bench" / "results" / "llamacpp_baseline.json"

# (label, filename). Chosen to span the regimes: a model small enough to be
# compute-bound at modest batch, and one large enough that generation is
# nothing but bandwidth.
MODELS = [
    ("qwen2.5-0.5b q4_k_m", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ("qwen2.5-0.5b q8_0", "qwen2.5-0.5b-instruct-q8_0.gguf"),
    ("qwen2.5-0.5b f16", "qwen2.5-0.5b-instruct-fp16.gguf"),
    ("llama-3.2-1b q4_k_m", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    ("qwen2.5-7b q4_k_m", "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
]

# The model whose prompt length is swept to trace the roofline curve.
SWEEP_MODEL = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
SWEEP_PROMPTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# CPU-only comparison runs, for the small models only: a 7B on six cores adds
# minutes and tells us nothing the 1B does not.
CPU_MODELS = ["qwen2.5-0.5b-instruct-q4_k_m.gguf", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"]


def run_bench(model: Path, extra: list[str], reps: int) -> list[dict]:
    return run_llama_bench(model, ["-r", str(reps)] + extra)


def derive(row: dict, cost: dict, roof: dict, backend: str = "metal") -> dict:
    """Turn one llama-bench row into a roofline coordinate.

    Both axes come from the model's own tensor table (bench/gguf_info.py), not
    from the headline parameter count: the embedding table is a gather, and the
    output head runs once per decode call rather than once per prompt token.
    """
    # Each backend is scored against its own ceilings. The CPU and the GPU
    # share one memory system but reach different fractions of it, and the CPU
    # matmul path runs through Accelerate rather than the GPU's ALUs, so a
    # single denominator would flatter one and libel the other.
    m = roof["measured"]
    if backend == "cpu":
        peak_bw = m["cpu_peak_bandwidth_gbs"] * 1e9
        peak_flops = m["cpu_peak_gflops"] * 1e9
    else:
        peak_bw = m["peak_bandwidth_gbs"] * 1e9
        peak_flops = m["peak_fp16_gflops"] * 1e9
    ts = row["avg_ts"]

    if row["n_gen"] > 0:
        kind, width = "tg", row["n_gen"]
        # Generation runs one token per pass, over a cache that grows from
        # empty to `width`, so the cache traffic is charged at its mean.
        flops_per_pass = gguf_info.gen_flops(cost, width / 2)
        bytes_per_pass = gguf_info.gen_bytes(cost) + gguf_info.kv_bytes(cost, width / 2)
        passes_per_s = ts
    else:
        kind, width = "pp", row["n_prompt"]
        flops_per_pass = gguf_info.prompt_flops(cost, width)
        bytes_per_pass = gguf_info.gen_bytes(cost) + gguf_info.kv_bytes(cost, width)
        passes_per_s = ts / width

    bytes_per_s = bytes_per_pass * passes_per_s
    flops_per_s = flops_per_pass * passes_per_s

    return {
        "kind": kind,
        "width": width,
        "tokens_per_s": round(ts, 2),
        "stddev_tokens_per_s": round(row.get("stddev_ts", 0.0), 2),
        "reps": len(row.get("samples_ts", [])),
        # The quantity a candidate kernel has to beat is the mean, so what
        # matters is the error on the mean, not the spread of single runs.
        "stderr_pct": round(
            100.0 * row.get("stddev_ts", 0.0) / ts / max(len(row.get("samples_ts", [1])), 1) ** 0.5,
            2,
        ),
        "model_size_bytes": row["model_size"],
        "model_n_params": row["model_n_params"],
        "bytes_per_pass": int(bytes_per_pass),
        "gflop_per_pass": round(flops_per_pass / 1e9, 3),
        "arithmetic_intensity_flop_per_byte": round(flops_per_pass / bytes_per_pass, 2),
        "achieved_gbs": round(bytes_per_s / 1e9, 1),
        "achieved_gflops": round(flops_per_s / 1e9, 1),
        "pct_of_peak_bandwidth": round(100.0 * bytes_per_s / peak_bw, 1),
        "pct_of_peak_flops": round(100.0 * flops_per_s / peak_flops, 1),
    }


def bound_by(point: dict, roof: dict, backend: str = "metal") -> str:
    key = "cpu_ridge_flop_per_byte" if backend == "cpu" else "ridge_fp16_flop_per_byte"
    ridge = roof["measured"][key]
    return "memory" if point["arithmetic_intensity_flop_per_byte"] < ridge else "compute"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one rep, small models only")
    args = ap.parse_args()

    if not LLAMA_BENCH.exists():
        print(f"llama-bench not built at {LLAMA_BENCH}", file=sys.stderr)
        return 1
    if not ROOFLINE.exists():
        print("run bench/roofline.py first", file=sys.stderr)
        return 1
    roof = json.loads(ROOFLINE.read_text())

    reps = 1 if args.quick else 5
    models = MODELS[:2] if args.quick else MODELS
    prompts = [1, 32, 512] if args.quick else SWEEP_PROMPTS

    results = {"roofline": roof["measured"], "machine": roof["machine"], "runs": []}
    build = None
    costs = {}

    def cost_for(filename: str) -> dict:
        if filename not in costs:
            costs[filename] = gguf_info.cost_model(gguf_info.read(MODEL_DIR / filename))
        return costs[filename]

    # --- headline: pp512 and tg128, Metal, defaults ------------------------
    for label, filename in models:
        path = MODEL_DIR / filename
        if not path.exists():
            print(f"missing model {path}", file=sys.stderr)
            return 1
        print(f"[metal] {label}", flush=True)
        for row in run_bench(path, ["-p", "512", "-n", "128"], reps):
            build = build or {
                "llama_cpp_path": str(LLAMA_CPP),
                "llama_cpp_commit": row["build_commit"],
                "llama_cpp_build_number": row["build_number"],
                "cmake_flags": CMAKE_FLAGS,
                "backends": row["backends"],
                "gpu_info": row["gpu_info"],
                "cpu_info": row["cpu_info"],
                "n_threads": row["n_threads"],
                "flash_attn": row["flash_attn"],
                "type_k": row["type_k"],
                "type_v": row["type_v"],
            }
            point = derive(row, cost_for(filename), roof)
            point |= {"model": label, "backend": "metal", "bound_by": bound_by(point, roof)}
            results["runs"].append(point)

    # --- CPU-only, for the small models ------------------------------------
    if not args.quick:
        for label, filename in MODELS:
            if filename not in CPU_MODELS:
                continue
            print(f"[cpu]   {label}", flush=True)
            for row in run_bench(MODEL_DIR / filename, ["-p", "512", "-n", "128", "-ngl", "0"], reps):
                point = derive(row, cost_for(filename), roof, backend="cpu")
                point |= {
                    "model": label,
                    "backend": "cpu",
                    "bound_by": bound_by(point, roof, "cpu"),
                }
                results["runs"].append(point)

    # --- arithmetic intensity sweep ----------------------------------------
    print(f"[sweep] {SWEEP_MODEL}", flush=True)
    sweep_args = ["-p", ",".join(str(p) for p in prompts), "-n", "128"]
    results["sweep"] = []
    for row in run_bench(MODEL_DIR / SWEEP_MODEL, sweep_args, reps):
        point = derive(row, cost_for(SWEEP_MODEL), roof)
        point |= {"model": SWEEP_MODEL, "backend": "metal", "bound_by": bound_by(point, roof)}
        results["sweep"].append(point)

    results["build"] = build
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print("\n" + render(results))
    print(f"\nwritten to {OUT.relative_to(ROOT)}")
    return 0


def render(results: dict) -> str:
    r = results["roofline"]
    lines = [
        f"gpu roofline  {r['peak_bandwidth_gbs']} GB/s,"
        f" {r['peak_fp16_gflops'] / 1000:.2f} TFLOP/s fp16,"
        f" ridge {r['ridge_fp16_flop_per_byte']} flop/byte",
        f"cpu roofline  {r['cpu_peak_bandwidth_gbs']} GB/s,"
        f" {r['cpu_peak_gflops'] / 1000:.2f} TFLOP/s accelerate,"
        f" ridge {r['cpu_ridge_flop_per_byte']} flop/byte",
        "",
        "| model | backend | test | tok/s | achieved | of its ceiling | bound by |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in results["runs"]:
        if p["kind"] == "tg":
            achieved, pct = f"{p['achieved_gbs']} GB/s", f"{p['pct_of_peak_bandwidth']}%"
        else:
            achieved, pct = f"{p['achieved_gflops'] / 1000:.2f} TFLOP/s", f"{p['pct_of_peak_flops']}%"
        lines.append(
            f"| {p['model']} | {p['backend']} | {p['kind']}{p['width']} |"
            f" {p['tokens_per_s']} | {achieved} | {pct} | {p['bound_by']} |"
        )
    lines += [
        "",
        "arithmetic intensity sweep, llama-3.2-1b q4_k_m on metal",
        "",
        "| test | flop/byte | tok/s | GB/s | TFLOP/s | bound by |",
        "|---|---|---|---|---|---|",
    ]
    for p in results["sweep"]:
        lines.append(
            f"| {p['kind']}{p['width']} | {p['arithmetic_intensity_flop_per_byte']} |"
            f" {p['tokens_per_s']} | {p['achieved_gbs']} |"
            f" {p['achieved_gflops'] / 1000:.2f} | {p['bound_by']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
