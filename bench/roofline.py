#!/usr/bin/env python3
"""Machine roofline for this machine, measured.

Builds and runs the Metal probe in bench/metal/roofline_probe.mm, which
measures achieved streaming bandwidth and achieved FMA throughput on the GPU,
and pairs those with the arithmetic that produces the vendor spec ceiling for
the same machine. The measured numbers are the ones the baseline is scored
against; the spec numbers are carried only to show how much of the paper
ceiling any kernel can actually reach.

    .venv/bin/python bench/roofline.py            # measure, write results
    .venv/bin/python bench/roofline.py --show     # reprint the last result

Writes bench/results/roofline.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import machine_state  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "bench" / "metal" / "roofline_probe.mm"
BIN = ROOT / "bench" / ".cache" / "roofline_probe"
OUT = ROOT / "bench" / "results" / "roofline.json"

CLANG_ARGS = [
    "clang++", "-std=c++17", "-fobjc-arc", "-O2", "-DACCELERATE_NEW_LAPACK",
    "-framework", "Metal", "-framework", "Foundation", "-framework", "Accelerate",
]


@dataclass(frozen=True)
class SpecCeiling:
    """Vendor spec, reconstructed from its own arithmetic so it can be checked."""

    label: str
    bandwidth_gbs: float
    bandwidth_how: str
    fp32_gflops: float
    fp32_how: str


# Only the configurations this project actually baselines on need an entry.
SPECS = {
    "Mac15,7": SpecCeiling(
        label="Apple M3 Pro, 18-core GPU",
        bandwidth_gbs=153.6,
        bandwidth_how="LPDDR5-6400 on a 192-bit bus: 6400 MT/s * 24 B = 153.6 GB/s",
        fp32_gflops=6441.0,
        fp32_how="18 cores * 128 ALUs * 2 flop/FMA * 1.398 GHz = 6441 GFLOP/s",
    ),
}


def machine_facts() -> dict:
    """This machine and its power state, from bench/machine_state.py, mapped
    onto the key names roofline.json has always recorded.

    Power state belongs in the provenance: this is a laptop, and a ceiling
    measured under a battery power cap is not the ceiling a plugged-in
    machine has.
    """
    fp = machine_state.fingerprint()
    power = machine_state.power_state()
    return {
        "power_source": power["source"],
        "low_power_mode": power["low_power_mode"],
        "hw_model": fp["hw_model"],
        "cpu": fp["chip"],
        "cpu_cores": fp["cores"],
        "cpu_performance_cores": fp["performance_cores"],
        "cpu_efficiency_cores": fp["efficiency_cores"],
        "memory_bytes": fp["memory_bytes"],
        "os": fp["os"],
    }


def build_probe() -> None:
    BIN.parent.mkdir(parents=True, exist_ok=True)
    if BIN.exists() and BIN.stat().st_mtime > SRC.stat().st_mtime:
        return
    subprocess.run(CLANG_ARGS + [str(SRC), "-o", str(BIN)], check=True)


def measure() -> dict:
    build_probe()
    proc = subprocess.run([str(BIN)], capture_output=True, text=True, check=True)
    probe = json.loads(proc.stdout)

    facts = machine_facts()
    spec = SPECS.get(facts["hw_model"])

    bw = probe["bandwidth_gbs"]
    gf = probe["gflops"]
    # The ceilings a kernel is scored against: the best storage mode for
    # bandwidth, and the widest FMA issue rate the hardware sustains.
    peak_bw = max(bw.values())
    peak_f32 = gf["fma_f32"]
    peak_f16 = max(gf["fma_f16"], gf["fma_f16x4"])

    cpu = probe["cpu"]
    cpu_bw = max(cpu["triad_gbs_p_cores"], cpu["triad_gbs_all_cores"])
    result = {
        "machine": facts,
        "metal_device": probe["device"],
        "probe": probe,
        "measured": {
            "peak_bandwidth_gbs": round(peak_bw, 1),
            "peak_fp32_gflops": round(peak_f32, 1),
            "peak_fp16_gflops": round(peak_f16, 1),
            "ridge_fp32_flop_per_byte": round(peak_f32 / peak_bw, 1),
            "ridge_fp16_flop_per_byte": round(peak_f16 / peak_bw, 1),
            "cpu_peak_bandwidth_gbs": round(cpu_bw, 1),
            "cpu_peak_gflops": round(cpu["accelerate_sgemm_gflops"], 1),
            "cpu_ridge_flop_per_byte": round(cpu["accelerate_sgemm_gflops"] / cpu_bw, 1),
        },
    }
    if spec:
        result["spec"] = {
            "label": spec.label,
            "bandwidth_gbs": spec.bandwidth_gbs,
            "bandwidth_how": spec.bandwidth_how,
            "fp32_gflops": spec.fp32_gflops,
            "fp32_how": spec.fp32_how,
            "measured_fraction_bandwidth": round(peak_bw / spec.bandwidth_gbs, 3),
            "measured_fraction_fp32": round(peak_f32 / spec.fp32_gflops, 3),
        }
    return result


def render(result: dict) -> str:
    m = result["measured"]
    p = result["probe"]
    lines = [
        f"machine   {result['machine']['hw_model']}  {result['machine']['cpu']}"
        f"  {result['machine']['memory_bytes'] / (1 << 30):.0f} GiB unified",
        f"gpu       {result['metal_device']}",
        "",
        "| ceiling | measured | spec | measured / spec |",
        "|---|---|---|---|",
    ]
    if "spec" in result:
        s = result["spec"]
        lines += [
            f"| memory bandwidth | {m['peak_bandwidth_gbs']} GB/s |"
            f" {s['bandwidth_gbs']} GB/s | {s['measured_fraction_bandwidth']:.0%} |",
            f"| fp32 FMA | {m['peak_fp32_gflops'] / 1000:.2f} TFLOP/s |"
            f" {s['fp32_gflops'] / 1000:.2f} TFLOP/s | {s['measured_fraction_fp32']:.0%} |",
            f"| fp16 FMA | {m['peak_fp16_gflops'] / 1000:.2f} TFLOP/s | - | - |",
        ]
    lines += [
        "",
        "| probe | GB/s |",
        "|---|---|",
    ]
    for k, v in p["bandwidth_gbs"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "| probe | GFLOP/s |", "|---|---|"]
    for k, v in p["gflops"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "| cpu, same memory system | value |",
        "|---|---|",
        f"| triad, {p['cpu']['p_cores']} p-cores | {p['cpu']['triad_gbs_p_cores']} GB/s |",
        f"| triad, {p['cpu']['all_cores']} cores | {p['cpu']['triad_gbs_all_cores']} GB/s |",
        f"| accelerate sgemm fp32 | {p['cpu']['accelerate_sgemm_gflops']} GFLOP/s |",
        "",
        f"ridge point  gpu fp32 {m['ridge_fp32_flop_per_byte']} flop/byte,"
        f"  gpu fp16 {m['ridge_fp16_flop_per_byte']} flop/byte,"
        f"  cpu {m['cpu_ridge_flop_per_byte']} flop/byte",
        "a kernel below its ridge point is bandwidth-bound and cannot be fixed by",
        "cheaper arithmetic alone.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="reprint the stored result")
    args = ap.parse_args()

    if args.show:
        if not OUT.exists():
            print(f"no stored result at {OUT}", file=sys.stderr)
            return 1
        print(render(json.loads(OUT.read_text())))
        return 0

    if platform.system() != "Darwin":
        print("this probe targets the Metal backend on macOS", file=sys.stderr)
        return 1

    result = measure()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(render(result))
    print(f"\nwritten to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
