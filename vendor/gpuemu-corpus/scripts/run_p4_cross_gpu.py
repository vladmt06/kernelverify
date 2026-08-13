#!/usr/bin/env python3
"""Orchestrate the P4 cross-GPU sweep: provision → run p4_artifacts → collect → destroy
for each of the five GPU classes used in P1's cross-GPU sweep (rtx3060, a10, l40s,
a100_sxm4, h100_nvl), and then emit the make_figures/p4_correlation commands wired to
the resulting run-ids.

Runs are launched **sequentially** (not in parallel) to keep vast.ai
provision-attempts orderly: the harness's label-strict reaper still cleans up any
stuck instances on next invocation. Each call already does always-destroy on exit, so
a single crash leaks at most one instance and the next ``reap`` clears it.

Requires env: VASTAI_API_KEY, S3_ENDPOINT, S3_BUCKET, S3_KEY_PREFIX,
              AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.

Usage: python3 scripts/run_p4_cross_gpu.py [--iters 50] [--gpus rtx3060,a10,...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

# vast.ai short name → label we want as the column header in the cross-GPU figure.
DEFAULT_GPUS = {
    "rtx3060": "RTX_3060",
    "a10": "A10",
    "l40s": "L40S",
    "a100_sxm4": "A100_SXM4",
    "h100_nvl": "H100_NVL",
}


def _run_one(gpu_short: str, iters: int) -> str:
    """Provision the GPU, run p4_artifacts, return the run_id."""
    cmd = [
        sys.executable, "-m", "gpuemu_lab.cli", "run",
        "--driver", "p4_artifacts",
        "--paper", "p4",
        "--gpu", gpu_short,
        "--iters", str(iters),
    ]
    print(f"\n[p4_cross_gpu] launching {gpu_short}: {' '.join(cmd)}", flush=True)
    out = subprocess.check_output(cmd, cwd=str(ROOT / "harness"), text=True)
    print(out)
    return json.loads(out)["run_id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--gpus", default=",".join(DEFAULT_GPUS),
                    help="comma-separated vast.ai short names (default: all 5)")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    results: dict[str, str] = {}
    failed: list[str] = []
    for g in gpus:
        try:
            rid = _run_one(g, args.iters)
            label = DEFAULT_GPUS.get(g, g.upper())
            results[label] = rid
        except subprocess.CalledProcessError as e:
            print(f"[p4_cross_gpu] {g}: FAILED ({e})", file=sys.stderr)
            failed.append(g)

    print("\n[p4_cross_gpu] sweep complete")
    print(f"  succeeded: {len(results)} / {len(gpus)}")
    print(f"  failed:    {failed or 'none'}")
    if not results:
        return 1

    pairs = ",".join(f"{label}={rid}" for label, rid in results.items())
    print()
    print("Run the analysis with:")
    print(f"  python3 analysis/make_figures.py --p4-gpu-runs '{pairs}'")
    print(f"  python3 analysis/p4_correlation.py --gpu-runs '{pairs}'")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
