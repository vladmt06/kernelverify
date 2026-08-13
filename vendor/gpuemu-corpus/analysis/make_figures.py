#!/usr/bin/env python3
"""Generate the four paper figures from accumulated B2 run records.

Outputs go to papers/figures/ as PDF (LaTeX-friendly). Each figure pulls
from a designated run id, with sensible defaults but overridable on the CLI.

Usage:
    python3 analysis/make_figures.py [--p1 RID] [--p3 RID] [--p4 RID]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "papers" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Defaults — most recent in the bucket as of paper draft.
P1_DEFAULT = "run-20260611-095210-889b18"
P3_DEFAULT = "run-20260611-095211-0ed227"
P4_DEFAULT = "run-20260611-085027-0b8ddc"


def _load_rows(run_id: str) -> list[dict]:
    p = DATA / run_id / "results.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def fig_p1(run_id: str) -> Path:
    """Bar chart: fail-rate per kernel. Controls green, illusions red."""
    rows = _load_rows(run_id)
    summary = json.loads((DATA / run_id / "summary.json").read_text())
    kernels = sorted(summary["kernels"], key=lambda k: k["kernel"])
    names = [k["kernel"] for k in kernels]
    rates = [k["failed"] / max(1, k["total"]) for k in kernels]
    is_illusion = [k["benchmark_verdict"] == "pass" and k["failed"] > 0 for k in kernels]
    colors = ["tab:red" if x else "tab:green" for x in is_illusion]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    y = np.arange(len(names))
    ax.barh(y, rates, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("fail rate (gpuemu)")
    ax.set_title(f"P1: kernel-level verdict on the {len(names)}-op corpus (run {run_id[-8:]})")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="tab:green", edgecolor="black", label="control (correct kernel)"),
        Patch(facecolor="tab:red", edgecolor="black",
              label=f"correctness illusion (benchmark=pass, gpuemu=fail)")
    ], loc="lower right", fontsize=9)
    fig.tight_layout()
    out = FIG / "p1_verdicts.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_p2() -> Path:
    """Tightening factor (current / calibrated atol) per (op, dtype), log-scale."""
    # Reuse calibration data via the analysis script: easiest is to re-run its
    # core. Inline a minimal version here to avoid the markdown coupling.
    runs = sorted([p for p in DATA.glob("run-*/results.jsonl")])
    rows = []
    for r in runs:
        for ln in r.read_text().splitlines():
            if not ln.strip():
                continue
            j = json.loads(ln)
            es = j.get("error_stats")
            if j.get("failure_kind") == "exception" or es is None:
                continue
            ma = es.get("max_abs")
            if ma is None:
                continue
            rows.append({
                "kernel": j["kernel"], "dtype": j["dtype"],
                "passed": bool(j["passed"]), "max_abs": float(ma),
            })
    # current tol from corpus
    cur = {}
    for m in (ROOT / "corpus").glob("*/meta.json"):
        d = json.loads(m.read_text())
        for dt, t in (d.get("tolerances") or {}).items():
            cur[(d["name"], dt)] = float(t)

    by = defaultdict(list)
    for r in rows:
        by[(r["kernel"], r["dtype"])].append(r)
    pairs, ratios = [], []
    for (k, dt), rs in sorted(by.items()):
        if "buggy" in k:
            continue
        passing = [r["max_abs"] for r in rs if r["passed"]]
        if not passing or (k, dt) not in cur:
            continue
        p95 = sorted(passing)[max(0, int(0.95 * (len(passing) - 1)))]
        proposed = p95 * 1.5
        if proposed <= 0:
            continue
        ratio = cur[(k, dt)] / proposed
        pairs.append(f"{k} ({dt})")
        ratios.append(max(ratio, 0.1))  # log-safe

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    y = np.arange(len(pairs))
    bars = ax.barh(y, ratios, color="tab:blue", edgecolor="black", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(pairs, fontsize=8)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, label="no change")
    ax.set_xlabel(r"current atol / calibrated atol  (log scale; >1 means the hand-picked tol was looser)")
    ax.set_title("P2: hand-picked atol vs p95(controls) × 1.5 calibration")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.5, which="both")
    fig.tight_layout()
    out = FIG / "p2_tightening.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_p3(run_id: str) -> Path:
    """Heatmap of bug recall: strategy × buggy kernel."""
    rows = _load_rows(run_id)
    summary = json.loads((DATA / run_id / "summary.json").read_text())
    strategies = summary["strategies"]
    buggy = [k for k in summary["kernels"] if "buggy" in k]
    if not buggy:
        raise RuntimeError("no buggy kernels in this P3 run")
    grid = np.full((len(strategies), len(buggy)), np.nan)
    for i, s in enumerate(strategies):
        for j, k in enumerate(buggy):
            rs = [r for r in rows if r.get("strategy") == s and r.get("kernel") == k]
            if not rs:
                continue
            grid[i, j] = sum(1 for r in rs if not r.get("passed")) / len(rs) * 100.0

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(buggy) + 4), 0.45 * len(strategies) + 2.5))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(buggy)))
    ax.set_xticklabels(buggy, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=9)
    for i in range(len(strategies)):
        for j in range(len(buggy)):
            v = grid[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        fontsize=8, color=("white" if v < 40 else "black"))
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("bug recall (%)", fontsize=9)
    ax.set_title("P3: strategy × LLM-style buggy kernel — recall (%)")
    fig.tight_layout()
    out = FIG / "p3_strategy_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_p4(run_id: str) -> Path:
    """Scatter: Δinstrs vs Δperf% across correct→buggy pairs."""
    rows = _load_rows(run_id)
    by_name = {r["kernel"]: r for r in rows}

    def first(r):
        ptxs = r.get("ptx") or []
        perf = r.get("perf") or {}
        if not ptxs or not perf.get("available"):
            return None
        return ptxs[0]["instruction_count"], perf["ms_median"]

    pairs = []
    for n in by_name:
        if not n.endswith("_buggy"):
            continue
        correct = n[:-len("_buggy")]
        if correct not in by_name:
            continue
        c = first(by_name[correct]); b = first(by_name[n])
        if c is None or b is None:
            continue
        di = b[0] - c[0]
        dp = (b[1] - c[1]) / c[1] * 100.0
        pairs.append((n, di, dp))

    if not pairs:
        raise RuntimeError("no pairs with both static + perf data in P4 run")

    names = [p[0] for p in pairs]
    di = np.array([p[1] for p in pairs])
    dp = np.array([p[2] for p in pairs])

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.scatter(di, dp, s=80, c="tab:orange", edgecolors="black")
    for n, x, y in zip(names, di, dp):
        ax.annotate(n.replace("_triton_buggy", ""), (x, y),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Δinstrs (buggy − correct)")
    ax.set_ylabel("Δperf% (buggy − correct)")
    ax.set_title("P4: structural change vs measured perf delta (one point per kernel pair)")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = FIG / "p4_structural_vs_perf.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_p4_cross_gpu(gpu_runs: dict) -> Path:
    """Heatmap: GPU × (correct→buggy) pair. Cell = Δperf% (buggy − correct).

    Modeled on ``fig_p1_cross_gpu``. Structural pairs (gelu/silu/etc. that drop
    work) should show consistent negative Δperf% across all GPUs; semantic pairs
    (matmul/attention/softmax with identical PTX) should sit near 0 on every GPU.
    """
    gpus = list(gpu_runs.keys())
    # Collect all kernel rows from every GPU run and compute the (correct,buggy)
    # pairs that have both static + perf data on at least one GPU.
    per_gpu_rows: dict[str, dict[str, dict]] = {}
    for g, rid in gpu_runs.items():
        rows = _load_rows(rid)
        per_gpu_rows[g] = {r["kernel"]: r for r in rows}

    # Discover pair set: union of *_buggy kernels with a non-buggy sibling
    # appearing on at least one GPU.
    pair_names: list[str] = []
    seen: set[str] = set()
    for g in gpus:
        for n in per_gpu_rows[g]:
            if not n.endswith("_buggy"):
                continue
            correct = n[:-len("_buggy")]
            if correct in per_gpu_rows[g] and n not in seen:
                pair_names.append(n)
                seen.add(n)
    pair_names.sort()
    if not pair_names:
        raise RuntimeError("no correct/buggy pairs found across the P4 runs")

    def _perf(rec):
        ptxs = rec.get("ptx") or []
        perf = rec.get("perf") or {}
        if not ptxs or not perf.get("available"):
            return None
        return perf["ms_median"]

    arr = np.full((len(gpus), len(pair_names)), np.nan)
    for i, g in enumerate(gpus):
        by_name = per_gpu_rows[g]
        for j, buggy in enumerate(pair_names):
            correct = buggy[:-len("_buggy")]
            if correct not in by_name or buggy not in by_name:
                continue
            c_ms = _perf(by_name[correct])
            b_ms = _perf(by_name[buggy])
            if c_ms is None or b_ms is None or c_ms <= 0:
                continue
            arr[i, j] = (b_ms - c_ms) / c_ms * 100.0

    # Diverging colormap centred at 0: negative (faster — usually a structural
    # bug that drops work) on one side; positive (slower) on the other.
    vmax = float(np.nanmax(np.abs(arr))) if np.any(~np.isnan(arr)) else 50.0
    vmax = max(vmax, 5.0)
    short_pairs = [p.replace("_triton_buggy", "").replace("_buggy", "") for p in pair_names]
    # Compact aspect (closer to 1.6:1 than 2:1) so the figure renders large
    # when LaTeX scales it to textwidth (~120 mm). Larger fonts so cell
    # labels stay readable at that scale.
    fig, ax = plt.subplots(
        figsize=(max(7.5, 0.7 * len(short_pairs) + 2.0),
                 0.75 * len(gpus) + 2.0),
        constrained_layout=True,
    )
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(short_pairs)))
    ax.set_xticklabels(short_pairs, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(gpus)))
    ax.set_yticklabels(gpus, fontsize=11)
    for i in range(len(gpus)):
        for j in range(len(pair_names)):
            v = arr[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                        fontsize=9, color=("black" if abs(v) < 0.5 * vmax else "white"))
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Δperf% (buggy − correct)")
    ax.set_title("P4: cross-architecture Δperf% per correct→buggy pair",
                 fontsize=11)
    out = FIG / "p4_cross_gpu.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_p1_cross_gpu(gpu_runs: dict) -> Path:
    """Heatmap: GPU × kernel, split into two stacked panels so the figure
    fits inside an LNCS textwidth at a readable cell font.

    With 26 kernels and 5 GPUs the single-panel layout (used previously)
    becomes 17 inches wide. When LaTeX scales that to textwidth (~5 in)
    the per-cell labels render at ~2pt and are unreadable. Splitting the
    kernel axis in half gives two panels of 13 kernels each, stacked
    vertically. The figure ends up ~8x6 inches; scaled to textwidth the
    cells render at ~7pt and the labels stay legible without spilling
    onto its own landscape page.
    """
    gpus = list(gpu_runs.keys())
    kernels: list = []
    grids: list[list] = []
    for g, rid in gpu_runs.items():
        s = json.loads((DATA / rid / "summary.json").read_text())
        row = {k["kernel"]: (k["failed"] / max(1, k["total"]) * 100.0) for k in s["kernels"]}
        if not kernels:
            kernels = sorted(row)
        grids.append([row.get(k, np.nan) for k in kernels])
    arr = np.array(grids)

    # Split kernels into two contiguous halves so adjacent kernels stay
    # near each other on the same panel.
    half = (len(kernels) + 1) // 2
    splits = [
        (kernels[:half], arr[:, :half]),
        (kernels[half:], arr[:, half:]),
    ]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(8, 0.55 * half + 3.5), 1.0 * len(gpus) + 2.5),
        constrained_layout=True,
    )
    im = None
    for ax, (panel_kernels, panel_arr) in zip(axes, splits):
        im = ax.imshow(panel_arr, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
        ax.set_xticks(range(len(panel_kernels)))
        ax.set_xticklabels(panel_kernels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(gpus)))
        ax.set_yticklabels(gpus, fontsize=10)
        for i in range(len(gpus)):
            for j in range(len(panel_kernels)):
                v = panel_arr[i, j]
                if not math.isnan(v):
                    ax.text(j, i, f"{int(v)}", ha="center", va="center",
                            fontsize=8, color=("white" if v >= 50 else "black"))

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label="fail rate (%)  green = correct, red = illusion")
    fig.suptitle("P1: cross-GPU verdict consistency on the 26-op corpus",
                 fontsize=11)
    out = FIG / "p1_cross_gpu.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", default=P1_DEFAULT)
    ap.add_argument("--p3", default=P3_DEFAULT)
    ap.add_argument("--p4", default=P4_DEFAULT)
    ap.add_argument("--gpu-runs", default="",
                    help="comma-separated GPU=run_id pairs for P1 cross-GPU figure")
    ap.add_argument("--p4-gpu-runs", default="",
                    help="comma-separated GPU=run_id pairs for P4 cross-GPU figure")
    args = ap.parse_args()
    made = []
    for label, fn in [("p1", lambda: fig_p1(args.p1)),
                      ("p2", fig_p2),
                      ("p3", lambda: fig_p3(args.p3)),
                      ("p4", lambda: fig_p4(args.p4))]:
        try:
            p = fn()
            print(f"  {label}: {p.relative_to(ROOT)}")
            made.append(p)
        except Exception as e:
            print(f"  {label}: SKIP ({e})")
    if args.gpu_runs:
        pairs = dict(p.split("=") for p in args.gpu_runs.split(",") if "=" in p)
        try:
            p = fig_p1_cross_gpu(pairs)
            print(f"  p1_cross_gpu: {p.relative_to(ROOT)}")
            made.append(p)
        except Exception as e:
            print(f"  p1_cross_gpu: SKIP ({e})")
    if args.p4_gpu_runs:
        pairs = dict(p.split("=") for p in args.p4_gpu_runs.split(",") if "=" in p)
        try:
            p = fig_p4_cross_gpu(pairs)
            print(f"  p4_cross_gpu: {p.relative_to(ROOT)}")
            made.append(p)
        except Exception as e:
            print(f"  p4_cross_gpu: SKIP ({e})")
    print(f"made {len(made)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
