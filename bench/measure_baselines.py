#!/usr/bin/env python3
"""One command that measures this machine's baselines and appends them as JSONL.

Replaces hand-run numbers. Everything the per-chip matrix consumes is produced
here, and every row carries what is needed to decide whether to believe it.

Five rules are built in rather than remembered:

  interleaved     the arms of each workload cell alternate back to back, one
                  cell at a time, because GPU timings drift with power state
                  and two arms sampled minutes apart compare the clock, not
                  the stacks. A sampling group IS one A/B session.
  timing floor    a sample under a millisecond is reporting the power manager,
                  so it is recorded and marked non-binding (bench/machine_state.py).
  idle gate       load, power source, low power mode and thermal state are
                  sampled before and after; a row measured on a busy or
                  unplugged machine is kept, labelled, and never binds.
  dispersion      repeats that disagree by more than the limit do not bind,
                  because a one-minute load average cannot see a transient
                  that lands inside one sample and an idle machine is
                  therefore not evidence that the measurement succeeded.
  modelled bytes  utilisation is computed from the model's own tensor table,
                  never from file size, on both stacks.

    .venv/bin/python bench/measure_baselines.py              # everything
    .venv/bin/python bench/measure_baselines.py --quick      # fewer rounds
    .venv/bin/python bench/measure_baselines.py --only mlx   # one stack

Appends to bench/.baselines/<date>.jsonl. Schema is documented in
bench/.baselines/SCHEMA.md, which matrix.py consumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gguf_info
import machine_state
import mlx_info
import roofline

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "bench" / ".baselines"
HASH_CACHE = ROOT / "bench" / ".cache" / "model_hashes.json"

LLAMA_CPP = Path("/Users/vlad/llama.cpp")
LLAMA_BENCH = LLAMA_CPP / "build" / "bin" / "llama-bench"
GGUF_DIR = Path("/Users/vlad/models/gguf")

SCHEMA_VERSION = 3
PROVENANCE_TIER = "owner-run"

# The model both stacks run, so the two are comparable at all. logical_name
# is the cross-stack identity of what ran: one canonical spelling, because a
# variant would silently split a matrix cell in two.
GGUF_MODEL = "Qwen3-4B-Q4_K_M.gguf"
MLX_REPO = "mlx-community/Qwen3-4B-4bit"
LOGICAL_MODEL = "qwen3-4b"

# The matmul's n dimension, reached by prompt width on llama.cpp and by batch
# size on MLX. n=8 and n=16 are first-class rows because that is where both
# stacks collapse, not an appendix.
BATCH_POINTS = [1, 8, 16, 512]
MLX_BATCH_POINTS = [1, 8, 16]

# Serving decode: n_parallel independent streams decoded together. mlx-lm only
# this block (D6): llama-bench has no parallel mode, and the honest llama.cpp
# path (llama-batched-bench plus a parser) is deferred - TODOS.md carries the
# scoping. These cells are labeled mlx-only and grouped one arm per cell, so
# no cross-stack serving comparison can form from them.
MLX_PARALLEL_POINTS = [4, 8, 16]

PREFILL_TOKENS = 1024
DECODE_TOKENS = 128


# The consumer contract. matrix.py reads these rows, so a field going missing
# has to fail here, in the producer, rather than silently blanking a published
# column.
REQUIRED_TOP = (
    "schema_version", "run_id", "row_id", "measured_at", "provenance_tier",
    "machine", "idle_before", "idle_after", "stack", "measurement", "result",
    "binding", "binding_blockers",
)
REQUIRED_RESULT = ("metric", "median", "reps", "samples", "min_sample_ms",
                   "below_timing_floor")
REQUIRED_SAMPLING = ("interleaved", "group", "rounds")
PROVENANCE_TIERS = ("owner-run", "rental-run", "community-unattested")
BINDING_RESOURCES = ("compute", "memory", "unknown")
LOGICAL_MODEL_NAMES = ("qwen3-4b",)


def validate_row(row: dict) -> dict:
    missing = [k for k in REQUIRED_TOP if k not in row]
    if missing:
        raise ValueError(f"row {row.get('row_id')} missing {missing}")
    missing = [k for k in REQUIRED_RESULT if k not in row["result"]]
    if missing:
        raise ValueError(f"row {row.get('row_id')} result missing {missing}")
    if row["provenance_tier"] not in PROVENANCE_TIERS:
        raise ValueError(f"unknown provenance tier {row['provenance_tier']}")
    if row["binding"] and row["binding_blockers"]:
        raise ValueError("a row cannot bind and carry blockers")
    if row["measurement"]["kind"] != "ceiling":
        if "roofline" not in row:
            raise ValueError(f"row {row['row_id']} has no roofline placement")
        # Comparability is a separate property from bindingness, and it has its
        # own falsification: the kernels lane ran an A/B whose dispatches were
        # all past 5 ms and still got the sign wrong, because the two arms ran
        # in separate passes. Batching is not enough; the arms have to be
        # interleaved. A row that cannot say how it was sampled must not be
        # compared against another row.
        missing = [k for k in REQUIRED_SAMPLING if k not in row.get("sampling", {})]
        if missing:
            raise ValueError(f"row {row['row_id']} sampling missing {missing}")
        if row["schema_version"] >= 3:
            resource = (row.get("roofline") or {}).get("binding_resource")
            if resource not in BINDING_RESOURCES:
                raise ValueError(
                    f"row {row['row_id']} binding_resource {resource!r} is not "
                    f"one of {BINDING_RESOURCES}")
            logical = (row.get("model") or {}).get("logical_name")
            if logical not in LOGICAL_MODEL_NAMES:
                raise ValueError(
                    f"row {row['row_id']} model.logical_name {logical!r} is not "
                    f"one of {LOGICAL_MODEL_NAMES}; the cross-stack cell "
                    "identity needs one canonical spelling")
            if row["measurement"]["kind"] == "batch_decode":
                # D6: the llama.cpp serving arm is deferred to its own block,
                # so a serving row must both name its scope and stay on the
                # stack that can be measured honestly today. An unlabeled
                # serving aggregate is exactly the number a reader would set
                # beside the other stack's single-stream decode.
                if row["measurement"].get("stack_scope") != "mlx-only":
                    raise ValueError(
                        f"row {row['row_id']} is a batch_decode row without its "
                        "mlx-only stack_scope; the serving cells have no "
                        "cross-stack counterpart this block")
                if row["stack"]["name"] != "mlx-lm":
                    raise ValueError(
                        f"row {row['row_id']} claims a batch_decode cell on "
                        f"{row['stack']['name']}; that baseline is deferred "
                        "(D6, TODOS.md) and must not be written by this producer")
                n_parallel = row["measurement"].get("n_parallel")
                if not isinstance(n_parallel, int) or n_parallel < 2:
                    raise ValueError(
                        f"row {row['row_id']} batch_decode needs an integer "
                        f"n_parallel of at least 2, got {n_parallel!r}")
    return row


def reproduces(recorded: dict, rerun: dict) -> tuple[bool, str]:
    """The D3.7 regression rule: does a rerun row reproduce a recorded cell?

    Identity first, because equal medians on different workloads prove
    nothing. Then the producer's own repeat-disagreement limit is the bar
    between runs as well: a drift the dispersion gate would refuse inside one
    run cannot be waved through because a day passed. New cells only add;
    a row with a different workload identity is not a rerun of anything.
    """
    if (recorded["measurement"] != rerun["measurement"]
            or recorded["stack"]["name"] != rerun["stack"]["name"]):
        return False, "not a rerun of this cell: different workload or stack"
    old, new = recorded["result"]["median"], rerun["result"]["median"]
    drift = abs(new - old) / old * 100
    if drift > machine_state.MAX_SPREAD_PCT:
        return False, (f"median moved {drift:.1f}% ({old} to {new}), over the "
                       f"{machine_state.MAX_SPREAD_PCT}% repeat-disagreement limit")
    return True, ""


# --------------------------------------------------------------------------
# provenance


def model_hash(path: Path) -> str:
    """sha256 of a file or of a directory's safetensors, cached by size+mtime."""
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    stat_key = json.dumps(
        [[f.name, f.stat().st_size, int(f.stat().st_mtime)] for f in files],
        sort_keys=True,
    )
    key = hashlib.sha256(f"{path}|{stat_key}".encode()).hexdigest()

    HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(HASH_CACHE.read_text()) if HASH_CACHE.exists() else {}
    if key in cache:
        return cache[key]

    digest = hashlib.sha256()
    for f in files:
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                digest.update(chunk)
    cache[key] = digest.hexdigest()[:32]
    HASH_CACHE.write_text(json.dumps(cache, indent=1))
    return cache[key]


def llama_cpp_build() -> dict:
    commit = subprocess.run(
        ["git", "-C", str(LLAMA_CPP), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "name": "llama.cpp",
        "version": commit,
        "path": str(LLAMA_CPP),
        "build_flags": [
            "-DCMAKE_BUILD_TYPE=Release", "-DGGML_METAL=ON",
            "-DGGML_METAL_EMBED_LIBRARY=ON",
        ],
    }


def mlx_build() -> dict:
    import importlib.metadata as md

    return {
        "name": "mlx-lm",
        "version": md.version("mlx-lm"),
        "mlx_version": md.version("mlx"),
        "path": None,
        "build_flags": [],
    }


# --------------------------------------------------------------------------
# single measurements, each returning one sample


def run_llama_sample(model: Path, n_prompt: int, n_gen: int, depth: int = 0) -> dict:
    cmd = [
        str(LLAMA_BENCH), "-m", str(model), "-r", "1", "-o", "json",
        "-p", str(n_prompt), "-n", str(n_gen), "-d", str(depth),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed: {proc.stderr[-1200:]}")
    rows = [r for r in json.loads(proc.stdout) if (r["n_gen"] or r["n_prompt"])]
    row = rows[0]
    return {
        "tokens_per_s": row["avg_ts"],
        "sample_ms": min(row["samples_ns"]) / 1e6,
        "raw": {k: row[k] for k in ("n_prompt", "n_gen", "model_size", "model_n_params")},
    }


MLX_TRIAL = re.compile(
    r"prompt_tps=(?P<ptps>[\d.]+), generation_tps=(?P<gtps>[\d.]+),"
    r" peak_memory=(?P<mem>[\d.]+), total_time=(?P<total>[\d.]+)"
)


def run_mlx_sample(repo: str, n_prompt: int, n_gen: int, batch: int) -> dict:
    cmd = [
        sys.executable, "-m", "mlx_lm.benchmark", "--model", repo,
        "-p", str(n_prompt), "-g", str(n_gen), "-b", str(batch), "-n", "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"mlx_lm.benchmark failed: {proc.stderr[-1200:]}")
    m = None
    for line in proc.stdout.splitlines():
        found = MLX_TRIAL.search(line)
        if found and line.strip().startswith("Trial"):
            m = found
    if not m:
        raise RuntimeError(f"could not parse mlx output:\n{proc.stdout[-1200:]}")
    return {
        "prompt_tps": float(m["ptps"]),
        "generation_tps": float(m["gtps"]),
        "peak_memory_gb": float(m["mem"]),
        "sample_ms": float(m["total"]) * 1000.0,
    }


# --------------------------------------------------------------------------
# spec construction


def build_specs(only: str | None) -> list[dict]:
    specs = []
    gguf_path = GGUF_DIR / GGUF_MODEL
    mlx_path = mlx_info.resolve_hf_model(MLX_REPO)

    if only in (None, "llama.cpp"):
        # Decode at the same cache depth MLX carries, or the cross-stack row
        # compares a cold cache against a 1024-token one and the KV traffic
        # difference is charged to the stacks.
        specs.append({
            "id": "llamacpp/decode", "stack": "llama.cpp", "model_path": gguf_path,
            "kind": "decode", "n_prompt": 0, "n_gen": DECODE_TOKENS, "batch": 1,
            "depth": PREFILL_TOKENS,
        })
        specs.append({
            "id": "llamacpp/prefill", "stack": "llama.cpp", "model_path": gguf_path,
            "kind": "prefill", "n_prompt": PREFILL_TOKENS, "n_gen": 0, "batch": PREFILL_TOKENS,
        })
        for n in BATCH_POINTS:
            specs.append({
                "id": f"llamacpp/width{n}", "stack": "llama.cpp", "model_path": gguf_path,
                "kind": "matmul_width", "n_prompt": n, "n_gen": 0, "batch": n,
            })

    if only in (None, "mlx"):
        specs.append({
            "id": "mlx/decode", "stack": "mlx-lm", "model_path": mlx_path,
            "kind": "decode", "n_prompt": PREFILL_TOKENS, "n_gen": DECODE_TOKENS, "batch": 1,
        })
        for b in MLX_BATCH_POINTS:
            # A short prompt here on purpose: these rows are about the decode
            # matmul's width, and a 1024-token prefill per stream would bury
            # that in prefill time.
            #
            # 64 generated tokens rather than 16: at 16 the timed region was
            # short enough that samples came back in three separate clusters
            # (15, 93 and 188 tok/s for the same spec), which is startup and
            # memory-pressure noise rather than kernel throughput. A longer
            # timed region dilutes it.
            specs.append({
                "id": f"mlx/width{b}", "stack": "mlx-lm", "model_path": mlx_path,
                "kind": "matmul_width", "n_prompt": 32, "n_gen": 64, "batch": b,
            })
        for n in MLX_PARALLEL_POINTS:
            # The per-stream shape mirrors mlx/decode (1024-token cache, 128
            # generated), so the B=1 anchor of the n_parallel series is the
            # decode cell already in the record rather than a new shape.
            specs.append({
                "id": f"mlx/batch_decode{n}", "stack": "mlx-lm", "model_path": mlx_path,
                "kind": "batch_decode", "n_prompt": PREFILL_TOKENS,
                "n_gen": DECODE_TOKENS, "batch": n,
            })
    return specs


def measurement_fields(spec: dict) -> dict:
    """The measurement object this spec writes: the row's cell identity.

    Shared with the regression tests, because the D3.7 rule that a rerun must
    reproduce a recorded cell is only checkable while the producer still
    measures the same workload under the same spec id.
    """
    fields = {
        "kind": spec["kind"],
        "matmul_width": spec["batch"],
        "n_prompt": spec["n_prompt"],
        "n_gen": spec["n_gen"],
        "cache_depth": spec.get("depth", 0) + spec["n_prompt"],
        "width_mechanism": "prompt-width" if spec["stack"] == "llama.cpp"
                           else "batch-size",
    }
    if spec["kind"] == "batch_decode":
        fields |= {"n_parallel": spec["batch"], "stack_scope": "mlx-only"}
    return fields


def measure_once(spec: dict) -> dict:
    if spec["stack"] == "llama.cpp":
        return run_llama_sample(
            spec["model_path"], spec["n_prompt"], spec["n_gen"], spec.get("depth", 0)
        )
    return run_mlx_sample(MLX_REPO, spec["n_prompt"], spec["n_gen"], spec["batch"])


def sample_value(spec: dict, sample: dict) -> float:
    """The one number this spec is about, as an aggregate token rate.

    mlx_lm.benchmark's generation_tps is already aggregated across the batch,
    not per stream: at batch 8 it reports 16.07 for 128 tokens in 7.96 s.
    Verified before trusting it, because assuming per-stream would have
    multiplied the number by the batch size and turned a 3x collapse into a
    3x speedup.
    """
    if spec["stack"] == "llama.cpp":
        return sample["tokens_per_s"]
    if spec["kind"] == "prefill":
        return sample["prompt_tps"]
    return sample["generation_tps"]


# --------------------------------------------------------------------------
# the measurement loop: one workload cell at a time, arms alternating


def workload_cell(spec: dict) -> tuple:
    """WHAT this spec measures, mirroring the renderer's cell key.

    The specs sharing this key are the arms of one A/B, and they are the only
    rows that will ever share a sampling group, so a group is one real A/B
    session by construction rather than a relabel of a run-wide rotation.
    """
    return (spec["kind"], spec["batch"],
            spec["n_prompt"] if spec["kind"] == "prefill" else None)


def cell_label(key: tuple) -> str:
    kind, width, _ = key
    return f"{kind}-w{width}"


def group_cells(specs: list[dict]) -> list[tuple[str, list[dict]]]:
    """The specs grouped into workload cells, in spec order."""
    cells: dict[tuple, list[dict]] = {}
    for spec in specs:
        cells.setdefault(workload_cell(spec), []).append(spec)
    return [(cell_label(key), members) for key, members in cells.items()]


def measure_cells(specs: list[dict], rounds: int,
                  measure=measure_once) -> tuple[dict, list[str]]:
    """All samples, one workload cell run to completion at a time.

    Within a cell the arms run back to back in a fixed order every round, the
    same discipline as runners/compare.py: consecutive samples alternate arms,
    so a clock excursion lands on both arms of a pair instead of on whichever
    spec a rotation happened to reach. The old loop rotated ALL specs
    round-robin, which put a full rotation - minutes - between the two arms
    of every comparison.
    """
    samples: dict[str, list] = {s["id"]: [] for s in specs}
    sequence: list[str] = []
    for label, members in group_cells(specs):
        for r in range(rounds):
            for spec in members:
                sample = measure(spec)
                value = sample_value(spec, sample)
                samples[spec["id"]].append({"value": value, "sample": sample})
                sequence.append(spec["id"])
                print(f"  [{label}] round {r + 1} {spec['id']:22} {value:10.2f}",
                      flush=True)
    return samples, sequence


# --------------------------------------------------------------------------
# utilisation


def utilisation(spec: dict, value: float, ceilings: dict) -> dict:
    """Bytes and flops per pass from the model's tensor table, never file size."""
    read_gbs = ceilings["read_gbs"]
    flops_ceiling = ceilings["fp16_gflops"]

    if spec["stack"] == "llama.cpp":
        cost = gguf_info.cost_model(gguf_info.read(spec["model_path"]))
        gen_bytes, kv_bytes = gguf_info.gen_bytes, gguf_info.kv_bytes
    else:
        cost = mlx_info.cost_model(spec["model_path"])
        gen_bytes, kv_bytes = mlx_info.gen_bytes, mlx_info.kv_bytes

    if spec["kind"] == "decode":
        # Cache depth is part of the traffic, so it has to be part of the model:
        # llama.cpp reaches it with -d, MLX by prefilling.
        ctx = spec.get("depth", 0) + spec["n_prompt"] + spec["n_gen"] / 2
        bytes_per_pass = gen_bytes(cost) + kv_bytes(cost, ctx)
        passes_per_s = value
        denominator = "read"
    elif spec["kind"] == "batch_decode":
        # One forward pass advances every stream one token: the weights are
        # read once per pass, but each stream drags its own KV cache, so KV
        # is charged per stream at the average depth. Charging it once would
        # overstate utilisation by nearly the stream count at serving depth.
        streams = spec["batch"]
        ctx = spec["n_prompt"] + spec["n_gen"] / 2
        bytes_per_pass = gen_bytes(cost) + streams * kv_bytes(cost, ctx)
        passes_per_s = value / streams
        denominator = "read"
    else:
        width = max(spec["batch"], 1)
        ctx = spec["n_prompt"] if spec["kind"] == "prefill" else spec["n_prompt"]
        bytes_per_pass = gen_bytes(cost) + kv_bytes(cost, ctx)
        passes_per_s = value / width
        denominator = "read"

    achieved_gbs = bytes_per_pass * passes_per_s / 1e9
    out = {
        "bytes_per_pass": int(bytes_per_pass),
        "denominator_name": denominator,
        "denominator_gbs": read_gbs,
        "achieved_gbs": round(achieved_gbs, 1),
        "bandwidth_utilisation_pct": round(100 * achieved_gbs / read_gbs, 1),
        "byte_model": "tensor-table",
    }

    # Flops only where a parameter count exists; the MLX checkpoint gives bytes
    # but not an unambiguous parameter count, and guessing one would put a
    # fabricated number in a published column.
    if spec["stack"] == "llama.cpp":
        if spec["kind"] == "decode":
            flops = gguf_info.gen_flops(cost, ctx)
        else:
            flops = gguf_info.prompt_flops(cost, max(spec["batch"], 1))
        achieved_gflops = flops * passes_per_s / 1e9
        ai = flops / bytes_per_pass
        ridge = flops_ceiling / read_gbs
        out |= {
            "gflop_per_pass": round(flops / 1e9, 3),
            "arithmetic_intensity_flop_per_byte": round(ai, 2),
            "achieved_gflops": round(achieved_gflops, 1),
            "flops_ceiling_gflops": flops_ceiling,
            "roofline_ceiling_gflops": round(min(flops_ceiling, ai * read_gbs), 1),
            "roofline_utilisation_pct": round(
                100 * achieved_gflops / min(flops_ceiling, ai * read_gbs), 1
            ),
            # Which ceiling actually binds this row. A prefill row read as a
            # bandwidth percentage looks like a 1% failure when it is a 60%
            # compute result, so the consumer must be told which column to use.
            "binding_resource": "memory" if ai < ridge else "compute",
            "ridge_flop_per_byte": round(ridge, 1),
        }
    else:
        out |= {
            "achieved_gflops": None,
            "roofline_utilisation_pct": None,
            "binding_resource": "memory" if spec["kind"] != "prefill" else "unknown",
            "flops_model": "absent: the MLX checkpoint gives bytes but no "
                           "unambiguous parameter count",
        }
    return out


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["llama.cpp", "mlx"])
    ap.add_argument("--allow-busy", action="store_true",
                    help="measure anyway; rows are recorded as non-binding")
    args = ap.parse_args()
    rounds = 1 if args.quick else args.rounds

    fp = machine_state.fingerprint()
    before = machine_state.idle_check(fp["cores"])
    if not before["idle"] and not args.allow_busy:
        print("machine is not in a binding state:", file=sys.stderr)
        for b in before["blockers"]:
            print(f"  - {b}", file=sys.stderr)
        print("re-run when idle and on AC, or pass --allow-busy to record "
              "labelled non-binding rows", file=sys.stderr)
        return 2

    print("[roofline] measuring machine ceilings", flush=True)
    roof = roofline.measure()
    ceilings = {
        "read_gbs": roof["probe"]["bandwidth_gbs"]["read_shared"],
        "copy_gbs": roof["probe"]["bandwidth_gbs"]["copy_private"],
        "fp16_gflops": roof["measured"]["peak_fp16_gflops"],
        "fp32_gflops": roof["measured"]["peak_fp32_gflops"],
    }
    ceilings["read_gbs"] = max(ceilings["read_gbs"], roof["probe"]["bandwidth_gbs"]["read_private"])
    print(f"           read {ceilings['read_gbs']} GB/s, copy {ceilings['copy_gbs']} GB/s,"
          f" fp16 {ceilings['fp16_gflops']} GFLOP/s", flush=True)

    specs = build_specs(args.only)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"

    # One sampling group per workload cell, so a group is one A/B session and
    # can never be a relabel of an all-specs rotation.
    groups = {
        spec["id"]: {"group": f"{run_id}/{label}",
                     "group_members": [m["id"] for m in members]}
        for label, members in group_cells(specs) for spec in members
    }
    samples, _ = measure_cells(specs, rounds)

    after = machine_state.idle_check(fp["cores"])
    stacks = {"llama.cpp": llama_cpp_build(), "mlx-lm": mlx_build()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    rows = []

    # Ceiling rows first: every other row divides by them.
    for name, value, metric in [
        ("bandwidth_read", ceilings["read_gbs"], "gbs"),
        ("bandwidth_copy", ceilings["copy_gbs"], "gbs"),
        ("fp16_fma", ceilings["fp16_gflops"], "gflops"),
        ("fp32_fma", ceilings["fp32_gflops"], "gflops"),
    ]:
        timing = machine_state.timing_verdict(10.0)  # probe dispatches are ms-scale
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "row_id": f"{run_id}/ceiling/{name}",
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "provenance_tier": PROVENANCE_TIER,
            "machine": fp,
            "idle_before": before, "idle_after": after,
            "stack": {"name": "roofline-probe", "version": "bench/metal/roofline_probe.mm"},
            "model": None,
            "measurement": {"kind": "ceiling", "name": name},
            "result": {"metric": metric, "median": value, "spread_pct": None,
                       "reps": 1, "samples": [value], **timing},
            **machine_state.binding_verdict(before, after, timing),
        })

    for spec in specs:
        vals = [s["value"] for s in samples[spec["id"]]]
        sample_ms = min(s["sample"]["sample_ms"] for s in samples[spec["id"]])
        timing = machine_state.timing_verdict(sample_ms)
        median = statistics.median(vals)
        spread = round((max(vals) - min(vals)) / median * 100, 2) if median else None
        dispersion = machine_state.dispersion_verdict(spread)
        util = utilisation(spec, median, ceilings)
        model_path = spec["model_path"]
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "row_id": f"{run_id}/{spec['id']}",
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "provenance_tier": PROVENANCE_TIER,
            "machine": fp,
            "idle_before": before, "idle_after": after,
            "stack": stacks[spec["stack"]],
            "model": {
                "name": model_path.name,
                "logical_name": LOGICAL_MODEL,
                "path": str(model_path),
                "sha256_32": model_hash(model_path),
                "quant": "Q4_K_M" if spec["stack"] == "llama.cpp" else "mlx-affine-4bit",
            },
            "measurement": measurement_fields(spec),
            "result": {
                "metric": "tokens_per_s", "median": round(median, 2),
                "spread_pct": spread, "reps": len(vals),
                "samples": [round(v, 2) for v in vals], **timing, **dispersion,
            },
            "sampling": {
                "interleaved": True,
                "rotation": "arm-alternation",
                "rounds": rounds,
                **groups[spec["id"]],
            },
            "roofline": util,
            **machine_state.binding_verdict(before, after, timing, dispersion),
        })

    with out_path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(validate_row(row)) + "\n")

    print("\n" + render(rows))
    print(f"\n{len(rows)} rows appended to {out_path.relative_to(ROOT)}")

    # Exit non-zero when anything was rejected, so an automated runner notices
    # instead of publishing a table with quiet holes in it. The rows are still
    # written: a rejection is evidence, not an absence.
    rejected = [r for r in rows if not r["binding"]]
    if rejected:
        print(f"\n{len(rejected)} of {len(rows)} rows REJECTED, not binding:",
              file=sys.stderr)
        for r in rejected:
            print(f"  {r['row_id'].split('/', 1)[1]}: "
                  f"{'; '.join(r['binding_blockers'])}", file=sys.stderr)
        return 1
    return 0


def render(rows: list[dict]) -> str:
    out = [
        "| stack | measurement | width | tok/s | spread | GB/s | binds on | % of that ceiling | binding row |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["measurement"]["kind"] == "ceiling":
            continue
        m, res, roof = r["measurement"], r["result"], r["roofline"]
        resource = roof.get("binding_resource", "memory")
        if resource == "compute" and roof.get("roofline_utilisation_pct") is not None:
            pct = f"{roof['roofline_utilisation_pct']}%"
        else:
            pct = f"{roof['bandwidth_utilisation_pct']}%"
        # The scope travels with the kind here too: the operator reading a
        # run's summary is a reader like any other (SCHEMA.md renderer notes).
        kind = m["kind"]
        scope = m.get("stack_scope")
        if scope:
            kind = f"{kind} ({scope})"
        out.append(
            f"| {r['stack']['name']} | {kind} | {m['matmul_width']} |"
            f" {res['median']} | {res['spread_pct']}% | {roof['achieved_gbs']} |"
            f" {resource} | {pct} |"
            f" {'yes' if r['binding'] else 'NO'} |"
        )
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
