"""Phase 0: measure the escape rate of the gpuemu corpus bugs.

Why this exists
---------------
"The Correctness Illusion in LLM-Generated GPU Kernels" (arXiv 2606.20128)
reports that its seeded buggy kernels are certified correct by a standard
single-shape `allclose` oracle. In the released corpus, that claim is carried
by a field in each `meta.json`:

    "benchmark_verdict": "pass"

The field is hardcoded to "pass" for all 26 corpus entries, correct and buggy
alike, and nothing in the released code ever computes it. So the headline is
asserted, not measured. This script measures it.

What is measured
----------------
For each of the 10 buggy kernels, against the corpus's own fp64 reference
scripts and its own input distribution:

1. Does the bug survive a standard benchmark oracle at a single shape?
   Three oracles are reported: `allclose` at atol=rtol=1e-3, the same at 1e-2,
   and the corpus's own per-(op, dtype) absolute tolerance.

2. Across the operator's full declared schema, at what fraction of shapes does
   the bug stay hidden? This is the number that decides whether single-shape
   testing is lucky or doomed, and nobody has published it.

Controls
--------
The correct variant of every op is run through the same sweep. Any control
failure means the CPU port is wrong, and the run aborts rather than reporting
numbers built on a broken port.

Corpus pinned at commit 2f15310368118ddd8c76887675f8c6c0f76a0989,
vendored under vendor/gpuemu-corpus, dual-licensed MIT OR Apache-2.0.
"""

from __future__ import annotations

import base64
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_ports import BUGGY_TO_CONTROL, PORTS  # noqa: E402

CORPUS = Path(__file__).resolve().parents[1] / "vendor" / "gpuemu-corpus" / "gpuemu_corpus" / "data"

# KernelBench-style oracles. The corpus never states which tolerance the
# "benchmark verdict" refers to, so both common settings are reported.
BENCHMARK_TOLERANCES = (1e-3, 1e-2)

# KernelBench draws a handful of random inputs at one fixed shape.
BENCHMARK_DRAWS = 5

# The corpus fuzzer samples uniform [-10, 10]. That is an aggressive range, and
# it turns out to matter: for attention it saturates every softmax row to
# one-hot, and a saturated softmax is invariant to a positive rescale, which is
# exactly the bug being seeded. So both ranges are measured. Activations in a
# trained network sit far closer to unit scale than to +/-10.
DISTRIBUTIONS = {
    "corpus uniform[-10,10]": (-10.0, 10.0),
    "unit-scale uniform[-1,1]": (-1.0, 1.0),
}
CORPUS_DISTRIBUTION = "corpus uniform[-10,10]"

NUMPY_DTYPE = {"float16": np.float16, "float32": np.float32}


# ---------------------------------------------------------------------------
# Corpus access
# ---------------------------------------------------------------------------
def load_meta(name: str) -> dict:
    return json.loads((CORPUS / name / "meta.json").read_text())


def make_inputs(
    meta: dict, dims: dict[str, int], dtype: str, seed: int, distribution: str
) -> dict[str, np.ndarray]:
    low, high = DISTRIBUTIONS[distribution]
    rng = np.random.default_rng(seed)
    npdt = NUMPY_DTYPE[dtype]
    inputs = {}
    for spec in meta["op_schema"]["inputs"]:
        shape = [dims[d] for d in spec["dims"]]
        data = rng.random(int(np.prod(shape))) * (high - low) + low
        inputs[spec["name"]] = data.reshape(shape).astype(npdt)
    return inputs


_REFERENCE_CACHE: dict[tuple, np.ndarray] = {}


def reference(meta: dict, inputs: dict[str, np.ndarray], cache_key: tuple) -> np.ndarray:
    """Run the corpus's own fp64 reference script over its documented protocol."""
    if cache_key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[cache_key]

    script = CORPUS / meta["reference"]
    payload = {
        "inputs": {
            name: {
                "shape": list(arr.shape),
                "strides": list(arr.strides),
                "dtype": str(arr.dtype),
                "data": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode(),
            }
            for name, arr in inputs.items()
        },
        "kwargs": {},
    }
    done = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if done.returncode != 0:
        raise RuntimeError(f"reference {script} failed: {done.stderr.strip()}")
    out = json.loads(done.stdout)
    arr = np.frombuffer(base64.b64decode(out["data"]), dtype=np.dtype(out["dtype"]))
    arr = arr.reshape(out["shape"])
    _REFERENCE_CACHE[cache_key] = arr
    return arr


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------
def allclose_passes(candidate: np.ndarray, ref: np.ndarray, tol: float) -> bool:
    """torch.allclose / np.allclose semantics: |a - b| <= atol + rtol * |b|."""
    if candidate.shape != ref.shape:
        return False
    a = candidate.astype(np.float64)
    b = ref.astype(np.float64)
    if not np.all(np.isfinite(a)):
        return False
    return bool(np.all(np.abs(a - b) <= tol + tol * np.abs(b)))


def corpus_oracle_passes(candidate: np.ndarray, ref: np.ndarray, tol: float) -> bool:
    """The corpus validator: shape, NaN/Inf, then a plain absolute tolerance."""
    if candidate.shape != ref.shape:
        return False
    a = candidate.astype(np.float64)
    b = ref.astype(np.float64)
    if not np.all(np.isfinite(a)):
        return False
    return bool(np.all(np.abs(a - b) <= tol))


def max_abs_error(candidate: np.ndarray, ref: np.ndarray) -> float:
    diff = np.abs(candidate.astype(np.float64) - ref.astype(np.float64))
    return float(np.max(diff)) if diff.size else 0.0


# ---------------------------------------------------------------------------
# Shape selection
# ---------------------------------------------------------------------------
def benchmark_dims(meta: dict) -> dict[str, int]:
    """The shape a benchmark would pick: largest power-of-two candidate per dim.

    Falls back to the largest candidate when a dim has no power of two. On the
    softmax schema this yields H=256, which is the shape the paper names.
    """
    chosen = {}
    for dim in meta["op_schema"]["dims"]:
        candidates = sorted(dim["candidates"])
        powers = [c for c in candidates if c >= 2 and (c & (c - 1)) == 0]
        chosen[dim["name"]] = powers[-1] if powers else candidates[-1]
    return chosen


def all_dims(meta: dict):
    """Every combination the operator's declared schema allows."""
    names = [d["name"] for d in meta["op_schema"]["dims"]]
    for combo in itertools.product(*[d["candidates"] for d in meta["op_schema"]["dims"]]):
        yield dict(zip(names, combo))


# ---------------------------------------------------------------------------
# Verification of the ports themselves
# ---------------------------------------------------------------------------
def check_softmax_llm_port() -> None:
    """The one corpus kernel that runs on CPU unmodified: compare bit for bit."""
    import importlib.util

    path = CORPUS / "softmax_llm_buggy" / "kernel.py"
    spec = importlib.util.spec_from_file_location("corpus_softmax_llm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rng = np.random.default_rng(0)
    for n_cols in (3, 128, 256, 1025):
        x = (rng.random((2, 7, n_cols)) * 20.0 - 10.0).astype(np.float32)
        theirs = module.run({"input": x})
        ours = PORTS["softmax_llm_buggy"]({"input": x})
        if not np.array_equal(theirs, ours):
            raise AssertionError(
                f"softmax_llm_buggy port diverges from the corpus original at H={n_cols}"
            )
    print("port check: softmax_llm_buggy matches the corpus original exactly")


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure(buggy_name: str) -> dict:
    meta = load_meta(buggy_name)
    control_name = BUGGY_TO_CONTROL[buggy_name]
    buggy_fn = PORTS[buggy_name]
    control_fn = PORTS[control_name]
    dtypes = meta.get("dtypes", ["float32"])

    result = {
        "name": buggy_name,
        "benchmark_verdict_claimed": meta["benchmark_verdict"],
        "benchmark_dims": benchmark_dims(meta),
        "escaped": {},
        "max_err_at_benchmark": {},
        "control_failures": [],
        "sweep": {d: {"hidden": 0, "total": 0, "hiding_shapes": []} for d in DISTRIBUTIONS},
    }

    # --- 1. the single-shape benchmark oracle -----------------------------
    # Run under the corpus's own input distribution, which is the reading of
    # the "benchmark_verdict" claim most favourable to the paper.
    dims = benchmark_dims(meta)
    for dtype in dtypes:
        tol_corpus = float(meta["tolerances"].get(dtype, 1e-5))
        escapes = {f"allclose_{t:g}": True for t in BENCHMARK_TOLERANCES}
        escapes["corpus_tol"] = True
        worst = 0.0
        for seed in range(BENCHMARK_DRAWS):
            inputs = make_inputs(meta, dims, dtype, seed, CORPUS_DISTRIBUTION)
            key = (buggy_name, tuple(sorted(dims.items())), dtype, seed, CORPUS_DISTRIBUTION)
            ref = reference(meta, inputs, key)
            out = buggy_fn(inputs)
            worst = max(worst, max_abs_error(out, ref))
            for t in BENCHMARK_TOLERANCES:
                escapes[f"allclose_{t:g}"] &= allclose_passes(out, ref, t)
            escapes["corpus_tol"] &= corpus_oracle_passes(out, ref, tol_corpus)
        result["escaped"][dtype] = escapes
        result["max_err_at_benchmark"][dtype] = worst

    # --- 2. the full schema sweep, per distribution, plus port controls ----
    for distribution in DISTRIBUTIONS:
        bucket = result["sweep"][distribution]
        for dtype in dtypes:
            tol_corpus = float(meta["tolerances"].get(dtype, 1e-5))
            for dims in all_dims(meta):
                inputs = make_inputs(meta, dims, dtype, 1234, distribution)
                key = (buggy_name, tuple(sorted(dims.items())), dtype, 1234, distribution)
                ref = reference(meta, inputs, key)

                if not corpus_oracle_passes(control_fn(inputs), ref, tol_corpus):
                    result["control_failures"].append((dict(dims), dtype, distribution))

                bucket["total"] += 1
                if corpus_oracle_passes(buggy_fn(inputs), ref, tol_corpus):
                    bucket["hidden"] += 1
                    bucket["hiding_shapes"].append((dict(dims), dtype))

    return result


BUGGY_KERNELS = list(BUGGY_TO_CONTROL)


def main() -> int:
    print(f"corpus: {CORPUS}")
    print(f"inputs: {', '.join(DISTRIBUTIONS)}")
    print(f"reference: the corpus's own fp64 scripts, run over its own protocol\n")

    check_softmax_llm_port()
    print()

    results = [measure(name) for name in BUGGY_KERNELS]

    broken_ports = [(r["name"], r["control_failures"]) for r in results if r["control_failures"]]
    if broken_ports:
        print("ABORT: control kernels failed, so the CPU ports are not faithful.\n")
        for name, failures in broken_ports:
            print(f"  {name}: {len(failures)} control failures, first = {failures[0]}")
        return 1
    print("port check: all correct-variant controls pass their own schema sweep\n")

    header = f"{'kernel':<30} {'claimed':<8} {'1e-3':<6} {'1e-2':<6} {'own tol':<8} {'max abs err':>12}"
    print("=" * len(header))
    print("SINGLE-SHAPE BENCHMARK ORACLE, at the shape a benchmark would pick (float32)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    escaped_counts = {f"allclose_{t:g}": 0 for t in BENCHMARK_TOLERANCES}
    escaped_counts["corpus_tol"] = 0
    for r in results:
        e = r["escaped"]["float32"]
        for k in escaped_counts:
            escaped_counts[k] += bool(e[k])
        mark = lambda ok: "ESCAPE" if ok else "caught"
        print(
            f"{r['name']:<30} {r['benchmark_verdict_claimed']:<8} "
            f"{mark(e['allclose_0.001']):<6} {mark(e['allclose_0.01']):<6} "
            f"{mark(e['corpus_tol']):<8} {r['max_err_at_benchmark']['float32']:>12.4g}"
        )
    print("-" * len(header))
    total = len(results)
    for tol in BENCHMARK_TOLERANCES:
        k = f"allclose_{tol:g}"
        print(f"  measured escape rate, allclose atol=rtol={tol:g}: {escaped_counts[k]}/{total}")
    print(f"  measured escape rate, corpus's own tolerances:   {escaped_counts['corpus_tol']}/{total}")
    print(f"  rate asserted by the corpus metadata:            {total}/{total}")

    print()
    names = list(DISTRIBUTIONS)
    header2 = f"{'kernel':<30} {names[0]:>24} {names[1]:>26}"
    print("=" * len(header2))
    print("FULL SCHEMA SWEEP: fraction of the declared shape space that hides each bug")
    print("=" * len(header2))
    print(header2)
    print("-" * len(header2))
    totals = {d: [0, 0] for d in DISTRIBUTIONS}
    for r in results:
        cells = []
        for d in names:
            bucket = r["sweep"][d]
            totals[d][0] += bucket["hidden"]
            totals[d][1] += bucket["total"]
            pct = 100.0 * bucket["hidden"] / bucket["total"]
            cells.append(f"{bucket['hidden']:>4}/{bucket['total']:<4} {pct:>5.1f}%")
        print(f"{r['name']:<30} {cells[0]:>24} {cells[1]:>26}")
    print("-" * len(header2))
    cells = []
    for d in names:
        hidden, total = totals[d]
        cells.append(f"{hidden:>4}/{total:<4} {100.0 * hidden / total:>5.1f}%")
    print(f"{'ALL BUGS':<30} {cells[0]:>24} {cells[1]:>26}")
    print("=" * len(header2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
