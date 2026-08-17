import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from kernelverify.runners import MetalRunner  # noqa: E402

# One Metal probe per pytest run, shared by every Metal-gated module. Each
# probe spawns a worker process, so the per-file copies of this line paid
# that cost three to four times per collection.
METAL_DEVICE = MetalRunner().probe()

# One decorator carries BOTH facts about a Metal test, because they are the
# same fact read two ways and keeping them apart is how they drift:
#
#   skipif   - there may be no device at all (CI, a sandbox), so the test
#              cannot run and must not fail;
#   gpu      - there IS a device and something heavy is using it, so the test
#              must not run: this machine has one GPU and one machine-wide
#              measurement lock (bench/machine_state.py), and a 33-minute A/B
#              and a test batch dispatching against each other corrupts the
#              measurement and slows the tests. `pytest -m "not gpu"` is what
#              a lane runs during a measurement.
#
# Gating a test with `requires_metal` therefore also excuses it from the safe
# subset, with no second edit to remember.
_no_metal = pytest.mark.skipif(METAL_DEVICE is None,
                               reason="no Metal device on this machine")


def requires_metal(obj):
    """Gate a test on a Metal device AND excuse it from the safe subset.

    Applied as two marks, deliberately. `pytest.mark.gpu(pytest.mark.skipif(...))`
    reads like composition and is not: pytest takes the inner MarkDecorator as a
    positional ARGUMENT to `gpu`, so the decorated test carries one mark named
    `gpu` and no skipif at all. That failure is invisible on a machine that has
    Metal, because there the skip would be a no-op anyway - it only shows up on
    the machine the skip exists for, as a crash instead of a skip.
    """
    return pytest.mark.gpu(_no_metal(obj))

# Well-conditioned shapes: every dimension modest, no degenerate reduction, so
# any correct implementation should agree with the fp64 reference to near
# working precision and a disagreement means a broken member.
WELL_CONDITIONED = {"B": 2, "S": 3, "H": 16, "M": 5, "N": 96, "K": 32, "D": 16}


def unit_inputs(op: str, rng: np.random.Generator) -> dict:
    """Float32 standard-normal inputs for `op` at the well-conditioned dims.

    The rng is the caller's, so each suite keeps its own seed and its own
    draw sequence.
    """
    from measure_escape import load_meta

    spec = load_meta(op)["op_schema"]["inputs"]
    return {
        s["name"]: rng.standard_normal(
            [WELL_CONDITIONED[d] for d in s["dims"]]
        ).astype(np.float32)
        for s in spec
    }


# ---------------------------------------------------------------------------
# Baseline-row factories: one valid row per schema version. v2 rows feed the
# matrix renderer's gates, v3 rows feed the measure_baselines producer
# contract; each base is a fresh literal per call, exactly as the per-file
# factories built them.
# ---------------------------------------------------------------------------
def _row_v2() -> dict:
    return {
        "schema_version": 2,
        "run_id": "20260814T120000Z-abcdef12",
        "row_id": "20260814T120000Z-abcdef12/decode-llamacpp",
        "measured_at": "2026-08-14T12:00:00Z",
        "provenance_tier": "owner-run",
        "binding": True,
        "binding_blockers": [],
        "machine": {"chip": "Apple M3 Pro", "hw_model": "Mac15,6", "os": "macOS",
                    "os_build": "26.5.2"},
        "stack": {"name": "llama.cpp", "version": "a94d563"},
        "model": {"name": "Qwen3-4B", "quant": "Q4_K_M"},
        "measurement": {"kind": "decode", "matmul_width": 1,
                        "width_mechanism": "prompt-width"},
        "result": {"metric": "tok/s", "median": 45.67, "spread_pct": 1.2,
                   "reps": 3, "samples": [45.6, 45.67, 45.8],
                   "min_sample_ms": 21.9, "floor_ms": 1.0,
                   "below_timing_floor": False},
        "sampling": {"interleaved": True, "rotation": "round-robin", "rounds": 5,
                     "group": "qwen3-4b-decode", "group_members": 2},
        "roofline": {"bytes_per_pass": 2491000000, "byte_model": "tensor-table",
                     "denominator_name": "bandwidth_read", "denominator_gbs": 135.5,
                     "achieved_gbs": 114.2, "bandwidth_utilisation_pct": 84.3,
                     "binding_resource": "memory",
                     "roofline_utilisation_pct": None},
    }


def _row_v3() -> dict:
    def idle_state():
        return {
            "load1": 0.5, "load5": 0.5, "load15": 0.5, "load_threshold": 3.0,
            "power": {"source": "AC", "low_power_mode": "0",
                      "thermal_warning": False},
            "idle": True, "blockers": [],
        }

    return {
        "schema_version": 3, "run_id": "r", "row_id": "r/x", "measured_at": "t",
        "provenance_tier": "owner-run", "machine": {}, "idle_before": idle_state(),
        "idle_after": idle_state(), "stack": {"name": "llama.cpp"},
        "model": {"name": "Qwen3-4B-Q4_K_M.gguf", "logical_name": "qwen3-4b"},
        "measurement": {"kind": "decode"},
        "result": {"metric": "tokens_per_s", "median": 1.0, "reps": 3,
                   "samples": [1.0], "min_sample_ms": 50.0, "floor_ms": 1.0,
                   "below_timing_floor": False},
        "roofline": {"bandwidth_utilisation_pct": 80.0,
                     "binding_resource": "memory"},
        "sampling": {"interleaved": True, "rotation": "arm-alternation",
                     "rounds": 3, "group": "r/decode-w1", "group_members": ["x"]},
        "binding": True, "binding_blockers": [],
    }


_SCHEMA_ROW_BASES = {2: _row_v2, 3: _row_v3}


def schema_row(version: int = 2, **over) -> dict:
    """A valid baselines row at schema `version`; keyword overrides replace
    top-level keys, and dict-valued overrides merge one level deep."""
    base = _SCHEMA_ROW_BASES[version]()
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base
