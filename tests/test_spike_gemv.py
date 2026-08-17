"""The migrated GEMV spike's runner door, validated on the branch.

The spike's verify() routes through the pack's shared verdict path, which
lives on a branch that merges ahead of this one, so it cannot run here; the
consolidation merge review runs it as the blocking GEMV parity gate. What CAN
be pinned now is everything on this side of that import: the raw-MSL door
assembles, specializes, compiles, carries packed uint32 words through the
transport, and computes the same dequant-GEMV a plain numpy implementation
does on synthetic codes.
"""

import numpy as np
import pytest

from spike_dequant_gemv import gemv_case, gemv_template

from conftest import requires_metal

from kernelverify.runners import MetalRunner, specialize

RUNNER = MetalRunner()


def pack_nibbles(q: np.ndarray) -> np.ndarray:
    """8 4-bit codes per uint32, low nibble first: the spike kernel's layout."""
    rows, cols = q.shape
    q8 = q.reshape(rows, cols // 8, 8).astype(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * 4)[None, None, :]
    return np.bitwise_or.reduce(q8 << shifts, axis=2).astype(np.uint32)


def synthetic_gemv(d_out: int, d_in: int, seed: int = 3):
    """Codes, scales, biases and input drawn directly; no quantizer involved."""
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 16, size=(d_out, d_in), dtype=np.int64)
    groups = d_in // 64
    scales = (rng.random((d_out, groups), dtype=np.float32) * 0.05).astype(np.float16)
    biases = (rng.random((d_out, groups), dtype=np.float32) * 0.1 - 0.05).astype(np.float16)
    x = rng.standard_normal((1, d_in)).astype(np.float16)

    dequant = (q.astype(np.float64)
               * np.repeat(scales.astype(np.float64), 64, axis=1)
               + np.repeat(biases.astype(np.float64), 64, axis=1))
    reference = dequant @ x[0].astype(np.float64)
    return q, scales, biases, x, reference


def test_the_template_specializes_without_leftover_placeholders():
    spec = specialize(gemv_template(), {"T": "half"})
    assert "$T" not in spec.source
    assert "using T = half;" in spec.source
    assert spec.name == "kv_dequant_gemv-half"


@requires_metal
def test_the_runner_door_matches_a_numpy_dequant_gemv():
    spec = specialize(gemv_template(), {"T": "half"})
    for d_out, d_in in ((16, 128), (24, 2560)):
        q, scales, biases, x, reference = synthetic_gemv(d_out, d_in)
        case = gemv_case(x, pack_nibbles(q), scales, biases, d_out,
                         label=f"{d_out}x{d_in}")
        result = RUNNER.run_one(spec, case)
        assert result.ok, result.detail
        assert result.outputs[0].dtype == np.float16
        # The output is rounded to fp16; everything upstream accumulates in
        # fp32 over |dequant| <= 0.7, so half-epsilon of the result magnitude
        # bounds the legitimate error.
        scale = np.max(np.abs(reference))
        np.testing.assert_allclose(result.outputs[0].astype(np.float64),
                                   reference, atol=2e-3 * max(scale, 1.0))
