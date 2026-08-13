"""The shipped conditioning-aware tolerance: the working-precision ensemble floor.

Fixed per-operator tolerances are wrong in ill-conditioned regimes: on
constant-rows attention a provably correct fp32 kernel exceeds the published
1e-3 against the fp64 reference, because fp32 rounding of score intermediates
near magnitude 800 legitimately moves softmax outputs past it (ADR 0003).

The verdict a verifier can actually defend is therefore relative to what a
correct implementation could do on this exact case:

    floor(case) = max over an ensemble of provably-correct working-precision
                  implementations of |impl(inputs) - ref_fp64(inputs)|
    tolerance   = max(base_tol, K_ENSEMBLE * floor)

Ensemble members are the reference kernel itself plus worst-legitimate-order
variants: sequential (non-pairwise) accumulation, and a different flash tile
width for the attention family. They bound the rounding a correct kernel with
a different reduction structure can commit. K_ENSEMBLE = 1.5 was calibrated in
ADR 0004: the smallest grid value with zero false positives over held-out
diverse correct implementations (other tile widths, reversed summation, the
plain-vs-flash cross pair), with no fault detection lost.

An input-perturbation probe is NOT a substitute for this floor: it measures
input conditioning only, misses internal accumulation error by up to 64x, and
probing at fp16 epsilon absolves fp16-internal arithmetic, the exact fault
class the verifier exists to reject (ADR 0004 records the falsification).
"""

from __future__ import annotations

import functools

import numpy as np

from kernelverify.reference.kernels import (
    L2NORM_EPS,
    RMSNORM_EPS,
    attention,
    flash_attention,
    gelu,
    leaky_relu,
    matmul,
    rmsnorm,
    l2norm,
    silu,
    softmax,
)

K_ENSEMBLE = 1.5


def _f32(a):
    return a.astype(np.float32)


def serial_sum(a, axis):
    """Sequential left-to-right fp32 accumulation: the worst legitimate order."""
    return np.add.accumulate(_f32(a), axis=axis).take(-1, axis=axis)


def softmax_serial(inputs):
    x = _f32(inputs["input"])
    rows = x.reshape(-1, x.shape[-1])
    shifted = rows - rows.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    out = e / serial_sum(e, 1)[:, None]
    return out.reshape(x.shape).astype(inputs["input"].dtype)


def rmsnorm_serial(inputs):
    x = _f32(inputs["input"])
    ms = serial_sum(x * x, -1) / np.float32(x.shape[-1])
    out = x / np.sqrt(ms + np.float32(RMSNORM_EPS))[..., None]
    return out.astype(inputs["input"].dtype)


def l2norm_serial(inputs):
    x = _f32(inputs["input"])
    norm = np.sqrt(serial_sum(x * x, -1) + np.float32(L2NORM_EPS))
    return (x / norm[..., None]).astype(inputs["input"].dtype)


def matmul_serial(inputs):
    a, b = _f32(inputs["a"]), _f32(inputs["b"])
    m, kdim = a.shape
    n = b.shape[1]
    out = np.empty((m, n), dtype=np.float32)
    for n0 in range(0, n, 64):  # bound the products tensor to m*kdim*64 floats
        prod = a[:, :, None] * b[None, :, n0:n0 + 64]
        out[:, n0:n0 + 64] = np.add.accumulate(prod, axis=1)[:, -1, :]
    return out.astype(inputs["a"].dtype)


def attention_serial(inputs):
    q, k, v = _f32(inputs["q"]), _f32(inputs["k"]), _f32(inputs["v"])
    d = q.shape[-1]
    prod = q[:, None, :] * k[None, :, :]
    scores = np.add.accumulate(prod, axis=2)[:, :, -1] * np.float32(d ** -0.5)
    scores = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(scores)
    p = e / serial_sum(e, 1)[:, None]
    pv = p[:, :, None] * v[None, :, :]
    out = np.add.accumulate(pv, axis=1)[:, -1, :]
    return out.astype(inputs["q"].dtype)


_flash16 = functools.partial(flash_attention, block_n_cap=16)

# Corpus operator -> the provably-correct working-precision ensemble.
ENSEMBLES = {
    "gelu_triton": [gelu],
    "silu_triton": [silu],
    "leaky_relu_triton": [leaky_relu],
    "softmax_triton": [softmax, softmax_serial],
    "rmsnorm_triton": [rmsnorm, rmsnorm_serial],
    "l2norm_triton": [l2norm, l2norm_serial],
    "matmul_triton": [matmul, matmul_serial],
    "attention_triton": [attention, attention_serial, _flash16],
    "flash_attention_triton": [flash_attention, attention_serial, _flash16],
}


def _max_abs_error(candidate, ref) -> float:
    diff = np.abs(candidate.astype(np.float64) - ref.astype(np.float64))
    return float(np.max(diff)) if diff.size else 0.0


def ensemble_floor(op: str, inputs: dict, ref: np.ndarray) -> float:
    """Worst deviation of any provably-correct implementation from fp64 truth."""
    return max(_max_abs_error(fn(inputs), ref) for fn in ENSEMBLES[op])


def conditioned_tolerance(op: str, inputs: dict, ref: np.ndarray,
                          base_tol: float) -> float:
    """The shipped per-case tolerance: max(base_tol, K_ENSEMBLE * floor)."""
    return max(base_tol, K_ENSEMBLE * ensemble_floor(op, inputs, ref))
