"""The quantization contract: what a quantized kernel is supposed to compute.

An unquantized kernel's contract is implicit: fp64 of the original operator.
A quantized kernel's intended answer depends on the quantization artefact
(packed values, scales, biases), so the contract is explicit data:

    contract  = (scheme, bits, group_size)
    artefact  = quantize(W, contract)          - packed q, scales, biases
    R_contract(x) = fp64( x @ dequant(artefact) )

Anchoring verdicts on R_contract instead of fp64(x @ W) is the whole point
(ADR 0003's Experiment 2 measured the fp64-original anchor rejecting a correct
quantized layer on every case). Phase 0 (eng review D9-A) targets the
MLX-affine scheme because the launch channel is MLX; GGUF K-quants get their
own contract before any llama.cpp quant kernel.

The canonical quantizer here replicates MLX's affine scheme in numpy and is
verified bit-exact against mx.quantize by the Phase 0 experiment; if MLX ever
changes, the experiment fails loudly rather than the contract drifting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantContract:
    scheme: str = "mlx-affine"
    bits: int = 4
    group_size: int = 64


@dataclass(frozen=True)
class QuantArtefact:
    """The compiled quantization artefact, kept unpacked for clarity."""

    q: np.ndarray       # int levels, shape of W, values in [0, 2^bits - 1]
    scales: np.ndarray  # per group, shape (rows, groups)
    biases: np.ndarray  # per group, shape (rows, groups)
    contract: QuantContract


def canonical_quantize(w: np.ndarray, contract: QuantContract) -> QuantArtefact:
    """MLX-affine quantization, replicated in numpy.

    Per group of `group_size` along the last axis: the group's [min, max] maps
    affinely onto the integer levels [0, 2^bits - 1]:

        scale = (max - min) / (2^bits - 1)
        q     = round((w - min) / scale), clipped to the level range
        w~    = scale * q + min

    Scales and biases are stored in the weight dtype, matching mx.quantize.
    The Phase 0 experiment checks this function bit-exact against MLX.
    """
    g = contract.group_size
    levels = (1 << contract.bits) - 1
    rows, cols = w.shape
    assert cols % g == 0, f"d_in {cols} must be a multiple of group_size {g}"

    grouped = w.astype(np.float32).reshape(rows, cols // g, g)
    gmax = grouped.max(axis=2)
    gmin = grouped.min(axis=2)
    scale = (gmax - gmin) / np.float32(levels)
    safe_scale = np.where(scale == 0, np.float32(1.0), scale)

    q = np.rint((grouped - gmin[:, :, None]) / safe_scale[:, :, None])
    q = np.clip(q, 0, levels).astype(np.int32).reshape(rows, cols)
    return QuantArtefact(
        q=q,
        scales=scale.astype(w.dtype),
        biases=gmin.astype(w.dtype),
        contract=contract,
    )


def dequantize(a: QuantArtefact, dtype=np.float64) -> np.ndarray:
    """The intended weights W~, at the requested precision (fp64 = the truth)."""
    g = a.contract.group_size
    rows, cols = a.q.shape
    q = a.q.reshape(rows, cols // g, g).astype(dtype)
    w = a.scales.astype(dtype)[:, :, None] * q + a.biases.astype(dtype)[:, :, None]
    return w.reshape(rows, cols)


def r_contract(x: np.ndarray, a: QuantArtefact) -> np.ndarray:
    """The contract reference: fp64 evaluation of the intended quantized layer."""
    return x.astype(np.float64) @ dequantize(a, np.float64).T


# ---------------------------------------------------------------------------
# The legitimate ensemble: five correct working-precision implementations of
# x @ W~ that differ only in defensible evaluation choices. Their worst
# deviation from r_contract is the tolerance floor, exactly as floor.py's
# ensemble bounds the unquantized operators.
# ---------------------------------------------------------------------------
def _f32(x):
    return x.astype(np.float32)


def member_dequant_pairwise(x: np.ndarray, a: QuantArtefact) -> np.ndarray:
    """Dequantize to fp32, multiply with numpy's pairwise summation."""
    w = dequantize(a, np.float32)
    return (_f32(x) @ w.T).astype(x.dtype)


def member_dequant_serial(x: np.ndarray, a: QuantArtefact) -> np.ndarray:
    """Dequantize to fp32, accumulate strictly left to right (worst legal order)."""
    w = dequantize(a, np.float32)
    prod = _f32(x)[:, None, :] * w[None, :, :]
    return np.add.accumulate(prod, axis=2)[:, :, -1].astype(x.dtype)


def member_lut_gather(x: np.ndarray, a: QuantArtefact) -> np.ndarray:
    """Per-group lookup table of dequantized level values, gathered then MAC'd."""
    g = a.contract.group_size
    levels = (1 << a.contract.bits)
    rows, cols = a.q.shape
    table = (a.scales.astype(np.float32)[:, :, None]
             * np.arange(levels, dtype=np.float32)[None, None, :]
             + a.biases.astype(np.float32)[:, :, None])
    q = a.q.reshape(rows, cols // g, g)
    w = np.take_along_axis(table, q, axis=2).reshape(rows, cols)
    return (_f32(x) @ w.T).astype(x.dtype)


def member_factored_groups(x: np.ndarray, a: QuantArtefact) -> np.ndarray:
    """Integer-domain accumulation: s * sum(x*q) + b * sum(x), per group.

    Algebraically identical, numerically a different rounding sequence - the
    shape of every int-accumulate quantized GEMV kernel.
    """
    g = a.contract.group_size
    rows, cols = a.q.shape
    xg = _f32(x).reshape(x.shape[0], cols // g, g)
    qg = a.q.reshape(rows, cols // g, g).astype(np.float32)
    xq = np.einsum("bgk,rgk->brg", xg, qg, optimize=True)
    xs = xg.sum(axis=2)
    out = (xq * a.scales.astype(np.float32)[None, :, :]).sum(axis=2)
    out += xs @ a.biases.astype(np.float32).T
    return out.astype(x.dtype)


def member_dequant_reversed(x: np.ndarray, a: QuantArtefact) -> np.ndarray:
    """Serial accumulation in reversed feature order (a second legal order)."""
    w = dequantize(a, np.float32)
    prod = (_f32(x)[:, None, :] * w[None, :, :])[:, :, ::-1]
    return np.add.accumulate(prod, axis=2)[:, :, -1].astype(x.dtype)


ENSEMBLE = {
    "dequant-pairwise": member_dequant_pairwise,
    "dequant-serial": member_dequant_serial,
    "lut-gather": member_lut_gather,
    "factored-groups": member_factored_groups,
    "dequant-reversed": member_dequant_reversed,
}


# ---------------------------------------------------------------------------
# Artefact faults: the misalignment class no input-space axis can synthesise.
# Each returns a corrupted artefact; a verifier anchored on r_contract of the
# TRUE artefact must keep catching every one of these.
# ---------------------------------------------------------------------------
def fault_scales_rotated(a: QuantArtefact) -> QuantArtefact:
    """Group boundary misalignment: every group reads its neighbour's scale."""
    return QuantArtefact(a.q, np.roll(a.scales, 1, axis=1),
                         np.roll(a.biases, 1, axis=1), a.contract)


def fault_nibble_swapped(a: QuantArtefact) -> QuantArtefact:
    """Packing order fault: adjacent quantized values transposed."""
    q = a.q.reshape(a.q.shape[0], -1, 2)[:, :, ::-1].reshape(a.q.shape)
    return QuantArtefact(np.ascontiguousarray(q), a.scales, a.biases, a.contract)


def fault_bias_dropped(a: QuantArtefact) -> QuantArtefact:
    """Symmetric-scheme confusion: affine bias silently discarded."""
    return QuantArtefact(a.q, a.scales, np.zeros_like(a.biases), a.contract)


def fault_group_size_halved(a: QuantArtefact) -> QuantArtefact:
    """Wrong group_size interpretation: each scale applied to half its span."""
    scales = np.repeat(a.scales, 2, axis=1)[:, : a.scales.shape[1]]
    biases = np.repeat(a.biases, 2, axis=1)[:, : a.biases.shape[1]]
    return QuantArtefact(a.q, scales, biases, a.contract)


FAULTS = {
    "scales-rotated-one-group": fault_scales_rotated,
    "nibble-order-swapped": fault_nibble_swapped,
    "bias-dropped": fault_bias_dropped,
    "group-size-halved": fault_group_size_halved,
}
