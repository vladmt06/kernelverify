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


def _round_half_away(x: np.ndarray) -> np.ndarray:
    """Metal's round(): half away from zero, unlike numpy's half-to-even rint."""
    return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))


def canonical_quantize(w: np.ndarray, contract: QuantContract) -> QuantArtefact:
    """MLX-affine quantization, replicated exactly from the affine_quantize
    kernel in mlx/backend/metal/kernels/quantized.h (verified bit-exact by the
    Phase 0 experiment).

    The real algorithm differs from the naive min/max affine scheme:

    - w_max starts at 0, so an all-negative group clamps its max to zero;
    - scale = max((w_max - w_min) / (2^bits - 1), 1e-7), then takes a NEGATIVE
      sign when |w_max| >= |w_min| (the "side" convention);
    - the scale is snapped so the dominant edge is exactly representable:
      q0 = round(edge / scale); if q0 != 0, scale = edge / q0 and bias = edge,
      else bias = 0;
    - q = min(round((w - bias) / scale), 2^bits - 1), upper clip only, with
      round-half-away-from-zero;
    - all arithmetic in fp32; scales and biases are cast to the weight dtype
      only at storage (the stored fp16 values are NOT the ones q was computed
      with - that asymmetry is MLX's, and the contract preserves it).
    """
    g = contract.group_size
    n_bins = np.float32((1 << contract.bits) - 1)
    rows, cols = w.shape
    assert cols % g == 0, f"d_in {cols} must be a multiple of group_size {g}"

    grouped = w.astype(np.float32).reshape(rows, cols // g, g)
    w_min = grouped.min(axis=2)
    w_max = np.maximum(grouped.max(axis=2), np.float32(0.0))

    scale = np.maximum((w_max - w_min) / n_bins, np.float32(1e-7))
    side = np.abs(w_min) > np.abs(w_max)
    scale = np.where(side, scale, -scale).astype(np.float32)
    edge = np.where(side, w_min, w_max).astype(np.float32)
    q0 = _round_half_away((edge / scale).astype(np.float32))
    at_zero = q0 == 0.0
    scale = np.where(at_zero, scale, (edge / np.where(at_zero, 1.0, q0)).astype(np.float32))
    bias = np.where(at_zero, np.float32(0.0), edge)

    q = _round_half_away(((grouped - bias[:, :, None]) / scale[:, :, None]).astype(np.float32))
    q = np.minimum(q, n_bins).astype(np.int32).reshape(rows, cols)
    return QuantArtefact(
        q=q,
        scales=scale.astype(w.dtype),
        biases=bias.astype(w.dtype),
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

    Algebraically identical to the dequant-domain members, numerically a
    different rounding sequence. Both per-group sums are EXPLICIT fp32 CHAINS,
    accumulated left to right the way an int-accumulate GEMV kernel's inner
    loop runs; only the cross-group combination stays vectorized (that is the
    freedom `factored-serial` varies).

    The chains are the member's whole point and were once its bug. Formed by a
    pairwise reduction instead, the activation sum is EXACT on a constant row -
    64 identical fp32 values halve down a power-of-two tree with no rounding at
    all - so the member was most accurate exactly where a real kernel is least
    accurate, the floor it feeds was too tight there, and the shipped tolerance
    flagged a correct in-contract device kernel on 10 of 1,536 serving records
    (worst 1.743x; ADR 0016, tests/test_quant_contract_members.py).
    """
    g = a.contract.group_size
    rows, cols = a.q.shape
    groups = cols // g
    xg = _f32(x).reshape(x.shape[0], groups, g)
    qg = a.q.reshape(rows, groups, g).astype(np.float32)
    xq = np.zeros((x.shape[0], rows, groups), dtype=np.float32)
    xs = np.zeros((x.shape[0], groups), dtype=np.float32)
    for k in range(g):
        xq += xg[:, None, :, k] * qg[None, :, :, k]
        xs += xg[:, :, k]
    out = (xq * a.scales.astype(np.float32)[None, :, :]).sum(axis=2)
    out += xs @ a.biases.astype(np.float32).T
    return out.astype(x.dtype)


def member_dequant_reversed(x: np.ndarray, a: QuantArtefact) -> np.ndarray:
    """Serial accumulation in reversed feature order (a second legal order)."""
    w = dequantize(a, np.float32)
    prod = (_f32(x)[:, None, :] * w[None, :, :])[:, :, ::-1]
    return np.add.accumulate(prod, axis=2)[:, :, -1].astype(x.dtype)


def member_factored_serial(x: np.ndarray, a: QuantArtefact) -> np.ndarray:
    """Int-domain accumulation with strictly serial cross-group summation.

    The second member of the int-domain class. Every legitimate evaluation
    CLASS needs at least two ensemble members: leave-one-out K calibration
    otherwise removes the whole class and measures class absence (K blew up
    to 7 through exactly that artifact) instead of within-class spread.
    """
    g = a.contract.group_size
    rows, cols = a.q.shape
    xg = _f32(x).reshape(x.shape[0], cols // g, g)
    qg = a.q.reshape(rows, cols // g, g).astype(np.float32)
    xq = np.einsum("bgk,rgk->brg", xg, qg, optimize=True)
    per_group = (xq * a.scales.astype(np.float32)[None, :, :]
                 + xg.sum(axis=2)[:, None, :] * a.biases.astype(np.float32)[None, :, :])
    return np.add.accumulate(per_group, axis=2)[:, :, -1].astype(x.dtype)


ENSEMBLE = {
    "dequant-pairwise": member_dequant_pairwise,
    "dequant-serial": member_dequant_serial,
    "lut-gather": member_lut_gather,
    "factored-groups": member_factored_groups,
    "factored-serial": member_factored_serial,
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


def fault_unpack_mask_too_wide(a: QuantArtefact) -> QuantArtefact:
    """Unpack mask one bit too wide, so each code absorbs its neighbour's low bit.

    Measured on the MLX pack surface by the kv-kernels lane at 4.87e+2 against
    an output scale of 150. The mask is written as (1 << bits) - 1 in every
    kernel that unpacks a bit stream, and getting `bits` wrong by one is the
    single most available transcription error in that line.
    """
    n_bins = (1 << a.contract.bits) - 1
    neighbour = np.roll(a.q, -1, axis=1) & 1
    q = np.minimum(a.q + (neighbour << a.contract.bits) // 2, n_bins)
    return QuantArtefact(np.ascontiguousarray(q.astype(a.q.dtype)),
                         a.scales, a.biases, a.contract)


def fault_all_groups_read_group_zero(a: QuantArtefact) -> QuantArtefact:
    """Every group reads group 0's scale and bias: a dropped group stride.

    Measured by the kv-kernels lane at 2.75e+2. Distinct from
    `fault_scales_rotated`, which is off by one group and still varies down the
    row; this one collapses the whole row onto a single group's parameters, so
    it survives any test whose weights happen to be group-uniform.
    """
    scales = np.broadcast_to(a.scales[:, :1], a.scales.shape)
    biases = np.broadcast_to(a.biases[:, :1], a.biases.shape)
    return QuantArtefact(a.q, np.ascontiguousarray(scales),
                         np.ascontiguousarray(biases), a.contract)


FAULTS = {
    "scales-rotated-one-group": fault_scales_rotated,
    "nibble-order-swapped": fault_nibble_swapped,
    "bias-dropped": fault_bias_dropped,
    "group-size-halved": fault_group_size_halved,
    "unpack-mask-too-wide": fault_unpack_mask_too_wide,
    "all-groups-read-group-0": fault_all_groups_read_group_zero,
}
