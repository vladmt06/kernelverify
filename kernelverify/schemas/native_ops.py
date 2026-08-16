"""Native operator registry: schema, fp64 reference, and per-case tolerance.

Corpus operators get their ground truth from the vendored fp64 subprocess
scripts; native operators (Tranche 1.5) carry their own, in-process:

    NativeOp.meta       - the same schema shape corpus meta.json uses, so
                          case_space, the input modes, and every battery
                          policy work unchanged
    NativeOp.reference  - fp64 truth (contract-anchored where the operator
                          is quantized)
    NativeOp.tolerance  - the shipped conditioning-aware tolerance for this
                          operator family, with its own calibrated K
    NativeOp.augment    - deterministic non-random inputs a case implies
                          (e.g. the bits dimension delivered to the kernel)

Ground-truth honesty rule: every native reference is cross-checked against an
independent implementation in tests (MLX's own ops where they exist).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from kernelverify.reference.native_kernels import _topk_by_prob, moe_dispatch, TOP_K
from kernelverify.schemas.quant_contract import (
    ENSEMBLE as QUANT_ENSEMBLE,
    QuantContract,
    canonical_quantize,
    r_contract,
)

# Re-derived at 4.0 by the device-arithmetic calibration (ADR 0012): a correct
# device kernel (simdgroup-factored) legitimately exceeds the CPU-calibrated
# K=3 tolerance by up to 1.30x on constant-rows float32, and the joined
# membership demands 3.901. Measured adequate at 4.0: zero false positives on
# both draws, worst fault margin 161x, so detection is unchanged.
K_QUANT = 4.0
K_NATIVE = 1.5   # unquantized ensembles, same value ADR 0004 measured

_EPS = {"float16": 9.77e-4, "float32": 1.19e-7}


def _base_tol(dtype: str, ref: np.ndarray) -> float:
    scale = float(np.max(np.abs(ref))) if ref.size else 0.0
    return 4.0 * _EPS[dtype] * scale


def _max_err(candidate: np.ndarray, ref: np.ndarray) -> float:
    diff = np.abs(candidate.astype(np.float64) - ref.astype(np.float64))
    return float(diff.max()) if diff.size else 0.0


@dataclass(frozen=True)
class NativeOp:
    meta: dict
    reference: Callable
    tolerance: Callable
    augment: Callable | None = None


# ---------------------------------------------------------------------------
# quantized_matmul: contract-anchored (MLX-affine, group 64, bits as a
# schema dimension so the boundary policies cover the bit-width axis).
# ---------------------------------------------------------------------------
QUANTIZED_MATMUL_META = {
    "op_schema": {
        "inputs": [
            {"name": "x", "dims": ["B", "D_IN"]},
            {"name": "w", "dims": ["D_OUT", "D_IN"]},
        ],
        # Shapes are chosen for fault expression, not perf realism: dilution
        # (ADR 0001) means a short reduction carries the strongest absolute
        # signal, and 448 supplies the non-power-of-two the boundary policy
        # needs while staying a whole number of 64-wide quantization groups.
        "dims": [
            {"name": "B", "candidates": [1, 4]},
            {"name": "D_IN", "candidates": [128, 448, 1024]},
            {"name": "D_OUT", "candidates": [16, 512]},
            {"name": "BITS", "candidates": [2, 3, 4, 8]},
        ],
    },
    "dtypes": ["float32", "float16"],
    "tolerances": {"float32": 1e-3, "float16": 1e-2},
}


# One-slot artefact memo. Reference and tolerance each derive the artefact
# from the RAW weights (the anchoring property: a surface cannot hand the
# oracle an artefact that disagrees with the weights it passed), and a gate
# judges up to 32 cases per (shape, bits) against the same matrix - without
# reuse that is 64 canonical quantizations of a 389M-element matrix per
# lm_head group, whose transients the 2026-08-15 pricing instrumentation
# measured at 14-22 GB of ratcheted footprint.
#
# Keyed by IDENTITY, not by a content hash. The memo's strong reference keeps
# the array alive, so no later array can occupy its address and a hit can only
# ever be the same object - the anchoring property survives exactly as a hash
# would preserve it. Hashing would additionally cost a full copy of the weight
# bytes per case (`tobytes` always copies; 778 MB at lm_head in fp16), which is
# the transient this memo exists to avoid. Every caller holds one weight matrix
# across its case group and passes that same object in, so identity hits.
_qmm_memo: tuple | None = None


def _qmm_artefact(inputs):
    global _qmm_memo
    w, bits = inputs["w"], int(inputs["bits"][0])
    if _qmm_memo is not None and _qmm_memo[0] is w and _qmm_memo[1] == bits:
        return _qmm_memo[2]
    artefact = canonical_quantize(w, QuantContract(bits=bits, group_size=64))
    _qmm_memo = (w, bits, artefact)
    return artefact


def qmm_reference(inputs) -> np.ndarray:
    return r_contract(inputs["x"], _qmm_artefact(inputs))


def qmm_tolerance(case, inputs, ref) -> float:
    artefact = _qmm_artefact(inputs)
    floor = max(_max_err(fn(inputs["x"], artefact), ref)
                for fn in QUANT_ENSEMBLE.values())
    return max(_base_tol(case.dtype, ref), K_QUANT * floor)


WEIGHT_SCALE = 0.05


def _rescaled(arr: np.ndarray, target_peak: float) -> np.ndarray:
    peak = float(np.max(np.abs(arr.astype(np.float32)))) if arr.size else 0.0
    if peak == 0.0:
        return arr
    return (arr.astype(np.float32) * (target_peak / peak)).astype(arr.dtype)


def _as_weights(arr: np.ndarray) -> np.ndarray:
    """Rescale a generated tensor to a realistic weight magnitude.

    The input modes shape the ACTIVATIONS; a weight tensor drawn at the corpus
    range (uniform[-10,10]) is not an operating point any real model has, and
    at D_IN=1024 it overflows fp16 outright, which turns every fp16 case into
    a range failure rather than a fault measurement. The mode's structure
    (constant rows, opposed signs, near zero) is preserved by scaling; only the
    magnitude is made realistic. An all-zero tensor is left alone.
    """
    return _rescaled(arr, WEIGHT_SCALE)


def qmm_augment(case, inputs):
    inputs["bits"] = np.array([case.dim_map["BITS"]], dtype=np.int32)
    inputs["w"] = _as_weights(inputs["w"])
    return inputs


def moe_augment(case, inputs):
    inputs["router"] = _as_weights(inputs["router"])
    inputs["experts"] = _as_weights(inputs["experts"])
    return inputs


# ---------------------------------------------------------------------------
# moe_dispatch: Qwen3-class routing contract (softmax all, top-2 by prob,
# ties low, renormalize).
# ---------------------------------------------------------------------------
MOE_DISPATCH_META = {
    "op_schema": {
        "inputs": [
            {"name": "x", "dims": ["B", "D"]},
            {"name": "router", "dims": ["E", "D"]},
            {"name": "experts", "dims": ["E", "D_OUT", "D"]},
        ],
        "dims": [
            {"name": "B", "candidates": [1, 8]},
            {"name": "D", "candidates": [64, 100, 512]},
            {"name": "E", "candidates": [4, 16]},
            {"name": "D_OUT", "candidates": [32, 256]},
        ],
    },
    "dtypes": ["float32", "float16"],
    "tolerances": {"float32": 1e-3, "float16": 1e-2},
}


def moe_reference(inputs) -> np.ndarray:
    """The routing contract in fp64, sharing the kernel's tie-break helper so
    exact ties (every logit, on constant-rows inputs) resolve identically."""
    x = inputs["x"].astype(np.float64)
    router = inputs["router"].astype(np.float64)
    experts = inputs["experts"].astype(np.float64)
    logits = x @ router.T
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    top = _topk_by_prob(probs, TOP_K, tie_high=False)
    weights = np.take_along_axis(probs, top, axis=-1)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    out = np.zeros((x.shape[0], experts.shape[1]), dtype=np.float64)
    for row in range(x.shape[0]):
        for slot in range(TOP_K):
            out[row] += weights[row, slot] * (x[row] @ experts[int(top[row, slot])].T)
    return out


def _moe_member_reversed(inputs) -> np.ndarray:
    """Legitimate variant: expert contributions accumulated in reverse slot
    order at fp32 (a second member of the combine-order class)."""
    x = inputs["x"].astype(np.float32)
    router = inputs["router"].astype(np.float32)
    experts = inputs["experts"].astype(np.float32)
    logits = x @ router.T
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    top = _topk_by_prob(probs, TOP_K, tie_high=False)
    weights = np.take_along_axis(probs, top, axis=-1)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    out = np.zeros((x.shape[0], experts.shape[1]), dtype=np.float32)
    for row in range(x.shape[0]):
        for slot in reversed(range(TOP_K)):
            out[row] += weights[row, slot] * (x[row] @ experts[int(top[row, slot])].T)
    return out.astype(inputs["x"].dtype)


# Label -> member, mirroring KV_MEMBERS below; the labels feed the
# verdict-cache fingerprint, so they are part of the shipped identity.
MOE_MEMBERS = {
    "moe:default": moe_dispatch,
    "moe:reversed-slots": _moe_member_reversed,
}


def moe_tolerance(case, inputs, ref) -> float:
    floor = max(_max_err(fn(inputs), ref) for fn in MOE_MEMBERS.values())
    return max(_base_tol(case.dtype, ref), K_NATIVE * floor)


# ---------------------------------------------------------------------------
# kv_attention: one decode step over a contract-anchored quantized KV cache.
# ---------------------------------------------------------------------------
KV_ATTENTION_META = {
    "op_schema": {
        "inputs": [
            {"name": "q", "dims": ["B", "H", "DH"]},
            {"name": "k_cache", "dims": ["H", "T", "DH"]},
            {"name": "v_cache", "dims": ["H", "T", "DH"]},
            {"name": "new_k", "dims": ["B", "H", "DH"]},
            {"name": "new_v", "dims": ["B", "H", "DH"]},
        ],
        # The cache is shared across the batch (correctness simplification);
        # DH stays a whole number of 64-wide quantization groups; T=512 gives
        # the long-reduction regime the accumulator faults need.
        "dims": [
            {"name": "B", "candidates": [1, 4]},
            {"name": "H", "candidates": [2, 8]},
            {"name": "T", "candidates": [64, 512]},
            {"name": "DH", "candidates": [64, 128]},
            {"name": "BITS", "candidates": [4, 8]},
        ],
    },
    "dtypes": ["float32", "float16"],
    "tolerances": {"float32": 1e-3, "float16": 1e-2},
}

QUERY_SCALE = 0.6  # peak for q/new_k: puts score rms near 2, so softmax is
                   # neither uniform nor saturated at corpus input magnitudes


def kv_augment(case, inputs):
    inputs["bits"] = np.array([case.dim_map["BITS"]], dtype=np.int32)
    inputs["q"] = _rescaled(inputs["q"], QUERY_SCALE)
    inputs["new_k"] = _rescaled(inputs["new_k"], QUERY_SCALE)
    return inputs


# Two-slot memo, one per cache tensor of the case in hand (k and v): the
# reference and all four tolerance members dequantize the same two caches, so
# each is dequantized once per case instead of five times. Keyed by identity,
# with the strong reference keeping the keyed array alive, because the dequant
# is deterministic given (cache, bits).
_kv_dequant_memo: dict = {}


def _kv_dequant64(cache, bits):
    from kernelverify.reference.native_kernels import _cache_dequant
    key = (id(cache), int(bits))
    hit = _kv_dequant_memo.get(key)
    if hit is not None and hit[0] is cache:
        return hit[1]
    if len(_kv_dequant_memo) >= 2:  # a new case: the old case's caches are done
        _kv_dequant_memo.clear()
    value = _cache_dequant(cache, bits).astype(np.float64)
    _kv_dequant_memo[key] = (cache, value)
    return value


def kv_reference(inputs) -> np.ndarray:
    """fp64 truth, contract-anchored: the cached K/V are the canonical
    artefacts' dequantized values, the new entry arrives at full precision."""
    q = inputs["q"].astype(np.float64)
    bits = int(inputs["bits"][0])
    kd = _kv_dequant64(inputs["k_cache"], bits)
    vd = _kv_dequant64(inputs["v_cache"], bits)
    scale = 1.0 / np.sqrt(float(q.shape[-1]))
    scores = np.einsum("bhd,htd->bht", q, kd) * scale
    score_new = np.sum(q * inputs["new_k"].astype(np.float64), axis=-1) * scale
    scores = np.concatenate([scores, score_new[..., None]], axis=-1)
    shifted = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = np.einsum("bht,htd->bhd", probs[..., :-1], vd)
    return out + probs[..., -1][..., None] * inputs["new_v"].astype(np.float64)


def _kv_member(inputs, *, scores_order="pairwise", combine_order="pairwise"):
    """Legitimate fp32 implementations spanning two classes: the scores
    accumulation order and the combine accumulation order."""
    q = inputs["q"].astype(np.float32)
    bits = int(inputs["bits"][0])
    kd = _kv_dequant64(inputs["k_cache"], bits).astype(np.float32)
    vd = _kv_dequant64(inputs["v_cache"], bits).astype(np.float32)
    scale = np.float32(1.0 / np.sqrt(float(q.shape[-1])))
    if scores_order == "serial":
        prods = q[:, :, None, :] * kd[None, :, :, :]
        scores = np.add.accumulate(prods, axis=-1, dtype=np.float32)[..., -1] * scale
    else:
        scores = np.einsum("bhd,htd->bht", q, kd) * scale
    score_new = np.sum(q * inputs["new_k"].astype(np.float32), axis=-1) * scale
    scores = np.concatenate([scores, score_new[..., None]], axis=-1)
    shifted = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    pc = probs[..., :-1]
    if combine_order in ("serial", "reversed"):
        contrib = pc[..., None] * vd[None, ...]
        if combine_order == "reversed":
            contrib = contrib[:, :, ::-1, :]
        out = np.add.accumulate(contrib, axis=2, dtype=np.float32)[:, :, -1]
    else:
        out = np.einsum("bht,htd->bhd", pc, vd)
    out = out + probs[..., -1][..., None] * inputs["new_v"].astype(np.float32)
    return out.astype(inputs["q"].dtype)


KV_MEMBERS = {
    "kv:pairwise-pairwise": dict(scores_order="pairwise", combine_order="pairwise"),
    "kv:serial-scores": dict(scores_order="serial", combine_order="pairwise"),
    "kv:serial-combine": dict(scores_order="pairwise", combine_order="serial"),
    "kv:reversed-combine": dict(scores_order="pairwise", combine_order="reversed"),
}


def kv_tolerance(case, inputs, ref) -> float:
    """The kv_attention tolerance, whose K is BORROWED and uncalibrated.

    K_QUANT = 4.0 was derived by the quantized_matmul device calibration
    (ADR 0012, re-read by ADR 0016) over that operator's nine-name ensemble on
    that operator's grid. No harness has ever derived a K over KV_MEMBERS: the
    four members below have never been scored leave-one-out against a held-out
    implementation of this operator, at any shape or width.

    So ADR 0008's sentence about each family shipping "its own calibrated K"
    does not hold here, and this number is a reuse, not a measurement. It is
    not obviously wrong - both floors are ensembles of legitimate fp32
    reduction orders over a dequantized artefact - but nothing has measured
    whether 4.0 is loose, tight, or beside the point for an operator whose
    floor also carries a softmax. Calibrating it is queued in TODOS.md.
    """
    floor = max(_max_err(_kv_member(inputs, **kw), ref)
                for kw in KV_MEMBERS.values())
    return max(_base_tol(case.dtype, ref), K_QUANT * floor)


NATIVE_OPS = {
    "quantized_matmul": NativeOp(QUANTIZED_MATMUL_META, qmm_reference,
                                 qmm_tolerance, qmm_augment),
    "moe_dispatch": NativeOp(MOE_DISPATCH_META, moe_reference, moe_tolerance,
                             moe_augment),
    "kv_attention": NativeOp(KV_ATTENTION_META, kv_reference, kv_tolerance,
                             kv_augment),
}
