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

import weakref
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
from kernelverify.tolerance.floor import floored_tolerance

# Re-derived at 4.0 by the device-arithmetic calibration (ADR 0012): a correct
# device kernel (simdgroup-factored) legitimately exceeds the CPU-calibrated
# K=3 tolerance by up to 1.30x on constant-rows float32, and the joined
# membership demands 3.901. Measured adequate at 4.0: zero false positives on
# both draws, worst fault margin 161x, so detection is unchanged.
K_QUANT = 4.0
K_NATIVE = 1.5   # unquantized ensembles, same value ADR 0004 measured

# One ulp at 1.0 per dtype, by the convention 2**-(stored mantissa bits):
# float32 keeps 23, float16 keeps 10, bfloat16 keeps 7. bfloat16 is the
# training dtype (ADR 0019) and its eps is 60x float16's, so a tolerance
# written for float16 is not a bfloat16 tolerance.
_EPS = {"float16": 9.77e-4, "float32": 1.19e-7, "bfloat16": 2.0 ** -7}


def _base_tol(dtype: str, ref: np.ndarray) -> float:
    scale = float(np.max(np.abs(ref))) if ref.size else 0.0
    return 4.0 * _EPS[dtype] * scale


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
# Keyed by IDENTITY, not by a content hash. The slot holds a WEAK reference to
# the weight matrix whose callback clears the slot the moment the caller drops
# the matrix, so no later array can occupy a dead referent's address and be
# mistaken for it: a hit can only ever be the same, still-live object, and the
# anchoring property survives exactly as a hash would preserve it. Hashing would
# additionally cost a full copy of the weight bytes per case (`tobytes` always
# copies; 778 MB at lm_head in fp16), which is the transient this memo exists
# to avoid. Every caller holds one weight matrix across its case group and
# passes that same object in, so identity hits; and once the caller moves on,
# the weights and their artefact (~2.3 GB at lm_head) go with it instead of
# sitting resident under whatever runs next in the process.
_qmm_memo: tuple | None = None


def _forget_qmm_memo(_ref):
    global _qmm_memo
    _qmm_memo = None


def _qmm_artefact(inputs):
    global _qmm_memo
    w, bits = inputs["w"], int(inputs["bits"][0])
    if _qmm_memo is not None and _qmm_memo[0]() is w and _qmm_memo[1] == bits:
        return _qmm_memo[2]
    artefact = canonical_quantize(w, QuantContract(bits=bits, group_size=64))
    _qmm_memo = (weakref.ref(w, _forget_qmm_memo), bits, artefact)
    return artefact


def qmm_reference(inputs) -> np.ndarray:
    return r_contract(inputs["x"], _qmm_artefact(inputs))


def qmm_tolerance(case, inputs, ref) -> float:
    artefact = _qmm_artefact(inputs)
    return floored_tolerance(_base_tol(case.dtype, ref), K_QUANT,
                             (fn(inputs["x"], artefact) for fn in QUANT_ENSEMBLE.values()),
                             ref)


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


def _moe_route(inputs, dtype):
    """Softmax over all experts, top-k by probability (ties low, through the
    kernel's tie-break helper), renormalise: the routing contract at `dtype`.
    Returns (x cast to dtype, top indices, renormalised top-k weights)."""
    x = inputs["x"].astype(dtype)
    router = inputs["router"].astype(dtype)
    logits = x @ router.T
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    top = _topk_by_prob(probs, TOP_K, tie_high=False)
    weights = np.take_along_axis(probs, top, axis=-1)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return x, top, weights


def _moe_combine(x, experts, top, weights, slots, dtype):
    """Accumulate weight * (x @ expert.T) per row over `slots`, in that order.
    `slots` is walked once PER ROW, so it must be a re-iterable sequence (a
    range), never an iterator such as reversed(range(...))."""
    out = np.zeros((x.shape[0], experts.shape[1]), dtype=dtype)
    for row in range(x.shape[0]):
        for slot in slots:
            out[row] += weights[row, slot] * (x[row] @ experts[int(top[row, slot])].T)
    return out


def moe_reference(inputs) -> np.ndarray:
    """The routing contract in fp64, sharing the kernel's tie-break helper so
    exact ties (every logit, on constant-rows inputs) resolve identically."""
    x, top, weights = _moe_route(inputs, np.float64)
    experts = inputs["experts"].astype(np.float64)
    return _moe_combine(x, experts, top, weights, range(TOP_K), np.float64)


def _moe_member_reversed(inputs) -> np.ndarray:
    """Legitimate variant: expert contributions accumulated in reverse slot
    order at fp32 (a second member of the combine-order class)."""
    x, top, weights = _moe_route(inputs, np.float32)
    experts = inputs["experts"].astype(np.float32)
    out = _moe_combine(x, experts, top, weights, range(TOP_K - 1, -1, -1), np.float32)
    return out.astype(inputs["x"].dtype)


# Label -> member, mirroring KV_MEMBERS below; the labels feed the
# verdict-cache fingerprint, so they are part of the shipped identity.
MOE_MEMBERS = {
    "moe:default": moe_dispatch,
    "moe:reversed-slots": _moe_member_reversed,
}

# Bumped whenever a member's ARITHMETIC changes under an unchanged name, the
# way QUANT_ENSEMBLE_VERSION was for the factored-groups repair: the verdict
# cache keys on labels, so a same-name change is exactly what it cannot see.
MOE_ENSEMBLE_VERSION = "moe-ensemble-v1"


def moe_tolerance(case, inputs, ref) -> float:
    return floored_tolerance(_base_tol(case.dtype, ref), K_NATIVE,
                             (fn(inputs) for fn in MOE_MEMBERS.values()), ref)


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

# Same discipline as MOE_ENSEMBLE_VERSION above. This ensemble shares
# `_kv_member` and `_cache_dequant` with the reference, so a change to either
# moves every member at once under four unchanged names (the MoE ensemble
# shares `_moe_route` and `_moe_combine` with its reference the same way).
KV_ENSEMBLE_VERSION = "kv-ensemble-v1"


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
    return floored_tolerance(_base_tol(case.dtype, ref), K_QUANT,
                             (_kv_member(inputs, **kw) for kw in KV_MEMBERS.values()),
                             ref)


# ---------------------------------------------------------------------------
# train_attention: causal grouped-query self-attention at TRAINING shape,
# forward and backward.
#
# `kv_attention` above is the decode sibling: one query position against a
# quantized cache. Training is the other regime, and it differs in every way
# that matters to a kernel. Every position attends at once, the mask is
# causal rather than absent, nothing is quantized, and the operator carries a
# BACKWARD. The backward is where the opportunity is: MLX fuses this forward
# in a plain call and decomposes it into matmuls and a softmax inside any
# gradient trace, and implements no fused attention backward at all.
#
# Layout: query heads are held GROUPED under the key-value head they share,
# so q is (B, HKV, GQA, T, DH) against k and v of (B, HKV, T, DH). That is
# the structure grouped-query attention actually has, and holding it in the
# shape lets the schema sweep the group ratio as its own dimension. A kernel
# that indexes the group wrongly is silently correct at a ratio of 1 and
# wrong at every other, so the ratio has to be a dimension rather than a
# constant baked into one shape.
#
# The scale is 1/sqrt(DH), which is what `mlx_lm.models.qwen3.Attention`
# computes as `head_dim**-0.5` and passes to the seam this operator replaces.
# ---------------------------------------------------------------------------
TRAIN_ATTENTION_META = {
    "op_schema": {
        "inputs": [
            {"name": "q", "dims": ["B", "HKV", "GQA", "T", "DH"]},
            {"name": "k", "dims": ["B", "HKV", "T", "DH"]},
            {"name": "v", "dims": ["B", "HKV", "T", "DH"]},
        ],
        # Shapes are chosen for fault expression, not for speed realism. T=65
        # is the edge case a tiled kernel gets wrong: one row past a 64-wide
        # tile, so the last tile is a single position and the causal bound
        # falls inside a tile rather than on its boundary. The group ratios
        # are 2, 4 and 8 because a group-indexing fault is invisible at 1.
        "dims": [
            {"name": "B", "candidates": [1, 2]},
            {"name": "HKV", "candidates": [1, 2]},
            {"name": "GQA", "candidates": [2, 4, 8]},
            {"name": "T", "candidates": [65, 128]},
            {"name": "DH", "candidates": [64, 128]},
        ],
    },
    "dtypes": ["float32", "float16"],
    "tolerances": {"float32": 1e-3, "float16": 1e-2},
}

# Peak the queries are rescaled to, the same job QUERY_SCALE does for the
# decode operator: at the corpus range on both operands a score reaches
# several hundred and the softmax is a one-hot at every case, which measures
# nothing an attention kernel can get wrong. Keys and values stay at corpus
# scale, so the score rms lands near 2 whatever DH is.
ATTN_QUERY_SCALE = 0.6

# The flash tile widths the ensemble spans. The kernel's own key-loop width
# is a tuned knob, and a different width is a different rescaling schedule
# and therefore a different rounding, so the floor has to cover more than one
# of them or it would be a floor for one knob setting.
ATTN_TILES = (16, 32)


def _causal_mask(t_q: int, t_k: int) -> np.ndarray:
    """Position i may read key j only where j <= i. Square in training: the
    step attends over its own width, with no cache in front of it."""
    return np.tril(np.ones((t_q, t_k), dtype=bool))


def _attn_scores(inputs, dtype) -> np.ndarray:
    """Masked, scaled scores at `dtype`. Masked entries are -inf, which is
    exactly what a correct kernel's masked lane holds before its softmax."""
    q = inputs["q"].astype(dtype)
    k = inputs["k"].astype(dtype)
    scale = dtype(1.0 / np.sqrt(float(q.shape[-1])))
    scores = np.einsum("bhgid,bhjd->bhgij", q, k) * scale
    return np.where(_causal_mask(q.shape[-2], k.shape[-2]), scores, dtype(-np.inf))


def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    return probs / probs.sum(axis=-1, keepdims=True)


def attn_reference(inputs) -> np.ndarray:
    """fp64 truth for the forward: softmax over the causal row, then the
    value combine, with the group's queries each reading their own row."""
    probs = _softmax_rows(_attn_scores(inputs, np.float64))
    return np.einsum("bhgij,bhjd->bhgid", probs, inputs["v"].astype(np.float64))


def attn_lse_reference(inputs) -> np.ndarray:
    """fp64 truth for the row logsumexp the forward hands to its backward.

    A flash forward computes this on its way through the key loop and the
    backward rebuilds the probabilities from it, so it is a shipped output
    with its own correctness claim rather than an implementation detail. Its
    verdict is separate because an L that is wrong by a constant per row
    leaves the forward's own output exactly right: the constant cancels in
    the normalisation, and only the backward would ever see it.
    """
    scores = _attn_scores(inputs, np.float64)
    row_max = scores.max(axis=-1)
    return row_max + np.log(np.exp(scores - row_max[..., None]).sum(axis=-1))


def _online_combine(scores: np.ndarray, v: np.ndarray, tile: int) -> np.ndarray:
    """The flash order: one pass over key tiles, carrying a running maximum
    and a running normaliser, rescaling the accumulator whenever the maximum
    moves. Never materialises a normalised probability matrix."""
    b, h, g, t_q, _t_k = scores.shape
    dh = v.shape[-1]
    running_max = np.full((b, h, g, t_q), -np.inf, dtype=np.float32)
    running_sum = np.zeros((b, h, g, t_q), dtype=np.float32)
    acc = np.zeros((b, h, g, t_q, dh), dtype=np.float32)
    for start in range(0, scores.shape[-1], tile):
        block = scores[..., start:start + tile]
        new_max = np.maximum(running_max, block.max(axis=-1))
        # A row whose running maximum is still -inf has seen nothing yet, and
        # -inf minus -inf is a NaN rather than the zero the algebra wants.
        safe_max = np.where(np.isfinite(new_max), new_max, np.float32(0.0))
        correction = np.where(np.isfinite(running_max),
                              np.exp(running_max - safe_max), np.float32(0.0))
        probs = np.exp(block - safe_max[..., None])
        running_sum = running_sum * correction + probs.sum(axis=-1)
        acc = acc * correction[..., None] + np.einsum(
            "bhgij,bhjd->bhgid", probs, v[:, :, start:start + tile, :])
        running_max = new_max
    return acc / running_sum[..., None]


def _attn_member(inputs, *, order="standard", tile=32) -> np.ndarray:
    """Legitimate fp32 implementations of the same forward.

    `standard` builds the whole normalised row and combines it; `online` is
    the tiled rescaling order a flash kernel runs, at two tile widths;
    `reversed-keys` walks the key axis backwards, which is the same sum in a
    different associativity. Every one is a correct implementation, so the
    worst of their deviations from fp64 is the floor no correct kernel should
    be judged below.
    """
    scores = _attn_scores(inputs, np.float32)
    v = inputs["v"].astype(np.float32)
    if order == "online":
        out = _online_combine(scores, v, tile)
    else:
        probs = _softmax_rows(scores)
        if order == "reversed-keys":
            out = np.einsum("bhgij,bhjd->bhgid", probs[..., ::-1], v[:, :, ::-1, :])
        else:
            out = np.einsum("bhgij,bhjd->bhgid", probs, v)
    return out.astype(inputs["q"].dtype)


ATTN_MEMBERS = {
    "attn:standard": dict(order="standard"),
    "attn:online-16": dict(order="online", tile=ATTN_TILES[0]),
    "attn:online-32": dict(order="online", tile=ATTN_TILES[1]),
    "attn:reversed-keys": dict(order="reversed-keys"),
}

# Same discipline as KV_ENSEMBLE_VERSION, and it is load-bearing in the same
# place: these members share `_attn_scores` with the reference, so a change to
# the scoring moves every member at once under four unchanged names, and only a
# version bump says so. `kernelverify/battery/core.py::_oracle_member_labels`
# reads it into the verdict-cache fingerprint, which is what makes a bump
# rebuild the cached verdicts rather than leave them scored under the old
# arithmetic.
ATTN_ENSEMBLE_VERSION = "attn-ensemble-v1"


def attn_tolerance(case, inputs, ref) -> float:
    """The train_attention forward tolerance, whose K is BORROWED.

    K_NATIVE = 1.5 is ADR 0004's unquantized value, measured over the corpus
    operators' ensembles on the corpus grid. No harness has ever derived a K
    over ATTN_MEMBERS: these four have never been scored leave-one-out
    against a held-out implementation of this operator, at any shape, group
    ratio or width. The reuse is defensible in kind, since both floors are
    ensembles of legitimate fp32 reduction orders over unquantized operands,
    and it is still a reuse rather than a measurement, exactly as
    `kv_tolerance` records for its own borrow. Calibrating it is queued in
    TODOS.md.
    """
    return floored_tolerance(_base_tol(case.dtype, ref), K_NATIVE,
                             (_attn_member(inputs, **kw) for kw in ATTN_MEMBERS.values()),
                             ref)


def attn_augment(case, inputs):
    """Put the softmax in the regime a kernel can be wrong in.

    At the corpus range on both operands a score reaches several hundred, so
    every row is a one-hot and the value combine is a copy: a kernel could
    drop most of its arithmetic and pass. Rescaling the queries alone puts
    the score rms near 2 at every DH, because the 1/sqrt(DH) scale cancels
    the reduction's own growth. Keys and values stay where the mode put them.
    """
    inputs["q"] = _rescaled(inputs["q"], ATTN_QUERY_SCALE)
    return inputs


# ---------------------------------------------------------------------------
# The backward. Not a NATIVE_OPS entry of its own: it is the same operator's
# other half, judged through the same battery machinery by `verify.py`, with
# the cotangent carried in the inputs dict beside the primals.
# ---------------------------------------------------------------------------
def _attn_grad(inputs, dtype, *, reversed_order=False) -> dict:
    """The closed-form vjp of the forward above, at `dtype`.

    Written out rather than differentiated by a tool, because this IS the
    reference: a rewrite-class kernel computes its own backward and there is
    no stock backward to compare against (Amendment 19 clause 78). With
    P the causal softmax and O = P V and D the row sum of dO * O:

        dV = P^T dO                 summed over the group's queries
        dS = P * (dO V^T - D)
        dQ = scale * dS K
        dK = scale * dS^T Q         summed over the group's queries

    `reversed_order` reverses each contraction's own axis, which is the same
    sum in a different associativity and is what makes the fp32 pair an
    ensemble rather than one implementation run twice.
    """
    q = inputs["q"].astype(dtype)
    k = inputs["k"].astype(dtype)
    v = inputs["v"].astype(dtype)
    d_out = inputs["d_out"].astype(dtype)
    scale = dtype(1.0 / np.sqrt(float(q.shape[-1])))

    probs = _softmax_rows(_attn_scores(inputs, dtype))
    out = np.einsum("bhgij,bhjd->bhgid", probs, v)

    def contract(subscripts, left, right, axis_left, axis_right):
        if not reversed_order:
            return np.einsum(subscripts, left, right)
        flip_l = [slice(None)] * left.ndim
        flip_r = [slice(None)] * right.ndim
        flip_l[axis_left] = slice(None, None, -1)
        flip_r[axis_right] = slice(None, None, -1)
        return np.einsum(subscripts, left[tuple(flip_l)], right[tuple(flip_r)])

    # dV and dK sum over BOTH the group's members and the query positions:
    # every query in a group reads the same key-value head, so its gradient
    # lands on that one head. A kernel that forgets the group sum passes a
    # ratio-1 test and fails everywhere else.
    d_v = contract("bhgij,bhgid->bhjd", probs, d_out, 3, 3)
    d_probs = contract("bhgid,bhjd->bhgij", d_out, v, 4, 3)
    row = np.sum(d_out * out, axis=-1)
    d_scores = probs * (d_probs - row[..., None])
    d_q = contract("bhgij,bhjd->bhgid", d_scores, k, 4, 2) * scale
    d_k = contract("bhgij,bhgid->bhjd", d_scores, q, 3, 3) * scale
    return {"dq": d_q, "dk": d_k, "dv": d_v}


def attn_grad_reference(inputs) -> dict:
    """fp64 truth for the three gradients, keyed by the primal they belong to."""
    return _attn_grad(inputs, np.float64)


def _attn_grad_member(inputs, *, reversed_order=False) -> dict:
    grads = _attn_grad(inputs, np.float32, reversed_order=reversed_order)
    dtype = inputs["q"].dtype
    return {name: value.astype(dtype) for name, value in grads.items()}


ATTN_GRAD_MEMBERS = {
    "attn-grad:pairwise": dict(reversed_order=False),
    "attn-grad:reversed": dict(reversed_order=True),
}


def attn_grad_tolerances(case, inputs, refs) -> dict:
    """One tolerance per gradient, floored by the same ensemble rule as the
    forward and carrying the same borrowed K. Three separate floors rather
    than one: dQ, dK and dV have different magnitudes and different
    reductions, and a single tolerance would be the loosest of the three
    applied to all of them.
    """
    members = [_attn_grad_member(inputs, **kw) for kw in ATTN_GRAD_MEMBERS.values()]
    return {name: floored_tolerance(_base_tol(case.dtype, refs[name]), K_NATIVE,
                                    (member[name] for member in members), refs[name])
            for name in refs}


NATIVE_OPS = {
    "quantized_matmul": NativeOp(QUANTIZED_MATMUL_META, qmm_reference,
                                 qmm_tolerance, qmm_augment),
    "moe_dispatch": NativeOp(MOE_DISPATCH_META, moe_reference, moe_tolerance,
                             moe_augment),
    "kv_attention": NativeOp(KV_ATTENTION_META, kv_reference, kv_tolerance,
                             kv_augment),
    "train_attention": NativeOp(TRAIN_ATTENTION_META, attn_reference,
                                attn_tolerance, attn_augment),
}
