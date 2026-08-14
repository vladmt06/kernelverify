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

from kernelverify.reference.native_kernels import _topk_by_prob, TOP_K
from kernelverify.schemas.quant_contract import (
    ENSEMBLE as QUANT_ENSEMBLE,
    QuantContract,
    canonical_quantize,
    r_contract,
)

K_QUANT = 3.0    # Phase 0 calibration (bench/phase0_contract_k.py)
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
        "dims": [
            {"name": "B", "candidates": [1, 4]},
            {"name": "D_IN", "candidates": [256, 448, 2048]},
            {"name": "D_OUT", "candidates": [16, 2048]},
            {"name": "BITS", "candidates": [2, 3, 4, 8]},
        ],
    },
    "dtypes": ["float32", "float16"],
    "tolerances": {"float32": 1e-3, "float16": 1e-2},
}


def _qmm_artefact(inputs):
    bits = int(inputs["bits"][0])
    return canonical_quantize(inputs["w"], QuantContract(bits=bits, group_size=64))


def qmm_reference(inputs) -> np.ndarray:
    return r_contract(inputs["x"], _qmm_artefact(inputs))


def qmm_tolerance(case, inputs, ref) -> float:
    artefact = _qmm_artefact(inputs)
    floor = max(_max_err(fn(inputs["x"], artefact), ref)
                for fn in QUANT_ENSEMBLE.values())
    return max(_base_tol(case.dtype, ref), K_QUANT * floor)


def qmm_augment(case, inputs):
    inputs["bits"] = np.array([case.dim_map["BITS"]], dtype=np.int32)
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
            {"name": "E", "candidates": [4, 64]},
            {"name": "D_OUT", "candidates": [32, 512]},
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


def moe_tolerance(case, inputs, ref) -> float:
    from kernelverify.reference.native_kernels import moe_dispatch
    members = (moe_dispatch(inputs), _moe_member_reversed(inputs))
    floor = max(_max_err(m, ref) for m in members)
    return max(_base_tol(case.dtype, ref), K_NATIVE * floor)


NATIVE_OPS = {
    "quantized_matmul": NativeOp(QUANTIZED_MATMUL_META, qmm_reference,
                                 qmm_tolerance, qmm_augment),
    "moe_dispatch": NativeOp(MOE_DISPATCH_META, moe_reference, moe_tolerance),
}
