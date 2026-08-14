"""Native operators: references with seams for ops the corpus never had.

Tranche 1.5 (eng review decision 3A). These follow the same discipline as
kernels.py - correct by default, keyword seams introduce faults - but their
ground truth is ours: an in-process fp64 reference in native_ops.py,
cross-checked against MLX's own ops by tests, instead of a corpus subprocess.

quantized_matmul is contract-anchored: its correct answer depends on the
quantization artefact (ADR 0003 Experiment 2), so its seams include artefact
faults (the misalignment class no input axis can reach) alongside evaluation
faults like fp16 intermediate dequant, which Phase 0 measured outside the
contract tolerance on ~half of all cases.

moe_dispatch pins the routing contract Qwen3-class models use (softmax over
all logits, top-k, renormalize); its seams are the divergences that really
ship: renormalization dropped, expert index off by one, fp16 router, tie-break
direction (which expresses exactly on constant-rows inputs, where every logit
ties), and one predicted-equivalent entry (softmax-after-top-k is
mathematically identical to renormalization) to keep testing the exclusion
machinery.
"""

from __future__ import annotations

import numpy as np

from kernelverify.schemas.quant_contract import (
    QuantContract,
    canonical_quantize,
    dequantize,
    fault_bias_dropped,
    fault_group_size_halved,
    fault_nibble_swapped,
    fault_scales_rotated,
)


# ---------------------------------------------------------------------------
# Quantized matmul
# ---------------------------------------------------------------------------
def quantized_matmul(inputs, *, scales_rotated=False, nibble_swapped=False,
                     bias_dropped=False, group_size_halved=False,
                     dequant_dtype="float32", accum_dtype="float32"):
    """x @ dequant(quantize(w)).T with the canonical MLX-affine contract.

    Correct by default: canonical quantization of `w`, dequantized at fp32,
    numpy-pairwise MAC. Artefact seams corrupt the quantization artefact the
    way a compiler or packer would; `dequant_dtype` "float16" is the
    intermediate-precision fault the contract explicitly forbids.
    `accum_dtype` "float16" is the OTHER C1 violation - the MAC accumulator
    held in half precision - modelled as sequential fp16 accumulation of fp16
    products, the exact fault the pack's kernel-side probe showed escaping the
    shipped tolerance at typical fp16 operating points.
    """
    x, w = inputs["x"], inputs["w"]
    bits = int(inputs["bits"][0])
    artefact = canonical_quantize(w, QuantContract(bits=bits, group_size=64))
    if scales_rotated:
        artefact = fault_scales_rotated(artefact)
    if nibble_swapped:
        artefact = fault_nibble_swapped(artefact)
    if bias_dropped:
        artefact = fault_bias_dropped(artefact)
    if group_size_halved:
        artefact = fault_group_size_halved(artefact)
    wd = dequantize(artefact, np.float32)
    if dequant_dtype != "float32":
        wd = wd.astype(dequant_dtype).astype(np.float32)
    if accum_dtype in ("float16", "float16-seq"):
        products = (x.astype(np.float16)[:, None, :]
                    * wd.astype(np.float16)[None, :, :])
        if accum_dtype == "float16-seq":
            # Sequential left-to-right fp16 adds: the naive
            # one-thread-per-output kernel shape. Error grows with D and the
            # battery separates it broadly, even at fp16 activations.
            out = np.add.accumulate(products, axis=-1, dtype=np.float16)[..., -1]
        else:
            # Pairwise (tree) fp16 adds: the simdgroup-reduction shape. Error
            # grows with the tree depth only and measured 0/280 separable at
            # fp16 activations - the structural-attestation class.
            out = np.add.reduce(products, axis=-1, dtype=np.float16)
        return out.astype(x.dtype)
    return (x.astype(np.float32) @ wd.T).astype(x.dtype)


# ---------------------------------------------------------------------------
# MoE dispatch
# ---------------------------------------------------------------------------
TOP_K = 2


def _topk_by_prob(probs: np.ndarray, k: int, tie_high: bool) -> np.ndarray:
    """Indices of the k largest entries per row, ties to the LOWER index by
    default (the convention shared with the fp64 reference so that exact ties,
    which constant-rows inputs produce on every logit, resolve identically).
    """
    order = np.argsort(-probs, axis=-1, kind="stable")
    if tie_high:
        rev = np.argsort(-probs[:, ::-1], axis=-1, kind="stable")
        order = probs.shape[-1] - 1 - rev
    return order[:, :k]


def moe_dispatch(inputs, *, renormalize=True, expert_offset=0,
                 gate_dtype="float32", k_offset=0, tie_high=False,
                 softmax_topk_order=False):
    """Top-2 mixture-of-experts routing and combine, Qwen3-class contract.

    Correct path: logits = x @ router.T in fp32; softmax over ALL experts;
    top-k by probability (ties to lower index); renormalize the top-k weights;
    output = sum of weight_e * (x @ expert_e.T).

    `softmax_topk_order` computes softmax only over the selected logits
    instead of renormalizing the full softmax - predicted equivalent
    (exp(l_i)/sum_topk exp(l_j) either way) and included to test exclusion.
    """
    x, router, experts = inputs["x"], inputs["router"], inputs["experts"]
    xf = x.astype(np.float32)
    logits = xf @ router.astype(np.float32).T
    if gate_dtype != "float32":
        logits = logits.astype(gate_dtype).astype(np.float32)

    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)

    k = TOP_K + k_offset
    top = _topk_by_prob(probs, k, tie_high)

    if softmax_topk_order:
        chosen_logits = np.take_along_axis(logits, top, axis=-1)
        cs = chosen_logits - chosen_logits.max(axis=-1, keepdims=True)
        weights = np.exp(cs)
        weights /= weights.sum(axis=-1, keepdims=True)
    else:
        weights = np.take_along_axis(probs, top, axis=-1)
        if renormalize:
            weights = weights / weights.sum(axis=-1, keepdims=True)

    n_experts = experts.shape[0]
    out = np.zeros((x.shape[0], experts.shape[1]), dtype=np.float32)
    for row in range(x.shape[0]):
        for slot in range(k):
            expert = (int(top[row, slot]) + expert_offset) % n_experts
            out[row] += weights[row, slot] * (
                xf[row] @ experts[expert].astype(np.float32).T)
    return out.astype(x.dtype)


NATIVE_KERNELS = {
    "quantized_matmul": quantized_matmul,
    "moe_dispatch": moe_dispatch,
}
