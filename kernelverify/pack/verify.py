"""One verification path for every pack surface.

Every kernel in the pack is gated by the battery's own verdict: NATIVE_OPS
carries the shipped fp64 reference and the conditioning-aware tolerance for
each operator family, and this module is the only route between a pack
surface and a pass/fail claim. No pack test or bench gate states its own
tolerance.

The reason this is load-bearing rather than tidy: the verdict formula is
still evolving (K recalibration, ensemble membership, possible accum-dtype
enforcement), and a hand-rolled copy in a test or bench gate keeps applying
the OLD formula, silently, from the day the shipped one changes. Routing
through NATIVE_OPS makes that drift impossible: the pack is judged by
exactly the code that judges the battery.

Two call shapes. A bench gate needs the reference before the kernel runs,
because cases are batched through the crash-isolated runner first and
judged after:

    inputs = qmv_inputs(x, w, bits)            # or moe_inputs(...)
    ref, tol = reference_and_tolerance("quantized_matmul", inputs)
    ...run the kernel...
    verdict = judge(candidate, ref, tol)

A test that already holds the candidate uses the one-step form:

    verdict = verify_output("quantized_matmul", inputs, candidate)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from kernelverify.pack.evidence import CaseEvidence, output_fingerprint
from kernelverify.schemas.native_ops import NATIVE_OPS
from kernelverify.schemas.quant_contract import QuantArtefact, dequantize


@dataclass(frozen=True)
class _Case:
    """Stand-in for a battery case: dtype is the one field the NATIVE_OPS
    tolerance callables read."""

    dtype: str


@dataclass(frozen=True)
class Verdict:
    ok: bool
    err: float
    tol: float

    def __str__(self) -> str:
        return (f"err {self.err:9.3e}  tol {self.tol:9.3e}  "
                f"{'pass' if self.ok else 'FAIL'}")


def qmv_inputs(x: np.ndarray, w: np.ndarray, bits: int) -> dict:
    """Native-op inputs for a quantized matvec/matmul surface.

    Takes the ORIGINAL weights, not the artefact: the shipped operator
    derives the artefact itself through the canonical quantizer, so a
    surface cannot hand the oracle an artefact that disagrees with the one
    it packed for the kernel.
    """
    return {"x": x, "w": w, "bits": np.array([bits], dtype=np.int32)}


def moe_inputs(x: np.ndarray, router: np.ndarray,
               expert_artefacts: Sequence[QuantArtefact]) -> dict:
    """Native-op inputs for the MoE dispatch surface.

    Once the artefact is fixed, a quantized expert IS a dense expert whose
    entries happen to be s*q + b, so the shipped dense-expert contract
    judges a quantized kernel by substituting the exact dequantized weights.
    """
    experts = np.stack([dequantize(a, np.float64) for a in expert_artefacts])
    return {"x": x, "router": router, "experts": experts}


def kv_inputs(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray,
              new_k: np.ndarray, new_v: np.ndarray, bits: int) -> dict:
    """Native-op inputs for the quantized-KV attention decode surface.

    Takes the RAW float caches, not the packed artefacts: the shipped
    operator canonically quantizes the cache itself, so a surface cannot
    hand the oracle a cache that disagrees with the artefact it packed for
    the kernel. Same anchoring as `qmv_inputs`.
    """
    return {"q": q, "k_cache": k_cache, "v_cache": v_cache,
            "new_k": new_k, "new_v": new_v,
            "bits": np.array([bits], dtype=np.int32)}


def group_heads(x: np.ndarray, n_kv_heads: int) -> np.ndarray:
    """(B, HQ, T, DH) -> (B, HKV, GQA, T, DH), the layout the training-attention
    reference reads.

    The grouping is not a presentation choice. Query head h reads key-value head
    h // GQA, which is the mapping MLX's own quantized attention spells out when
    it reshapes queries to (B, n_kv_heads, n_repeats, L, D), and holding it in
    the shape is what lets the group ratio be a swept dimension rather than a
    constant baked into one geometry.
    """
    batch, n_q_heads, t_len, head_dim = x.shape
    if n_kv_heads < 1 or n_q_heads % n_kv_heads:
        raise ValueError(f"{n_q_heads} query heads do not group into "
                         f"{n_kv_heads} key-value heads")
    return x.reshape(batch, n_kv_heads, n_q_heads // n_kv_heads, t_len, head_dim)


def attn_inputs(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                d_out: np.ndarray | None = None) -> dict:
    """Native-op inputs for the training-attention surface.

    Takes the arrays in the layout the KERNEL sees, (B, HQ, T, DH) queries
    against (B, HKV, T, DH) keys and values, and groups the query heads for the
    reference. `d_out` is the cotangent a backward is judged against and is
    absent for a forward-only case, which is why it is optional rather than a
    separate inputs helper.

    Values arrive as float32 carrying exactly what the device stored: bfloat16
    is the dtype the seam actually receives and numpy cannot hold it, so a
    caller converts on the device, where the widening is exact.
    """
    inputs = {"q": group_heads(q, k.shape[1]), "k": k, "v": v}
    if d_out is not None:
        inputs["d_out"] = group_heads(d_out, k.shape[1])
    return inputs


def reference_and_tolerance(op_name: str, inputs: dict,
                            dtype: str = "float16") -> tuple[np.ndarray, float]:
    """The shipped reference and tolerance for one case, straight from
    NATIVE_OPS. `dtype` is the candidate kernel's output dtype: the decode
    surfaces emit float16, and the training-attention surface emits bfloat16
    too, whose eps is eight times float16's and whose tolerance is therefore a
    different number on the same case."""
    op = NATIVE_OPS[op_name]
    ref = op.reference(inputs)
    return ref, float(op.tolerance(_Case(dtype), inputs, ref))


def judge(candidate, ref: np.ndarray, tol: float) -> Verdict:
    """Max-abs error of the candidate against the reference, at fp64."""
    err = float(np.max(np.abs(np.asarray(candidate).astype(np.float64) - ref)))
    return Verdict(ok=err <= tol, err=err, tol=tol)


def verify_output(op_name: str, inputs: dict, candidate,
                  dtype: str = "float16") -> Verdict:
    """One-step gate: reference, tolerance, and verdict for one case."""
    ref, tol = reference_and_tolerance(op_name, inputs, dtype)
    return judge(candidate, ref, tol)


def judge_case(spec_ev, result, label: str, ref: np.ndarray, tol: float,
               aux: dict | None = None,
               input_sha256: dict | None = None) -> Verdict | None:
    """One crash-isolated runner result, judged into gate evidence.

    The bench gates batch their cases through the runner first and judge
    afterwards; this is that judging step, shared so the evidence a refused
    runner leaves and the fields a judged case carries cannot drift between
    gates. A runner that did not finish is recorded as a failed case with the
    runner's own status, and None is returned in place of a verdict.

    `input_sha256` is the per-input fingerprint a gate that drops verified
    input arrays must leave behind (ruling D2): it is recorded on the passing
    AND the failing path, since a failed case's arrays are only meaningful
    beside the hashes of what went in.
    """
    fingerprints = {"input_sha256": input_sha256} if input_sha256 else {}
    if not result.ok:
        spec_ev.cases.append(CaseEvidence(
            label=label, passed=False, tol=tol, **fingerprints,
            detail=f"runner {result.status.value}: {result.detail}"))
        return None
    v = judge(result.outputs[0], ref, tol)
    spec_ev.cases.append(CaseEvidence(
        label=label, passed=v.ok, err=v.err, tol=v.tol,
        output_sha256=output_fingerprint(result.outputs[0]),
        **fingerprints,
        **({"aux": aux} if aux is not None else {})))
    return v
