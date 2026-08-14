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


def reference_and_tolerance(op_name: str, inputs: dict,
                            dtype: str = "float16") -> tuple[np.ndarray, float]:
    """The shipped reference and tolerance for one case, straight from
    NATIVE_OPS. `dtype` is the candidate kernel's output dtype; every pack
    surface today emits float16."""
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
