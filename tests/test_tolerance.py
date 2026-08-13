"""The shipped conditioning-aware tolerance (ADR 0004).

The last test permanently pins the motivating case: constant-rows attention at
D=64, M=8, N=256, where a provably correct fp32 kernel exceeds the published
tolerance, an input-perturbation probe cannot see why, and the ensemble floor
covers it without absolving the fp16-score precision canary.
"""

import numpy as np
import pytest

from measure_escape import load_meta, max_abs_error, reference
from score_oracles import SWEEP_SEED, make_mode_inputs
from kernelverify.reference.kernels import attention, flash_attention
from kernelverify.tolerance.floor import (
    ENSEMBLES,
    K_ENSEMBLE,
    conditioned_tolerance,
    ensemble_floor,
)

RNG = np.random.default_rng(7)


def unit_inputs(op):
    spec = load_meta(op)["op_schema"]["inputs"]
    dims = {"B": 2, "S": 3, "H": 16, "M": 5, "N": 96, "K": 32, "D": 16}
    return {
        s["name"]: RNG.standard_normal([dims[d] for d in s["dims"]]).astype(np.float32)
        for s in spec
    }


@pytest.mark.parametrize("op", sorted(ENSEMBLES))
def test_ensemble_members_agree_on_well_conditioned_input(op):
    inputs = unit_inputs(op)
    outputs = [fn(inputs).astype(np.float64) for fn in ENSEMBLES[op]]
    for other in outputs[1:]:
        np.testing.assert_allclose(outputs[0], other, atol=1e-5)


def test_tolerance_stays_at_base_when_well_conditioned():
    meta = load_meta("gelu_triton")
    inputs = unit_inputs("gelu_triton")
    ref = ENSEMBLES["gelu_triton"][0](inputs).astype(np.float64)
    base = float(meta["tolerances"]["float32"])
    assert conditioned_tolerance("gelu_triton", inputs, ref, base) == base


def test_the_motivating_constant_rows_attention_case():
    op = "attention_triton"
    meta = load_meta(op)
    dims = {"D": 64, "M": 8, "N": 256}
    inputs = make_mode_inputs(meta, dims, "float32", SWEEP_SEED, "constant rows")
    key = (op, tuple(sorted(dims.items())), "float32", SWEEP_SEED, "constant rows")
    ref = reference(meta, inputs, key)
    base = float(meta["tolerances"]["float32"])

    control_err = max_abs_error(attention(inputs), ref)
    assert control_err > base, "the ill-conditioning this machinery exists for is gone"

    tol = conditioned_tolerance(op, inputs, ref, base)
    assert control_err <= tol, "the ensemble floor no longer covers the correct kernel"

    diverse_err = max_abs_error(
        flash_attention(inputs, block_n_cap=128), ref
    )
    assert diverse_err <= tol, "a held-out correct implementation false-positives"

    canary_err = max_abs_error(attention(inputs, scores_dtype="float16"), ref)
    assert canary_err > tol, "the tolerance absolved the fp16-score precision fault"

    # The falsified v1 design, pinned: a 1-ULP input probe of the fp64
    # reference moves it far less than the correct kernel's own rounding, so
    # no modest K on a probe floor can cover this case (ADR 0004).
    probe_floor = 0.0
    for sign, tag in ((1.0, "plus"), (-1.0, "minus")):
        probed = {
            name: arr.astype(np.float64) * (1.0 + sign * np.finfo(np.float32).eps)
            for name, arr in inputs.items()
        }
        moved = reference(meta, probed, key + (tag,))
        probe_floor = max(probe_floor, max_abs_error(moved, ref))
    assert control_err > 8 * probe_floor, (
        "the probe floor now covers internal accumulation error; "
        "revisit ADR 0004's two-tier design"
    )
    assert K_ENSEMBLE * ensemble_floor(op, inputs, ref) >= control_err