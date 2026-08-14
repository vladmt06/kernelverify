"""The admissible-implementation contract and the K anchored to it (ADR 0005).

Two things need pinning here, and they pull in opposite directions.

Every member of the contract population must be a correct implementation,
because the whole anchor is the claim that the worst deviation measured across
that population is a deviation the verifier promised to tolerate. A buggy
member would inflate the floor and read as "K must be raised" when the truth
is "that member does not belong".

Nothing outside the contract may creep in, because a floor that covers an
inadmissible implementation is a tolerance that absolves a fault. The three
exclusions the contract names - narrow intermediates, unshifted exponential
sums, alternative formulas - each have a fault in the catalogue standing on
them, so each is tested by showing that fault is still caught.
"""

import numpy as np
import pytest

from measure_escape import load_meta, max_abs_error, reference
from score_oracles import SWEEP_SEED, make_mode_inputs
from kernelverify.reference.kernels import attention, gelu, softmax
from kernelverify.tolerance.contract import (
    OPERATORS,
    contract_ensemble,
    contract_implementations,
)
from kernelverify.tolerance.floor import (
    ENSEMBLE_BUDGET,
    ENSEMBLES,
    K_ENSEMBLE,
    conditioned_tolerance,
    ensemble_floor,
    ensemble_labels,
)

RNG = np.random.default_rng(11)

# Well-conditioned shapes: every dimension modest, no degenerate reduction, so
# any correct implementation should agree with the fp64 reference to near
# working precision and a disagreement means a broken member.
WELL_CONDITIONED = {"B": 2, "S": 3, "H": 16, "M": 5, "N": 96, "K": 32, "D": 16}


def unit_inputs(op):
    spec = load_meta(op)["op_schema"]["inputs"]
    return {
        s["name"]: RNG.standard_normal(
            [WELL_CONDITIONED[d] for d in s["dims"]]
        ).astype(np.float32)
        for s in spec
    }


@pytest.mark.parametrize("op", OPERATORS)
def test_every_contract_member_is_a_correct_implementation(op):
    """The anchor is only as good as the population's correctness."""
    meta = load_meta(op)
    inputs = unit_inputs(op)
    ref = reference(meta, inputs, ("contract-correctness", op))
    base = float(meta["tolerances"]["float32"])
    for label, fn in contract_implementations(op, n_random=4):
        err = max_abs_error(fn(inputs), ref)
        assert err <= base, (
            f"{op} contract member {label} is not a correct implementation: "
            f"error {err:.3g} exceeds the base tolerance {base:.3g} on a "
            "well-conditioned case"
        )


@pytest.mark.parametrize("op", OPERATORS)
def test_the_population_is_deterministic(op):
    """A calibration nobody can rerun is not a calibration."""
    first = contract_implementations(op, n_random=4)
    second = contract_implementations(op, n_random=4)
    assert [label for label, _ in first] == [label for label, _ in second]

    inputs = unit_inputs(op)
    for (_, a), (_, b) in zip(first, second):
        np.testing.assert_array_equal(a(inputs), b(inputs))


@pytest.mark.parametrize("op", OPERATORS)
def test_the_shipped_ensemble_is_a_contract_prefix(op):
    """The ensemble is derived from the stated promise, not chosen by hand."""
    prefix = contract_ensemble(op, ENSEMBLE_BUDGET)
    assert ensemble_labels(op) == [label for label, _ in prefix]
    assert len(ENSEMBLES[op]) == min(
        ENSEMBLE_BUDGET, len(contract_implementations(op, n_random=ENSEMBLE_BUDGET))
    )


def test_the_reference_kernel_leads_every_ensemble():
    """A one-member floor must be the status quo ante, or the sweep is not nested."""
    for op in OPERATORS:
        inputs = unit_inputs(op)
        ref = reference(load_meta(op), inputs, ("lead", op))
        lead = ENSEMBLES[op][0]
        assert max_abs_error(lead(inputs), ref) <= float(
            load_meta(op)["tolerances"]["float32"]
        )


# ---------------------------------------------------------------------------
# The exclusions. Each names a fault that must survive the anchored tolerance.
# ---------------------------------------------------------------------------
def _attention_constant_rows():
    op = "attention_triton"
    meta = load_meta(op)
    dims = {"D": 64, "M": 8, "N": 256}
    inputs = make_mode_inputs(meta, dims, "float32", SWEEP_SEED, "constant rows")
    key = (op, tuple(sorted(dims.items())), "float32", SWEEP_SEED, "constant rows")
    return op, meta, inputs, reference(meta, inputs, key)


def test_c1_excludes_narrow_intermediates_so_the_canary_stays_caught():
    """fp16 score matrix: out of contract by C1, and still detected."""
    op, meta, inputs, ref = _attention_constant_rows()
    tol = conditioned_tolerance(op, inputs, ref, float(meta["tolerances"]["float32"]))
    canary = max_abs_error(attention(inputs, scores_dtype="float16"), ref)
    assert canary > tol, "the anchored tolerance absolved the fp16-score fault"


def test_c4_excludes_unshifted_exponential_sums():
    """Admitting an unshifted sum would make the class unbounded in error.

    At the corpus's own uniform[-10,10] an unshifted softmax is numerically
    harmless, which is exactly why the catalogue classes it an equivalent
    mutant. The clause is not about that regime. It is about the one where the
    sum overflows: there the shifted class stays exact and an unshifted
    implementation returns nothing at all, so a contract that admitted it
    would have no finite worst case to take a maximum over.
    """
    for op in ("softmax_triton", "attention_triton", "flash_attention_triton"):
        labels = [label for label, _ in contract_implementations(op, n_random=32)]
        assert not any("unshifted-exp" in label for label in labels)

    op = "softmax_triton"
    meta = load_meta(op)
    # Positive and large: exp overflows binary32 above about 88.
    rows = (RNG.random((2, 7, 256)) * 200.0).astype(np.float32)
    inputs = {"input": rows}
    ref = reference(meta, inputs, ("c4-overflow", op))

    base = float(meta["tolerances"]["float32"])
    for label, fn in contract_implementations(op, n_random=4):
        err = max_abs_error(fn(inputs), ref)
        assert err <= base, f"shifted member {label} broke at overflow scale: {err:.3g}"

    with np.errstate(over="ignore", invalid="ignore"):
        unshifted = softmax(inputs, subtract_max=False)
    assert not np.all(np.isfinite(unshifted.astype(np.float64))), (
        "the overflow regime this clause exists for is gone; C4 needs restating"
    )


def test_c5_excludes_alternative_formulas():
    """The erf GELU is a correct function and an inadmissible GELU."""
    op = "gelu_triton"
    meta = load_meta(op)
    inputs = unit_inputs(op)
    ref = reference(meta, inputs, ("c5", op))
    tol = conditioned_tolerance(op, inputs, ref, float(meta["tolerances"]["float32"]))
    assert max_abs_error(gelu(inputs, variant="erf"), ref) > tol


# ---------------------------------------------------------------------------
# The anchor itself
# ---------------------------------------------------------------------------
def test_k_is_anchored_on_the_case_that_binds():
    """ADR 0005's binding case: the class demands exactly 1.000 here.

    This is the one case in 4,060 where the floor, not the base tolerance,
    decides the verdict. If a future change makes the shipped ensemble
    under-cover it again, K goes back to compensating for the ensemble rather
    than being margin over it, and the anchor is gone.
    """
    op, meta, inputs, ref = _attention_constant_rows()
    base = float(meta["tolerances"]["float32"])

    floor = ensemble_floor(op, inputs, ref)
    assert K_ENSEMBLE * floor > base, "the case no longer binds; ADR 0005 needs rerunning"

    worst = max(max_abs_error(fn(inputs), ref)
                for _, fn in contract_implementations(op, n_random=24))
    assert worst <= floor, (
        f"the shipped ensemble under-covers its own contract here by "
        f"{worst / floor:.3f}x, so K is absorbing ensemble error again"
    )
    assert worst <= K_ENSEMBLE * floor
