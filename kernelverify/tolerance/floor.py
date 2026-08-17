"""The shipped conditioning-aware tolerance: the working-precision ensemble floor.

Fixed per-operator tolerances are wrong in ill-conditioned regimes: on
constant-rows attention a provably correct fp32 kernel exceeds the published
1e-3 against the fp64 reference, because fp32 rounding of score intermediates
near magnitude 800 legitimately moves softmax outputs past it (ADR 0003).

The verdict a verifier can actually defend is therefore relative to what a
correct implementation could do on this exact case:

    floor(case) = max over an ensemble of provably-correct working-precision
                  implementations of |impl(inputs) - ref_fp64(inputs)|
    tolerance   = max(base_tol, K_ENSEMBLE * floor)

Who is in the ensemble
----------------------
ADR 0004 hand-wrote three members on the premise that sequential accumulation
is the worst legitimate summation order, so bounding it bounds the class. ADR
0005 measured that premise and it is false: seeded random permutations of the
same reduction beat that ensemble by up to 8.45x on the attention family, and
the member that sets the worst case is a permutation, never the
magnitude-sorted order that theory nominates.

So the ensemble is no longer a hand-written list. It is a prefix of the
admissible-implementation contract in `kernelverify/tolerance/contract.py`,
which states in full which implementations this verifier promises never to
flag. Measured over the whole battery with the floor built from one draw of
that population and scored against an independent draw (ADR 0005):

- a one-member ensemble is not enough: it needs K = 2.119 and ships a false
  positive at the shipped K;
- from two members on, the held-out class needs exactly K = 1.000, so K stops
  compensating for an unrepresentative ensemble and becomes pure margin;
- the residual under-coverage on near-binding cases falls 8.45x -> 3.20x
  between the three members ADR 0004 shipped and this budget of eight, and
  only reaches 2.00x by twenty-four, which is why eight is the knee;
- not one (case, fault) verdict in the battery moves, so every table in ADR
  0002 through 0004 stands unchanged.

K_ENSEMBLE = 1.5 is unchanged, and is now anchored rather than fitted. Over
322,200 admissible implementations evaluated across the battery, with zero
false positives: the class demanded at least 1.147 under the ADR 0004
ensemble, and demands exactly 1.000 under this one.

An input-perturbation probe is NOT a substitute for this floor: it measures
input conditioning only, misses internal accumulation error by up to 64x, and
probing at fp16 epsilon absolves fp16-internal arithmetic, the exact fault
class the verifier exists to reject (ADR 0004 records the falsification).
"""

from __future__ import annotations

import numpy as np

from kernelverify.tolerance.contract import OPERATORS, contract_ensemble

K_ENSEMBLE = 1.5

# How many contract members the shipped floor pays for per case. Chosen by the
# budget sweep in ADR 0005, not by taste: it is where the marginal member stops
# buying coverage of the class.
ENSEMBLE_BUDGET = 8

# Corpus operator -> the provably-correct working-precision ensemble, taken as
# a prefix of the contract population so the members are derived from the
# stated promise rather than chosen by hand.
ENSEMBLE_MEMBERS = {op: contract_ensemble(op, ENSEMBLE_BUDGET) for op in OPERATORS}
ENSEMBLES = {op: [fn for _, fn in members] for op, members in ENSEMBLE_MEMBERS.items()}


def ensemble_labels(op: str) -> list[str]:
    """The names of `op`'s ensemble members, for fingerprints and reports."""
    return [label for label, _ in ENSEMBLE_MEMBERS[op]]


def max_abs_error(candidate, ref) -> float:
    """The one metric every tolerance in the project is built on: fp64 max-abs
    deviation of `candidate` from `ref`, 0.0 on an empty array. The native
    operators' tolerances import it too, so the corpus and native floors can
    never measure two different things."""
    diff = np.abs(candidate.astype(np.float64) - ref.astype(np.float64))
    return float(np.max(diff)) if diff.size else 0.0


def floored_tolerance(base_tol: float, k: float, member_outputs, ref: np.ndarray) -> float:
    """The shipped verdict formula, spelled once for the corpus and every
    native operator family: max(base_tol, k * floor), where the floor is the
    worst max_abs_error of any ensemble member's output from fp64 truth."""
    return max(base_tol, k * max(max_abs_error(out, ref) for out in member_outputs))


def ensemble_floor(op: str, inputs: dict, ref: np.ndarray) -> float:
    """Worst deviation of any provably-correct implementation from fp64 truth."""
    return max(max_abs_error(fn(inputs), ref) for fn in ENSEMBLES[op])


def conditioned_tolerance(op: str, inputs: dict, ref: np.ndarray,
                          base_tol: float) -> float:
    """The shipped per-case tolerance: max(base_tol, K_ENSEMBLE * floor)."""
    return floored_tolerance(base_tol, K_ENSEMBLE, (fn(inputs) for fn in ENSEMBLES[op]), ref)
