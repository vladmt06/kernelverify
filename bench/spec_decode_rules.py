"""Pure decision arithmetic for the speculative-decode end-to-end run.

This module deliberately imports only the standard library so the registered
outcomes remain testable on machines where importing ``mlx.nn`` aborts.
"""

from __future__ import annotations

import math
import statistics

from attribution import composed_attribution
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Mapping, Sequence


K_GRID = (2, 4, 6, 8, 10)
PRIMARY_K = 6

_NULL_KS = frozenset({2, 10})
_PER_STEP_GAIN_PCT = {
    2: 0.0,
    4: 5.70,   # Binding A/B at M=5.
    6: 16.44,  # Meeting-point spike at M=7.
    8: 15.65,  # Binding M=8 stands in for the unmeasured M=9 cell.
    10: 0.0,
}
_IDENTITY_EXCLUSIONS = {
    "O1": frozenset({"mlx-m-dependent"}),
    "O2": frozenset({"kernel-diverged"}),
    "O3": frozenset({"kernel-diverged"}),
    # Ruled 2026-08-18 before any measurement, and amended into section 6:
    # O4 claims a user gets stock-4-bit speed from the 3-bit model with our
    # kernel, so a round where our kernel emitted different tokens timed
    # something other than what the claim is about. Expected cost is a RARE
    # round, not zero: our kernel is equal to stock within the contract
    # tolerance rather than bit-equal, and an argmax over near-tied logits can
    # flip on far less than that.
    "O4": frozenset({"kernel-diverged"}),
}


class RunInvalid(RuntimeError):
    """The registered run cannot produce a decision from these records."""


class ExactCountMismatch(RunInvalid):
    """Observed routed calls differ from the per-site expected count."""


class NoVerifyPasses(RunInvalid):
    """A speculative arm completed without an observed verification pass."""


@dataclass(frozen=True)
class RoundSample:
    """The decision inputs from one aligned five-arm round."""

    generation_tps: Mapping[int, float]
    identity: Mapping[str, int] = field(default_factory=dict)
    valid: bool = True


@dataclass(frozen=True)
class _Comparison:
    numerator_tps: float
    denominator_tps: float
    delta_pct: float
    noise_floor_pct: float
    rounds: tuple[RoundSample, ...]


def in_window(k: int) -> bool:
    """Whether the registered K presents an M = K + 1 routed width."""
    return 5 <= k + 1 <= 9


def expected_routed_calls(
    shapes: Sequence[Sequence[int]], routed_sites_at: Mapping[int, int]
) -> int:
    """Sum routed sites over every observed pass, by that pass's width.

    The shapes are TOKEN-IDENTIFIER shapes, not activation shapes: mlx_lm
    verifies with ``model(y[None], cache=cache)`` where ``y`` holds the K + 1
    candidate tokens, so the counted seam receives ``(1, K + 1)`` and the
    embedding happens below it. The width a routed projection then sees is
    every dimension of that array multiplied out (doc, section 4 amendment of
    2026-08-18). A rank-3 shape is refused rather than measured, because it
    means the seam moved and the old ``prod(shape[:-1])`` reading of it would
    report width 1 for every pass instead of failing.
    """
    widths = []
    for shape in shapes:
        if len(shape) != 2:
            raise RunInvalid(
                f"observed pass shape {tuple(shape)} is rank {len(shape)}; the "
                "counted seam takes rank-2 token identifiers"
            )
        widths.append(math.prod(shape))
    return sum(routed_sites_at[width] for width in widths)


def accepted_per_pass(generation_tokens: int, verify_passes: int) -> float:
    if verify_passes == 0:
        raise NoVerifyPasses("speculative arm observed zero verification passes")
    return generation_tokens / verify_passes


def _time_total(value: float | Sequence[float]) -> float:
    if isinstance(value, (int, float)):
        return value
    return sum(value)


def verify_share(
    verify_time: float | Sequence[float],
    generation_time: float | Sequence[float],
) -> float:
    return _time_total(verify_time) / _time_total(generation_time)


def ceiling_for(k: int, share: float) -> float:
    try:
        gain = _PER_STEP_GAIN_PCT[k]
    except KeyError as error:
        raise RunInvalid(f"K={k} is outside the registered grid") from error
    return share * gain


def is_decider(ceiling: float, floor: float) -> bool:
    return ceiling > floor


def spread_pct(samples: Sequence[float]) -> float:
    if not samples:
        raise RunInvalid("no eligible paired round")
    return 100 * (max(samples) - min(samples)) / statistics.median(samples)


def delta_pct(t_a: float, t_b: float) -> float:
    return 100 * (t_a / t_b - 1)


def noise_floor(samples_a: Sequence[float], samples_b: Sequence[float]) -> float:
    return max(spread_pct(samples_a), spread_pct(samples_b))


def assert_exact_count(observed: int, expected: int, cell: str) -> None:
    if observed != expected:
        raise ExactCountMismatch(
            f"{cell}: observed {observed} routed calls, expected {expected}"
        )


_MISSING = object()


def _first_mismatch(left: Sequence[object], right: Sequence[object]) -> int | None:
    for position, (a, b) in enumerate(
        zip_longest(left, right, fillvalue=_MISSING)
    ):
        if a is _MISSING or b is _MISSING or a != b:
            return position
    return None


def identity_labels(
    *,
    a1: Sequence[object],
    a2: Sequence[object],
    a0: Sequence[object],
    a4: Sequence[object],
) -> dict[str, int]:
    """Name the two registered token mismatches, or reject a bad control."""
    control_mismatch = _first_mismatch(a4, a2)
    if control_mismatch is not None:
        raise RunInvalid(
            f"arm 4 differs from arm 2 at token position {control_mismatch}"
        )

    labels = {}
    kernel_mismatch = _first_mismatch(a1, a2)
    if kernel_mismatch is not None:
        labels["kernel-diverged"] = kernel_mismatch
    mlx_mismatch = _first_mismatch(a2, a0)
    if mlx_mismatch is not None:
        labels["mlx-m-dependent"] = mlx_mismatch
    return labels


def eligible_rounds(
    rounds: Sequence[RoundSample], outcome: str
) -> tuple[RoundSample, ...]:
    """The rounds one outcome may read, after its own exclusions.

    Public because the harness needs the same set the verdicts read: section 5
    sums arm 2's verify and generation times over the O2-eligible rounds, and a
    harness that re-derived "valid, and not carrying this outcome's identity
    label" would be a second copy of the exclusion rule with nothing binding it
    to this one (doc, section 5 amendment of 2026-08-18).
    """
    excluded = _IDENTITY_EXCLUSIONS[outcome]
    eligible = tuple(
        round_sample
        for round_sample in rounds
        if round_sample.valid
        and excluded.isdisjoint(round_sample.identity)
    )
    if not eligible:
        raise RunInvalid(f"{outcome} has no eligible paired round")
    return eligible


def _comparison(
    rounds: Sequence[RoundSample], outcome: str, numerator: int, denominator: int
) -> _Comparison:
    eligible = eligible_rounds(rounds, outcome)
    samples_a = [r.generation_tps[numerator] for r in eligible]
    samples_b = [r.generation_tps[denominator] for r in eligible]
    t_a = statistics.median(samples_a)
    t_b = statistics.median(samples_b)
    return _Comparison(
        numerator_tps=t_a,
        denominator_tps=t_b,
        delta_pct=delta_pct(t_a, t_b),
        noise_floor_pct=noise_floor(samples_a, samples_b),
        rounds=eligible,
    )


def decide_o1(
    cells: Mapping[int, Sequence[RoundSample]],
) -> dict[str, object]:
    """Select stock speculation's best K, then compare arm 2 with arm 0."""
    comparisons = {
        k: _comparison(cells.get(k, ()), "O1", 2, 0)
        for k in K_GRID
    }
    primary_k = max(
        K_GRID,
        key=lambda k: (comparisons[k].numerator_tps, -k),
    )
    comparison = comparisons[primary_k]
    if comparison.delta_pct > comparison.noise_floor_pct:
        verdict = "WIN"
    elif comparison.delta_pct < -comparison.noise_floor_pct:
        verdict = "DECELERATES"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "primary_k": primary_k,
        "arm2_tps": comparison.numerator_tps,
        "arm0_tps": comparison.denominator_tps,
        "delta_pct": comparison.delta_pct,
        "noise_floor_pct": comparison.noise_floor_pct,
        "eligible_rounds": len(comparison.rounds),
        "verdict": verdict,
    }


def decide_o2(
    k: int,
    rounds: Sequence[RoundSample],
    *,
    ceiling: float,
    routed_calls: int,
) -> dict[str, object]:
    """Read arm 1 against arm 2 at one registered K cell."""
    comparison = _comparison(rounds, "O2", 1, 2)
    result = {
        "k": k,
        "arm1_tps": comparison.numerator_tps,
        "arm2_tps": comparison.denominator_tps,
        "delta_pct": comparison.delta_pct,
        "noise_floor_pct": comparison.noise_floor_pct,
        "ceiling_pct": ceiling,
        "eligible_rounds": len(comparison.rounds),
    }

    if k in _NULL_KS:
        if routed_calls != 0 or abs(comparison.delta_pct) > comparison.noise_floor_pct:
            raise RunInvalid(
                f"K={k} mandatory null control had {routed_calls} routed calls "
                f"and delta {comparison.delta_pct}% at floor "
                f"{comparison.noise_floor_pct}%"
            )
        return {**result, "decider": False, "verdict": "NULL"}

    decider = is_decider(ceiling, comparison.noise_floor_pct)
    if not decider:
        return {**result, "decider": False, "verdict": "not-a-decider"}
    if comparison.delta_pct > comparison.noise_floor_pct:
        verdict = "WIN"
    elif comparison.delta_pct < -comparison.noise_floor_pct:
        verdict = "LOSS"
    else:
        verdict = "NULL"
    return {**result, "decider": True, "verdict": verdict}




def decide_o3(
    cells: Mapping[int, Sequence[RoundSample]],
) -> dict[str, object]:
    """Read the fixed K=6 product number and the labelled grid maximum."""
    # The PRIMARY cell is computed first and on its own. The exploratory
    # maximum below is labelled "never the quoted number", so it must not be
    # able to take the quoted number down with it: a K cell whose rounds were
    # all excluded by the identity rule is skipped there rather than raising.
    primary = _comparison(cells.get(PRIMARY_K, ()), "O3", 1, 0)
    comparisons = {}
    for k in K_GRID:
        try:
            comparisons[k] = _comparison(cells.get(k, ()), "O3", 1, 0)
        except RunInvalid:
            continue
    arm2_tps = statistics.median(
        r.generation_tps[2] for r in primary.rounds
    )
    speculation_pct = delta_pct(arm2_tps, primary.denominator_tps)
    kernel_pct = delta_pct(primary.numerator_tps, arm2_tps)
    exploratory_delta = max(c.delta_pct for c in comparisons.values())
    exploratory_ks = tuple(
        k for k in K_GRID if comparisons[k].delta_pct == exploratory_delta
    )
    return {
        "primary_k": PRIMARY_K,
        "arm1_tps": primary.numerator_tps,
        "arm2_tps": arm2_tps,
        "arm0_tps": primary.denominator_tps,
        "composed_pct": primary.delta_pct,
        "speculation_pct": speculation_pct,
        "kernel_pct": kernel_pct,
        "noise_floor_pct": primary.noise_floor_pct,
        "eligible_rounds": len(primary.rounds),
        "attribution": composed_attribution(
            primary.delta_pct, speculation_pct, primary.noise_floor_pct
        ),
        "exploratory": {
            "delta_pct": exploratory_delta,
            "ks": exploratory_ks,
            "label": "exploratory, selection-biased",
        },
    }


def decide_o4(
    cells: Mapping[int, Sequence[RoundSample]],
) -> dict[str, object]:
    """Compare speculative 3-bit arm 1 with speculative stock 4-bit arm 3."""
    comparison = _comparison(cells.get(PRIMARY_K, ()), "O4", 1, 3)
    arm2_tps = statistics.median(
        r.generation_tps[2] for r in comparison.rounds
    )
    base_pct = delta_pct(arm2_tps, comparison.denominator_tps)
    return {
        "primary_k": PRIMARY_K,
        "arm1_tps": comparison.numerator_tps,
        "arm2_tps": arm2_tps,
        "arm3_tps": comparison.denominator_tps,
        "ratio_composed_vs_4bit": (
            comparison.numerator_tps / comparison.denominator_tps
        ),
        "composed_pct": comparison.delta_pct,
        "base_pct": base_pct,
        "noise_floor_pct": comparison.noise_floor_pct,
        "eligible_rounds": len(comparison.rounds),
        "attribution": composed_attribution(
            comparison.delta_pct, base_pct, comparison.noise_floor_pct
        ),
    }
