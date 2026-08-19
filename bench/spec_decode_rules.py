"""Pure decision arithmetic for the speculative-decode end-to-end run.

It imports the standard library and two CPU-only siblings, ``attribution``
and ``decode_rules``, and nothing else, so the registered outcomes stay
testable on machines where importing ``mlx.nn`` aborts the interpreter.
The siblings own the shared attribution vocabulary and generic decode rules.
A test walks the import graph transitively and allows ``decode_rules`` to
reach only the stdlib-only ``machine_state`` sibling.
"""

from __future__ import annotations

import math
import statistics

from attribution import composed_attribution
from decode_rules import (
    Comparison,
    ExactCountMismatch,
    RoundSample,
    RunInvalid,
    assert_exact_count,
    classify_passes,
    comparison,
    delta_pct,
    eligible_rounds as select_eligible_rounds,
    first_mismatch,
    is_decider,
    noise_floor,
    result_line,
    spread_pct,
)
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


class NoVerifyPasses(RunInvalid):
    """A speculative arm completed without an observed verification pass."""


def in_window(k: int) -> bool:
    """Whether the registered K presents an M = K + 1 routed width."""
    return 5 <= k + 1 <= 9


def _token_width(shape: Sequence[int]) -> int:
    """The width of one observed target call at the token-identifier seam."""
    if len(shape) != 2:
        raise RunInvalid(
            f"observed pass shape {tuple(shape)} is rank {len(shape)}; the "
            "counted seam takes rank-2 token identifiers"
        )
    return math.prod(shape)


def verification_passes(
    shapes: Sequence[Sequence[int]], *, k: int, prompt_t: int
) -> list[tuple[int, ...]]:
    """The target calls that are verification passes, in order.

    A pass is a call whose width is at most K + 1; a wider call is prefill
    (doc, section 4 amendment of 2026-08-18). The separation is asserted
    rather than trusted: at most one call may be wider than K + 1 and it must
    be the single width-(PROMPT_T - 1) call `_prefill` makes, or the seam has
    moved and the cell is invalid. A final pass shorter than K + 1 is still a
    pass, which is the case the exact count has to survive.
    """
    passes, _ = classify_passes(
        shapes,
        is_pass=lambda width: width <= k + 1,
        prefill_width=prompt_t - 1,
    )
    return list(passes)


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
    return sum(routed_sites_at[_token_width(shape)] for shape in shapes)


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


def identity_labels(
    *,
    a1: Sequence[object],
    a2: Sequence[object],
    a0: Sequence[object],
    a4: Sequence[object],
) -> dict[str, int]:
    """Name the two registered token mismatches, or reject a bad control."""
    control_mismatch = first_mismatch(a4, a2)
    if control_mismatch is not None:
        raise RunInvalid(
            f"arm 4 differs from arm 2 at token position {control_mismatch}"
        )

    labels = {}
    kernel_mismatch = first_mismatch(a1, a2)
    if kernel_mismatch is not None:
        labels["kernel-diverged"] = kernel_mismatch
    mlx_mismatch = first_mismatch(a2, a0)
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
    return select_eligible_rounds(
        rounds,
        _IDENTITY_EXCLUSIONS[outcome],
        what=outcome,
    )


def _comparison(
    rounds: Sequence[RoundSample], outcome: str, numerator: int, denominator: int
) -> Comparison:
    return comparison(
        rounds,
        numerator,
        denominator,
        excluded_labels=_IDENTITY_EXCLUSIONS[outcome],
        what=outcome,
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
    routed_sites_at_width: int | None = None,
) -> dict[str, object]:
    """Read arm 1 against arm 2 at one registered K cell.

    ``routed_sites_at_width`` is the routing table's site count at this cell's
    verification width K + 1, and is required at the two null cells: their
    control is a claim about the TABLE, not about the observed count, which a
    short final pass can legitimately make non-zero (doc, section 4 amendment
    of 2026-08-18).
    """
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
        if routed_sites_at_width is None:
            raise RunInvalid(
                f"K={k} is a null control and needs the routing table's site "
                f"count at width {k + 1} to be read"
            )
        if routed_sites_at_width != 0:
            raise RunInvalid(
                f"K={k} mandatory null control: the routing table routes at "
                f"{routed_sites_at_width} sites at width {k + 1}, and the "
                "registration says it routes at none"
            )
        if routed_calls:
            # A short final pass landed inside the routed window, which is
            # where max_tokens fell rather than anything the interception did.
            # Section 4's exact-count assertion already bound this number; the
            # cell simply stops being a control for this run.
            return {**result, "decider": False, "routed_calls": routed_calls,
                    "verdict": "NULL-uncontrolled"}
        if abs(comparison.delta_pct) > comparison.noise_floor_pct:
            raise RunInvalid(
                f"K={k} mandatory null control routed nothing yet had delta "
                f"{comparison.delta_pct}% at floor "
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




def _require_registered_primary(primary_k: int) -> None:
    """Refuse a primary cell the grid does not register.

    The primary cell defaults to the parent pre-registration's K = 6 and may
    be named otherwise only by a pre-registration that fixes it before the
    run, which is what the K = 4 follow-up does (docs/research/
    2026-08-18-spec-decode-k4-followup.md, section 3). Off the grid there are
    no rounds to read, and reading one anyway reports a confident verdict on
    a cell nothing registered.
    """
    if primary_k not in K_GRID:
        raise RunInvalid(f"primary cell K={primary_k} is outside the grid")


def decide_o3(
    cells: Mapping[int, Sequence[RoundSample]],
    *,
    primary_k: int = PRIMARY_K,
) -> dict[str, object]:
    """Read the registered primary cell against plain decode.

    The labelled grid maximum is reported beside it and is never the
    quoted number.
    """
    _require_registered_primary(primary_k)
    # The PRIMARY cell is computed first and on its own. The exploratory
    # maximum below is labelled "never the quoted number", so it must not be
    # able to take the quoted number down with it: a K cell whose rounds were
    # all excluded by the identity rule is skipped there rather than raising.
    primary = _comparison(cells.get(primary_k, ()), "O3", 1, 0)
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
    # Over the cells that HAVE a comparison, not over K_GRID: the loop above
    # skips a cell whose rounds were all excluded, so indexing K_GRID here
    # raised KeyError rather than RunInvalid, and main() catches only the
    # latter - the exploratory report would have taken the run down with a
    # traceback at the verdict stage, after every GPU minute was spent.
    exploratory_ks = tuple(
        k for k in K_GRID
        if k in comparisons and comparisons[k].delta_pct == exploratory_delta
    )
    return {
        "primary_k": primary_k,
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
    *,
    primary_k: int = PRIMARY_K,
) -> dict[str, object]:
    """Compare speculative 3-bit arm 1 with speculative stock 4-bit arm 3."""
    _require_registered_primary(primary_k)
    comparison = _comparison(cells.get(primary_k, ()), "O4", 1, 3)
    arm2_tps = statistics.median(
        r.generation_tps[2] for r in comparison.rounds
    )
    base_pct = delta_pct(arm2_tps, comparison.denominator_tps)
    return {
        "primary_k": primary_k,
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
