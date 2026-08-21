"""Pure decision arithmetic for the Day 1 stock training profile.

Every constant here names the pre-registration section that fixed it, and
nothing in this module measures anything: it takes numbers a harness recorded
and applies rules that were written down before the harness ran. That split is
the point - the rules can be read, argued with and tested without a GPU, and
the harness cannot quietly become the thing that decides.

What this module certifies, and what it only reports
----------------------------------------------------
Amendment 6's clause 31 registered eight sites at which a rule takes an action,
and each certified action costs a margin that has to survive the whole box its
own measurements describe. Amendment 7 keeps three of them and demotes five,
because only three bear on which operation the sprint builds first:

  CERTIFIED   clause 24's shipping floor, clause 16's per-width pair, and
              clause 16's two-point boundary. Every one of these reads
              candidates L and Q and nothing else.
  REPORTED    clause 27's two exclusion routes, clause 23's kill rule and
              section 4.3's retained order. Each is still computed, still
              recorded with the readings it rests on, and reaches no terminal.
  RETIRED     clause 15's dial price, because clause 35 names the dial by rule
              and there is no price comparison left to make.

A REPORTED quantity is not a weaker certification and must never be written as
one. It also is not harmless: it reaches no terminal and it can still REFUSE
the run, because a share that is not a fraction proves the instrument is wrong
whichever candidate it was measured on.

Everything here works in the SAVING domain rather than the gain domain, which
is clause 31's doing. A gain is a ratio of two uncertain numbers and its
uncertainty is not a fraction of anything; a saving `K = M * (1 - F/N)` is a
time, its bounds are exact over the box, and a gain of 1.10 IS a saving of
`T/11`. Scores are still reported, because a score is what a reader wants, and
no rule reads one.

The document is docs/research/2026-08-19-metalrunner-sprint1-prereg.md,
sections 3, 4 and 5, as amended by Amendments 4 through 9.
"""

from __future__ import annotations

import math
import statistics

from dataclasses import dataclass
from typing import Mapping, Sequence

from decode_rules import RunInvalid, spread_pct

# Section 3.2. The four cells, and which one decides.
CELLS = {
    "A": {"batch": 1, "supervision": "masked"},
    "B": {"batch": 4, "supervision": "masked"},
    "C": {"batch": 1, "supervision": "all-tokens"},
    "D": {"batch": 4, "supervision": "all-tokens"},
}
PRIMARY_CELL = "B"
SECONDARY_CELL = "C"

# Section 3.2. Fixed across every cell.
SEQ_LEN = 2048
LORA_RANK = 8

# Section 3.2, "the adapter layer count mlx-lm defaults to". It decides how
# much of the step has a backward at all: mlx-lm adapts the LAST this-many
# blocks, and every block below the first adapted one has no trainable
# parameter beneath it and no backward. A profile that assumed a backward per
# layer would overstate every share built on one.
LORA_LAYERS = 16

# Section 3.2 plus Amendment 5 clause 7. The SIX distinct projection shapes
# the model contains, as (d_out, d_in). Section 3.2 listed five and the model
# has six: the key and value projections at (1024, 2560), eight key-value
# heads at head dimension 128, are counted in candidate Q's share and were
# omitted from the registered list, so a floor over five shapes would credit a
# ratio measured on part of the operation and divide it into a share measured
# on all of it.
SHAPES = {
    "S1": (4096, 2560),
    "S2": (2560, 4096),
    "S3": (9728, 2560),
    "S4": (2560, 9728),
    "S5": (151936, 2560),
    "S6": (1024, 2560),
}

# Amendment 5 clause 14 and clause 20. TWO registered widths, both drawn from
# one corpus so that width is the only thing separating them, and both derived
# by the band rule at the pinned revision rather than chosen. `SEQ_LEN` above
# is the cap mlx-lm is given and this corpus never approaches it: mlx-lm pads
# each batch to one plus the next multiple of 32 above its own longest row, so
# the band edge is what actually sets a step's width.
#
# `batch_width` is the padded width mlx-lm produces, and `operation_width` is
# what every matmul in the step actually sees, because `default_loss` trains on
# `batch[:, :-1]`. The two are carried apart rather than one derived at each
# call site, since a floor measured at the padded width would be measuring a
# shape the step never produces while agreeing about everything else.
WIDTHS = {
    "short": {"band": 64, "batch_width": 65, "operation_width": 64,
              "data": "ultrachat-64"},
    "long": {"band": 1056, "batch_width": 1057, "operation_width": 1056,
             "data": "ultrachat-1056"},
}

# Amendment 6 clause 26 and Amendment 7 clause 35: candidate A's dial fits at
# the long width and not at the short, so the deciding cell measures both and
# candidate A appears at one of them. The order is the order every rule-facing
# table is serialised in, so two artifacts can be compared row by row.
WIDTH_ORDER = ("short", "long")

# Section 4.3, which is R10's shipping floor read as a gain.
GAIN_FLOOR = 1.10

# Section 4.3. Gain read as a percentage; two points of it is a tie.
TIE_GAIN = 0.02

# Section 4.2, in the order the table lists them, which is the last tie-break.
CANDIDATES = ("L", "A", "Q")

# A credited ratio is weighted per direction, because a region's forward and
# backward call counts genuinely differ under LoRA.
DIRECTIONS = ("forward", "backward")

# Section 5 as restated by Amendment 5 clause 23. Q is at the ceiling if stock
# is within this ratio of the dense fp16 ceiling at ALL BUT ONE shape.
#
# The count is 5 of 6 and not 4 of 6. Section 5 registered "4 of the 5 shapes",
# which is all but one; inheriting the number 4 once S6 exists would have
# loosened the rule from four-fifths to two-thirds without saying so, so the
# FRACTION is what carries forward and the number moves with the shape count.
KILL_RATIO = 1.10
KILL_SHAPES = len(SHAPES) - 1

# Amendment 7 clause 34. Which sites take a certified margin and which are
# measured, published and consulted by no rule that reaches a terminal. Held
# as data so the decision artifact can state it rather than a reader having to
# infer it from which function was called.
CERTIFIED_SITES = {
    "clause 24, the shipping floor":
        "it decides whether building anything at all is worth the sprint",
    "clause 16, the per-width pair":
        "this comparison IS the choice between candidates L and Q",
    "clause 16, the two-point boundary":
        "band membership is part of that same choice",
}
REPORTED_SITES = {
    "clause 27, route 1":
        "its only action was excluding candidate A, which clause 35 stops "
        "gating",
    "clause 27, route 2": "the same",
    "clause 23, the kill rule":
        "clause 36: its verdict removes no candidate, and a candidate Q at "
        "the ceiling everywhere fails the shipping floor without it",
    "section 4.3, the retained order":
        "it sets the Day 3 build order of the candidates the sprint does not "
        "build first, which no measurement this run takes has to settle",
}


def gain(share: float, ratio: float) -> float:
    """Amdahl's relation for one accelerated part, section 4.1.

        gain = 1 / (1 - f * (1 - 1/r))

    Refuses rather than returning a number for inputs the formula does not
    describe: a share outside [0, 1] is not a share, and a ratio at or below
    zero is not a ratio. A denominator at or below zero means the part being
    replaced is the whole step and infinitely fast, which no measurement can
    produce and which would otherwise come back as a negative gain that reads
    like a slowdown.
    """
    # STRICT at the upper end, which committed code was not and clause 27
    # registers as step 8's obligation. A share of exactly 1 is reachable
    # through clause 26's conditions - a slope equal to the whole step clears
    # every excursion condition - and it says the candidate is the entire step
    # with nothing left for the adapters, the norms, the elementwise work and
    # the optimizer, which no step can be. Committed code returned `r` there
    # and raised nothing.
    if not 0.0 <= share < 1.0:
        raise RunInvalid(f"share {share} is not a fraction of the step")
    if ratio <= 0.0:
        raise RunInvalid(f"ratio {ratio} is not a ratio")
    denominator = 1.0 - share * (1.0 - 1.0 / ratio)
    if denominator <= 0.0:
        raise RunInvalid(
            f"share {share} at ratio {ratio} leaves no step to divide by; "
            f"the formula does not describe this input")
    return 1.0 / denominator


def ratio_lo(numerator: Sequence[float], denominator: Sequence[float]) -> float:
    """The worst pairing the samples permit, section 4.2.

    Smallest numerator over largest denominator, the way the pricing probe
    computes its own `ratio_lo`, so a candidate is credited with the least its
    measurements support rather than their midpoint.
    """
    if not numerator or not denominator:
        raise RunInvalid("a credited ratio needs samples on both sides")
    smallest, largest = min(numerator), max(denominator)
    if largest <= 0.0:
        raise RunInvalid("a measured floor of zero cannot bound a ratio")
    return smallest / largest


def collapse_ratio_lo(per_shape: Mapping[str, Mapping[str, object]]) -> dict:
    """One credited ratio out of many shapes, weighted per direction.

    A candidate's floor is measured shape by shape, but the operation it would
    replace is the whole set of them, and the shapes do not appear equally
    often: the quantized matmul runs seven times per layer across six distinct
    shapes and the output head runs once per step. A ratio taken over shapes
    without weighting would credit a shape that runs once as heavily as one
    that runs 252 times.

    Forward and backward are weighted SEPARATELY, and that is not a
    refinement. Measured on 2026-08-20 inside a real LoRA step, the quantized
    projections ran 196 times forward and 25 times backward, because the
    blocks below the first adapted one have no backward at all. One count
    cannot describe both, and a single count applied to samples that already
    combine the two directions produces a number that is not a bound on
    anything.

    The worst pairing is preserved at the aggregate, exactly as `ratio_lo`
    preserves it per shape: the weighted sum of the SMALLEST numerator each
    shape and direction produced, over the weighted sum of the LARGEST
    denominator. A candidate is therefore credited with the least its
    measurements support, and no shape's optimistic sample can be paired with
    another shape's pessimistic one to manufacture headroom.

    Counts come from the instrument's own per-shape, per-direction call counts
    rather than from the model's declared geometry, because a share and a
    ratio that disagree about how often an operation ran are two readings of
    different workloads. A direction with no calls contributes nothing and is
    not an error: that is what a region absent from the backward looks like.

    Each shape carries `{"forward": {...}, "backward": {...}}`, and each of
    those carries `count`, `numerator` and `denominator`.
    """
    if not per_shape:
        raise RunInvalid("a collapsed ratio needs at least one shape")
    numerator_total = 0.0
    denominator_total = 0.0
    contributions: dict[str, dict] = {}
    for shape, entry in sorted(per_shape.items()):
        unknown = set(entry) - set(DIRECTIONS)
        if unknown:
            raise RunInvalid(
                f"shape {shape} carries {sorted(unknown)}, and a ratio is "
                f"weighted per direction: expected {list(DIRECTIONS)}")
        per_direction = {}
        for direction in DIRECTIONS:
            side = entry.get(direction)
            if side is None:
                # Absent is not zero. A shape that genuinely never ran in a
                # direction says so with an explicit count of zero, and a
                # shape whose direction was never measured must not be
                # silently weighted as if it had been measured and found
                # empty: the first contributes nothing because there is
                # nothing, the second would hide a hole in the floor.
                raise RunInvalid(
                    f"shape {shape} names no {direction}; a direction that "
                    f"never ran is written as a count of zero, and one that "
                    f"was never measured cannot be weighted at all")
            count = side.get("count")
            if (not isinstance(count, int) or isinstance(count, bool)
                    or count < 0):
                raise RunInvalid(
                    f"shape {shape} ran {count!r} times in {direction}, which "
                    f"is not a count")
            if count == 0:
                per_direction[direction] = {"count": 0, "numerator": 0.0,
                                            "denominator": 0.0}
                continue
            numerator = side.get("numerator") or ()
            denominator = side.get("denominator") or ()
            if not numerator or not denominator:
                raise RunInvalid(
                    f"shape {shape} in {direction} has samples on only one "
                    f"side, so it cannot contribute a ratio")
            smallest, largest = min(numerator), max(denominator)
            if largest <= 0.0:
                raise RunInvalid(
                    f"shape {shape} in {direction} has a measured floor of "
                    f"zero, which cannot bound a ratio")
            numerator_total += count * smallest
            denominator_total += count * largest
            per_direction[direction] = {"count": count,
                                        "numerator": count * smallest,
                                        "denominator": count * largest}
        contributions[shape] = per_direction
    if denominator_total <= 0.0:
        raise RunInvalid("the weighted floor is zero, so no ratio is bounded")
    return {"ratio_lo": numerator_total / denominator_total,
            "numerator_total": numerator_total,
            "denominator_total": denominator_total,
            "shapes": contributions}


# VOID, and removed rather than left callable.
#
# Three rules stood here and Amendment 5 voids all three. Section 3.3's
# reconciliation is void by clause 2: there are no spans to sum and the check
# it was written for cannot be posed. Amendment 4's compile transfer is void by
# clause 4: the profile runs compiled, so there is no uncompiled total to
# transfer to. The instrument-cost band is void because there is no timing
# device inside the step left to price, and its `INSTRUMENT_COST_BAND` is gone
# with it.
#
# They are deleted rather than deprecated because each of them RETURNS A
# VERDICT, and a voided rule that still answers is a rule someone can call and
# believe. What replaces the reconciliation is clause 22's partition, which
# lives in the harness because it reads shares the harness reduces; what
# replaces the instrument band is the scaffold price, which lives in
# `profile_knobs` beside the fit it guards.


# Amendment 6 clause 33. The registered false-separation rate, fixed before
# the null is measured. The multiplier on `R` is NOT registered: step 10
# measures the null and reports the empirical critical value that holds a
# future exchangeable null exceedance at or below this rate.
ALPHA = 0.05


def null_contrast(arm_a: Sequence[float], arm_b: Sequence[float]) -> float:
    """ONE block's null: two identical arms reduced the way the rules reduce.

    Committed clause 21's wording admits two readings and Amendment 6's
    clause 33 fixes this one, the absolute difference of the two arms'
    MEDIANS, because nothing in these rules consumes a single round: the
    registered slope is the median of the per-round slopes, every fit runs
    over arm medians, and every share and gain descends from those. The null
    has to be carried through the same reduction the rule uses or it is
    answering a question no rule asks.

    An earlier version of this module took the median of the PAIRED absolute
    differences, which measures single-round jitter and is about 2.46 times
    larger under an ideal null. It looked conservative and was so only by
    measuring a different quantity.
    """
    if len(arm_a) != len(arm_b):
        raise RunInvalid(
            f"the two arms have {len(arm_a)} and {len(arm_b)} rounds; a null "
            f"contrast compares two arms measured over the same rounds")
    if not arm_a:
        raise RunInvalid("a null contrast needs at least one round")
    return abs(statistics.median(arm_a) - statistics.median(arm_b))


def resolution(blocks: Sequence[float]) -> float:
    """`R`: the median null contrast over `m` INDEPENDENT complete blocks.

    One block cannot estimate this. A single block's contrast is one draw
    from a distribution rather than a scale estimate, and its relative spread
    does not shrink with the round count: under an ideal null it converges to
    an absolute Normal draw whose coefficient of variation is about 0.76 for
    any number of rounds. Simulated over 400000 nine-round blocks, the
    central 90% of single-block contrasts spans 0.09 to 2.91 times their own
    median.

    `R` feeds the READABILITY gates, clause 21's `3R` scaffold offset and
    clause 26's `10R` excursion. It does NOT set the selection demand; that
    is `critical_value` below, and the two are not interchangeable.
    """
    if not blocks:
        raise RunInvalid("`R` is a median over blocks and needs at least one")
    return statistics.median(blocks)


def critical_value(blocks: Sequence[float], alpha: float = ALPHA) -> float:
    """The demand clause 31's sites take, calibrated rather than judged.

    Committed clause 16 demands `2 * R`, registered as a judgement before
    anything had measured what it buys. Simulated under an ideal null with
    `R` at its true value, a fresh null contrast exceeds `2 * R` 17.7% of the
    time, so that test admits noise as a separation about one pair in six.

    This returns the order statistic that holds a future exchangeable null
    exceedance at or below `alpha`: the `ceil((m + 1) * (1 - alpha))`-th
    smallest of `m` measured null contrasts. It assumes nothing about the
    null's shape, which matters because timing noise is not Gaussian.

    Where the block count is too small for the rate to be attainable, the
    largest observed contrast is returned and the caller is entitled to
    nothing better: with `m` blocks the smallest attainable rate is
    `1 / (m + 1)`.
    """
    if not blocks:
        raise RunInvalid("a critical value needs at least one block")
    if not 0.0 < alpha < 1.0:
        raise RunInvalid(f"alpha {alpha} is not a rate in (0, 1)")
    ordered = sorted(blocks)
    index = math.ceil((len(ordered) + 1) * (1.0 - alpha))
    return ordered[min(index, len(ordered)) - 1]


def attainable_rate(block_count: int) -> float:
    """The smallest false-separation rate `m` blocks can hold, `1/(m + 1)`."""
    if block_count < 1:
        raise RunInvalid("a rate needs at least one block")
    return 1.0 / (block_count + 1)


def resolution_admissible(value: float) -> list[str]:
    """Amendment 6: every registered `R` must be POSITIVE and finite.

    A floor of zero claims the machine can resolve any difference at all,
    which would let every positive slope clear `10R` and every strict excess
    clear clause 31's demand. Whether this machine can produce a median of
    exactly zero is a device question nobody has answered, so the refusal is
    registered rather than assumed unnecessary.
    """
    problems = []
    if not math.isfinite(value):
        problems.append(f"the resolution floor is {value}, which is not finite")
    elif value <= 0.0:
        problems.append(
            f"the resolution floor measured {value}, and a floor of zero or "
            f"below claims the machine resolves any difference at all")
    return problems


def floor_is_clearable(step_total: float, demand: float) -> bool:
    """A DIAGNOSTIC, never a gate: could the largest saving the floor can
    credit exceed this cell's demand at all?

    An earlier version of this module offered it as an acceptance rule, on the
    reading that a score ships only if `T * (1/1.10 - 1/g)` reaches the
    demand, and that as `g` grows this approaches `T / 1.10`. The arithmetic
    is right and the RULE is withdrawn: under Amendment 6 a candidate's margin
    is propagated from its own `M`, `N` and `F`, whose sensitivities differ
    per candidate, so no candidate-independent effective floor exists for a
    cell and none is claimed.

    What survives is reporting. A cell where this is false is one where no
    candidate could have shipped whatever it measured, and saying so beside
    the result is how an empty kept list carries its reason.
    """
    return demand < step_total / GAIN_FLOOR


def smallest_shippable_gain(step_total: float, demand: float) -> float | None:
    """The EFFECTIVE shipping floor, which sits above the registered 1.10.

    Returns None where the floor is unclearable at any gain.
    """
    if not floor_is_clearable(step_total, demand):
        return None
    return 1.0 / (1.0 / GAIN_FLOOR - demand / step_total)


def credited_saving(median_numerator: float, min_numerator: float,
                    max_denominator: float) -> float:
    """`K`, the part of the step a candidate's credit says it removes.

    Amendment 6 clause 31. The registered gain is `T/(T - K)` with

        K = M * (1 - F/N)

    where `M` is the MEDIAN per-round credited numerator that clause 21's
    statistics table reports as the share's slope, `N` is the SMALLEST such
    numerator and `F` the LARGEST credited denominator, which is what clause
    9's `ratio_lo` pairs. Those are different reductions of one measurement,
    each chosen to make its own rule harder to take, and writing the saving
    this way is what makes the difference visible instead of hidden inside a
    gain.

    Its consequence is the whole reason clause 31 was rewritten: because
    `dK/dF = -M/N`, the predicted step moves by `M/N` times a floor movement,
    not one for one. Reproduced at `M=50`, `N=10`: a floor movement of 0.11
    moves the predicted step by 0.55.
    """
    if min_numerator <= 0.0:
        raise RunInvalid(
            f"the smallest credited numerator is {min_numerator}, and a "
            f"saving divides by it")
    if not 0.0 < max_denominator < min_numerator:
        raise RunInvalid(
            f"the credited denominator {max_denominator} is not inside "
            f"(0, {min_numerator}); outside that the saving's bounds below "
            f"are not the ones clause 31 registers")
    return median_numerator * (1.0 - max_denominator / min_numerator)


def saving_bounds(median_numerator: tuple[float, float],
                  min_numerator: tuple[float, float],
                  max_denominator: tuple[float, float]) -> tuple[float, float]:
    """The exact range of `K` over a box of its three reduced operands.

    Each argument is a `(low, high)` interval. Exact while `0 < F < N` holds
    across the box, which is checked rather than assumed.

    This bounds the REDUCED scalars. Clause 31 registers that the
    authoritative construction recomputes `M`, `N` and `F` from their shared
    source samples instead, because independent intervals admit combinations
    the raw rounds cannot produce; this is the conservative outer form and is
    labelled as such wherever it is used.
    """
    m_lo, m_hi = median_numerator
    n_lo, n_hi = min_numerator
    f_lo, f_hi = max_denominator
    for name, (lo, hi) in (("median numerator", median_numerator),
                           ("smallest numerator", min_numerator),
                           ("largest denominator", max_denominator)):
        if lo > hi:
            raise RunInvalid(f"the {name} interval is inverted: {lo} > {hi}")
    if not 0.0 < f_hi < n_lo:
        raise RunInvalid(
            f"the denominator box reaches {f_hi} against a smallest "
            f"numerator of {n_lo}; clause 31's bounds need `0 < F < N` "
            f"everywhere in the box")
    return (m_lo * (1.0 - f_hi / n_lo), m_hi * (1.0 - f_lo / n_hi))


def ships_above_floor(saving_low: float, step_total_high: float) -> bool:
    """Clause 24 under clause 31: a gain of 1.10 IS a saving of `T/11`.

    The action is shipping, so it is supported only where the credited saving
    exceeds `T/11` everywhere in the box, which is its lowest saving against
    the largest step it might be a fraction of.
    """
    return saving_low - step_total_high / 11.0 > 0.0


def excluded_by_arithmetic(share_high: float, step_total_low: float,
                           median_numerator_high: float) -> bool:
    """Clause 27's route 1 under clause 31, on candidate A's share alone.

    The action is exclusion, so it is supported only where `T/11` exceeds the
    attributed cost everywhere in the box.
    """
    del share_high  # the margin is on the cost, not the fraction
    return step_total_low / 11.0 - median_numerator_high > 0.0


def separable(saving_low: float, other_saving_high: float) -> bool:
    """Clause 16's per-width pair under clause 31.

    At a shared width the sign of a gain difference is the sign of a saving
    difference, and the time conversion clause 16 registers is EXACTLY the
    saving difference. Verified over 200000 random draws with no mismatch and
    a worst discrepancy of 7e-13.
    """
    return saving_low - other_saving_high > 0.0


def outside_band(anchor_fraction_low: float, lower_fraction_high: float) -> bool:
    """Clause 16's two-point boundary, on saving FRACTIONS `E = K/T`.

    `E_a - E_l - 0.02 * (1 - E_a) * (1 - E_l)` has the same sign as
    `S_a - S_l - 0.02` where `S = 1/(1 - E)`, verified over 200000 draws with
    no mismatch, so the band can be tested without leaving the saving domain.
    """
    return (anchor_fraction_low - lower_fraction_high
            - TIE_GAIN * (1.0 - anchor_fraction_low)
            * (1.0 - lower_fraction_high)) > 0.0


def shape_counts_toward_kill(denominator_low: float,
                             numerator_high: float) -> bool:
    """Clause 23 under clause 31, as an operand margin and not a ratio.

    A shape counts only where its floor cost exceeds its stock cost divided by
    the kill ratio, everywhere in the box. Converting the ratio through a
    shape's stock slope instead would amplify the floor's uncertainty by that
    slope over the round's own numerator, which is the fault clause 31 exists
    to remove.
    """
    return denominator_low - numerator_high / KILL_RATIO > 0.0


def round_is_eligible(samples: Sequence[float], limit_pct: float) -> bool:
    """Section 3.4: a spread past the class limit rejects the round as the
    machine rather than the workload."""
    return spread_pct(samples) <= limit_pct


@dataclass(frozen=True)
class ShapeAtWidth:
    """One shape's kill evidence at one width, already reduced per clause 8.

    `ratio` is the LARGEST of the per-round ratios of summed costs, which is
    the reduction clause 8 registers: the largest is the one furthest from the
    ceiling and therefore the least likely to kill, which is the conservative
    direction for a rule whose effect is to remove a candidate. That is
    deliberately the opposite reduction from `ratio_lo`, which takes the worst
    pairing because its effect is to CREDIT a candidate.

    The two operands are what clause 31 reads instead of the ratio: stock's
    summed cost at the top of its box and the floor's summed cost at the
    bottom. Converting the ratio through a shape's stock slope would amplify
    the floor's uncertainty by that slope over the round's own numerator,
    which is the fault clause 31 exists to remove.
    """

    ratio: float
    numerator_high: float
    denominator_low: float


def largest_round_ratio(per_round: Sequence[float]) -> float:
    """Clause 8's reduction over rounds, registered here rather than assumed.

    The sums are taken WITHIN each round, giving one shape ratio per round,
    and the ratio the kill rule reads is the largest of them. It is a named
    function and not a bare `max` at a call site because a rule the harness
    can forget to apply is a rule that decides nothing.
    """
    if not per_round:
        raise RunInvalid("a shape's kill ratio needs at least one round")
    if any(not math.isfinite(value) for value in per_round):
        raise RunInvalid(
            f"a per-round shape ratio is not finite: {list(per_round)}")
    return max(per_round)


def kill_q(per_shape: Mapping[str, Mapping[str, ShapeAtWidth]]
           ) -> dict[str, object]:
    """Section 5 over six shapes and two widths, REPORTED and never a gate.

    `per_shape` is `{shape: {width: ShapeAtWidth}}`. A shape counts toward the
    kill only where stock is within `KILL_RATIO` of the dense fp16 ceiling at
    BOTH registered widths, so headroom at either width keeps the candidate
    alive, and the verdict is `killed` at all but one shape.

    Amendment 7 clause 36 demotes the ACTION and keeps the MEASUREMENT. The
    verdict removes no candidate from scoring; it is recorded beside the
    ruling and a reader who acts on it is acting outside this
    pre-registration. What the demotion concedes is a real band and it is
    named rather than waved past: a candidate Q with five shapes at the
    ceiling and one with headroom can score well on the collapsed
    `ratio_lo` while the per-shape evidence says the opposite and no rule
    reads it. Measured over these six shapes, that band opens at a
    concentration of 1.910199 when the fast shape's ratio is 4.0 and WIDENS as
    that shape gets faster, reaching 1.297 at a ratio of 100.

    What the demotion does not concede is a candidate Q that is dead
    everywhere: the gain is increasing in the share with limit the ratio, so a
    collapsed ratio at or below 1.10 forces a score below 1.10 at every valid
    share, and the shipping floor removes it without the kill rule.
    """
    missing = sorted(set(SHAPES) - set(per_shape))
    if missing:
        raise RunInvalid(f"the kill rule reads all {len(SHAPES)} registered "
                         f"shapes; missing {missing}")
    # Extra keys are refused, not ignored. Counting them would let shapes
    # nobody registered reach the kill threshold, and the verdict would then
    # say "5 of 6 shapes" about a set that was never six.
    extra = sorted(set(per_shape) - set(SHAPES))
    if extra:
        raise RunInvalid(
            f"the kill rule counts only registered shapes, and these are not "
            f"registered: {extra}; add them to SHAPES by amendment before "
            f"they can decide anything")

    evidence = {}
    at_ceiling = []
    for shape in sorted(SHAPES):
        widths = per_shape[shape]
        absent = sorted(set(WIDTH_ORDER) - set(widths))
        if absent:
            raise RunInvalid(
                f"shape {shape} was measured at {sorted(widths)} and the kill "
                f"rule reads both registered widths; a shape measured at one "
                f"cannot be shown within the ceiling at both")
        unknown = sorted(set(widths) - set(WIDTH_ORDER))
        if unknown:
            raise RunInvalid(
                f"shape {shape} names widths {unknown}, which are not "
                f"registered")
        per_width = {}
        for width in WIDTH_ORDER:
            entry = widths[width]
            for name, value in (("ratio", entry.ratio),
                                ("numerator_high", entry.numerator_high),
                                ("denominator_low", entry.denominator_low)):
                if not math.isfinite(value):
                    raise RunInvalid(
                        f"shape {shape} at the {width} width has a "
                        f"non-finite {name}: {value}")
            per_width[width] = {
                "ratio": entry.ratio,
                "counts": shape_counts_toward_kill(entry.denominator_low,
                                                   entry.numerator_high),
                "margin": (entry.denominator_low
                           - entry.numerator_high / KILL_RATIO),
            }
        both = all(per_width[width]["counts"] for width in WIDTH_ORDER)
        evidence[shape] = {"widths": per_width, "counts_toward_kill": both}
        if both:
            at_ceiling.append(shape)

    killed = len(at_ceiling) >= KILL_SHAPES
    return {
        "killed": killed,
        "certified": False,
        "at_ceiling": at_ceiling,
        "needed": KILL_SHAPES,
        "limit": KILL_RATIO,
        "shapes": evidence,
        "reads": "all but one of the six registered shapes, within "
                 f"{KILL_RATIO} of the dense fp16 ceiling at BOTH widths",
        "gates_nothing": (
            "Amendment 7 clause 36: this verdict is reported and removes no "
            "candidate from scoring. A candidate Q at the ceiling everywhere "
            "fails the shipping floor without it, and one at the ceiling at "
            "five shapes with headroom at the sixth can still be selected"),
        "reason": (
            f"stock is within {KILL_RATIO} of the dense fp16 ceiling at "
            f"{len(at_ceiling)} of {len(SHAPES)} shapes, at both widths"),
    }


# Clause 30's typed absences, after the per-width reduction that clause fixes.
# `killed` was a third type under Amendment 6 and is DEMOTED by Amendment 7 to
# a recorded flag, so the precedence is these two and nothing else.
MISSING_SHARE = "missing_share"
MISSING_RATIO = "missing_ratio"
ABSENCES = (MISSING_SHARE, MISSING_RATIO)

# Amendment 7 clause 38's three terminals. UNRESOLVED is RETIRED, and the word
# is used because states that reached it now reach SELECTED or the
# no-selection record instead.
SELECTED = "SELECTED"
NO_SELECTION = "NO SINGLE OPERATION REACHES THE FLOOR"
INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True)
class Interval:
    """A measured value with the range clause 31 propagates its rule over.

    Every rule below acts on the bound that makes its own action hardest to
    take, so the two ends are not decoration: an action supported at the
    measured value and not across the box is an action the measurements do not
    support. The measured value travels with them because a score is what a
    reader wants and no rule reads one.
    """

    value: float
    low: float
    high: float

    def __post_init__(self):
        for name, one in (("value", self.value), ("low", self.low),
                          ("high", self.high)):
            if not isinstance(one, (int, float)) or isinstance(one, bool) \
                    or not math.isfinite(one):
                raise RunInvalid(
                    f"the interval's {name} is {one!r}; every reading these "
                    f"rules consume must be finite, because a NaN satisfies "
                    f"neither branch of any test and makes the terminal table "
                    f"neither total nor disjoint")
        if not self.low <= self.value <= self.high:
            raise RunInvalid(
                f"the measured value {self.value} is outside its own box "
                f"[{self.low}, {self.high}]")


@dataclass(frozen=True)
class ScoredEntry:
    """Candidate L or Q at one width: a step total and a credited saving.

    The saving is clause 31's `K = M * (1 - F/N)`, already reduced from that
    width's own samples, and it is a TIME. Nothing here divides two uncertain
    numbers to make a gain and then tries to bound the result.
    """

    width: str
    step_total: Interval
    saving: Interval


@dataclass(frozen=True)
class CeilingEntry:
    """Candidate A at one width: a share, a step total, and never a ratio.

    `attributed` is `M`, the credited numerator the share is built from, and
    it is carried apart from the share because clause 27's route 1 takes its
    margin on the COST rather than on the fraction: converting through the
    step total would amplify the step's uncertainty into a test about the
    operation.
    """

    width: str
    step_total: Interval
    share: Interval
    attributed: Interval


@dataclass(frozen=True)
class CandidateInput:
    """One candidate as the selection rule receives it, clause 30's shape.

    A candidate is present with its numbers or absent with ONE typed reason,
    never both and never neither, and never simply missing from the input. A
    candidate that could disappear between the profile and the ruling is the
    fault clause 30 exists to make impossible.
    """

    candidate: str
    entries: tuple = ()
    absence: str | None = None
    killed: bool = False
    footprint_delta: int | None = None


def _fraction(entry: ScoredEntry) -> Interval:
    """`E = K/T`, the saving as a fraction of the step it was measured in.

    `K/T` is increasing in `K` and monotone in `T` for fixed `K`, so the
    extremes sit at corners in `K` and at whichever `T` corner the sign of `K`
    selects. Pairing the smallest saving with the largest step unconditionally
    is right only while `K` is positive: at a saving box reaching -1 over a
    step box of [94, 106] it returns -0.0094 where the true minimum is
    -0.0106, which is the LESS conservative end, and the fraction's low is
    what clause 16's band test and clause 27's route 2 read to decide whether
    a candidate is certified better. A wide box is exactly where a saving
    reaches below zero, and a wide box is exactly where these tests matter.
    """
    corners = (entry.step_total.low, entry.step_total.high)
    return Interval(value=entry.saving.value / entry.step_total.value,
                    low=min(entry.saving.low / one for one in corners),
                    high=max(entry.saving.high / one for one in corners))


def _score(fraction: Interval) -> float:
    """`S = 1/(1 - E)`, the registered gain written in the saving domain."""
    if fraction.value >= 1.0:
        raise RunInvalid(
            f"a credited saving of {fraction.value:.4f} of the step leaves "
            f"nothing for the adapters, the norms, the elementwise work and "
            f"the optimizer, which no step can be")
    return 1.0 / (1.0 - fraction.value)


def _refuse_bad_readings(candidate: CandidateInput) -> None:
    """Clause 27's refusal class, which is not a terminal and takes no margin.

    A share at or above one or at or below zero, at ANY width and for ANY
    candidate, refuses the profile outright. The sibling width is not used,
    because it came from the same instrument and the same machinery: a share
    of 1.401 is exactly what exposed the instrument Amendment 5 exists to
    replace, and continuing on its other reading would be believing the same
    instrument twice.
    """
    for entry in candidate.entries:
        if entry.step_total.low <= 0.0:
            raise RunInvalid(
                f"candidate {candidate.candidate} at the {entry.width} width "
                f"has a step total reaching {entry.step_total.low}, and a "
                f"step with no measured time cannot be a denominator")
        share = getattr(entry, "share", None)
        if share is None:
            continue
        for name, value in (("measured", share.value), ("low", share.low),
                            ("high", share.high)):
            if not 0.0 < value < 1.0:
                raise RunInvalid(
                    f"candidate {candidate.candidate} at the {entry.width} "
                    f"width has a {name} share of {value}, which is not a "
                    f"fraction of a step; this is an instrument fault and no "
                    f"reading from the same instrument is used instead")


def _validate(candidates: Sequence[CandidateInput]) -> dict:
    """Clause 30: exactly the registered candidates, each typed exactly once."""
    if not candidates:
        raise RunInvalid("the selection rule needs the registered candidates")
    names = [one.candidate for one in candidates]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise RunInvalid(
            f"more than one input for {duplicated}: a candidate has one state "
            f"after clause 30's reduction, so the rule cannot say which of "
            f"two is the one it was asked about")
    unknown = sorted(set(names) - set(CANDIDATES))
    if unknown:
        raise RunInvalid(f"not candidates of this pre-registration: {unknown}")
    absent = sorted(set(CANDIDATES) - set(names))
    if absent:
        raise RunInvalid(
            f"the input names no state for {absent}; a candidate is present "
            f"with its numbers or absent with a typed reason, and never "
            f"simply unmentioned")

    by_name = {}
    for one in candidates:
        if one.absence is not None and one.absence not in ABSENCES:
            raise RunInvalid(
                f"candidate {one.candidate} carries the untyped absence "
                f"{one.absence!r}; clause 30 registers {list(ABSENCES)}")
        if (one.absence is None) == (not one.entries):
            fault = ("both a typed absence and readings" if one.absence
                     else "neither readings nor a typed absence")
            raise RunInvalid(f"candidate {one.candidate} carries {fault}")
        widths = [entry.width for entry in one.entries]
        if len(set(widths)) != len(widths):
            raise RunInvalid(
                f"candidate {one.candidate} carries two readings at one width")
        stray = sorted(set(widths) - set(WIDTH_ORDER))
        if stray:
            raise RunInvalid(
                f"candidate {one.candidate} names widths {stray}, which are "
                f"not registered")
        if one.candidate == "A" and one.absence == MISSING_RATIO:
            raise RunInvalid(
                "candidate A is typed `missing_ratio`, and clause 30 records "
                "that for candidate A this is not an absence at all: it is "
                "the amendment's permanent state, and typing it here would "
                "discard the share the ceiling is built from")
        if one.killed and one.candidate != "Q":
            raise RunInvalid(
                f"candidate {one.candidate} carries a kill flag, and clause "
                f"23's rule reaches candidate Q alone; a killed candidate "
                f"{one.candidate} is not a state this profile can be in")
        _refuse_bad_readings(one)
        by_name[one.candidate] = one

    for name in ("L", "Q"):
        one = by_name[name]
        if one.absence is None and set(entry.width for entry in one.entries) \
                != set(WIDTH_ORDER):
            raise RunInvalid(
                f"candidate {name} is scored at "
                f"{sorted(entry.width for entry in one.entries)} and clause "
                f"14 takes its gain at the WORSE of both registered widths, "
                f"so a candidate measured at one has no worse to take")
    return by_name


def _ceiling(entry: CeilingEntry) -> Interval:
    """`U = 1/(1 - f)`, the best score any positive finite ratio could earn.

    Clause 27: `gain = 1/(1 - f*(1 - 1/r))` is increasing in `r` over positive
    finite `r` with supremum `1/(1 - f)`, attained at no finite ratio. Its
    bounds are monotone in `f`, which validity has already put strictly inside
    (0, 1), so the low end of the share gives the low end of the ceiling.
    """
    return Interval(value=1.0 / (1.0 - entry.share.value),
                    low=1.0 / (1.0 - entry.share.low),
                    high=1.0 / (1.0 - entry.share.high))


def candidate_a_report(entry_by_width: Mapping[str, CeilingEntry],
                       winner: Mapping[str, object] | None) -> dict:
    """Clause 27's ceiling and both exclusion routes, REPORTED and not a gate.

    Every construction is clause 27's, unchanged. What Amendment 7 clause 35
    changes is where the answers go: neither route reaches a terminal, and a
    winner that fails to beat the ceiling produces an OPEN QUESTION recorded
    against it rather than a refusal to select.

    The bias is stated where the number appears rather than in a footnote.
    While exclusion was the only action, an inflated share made exclusion
    harder, which was the safe direction. With no action left, the same
    inflation only makes candidate A look BETTER in the report than it is.
    """
    if not entry_by_width:
        return {"certified": False, "ceiling": None, "u_min": None,
                "valid_widths": [], "route_1": None, "route_2": None,
                "reason": "candidate A has no valid share at either width, so "
                          "no ceiling was available and none is reported"}

    ceilings = {width: _ceiling(entry)
                for width, entry in sorted(entry_by_width.items())}
    lowest = min(ceilings[width].value for width in ceilings)
    setters = sorted(width for width in ceilings
                     if ceilings[width].value == lowest)
    u_min = ceilings[setters[0]]

    route_1 = {}
    for width, entry in sorted(entry_by_width.items()):
        route_1[width] = {
            "excludes": excluded_by_arithmetic(
                share_high=entry.share.high,
                step_total_low=entry.step_total.low,
                median_numerator_high=entry.attributed.high),
            "share": entry.share.value,
            "margin_ms": (entry.step_total.low / 11.0
                          - entry.attributed.high),
        }
    excluded_1 = sorted(w for w, one in route_1.items() if one["excludes"])

    route_2 = None
    if winner is not None:
        # The excess converts to a time at the width that SETS `U_min`, and
        # the conversion is monotone in the fraction difference, so the sign
        # of the certified excess is the sign of `E_winner_low - f_A_high`.
        # Clause 27 registers the SELECTED candidate rather than the top
        # scorer, on the same principle `ratio_lo` follows: each reduction is
        # the one that makes its own action harder to take.
        setter = entry_by_width[setters[0]]
        difference = winner["fraction_low"] - setter.share.high
        route_2 = {
            "excludes": difference > 0.0,
            "against": winner["candidate"],
            "u_min": u_min.value,
            "set_by": setters,
            "winner_score": winner["score"],
            "excess_ms": (setter.step_total.value
                          * (winner["fraction_value"] - setter.share.value)),
        }

    return {
        "certified": False,
        "ceiling": {width: one.value for width, one in ceilings.items()},
        "u_min": u_min.value,
        "u_min_set_by": setters,
        "valid_widths": sorted(entry_by_width),
        "route_1": route_1,
        "route_1_excludes": excluded_1,
        "route_2": route_2,
        "gates_nothing": (
            "Amendment 7 clause 35: candidate A's share and ceiling are "
            "computed and recorded, both exclusion routes are evaluated, and "
            "neither reaches a terminal. Candidate A could not be ruled IN "
            "under Amendment 6 either, so what was removed is a veto and not "
            "a candidate"),
        "stated_bias": (
            "the named dial's scaffold is the most expensive of the three at "
            "the long width, so this share reads high and this ceiling is "
            "inflated. While exclusion was an action that was the safe "
            "direction; with no action left it only makes candidate A look "
            "better here than it is"),
    }


def break_band(band: Sequence[Mapping[str, object]]
               ) -> tuple[list, str, bool]:
    """Clause 12's tie-break, with BOTH of its fall-through conditions.

    Committed code carried one of the two. A band is decided on footprint only
    where every member has a measured delta AND candidate L is not in it
    against another candidate: candidate L's baseline is measured on the bench
    and the other two in the step, so its delta is not comparable with theirs.

    It is a rule of its own rather than a branch inside the selection because
    both of its conditions are registered text, and today only one of them can
    ever fire. Candidates L and Q are the only ones that score, so any band
    with more than one member contains candidate L, the second condition
    always fires, and the tie-break is unreachable in practice - which is what
    the Amendment 7 ledger records about clause 12. A later amendment that
    adds a scoreable candidate gets the rule rather than a rediscovery.
    """
    measured = all(row["footprint_delta"] is not None for row in band)
    l_against_others = (len(band) > 1
                        and any(row["candidate"] == "L" for row in band))
    if measured and not l_against_others:
        return (sorted(band, key=lambda row: (row["footprint_delta"],
                                              CANDIDATES.index(
                                                  row["candidate"]))),
                "smaller peak-footprint delta then table order", True)
    return (sorted(band, key=lambda row: CANDIDATES.index(row["candidate"])),
            "table order, because candidate L's peak-footprint delta is "
            "measured on the bench and the others' in the step, so they are "
            "not comparable" if l_against_others else
            "table order, because the peak-footprint delta is not measured "
            "for every candidate in the band",
            False)


def select_first_operation(candidates: Sequence[CandidateInput]
                           ) -> dict[str, object]:
    """Section 4.3 under Amendments 6 and 7. The rule picks and no argument may
    substitute for it.

    Three terminals, and the precedence is INCOMPLETE first:

      INCOMPLETE     candidate L or candidate Q reduces to `missing_share`, so
                     a required measurement does not exist and no remaining
                     score can substitute for it.
      SELECTED       a winner whose credited saving exceeds `T/11` everywhere
                     in its own box, with candidate A's ceiling and candidate
                     Q's kill verdict recorded beside it.
      NO SELECTION   every score below the floor, or no scores at all.

    `missing_ratio` leaves a candidate excluded, visible and continuable, and
    a `killed` candidate Q is neither: Amendment 7 demotes the kill rule to a
    recorded flag, so it removes nothing from scoring.

    Every comparison here is certified over the box: the floor, the pair test
    and the band all act only where they hold at every point the measurements
    admit. A gain of 1.10 IS a saving of `T/11`, and at a shared width the
    sign of a gain difference is the sign of a saving difference, which is
    what lets the whole rule run without ever bounding a ratio of two
    uncertain numbers.
    """
    by_name = _validate(candidates)

    incomplete = [name for name in ("L", "Q")
                  if by_name[name].absence == MISSING_SHARE]
    if incomplete:
        return {
            "verdict": INCOMPLETE, "selected": None,
            "incomplete": incomplete,
            "absences": {name: by_name[name].absence for name in CANDIDATES
                         if by_name[name].absence is not None},
            # Recorded even here. A reported quantity reaches no terminal and
            # is still evidence, and a run that measured candidate A and then
            # failed on candidate L has no reason to throw the measurement
            # away.
            "candidate_a": candidate_a_report(
                {entry.width: entry for entry in by_name["A"].entries}, None),
            "certified_sites": dict(CERTIFIED_SITES),
            "reason": (
                f"candidate {' and '.join(incomplete)} has no valid share at "
                f"a width the rule needs, so the profile refuses SELECTED "
                f"whatever the remaining scores say"),
        }

    scored = []
    for name in ("L", "Q"):
        one = by_name[name]
        if one.absence is not None:
            continue
        by_width = {entry.width: entry for entry in one.entries}
        fractions = {width: _fraction(entry)
                     for width, entry in by_width.items()}
        # Clause 14: the score is the candidate's gain at its WORSE of the two
        # widths, so the reduced fraction is the smallest of them.
        worst = min(WIDTH_ORDER, key=lambda w: fractions[w].value)
        scored.append({
            "candidate": name,
            "score": _score(fractions[worst]),
            "fraction_value": fractions[worst].value,
            "fraction_low": fractions[worst].low,
            "fraction_high": fractions[worst].high,
            "worst_width": worst,
            "per_width": {w: {"saving_ms": by_width[w].saving.value,
                              "step_total_ms": by_width[w].step_total.value,
                              "fraction": fractions[w].value,
                              "score": _score(fractions[w])}
                          for w in sorted(by_width)},
            "ships": all(ships_above_floor(by_width[w].saving.low,
                                           by_width[w].step_total.high)
                         for w in WIDTH_ORDER),
            "floor_margin_ms": {
                w: by_width[w].saving.low - by_width[w].step_total.high / 11.0
                for w in WIDTH_ORDER},
            "killed": one.killed,
            "footprint_delta": one.footprint_delta,
            "_by_width": by_width,
        })

    ceiling_entries = {entry.width: entry
                       for entry in by_name["A"].entries}
    score_order = sorted(scored, key=lambda row: (-row["score"],
                                                  CANDIDATES.index(
                                                      row["candidate"])))
    absences = {name: by_name[name].absence for name in CANDIDATES
                if by_name[name].absence is not None}
    above = [row for row in scored if row["ships"]]

    if not above:
        # Clause 34 REVERSES the committed kept list. It used to be the top
        # two in score order, and section 4.3 makes that list the Day 2 build
        # order, so a quantity that certifies nothing arrived somewhere as an
        # instruction: flipping which candidate scored higher flipped the
        # payload while the terminal label never moved. It is now the whole
        # scored set, unordered, presented in section 4.2's table order, with
        # the score order beside it as evidence and explicitly not a build
        # order.
        return {
            "verdict": NO_SELECTION, "selected": None,
            "floor": GAIN_FLOOR,
            "keep": sorted((row["candidate"] for row in scored),
                           key=CANDIDATES.index),
            "keep_is_unordered": (
                "an unordered set in section 4.2's table order; the score "
                "order below is evidence and is NOT a build order, and where "
                "a build order is needed section 4.2's table order supplies "
                "it"),
            "score_order": [row["candidate"] for row in score_order],
            "scored": [_public(row) for row in score_order],
            "absences": absences,
            "candidate_a": candidate_a_report(ceiling_entries, None),
            "certified_sites": dict(CERTIFIED_SITES),
            "reason": (
                f"no candidate's credited saving exceeds T/11 everywhere in "
                f"its own box at both widths, so none reaches the "
                f"{GAIN_FLOOR} floor" if scored else
                "no candidate has a score at all, which is an answer rather "
                "than an absence: every scored candidate is excluded by type"),
        }

    # Clause 24: the floor is applied FIRST and the band is drawn only from
    # what survives it, because a tie-break must never hand SELECTED to a
    # candidate the floor already ruled out.
    #
    # Clause 16's anchor is the highest-scoring survivor, and where two share
    # the highest score exactly the anchor is the earlier of them in section
    # 4.2's table. Without that, two candidates on an identical score give two
    # different anchors and two different bands.
    ranked = sorted(above, key=lambda row: (-row["score"],
                                            CANDIDATES.index(row["candidate"])))
    anchor = ranked[0]
    band, band_evidence = [anchor], {}
    for row in ranked[1:]:
        # Both tie routes are UNIONS: a candidate joins the band unless it is
        # certified outside the two-point band AND certified resolvably worse.
        # An earlier draft used the second route alone, which left a candidate
        # inside the two-point band but resolvably worse simultaneously in and
        # out of it.
        outside = outside_band(anchor["fraction_low"], row["fraction_high"])
        resolvable = all(
            separable(anchor["_by_width"][w].saving.low,
                      row["_by_width"][w].saving.high)
            for w in WIDTH_ORDER)
        band_evidence[row["candidate"]] = {
            "certified_outside_the_two_point_band": outside,
            "certified_resolvably_worse": resolvable,
            "in_band": not (outside and resolvable),
        }
        if not (outside and resolvable):
            band.append(row)

    tied, broken_by, measured = break_band(band)
    winner = tied[0]

    candidate_a = candidate_a_report(ceiling_entries, winner)
    open_questions = []
    if candidate_a["u_min"] is not None:
        route_2 = candidate_a["route_2"]
        if not candidate_a["route_1_excludes"] and not route_2["excludes"]:
            open_questions.append({
                "question": "candidate A's ceiling is not resolvably below "
                            "the selected candidate's score",
                "u_min": candidate_a["u_min"],
                "set_by": candidate_a["u_min_set_by"],
                "winner_score": winner["score"],
                "shortfall": candidate_a["u_min"] - winner["score"],
                "carried": "a known risk carried into Day 2 rather than a "
                           "refusal to start it",
            })

    return {
        "verdict": SELECTED, "selected": winner["candidate"],
        "floor": GAIN_FLOOR,
        "tied_with": [row["candidate"] for row in tied[1:]],
        "band": [row["candidate"] for row in band],
        "band_evidence": band_evidence,
        "anchor": anchor["candidate"],
        "footprint_measured": measured,
        "scored": [_public(row) for row in score_order],
        "score_order": [row["candidate"] for row in score_order],
        "absences": absences,
        "kill": {row["candidate"]: row["killed"] for row in scored},
        "candidate_a": candidate_a,
        "open_questions": open_questions,
        "certified_sites": dict(CERTIFIED_SITES),
        "reason": (
            f"the largest certified score at cell {PRIMARY_CELL}"
            if len(tied) == 1 else
            f"inside clause 16's band around {anchor['candidate']}, broken by "
            f"{broken_by}"),
    }


def _public(row: Mapping[str, object]) -> dict:
    """One scored candidate as the artifact carries it, without the operands.

    The `_by_width` entries are the rule's own working and are dropped here
    rather than serialised, because a reader who wants them wants the arms the
    recording already holds, not a second copy the ruling could disagree with.
    """
    return {key: value for key, value in row.items()
            if not key.startswith("_")}
