"""Pure decision arithmetic for the Day 1 stock training profile.

Every constant here names the pre-registration section that fixed it, and
nothing in this module measures anything: it takes numbers a harness recorded
and applies rules that were written down before the harness ran. That split is
the point - the rules can be read, argued with and tested without a GPU, and
the harness cannot quietly become the thing that decides.

The document is docs/research/2026-08-19-metalrunner-sprint1-prereg.md,
sections 3, 4 and 5, as amended by its Amendment 4.
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

# Section 3.2. The five distinct projection shapes the model contains,
# as (d_out, d_in), with the token count M the cell produces.
SHAPES = {
    "S1": (4096, 2560),
    "S2": (2560, 4096),
    "S3": (9728, 2560),
    "S4": (2560, 9728),
    "S5": (151936, 2560),
}

# Section 3.3. The named regions plus one remainder must account for the step.
RECONCILE_PCT = 2.0

# Amendment 4. Declared uncalibrated: a judgement about how much redistribution
# makes a share meaningless, not a measured limit.
COMPILE_RATIO_BAND = (0.90, 1.10)

# Section 4.3, which is R10's shipping floor read as a gain.
GAIN_FLOOR = 1.10

# Section 4.3. Gain read as a percentage; two points of it is a tie.
TIE_GAIN = 0.02

# Section 4.2, in the order the table lists them, which is the last tie-break.
CANDIDATES = ("L", "A", "Q")

# A credited ratio is weighted per direction, because a region's forward and
# backward call counts genuinely differ under LoRA.
DIRECTIONS = ("forward", "backward")

# Section 5. Q dies if stock is already this close to the dense ceiling at
# this many of the five shapes.
KILL_RATIO = 1.10
KILL_SHAPES = 4


@dataclass(frozen=True)
class Reading:
    """One candidate's inputs to the gain formula, and where they came from.

    `footprint_delta` is bytes, the first tie-break in section 4.3, and it is
    None when nothing measured it. The tie-break asks for the peak-footprint
    delta of an implementation that does not exist yet, so what stands in for
    it is the delta measured across that candidate's own floor arms, which is
    a statement about the floor rather than about a kernel; Amendment 5 names
    that substitution and requires it labelled wherever it is reported.
    """

    candidate: str
    share: float                 # f, section 3.3
    ratio_lo: float              # r_lo, section 4.2
    footprint_delta: int | None = None


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
    if not 0.0 <= share <= 1.0:
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


def reconciles(regions: Mapping[str, float], remainder: float,
               step_total: float) -> dict[str, object]:
    """Section 3.3: the decomposition must account for the step it describes.

    Compared against the step timed WITHOUT the interior boundaries, per
    Amendment 4, so what this catches is the boundaries moving the workload
    they were inserted to describe.
    """
    if step_total <= 0.0:
        raise RunInvalid("a step with no measured time cannot be reconciled")
    accounted = sum(regions.values()) + remainder
    gap_pct = 100.0 * abs(accounted - step_total) / step_total
    return {"accounted": accounted, "step_total": step_total,
            "gap_pct": gap_pct, "ok": gap_pct <= RECONCILE_PCT,
            "limit_pct": RECONCILE_PCT}


def compile_transfer(uncompiled_total: float,
                     compiled_total: float) -> dict[str, object]:
    """Amendment 4: what the shares transfer to, priced rather than assumed.

    Nothing is scaled by this ratio. It is reported beside every share as the
    stated bound on reading an uncompiled decomposition as a claim about the
    compiled step the product actually runs.
    """
    if compiled_total <= 0.0:
        raise RunInvalid("a compiled step with no measured time cannot bound a "
                         "transfer")
    ratio = uncompiled_total / compiled_total
    low, high = COMPILE_RATIO_BAND
    return {"ratio": ratio, "band": COMPILE_RATIO_BAND,
            "ok": low <= ratio <= high,
            "meaning": "uncompiled step total over compiled step total, on the "
                       "same batch in the same run; a ratio far from 1 means "
                       "the decomposition describes a workload the product "
                       "does not run"}


# Amendment 5. What the marks are allowed to cost, as a multiple of the same
# step measured without them.
#
# Deliberately unregistered until the calibration run fills it. Section 3.3
# registered a 2% reconciliation limit, and every instrument that can actually
# put a clock inside an MLX backward costs far more than that: marking a
# synthetic six-region chain cost +40.5% and marking a real 0.6B training step
# cost +110.8%, both measured 2026-08-20. A limit no working instrument can
# meet rejects the instrument rather than the run, so the cost is measured
# first at the real cell and the band is written here by amendment.
#
# While this is None a binding profile refuses. That is the point: a cost with
# no registered limit is a number nobody agreed to accept, and accepting it
# after seeing it is the one thing the pre-registration exists to prevent.
INSTRUMENT_COST_BAND: tuple[float, float] | None = None


def instrument_cost(instrumented_total: float, plain_total: float, *,
                    band: Sequence[float] | None = None) -> dict[str, object]:
    """Amendment 5: what the marks cost, against the limit registered for them.

    The numerator and the denominator are the same step on the same batch in
    the same child, one pass with the marks installed and one without, so the
    ratio is the instrument and nothing else.

    Nothing is scaled by it. A share already takes its denominator from the
    plain pass, so the marks' cost does not enter the share arithmetic; this
    ratio is the check that the marked pass still describes the workload the
    plain one ran, which is the claim a share silently makes.

    `band` overrides the registered limit so the rule is testable before the
    amendment exists. Passing nothing reads `INSTRUMENT_COST_BAND`, and while
    that is None the verdict is None rather than True: unregistered is not the
    same as passed, and a caller that treats it as passed is refusing to see
    the difference.
    """
    if plain_total <= 0.0:
        raise RunInvalid("a plain step with no measured time cannot price the "
                         "instrument")
    if instrumented_total <= 0.0:
        raise RunInvalid("an instrumented step with no measured time is not a "
                         "measurement of anything")
    ratio = instrumented_total / plain_total
    limits = INSTRUMENT_COST_BAND if band is None else band
    if limits is None:
        return {"ratio": ratio, "band": None, "ok": None,
                "meaning": "instrumented step total over plain step total, on "
                           "the same batch in the same child",
                "reason": "no band is registered: the calibration run measures "
                          "this cost and Amendment 5 writes the limit before "
                          "any binding profile may read it"}
    low, high = limits
    return {"ratio": ratio, "band": (float(low), float(high)),
            "ok": low <= ratio <= high,
            "meaning": "instrumented step total over plain step total, on the "
                       "same batch in the same child",
            "reason": "a ratio outside the band means the marked pass and the "
                      "plain pass are not describing the same step"}


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


def kill_q(ceiling_ratios: Mapping[str, float]) -> dict[str, object]:
    """Section 5, computed at the primary cell's token count.

    `ceiling_ratios` is stock's quantized matmul time over the dense fp16
    ceiling at the same logical shape, per shape. A ratio at or below the kill
    line means stock is already that close to a ceiling that does strictly
    less work, so there is no headroom worth a sprint.
    """
    missing = set(SHAPES) - set(ceiling_ratios)
    if missing:
        raise RunInvalid(f"the kill rule reads all {len(SHAPES)} registered "
                         f"shapes; missing {sorted(missing)}")
    # Extra keys are refused, not ignored. Counting them would let shapes
    # nobody registered reach the kill threshold, and the verdict would then
    # say "4 of 5 shapes" about a set that was never five.
    extra = set(ceiling_ratios) - set(SHAPES)
    if extra:
        raise RunInvalid(
            f"the kill rule counts only registered shapes, and these are not "
            f"registered: {sorted(extra)}; add them to SHAPES by amendment "
            f"before they can decide anything")
    at_ceiling = sorted(name for name, r in ceiling_ratios.items()
                        if r <= KILL_RATIO)
    killed = len(at_ceiling) >= KILL_SHAPES
    return {"killed": killed, "at_ceiling": at_ceiling,
            "needed": KILL_SHAPES, "limit": KILL_RATIO,
            "reason": (f"stock is within {KILL_RATIO} of the dense fp16 "
                       f"ceiling at {len(at_ceiling)} of {len(SHAPES)} shapes")
            if killed else
            (f"stock is within {KILL_RATIO} of the ceiling at only "
             f"{len(at_ceiling)} of {len(SHAPES)} shapes")}


def select_first_operation(readings: Sequence[Reading]) -> dict[str, object]:
    """Section 4.3. The rule picks, and no argument may substitute for it.

    Returns the ruling with everything it rested on, including the branch
    where no single operation reaches R10's floor: that branch does not pick
    one, it records the reading and keeps the top two in gain order, so a weak
    profile produces an honest plan rather than a hopeful pick.
    """
    if not readings:
        raise RunInvalid("the selection rule needs at least one candidate")
    unknown = [r.candidate for r in readings if r.candidate not in CANDIDATES]
    if unknown:
        raise RunInvalid(f"not candidates of this pre-registration: {unknown}")
    # One reading per candidate, by construction: a candidate has one share at
    # the primary cell and one credited ratio. Two readings for one candidate
    # means a harness produced a duplicate, and the rule would then silently
    # rank the better of them and report a table with the same name twice.
    seen = [r.candidate for r in readings]
    duplicated = sorted({name for name in seen if seen.count(name) > 1})
    if duplicated:
        raise RunInvalid(
            f"more than one reading for {duplicated}: a candidate has one "
            f"share and one credited ratio, so the rule cannot say which of "
            f"two readings is the one it was asked about")

    scored = sorted(
        ({"candidate": r.candidate, "gain": gain(r.share, r.ratio_lo),
          "share": r.share, "ratio_lo": r.ratio_lo,
          "footprint_delta": r.footprint_delta} for r in readings),
        key=lambda row: (-row["gain"], CANDIDATES.index(row["candidate"])),
    )
    largest = scored[0]["gain"]

    if largest < GAIN_FLOOR:
        return {
            "selected": None, "ranked": scored, "floor": GAIN_FLOOR,
            "verdict": "NO SINGLE OPERATION REACHES THE FLOOR",
            "keep": [row["candidate"] for row in scored[:2]],
            "reason": (f"the largest gain at cell {PRIMARY_CELL} is "
                       f"{largest:.4f}, below {GAIN_FLOOR}; the sprint keeps "
                       f"the top two and builds them in gain order"),
        }

    # The tie rule is not a sort key. Gain decides only where the candidates
    # differ by two points or more; inside that band the reading is not
    # precise enough to order them, so the registered tie-breaks do it and a
    # hair more gain wins nothing.
    #
    # Amendment 5 clause 24: the shipping floor applies to the SELECTED
    # candidate, not to the largest gain. The band is therefore drawn only
    # from candidates at or above the floor, because a tie-break must never
    # hand SELECTED to a candidate the floor already ruled out. Below-floor
    # candidates stay in the ranked evidence.
    band = [row for row in scored
            if row["gain"] >= GAIN_FLOOR and largest - row["gain"] < TIE_GAIN]

    # The footprint tie-break needs every tied candidate measured. Ranking a
    # measured delta against an unmeasured one would decide the sprint on
    # which candidate happened to get a number, so an unmeasured member sends
    # the whole band to table order and the ruling says that is what happened.
    measured = all(row["footprint_delta"] is not None for row in band)
    if measured:
        tied = sorted(band, key=lambda row: (row["footprint_delta"],
                                             CANDIDATES.index(row["candidate"])))
        broken_by = ("smaller peak-footprint delta then table order")
    else:
        tied = sorted(band, key=lambda row: CANDIDATES.index(row["candidate"]))
        broken_by = ("table order, because the peak-footprint delta is not "
                     "measured for every tied candidate")
    winner = tied[0]
    return {
        "selected": winner["candidate"], "ranked": scored, "floor": GAIN_FLOOR,
        "verdict": "SELECTED",
        "tied_with": [row["candidate"] for row in tied[1:]],
        "footprint_measured": measured,
        "reason": (f"largest gain at cell {PRIMARY_CELL}"
                   if len(tied) == 1 else
                   f"tied within {TIE_GAIN} of the largest gain, broken by "
                   f"{broken_by}"),
    }
