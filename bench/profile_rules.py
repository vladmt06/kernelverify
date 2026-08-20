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

# Section 5. Q dies if stock is already this close to the dense ceiling at
# this many of the five shapes.
KILL_RATIO = 1.10
KILL_SHAPES = 4


@dataclass(frozen=True)
class Reading:
    """One candidate's inputs to the gain formula, and where they came from."""

    candidate: str
    share: float          # f, section 3.3
    ratio_lo: float       # r_lo, section 4.2
    footprint_delta: int  # bytes, the first tie-break in section 4.3


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
        raise RunInvalid(f"the kill rule reads all five shapes; missing "
                         f"{sorted(missing)}")
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
    tied = sorted(
        (row for row in scored if largest - row["gain"] < TIE_GAIN),
        key=lambda row: (row["footprint_delta"],
                         CANDIDATES.index(row["candidate"])),
    )
    winner = tied[0]
    return {
        "selected": winner["candidate"], "ranked": scored, "floor": GAIN_FLOOR,
        "verdict": "SELECTED",
        "tied_with": [row["candidate"] for row in tied[1:]],
        "reason": (f"largest gain at cell {PRIMARY_CELL}"
                   if len(tied) == 1 else
                   f"tied within {TIE_GAIN} of the largest gain, broken by "
                   f"smaller peak-footprint delta then table order"),
    }
