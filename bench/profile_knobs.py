"""The Day 1 profile's instrument: a dial, a fitted line, and no clock inside the step.

Why this module exists at all
-----------------------------
Section 3.3 of the pre-registration measured an operation's share by placing
evaluation boundaries around it. Amendment 5 voids that, because a boundary
forces pending work to finish and THEN reads the clock, so its cost lands
inside the span forming a share's numerator and outside the plain step forming
its denominator. The error grows with how many boundaries a region carries,
which is exactly what separates the three candidates: measured on the 0.6B
proxy, two boundaries read 7% high, thirty-two read 329% high, and a hundred
and ninety-seven produced 1.401, which is not a fraction of anything.

A share is now a fitted slope. Shrink the operation on a dial, time the WHOLE
step at each setting, fit `T(phi) = a + b*phi`. Nothing times anything inside
the step, which is why compilation goes back on and this instrument measures
the arrangement `mlx_lm.lora` actually runs.

The arithmetic, and why it is exact rather than close
-----------------------------------------------------
Section 4.1's formula is `gain = 1/(1 - f*(1 - 1/r))`, which rearranges to
`T_new = T - f*T + f*T/r`. That is exactly right whenever

    f := A / T        A is what the candidate's operation costs the step
    r := A / F        F is what the replacement would cost the step

because then `T_new = T - A + F`, which is what replacing an operation does.
Amendment 5 clause 18 defines A per candidate KIND, read off the names section
4.2 already gives them:

    retune  (Q)   A = b            F = d
    rewrite (A,L) A = b + c        F = d + c_floor

`b` is the fitted slope, `c` the non-scaling residue a rewrite also deletes.
Pairing `A = b + c` with `r = b/d`, as an earlier draft did, silently assumes
`c_floor = c*d/b`; on T=100, b=20, c=10, d=5 that returns 77.5 where deleting
the residue returns 75 and preserving it returns 85.

What the dial actually moves, stated because it is not what it looks like
-------------------------------------------------------------------------
Shrinking a matmul's input width cuts its arithmetic AND the weight bytes it
reads, and a retuned kernel can improve how it reads those bytes but cannot
decide not to read them. So `f` is the fraction of the step that scales with
the operation's SIZE, not the fraction that is arithmetic. That is consistent
rather than loose, because the floor is measured with the same dial and
whatever traffic is irreducible sits in both slopes and divides out of `r`.

The compile-cache trap, which is the reason arms are built the way they are
---------------------------------------------------------------------------
MLX keys its compilation cache on the underlying callable and the input
signature, NOT on the wrapper `mx.compile` returns. Wrapping one shared step
function twice therefore serves one arm's traced graph to another, and a dial
that silently does nothing produces a clean fit through a horizontal line,
which is worse than a bad fit because it looks fine.

Three requirements follow, and clause 25 registers them:

    every arm at every width owns a freshly built closure, retained for the run
    the optimizer's state is initialised and evaluated BEFORE any arm is built
    a trace counter refuses the run if any trace occurs during a timed round

The second is not paranoia: Adam allocates its `m` and `v` on the first update,
which grows the captured state tree and forces a second trace. If that trace
lands after the seam has been removed, the arm silently becomes stock.

The third is the general guard, and it is why the first two do not have to be
exhaustive. A trace during timing is always a fault, whatever caused it. The
counter works because a compiled function's Python body executes only while
tracing, so a counter incremented in the body counts traces exactly.

    python bench/profile_knobs.py --self-check
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from decode_rules import RunInvalid

# Amendment 5 clause 1. The dial's settings, dimensionless: the ratio of the
# dialled size to the full size, so 1.0 is stock's own size.
PHIS = (1.00, 0.75, 0.50, 0.25)

# Clause 1. Five rounds, and EXACTLY three warm-ups rather than at least
# three: at one warm-up the uniqueness probe returned two mutually
# contradictory answers on consecutive runs, and a count left free is a knob
# an implementer can turn.
ROUNDS = 5
WARMUPS = 3

# Clause 21. Three of the four limits are pure ratios and need no measured
# scale, so they are fixed here. The fourth needs the resolution floor R and
# is supplied by the addendum.
SCAFFOLD_SLOPE_LIMIT = 0.10       # |scaffold slope| <= this * |knob slope|
R_SQUARED_FLOOR = 0.99
RESIDUAL_LIMIT = 0.05             # |largest residual| <= this * |slope|
SCAFFOLD_OFFSET_R_MULTIPLE = 3.0  # |offset| <= this * R
EXCURSION_R_MULTIPLE = 10.0       # |slope| * span >= this * R, clause 26

# Clause 15. Two attention dials whose scaffold prices differ by less than
# this are a tie, and a tie goes to the more complete dial. The statistic is
# dimensionless, so its threshold is too; comparing it against the
# time-valued R would be a units error.
DIAL_TIE = 0.01

# Clause 15's three attention dials, in registered completeness order, MOST
# COMPLETE FIRST, which is the order a tie is broken in.
ATTENTION_DIALS = ("kv-length", "head-dim-qkv", "head-dim-qk")

# Clause 15. A dial that cannot place three distinct realisable settings on
# the ladder is not eligible whatever its price. Two points fit a line with no
# residual and an R-squared of exactly 1, so the linearity gate could never
# refuse such a dial and would report a slope nothing had checked.
MIN_DIAL_SETTINGS = 3

RETUNE = "retune"
REWRITE = "rewrite"
KINDS = (RETUNE, REWRITE)


@dataclass(frozen=True)
class Fit:
    """A least-squares line through one dial's arm times."""

    slope: float
    intercept: float
    r_squared: float
    max_residual: float

    @property
    def is_a_dial(self) -> bool:
        """Does this line describe a dial, or something that does not move?

        A non-positive slope is refused explicitly rather than left to fail a
        numeric gate by accident: a dial that does not shrink the step is not
        measuring the operation it claims to.
        """
        return self.slope > 0.0


def fit(phis: Sequence[float], times: Sequence[float]) -> Fit:
    """Least squares over the dial, with the residual reported beside it.

    `r_squared` is 1.0 for a perfect line and is defined as 1.0 when every
    time is identical, which is the degenerate case a horizontal line
    produces; the slope test above is what catches that, not this one.
    """
    if len(phis) != len(times):
        raise RunInvalid(
            f"the dial has {len(phis)} settings and {len(times)} times, so "
            f"there is no line to fit")
    if len(phis) < 3:
        raise RunInvalid(
            f"a dial needs at least three settings to be shown linear, and "
            f"this one has {len(phis)}")

    n = len(phis)
    mean_x = sum(phis) / n
    mean_y = sum(times) / n
    sxx = sum((x - mean_x) ** 2 for x in phis)
    if sxx == 0.0:
        raise RunInvalid("every dial setting is the same value")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(phis, times))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    residuals = [y - (intercept + slope * x) for x, y in zip(phis, times)]
    ss_res = sum(r * r for r in residuals)
    ss_tot = sum((y - mean_y) ** 2 for y in times)
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return Fit(slope=slope, intercept=intercept, r_squared=r_squared,
               max_residual=max(residuals, key=abs))


@dataclass(frozen=True)
class KnobReading:
    """One candidate's measured inputs, and everything they were derived from.

    Held as raw samples plus derived properties rather than as numbers alone,
    because a share that cannot say which reduction produced it is a share
    nobody can check.
    """

    candidate: str
    kind: str
    stock_median: float                    # T, stock with NO seam installed
    per_round: tuple[Fit, ...]             # one fit per round
    pooled: Fit                            # one fit over the arm medians
    scaffold: Fit                          # the scaffold-only arm's own line
    scaffold_offset: float                 # phi=1 arm minus stock
    ablated_median: float | None = None    # rewrite candidates only
    resolution_floor: float | None = None  # R, from the instrument-only stage
    span: float | None = None              # highest minus lowest ACTUAL phi
    scaffold_range: float | None = None    # the scaffold's own median range

    def __post_init__(self):
        if self.kind not in KINDS:
            raise RunInvalid(
                f"candidate {self.candidate} is a {self.kind!r}, which is "
                f"neither of the two kinds section 4.2's names allow: "
                f"{KINDS}")
        if self.kind == REWRITE and self.ablated_median is None:
            raise RunInvalid(
                f"candidate {self.candidate} is a rewrite, so its credit is "
                f"its slope PLUS its residue, and no ablated arm was "
                f"measured to give the residue")
        if self.kind == RETUNE and self.ablated_median is not None:
            raise RunInvalid(
                f"candidate {self.candidate} is a retune, so its credit is "
                f"its slope alone, and an ablated arm was measured that "
                f"nothing may use")

    @property
    def slope(self) -> float:
        """`b`, the median of the per-round slopes."""
        return statistics.median(f.slope for f in self.per_round)

    @property
    def residue(self) -> float:
        """`c`, the non-scaling cost a rewrite also deletes.

        Zero for a retune by clause 18, and zero for a rewrite whose measured
        residue is smaller than the machine can resolve, because a number
        below the resolution floor is not a measurement.
        """
        if self.kind == RETUNE:
            return 0.0
        raw = self.raw_residue
        if self.resolution_floor is not None and abs(raw) < self.resolution_floor:
            return 0.0
        return raw

    @property
    def raw_residue(self) -> float:
        """The residue before clause 18's clamp, which the record retains."""
        if self.kind == RETUNE:
            return 0.0
        return self.pooled.intercept - self.ablated_median

    @property
    def residue_is_a_fault(self) -> bool:
        """Clause 33: a residue below `-R` refuses the profile.

        Committed clause 18 says such a residue "REJECTS that candidate's
        reading" and clause 27 puts impossible measurements in the refusal
        class rather than among the typed absences; clause 33 settles the two
        in favour of a FAULT. It says the step ran SLOWER with the operation
        removed than the fit predicts without its scaling part, which is not
        a measurement that can be true, so no rule carries it.
        """
        if self.kind == RETUNE or self.resolution_floor is None:
            return False
        return self.raw_residue < -self.resolution_floor

    @property
    def scaffold_case(self) -> str:
        """Clause 26's three cases, because a fit is a trend not a movement.

        An arm whose fitted slope is zero can still have moved: medians of
        100, 110, 110, 100 fit a slope of exactly zero while the arm moved by
        ten. So the RANGE decides first, and it is taken over the arm medians
        rather than the raw rounds, which is the same reduction the pooled
        fit uses.

        Clause 26 words its third case as "neither of the other two
        conditions holds", and read literally that leaves a gap where the
        excursion clears `R` but the shape limits fail. The clause also says
        the three cases are exhaustive, so the only consistent reading is
        that the third case is everything at or above `R` that the second
        does not take, and that is what runs here.
        """
        if self.resolution_floor is None or self.scaffold_range is None:
            return "unjudged"
        if self.scaffold_range < self.resolution_floor:
            return "clamped"
        if self.span is None:
            return "unjudged"
        excursion = abs(self.scaffold.slope) * self.span
        shaped = (self.scaffold.r_squared >= R_SQUARED_FLOOR
                  and abs(self.scaffold.max_residual)
                  <= RESIDUAL_LIMIT * abs(self.scaffold.slope))
        if excursion >= self.resolution_floor and shaped:
            return "fitted"
        return "unreadable"

    @property
    def effective_scaffold_slope(self) -> float:
        """Zero where clause 26 clamps, the fitted slope where it stands."""
        return 0.0 if self.scaffold_case == "clamped" else self.scaffold.slope

    @property
    def attributed(self) -> float:
        """`A`, what this candidate's operation costs the step."""
        return self.slope + self.residue

    @property
    def share(self) -> float:
        """`f`, the median reduction, per clause 18's statistics table."""
        return self.attributed / self.stock_median

    def blockers(self) -> list[str]:
        """Every reason this reading cannot be credited, in plain words.

        Gathered rather than raised: a knob that measured everything and then
        failed one gate is evidence, and the reason is worth naming.
        """
        problems = []
        # Clause 26 evaluates the SIGN rule first, on the knob's slope alone,
        # and demands it of BOTH estimators: an excursion condition written
        # on an absolute value cannot see a sign, so a dial whose registered
        # slope is negative would clear `10R` on magnitude and be admitted by
        # a rule committed text already refuses.
        if not self.pooled.is_a_dial:
            problems.append(
                f"candidate {self.candidate}: the fitted slope is "
                f"{self.pooled.slope:.6f}, which is not positive, so the "
                f"dial does not shrink the step and is not measuring the "
                f"operation it names")
        if self.slope <= 0.0:
            problems.append(
                f"candidate {self.candidate}: the median of the per-round "
                f"slopes is {self.slope:.6f}, which is not positive; that is "
                f"the slope the share is built from, so the dial does not "
                f"measure the operation it names whatever the pooled fit says")
        if self.pooled.r_squared < R_SQUARED_FLOOR:
            problems.append(
                f"candidate {self.candidate}: the fit's R-squared is "
                f"{self.pooled.r_squared:.4f}, below the registered "
                f"{R_SQUARED_FLOOR}, so the dial is not a dial")
        residual_limit = RESIDUAL_LIMIT * abs(self.pooled.slope)
        if abs(self.pooled.max_residual) > residual_limit:
            problems.append(
                f"candidate {self.candidate}: the largest residual is "
                f"{abs(self.pooled.max_residual):.6f} against a limit of "
                f"{residual_limit:.6f}, one twentieth of the slope")
        # Clause 26's clamp reaches clause 21's scaffold-slope limit rather
        # than sitting beside it: a scaffold whose whole excursion is below
        # what the machine resolves cannot be moving the knob's slope by a
        # resolvable amount, so the limit would guard against nothing.
        scaffold_limit = SCAFFOLD_SLOPE_LIMIT * abs(self.pooled.slope)
        case = self.scaffold_case
        if case == "unjudged":
            problems.append(
                f"candidate {self.candidate}: the scaffold's observed range "
                f"or the dial's actual span was not recorded, so clause 26's "
                f"three cases cannot be told apart and the scaffold's slope "
                f"cannot be read")
        elif case == "unreadable":
            problems.append(
                f"candidate {self.candidate}: the scaffold moved by "
                f"{self.scaffold_range:.6f}, at or above the resolution "
                f"floor, and its own line does not describe that movement, "
                f"so its price cannot be read and the dial is not eligible "
                f"at this width")
        elif (case == "fitted"
                and abs(self.scaffold.slope) > scaffold_limit):
            problems.append(
                f"candidate {self.candidate}: the scaffold's own slope is "
                f"{abs(self.scaffold.slope):.6f} against a limit of "
                f"{scaffold_limit:.6f}, one tenth of the knob's slope, so "
                f"the scaffold's cost is inside the number the knob reports")
        if self.resolution_floor is None:
            problems.append(
                f"candidate {self.candidate}: no resolution floor is "
                f"registered, so the scaffold offset of "
                f"{self.scaffold_offset:.6f} cannot be judged and the "
                f"residue cannot be tested against anything")
        else:
            offset_limit = SCAFFOLD_OFFSET_R_MULTIPLE * self.resolution_floor
            if abs(self.scaffold_offset) > offset_limit:
                problems.append(
                    f"candidate {self.candidate}: the scaffold offset is "
                    f"{abs(self.scaffold_offset):.6f} against a limit of "
                    f"{offset_limit:.6f}, three times the resolution floor")
            # Clause 26's fifth limit, demanded of BOTH estimators because a
            # condition on one certifies a number the rule does not consume.
            if self.span is None:
                problems.append(
                    f"candidate {self.candidate}: the dial's actual span was "
                    f"not recorded, so its excursion cannot be judged against "
                    f"the resolution floor")
            else:
                demand = EXCURSION_R_MULTIPLE * self.resolution_floor
                for name, value in (("pooled", self.pooled.slope),
                                    ("median", self.slope)):
                    excursion = abs(value) * self.span
                    if excursion < demand:
                        problems.append(
                            f"candidate {self.candidate}: the {name} "
                            f"excursion is {excursion:.6f} against a demand "
                            f"of {demand:.6f}, {EXCURSION_R_MULTIPLE:g} times "
                            f"the resolution floor, so the dial does not move "
                            f"the step by enough to be read")
            if self.residue_is_a_fault:
                problems.append(
                    f"candidate {self.candidate}: the raw residue is "
                    f"{self.raw_residue:.6f}, negative by more than the "
                    f"resolution floor of {self.resolution_floor:.6f}; the "
                    f"step ran slower with the operation removed than the fit "
                    f"predicts without its scaling part, which cannot be true")
        if not 0.0 < self.share < 1.0:
            problems.append(
                f"candidate {self.candidate}: the share is "
                f"{self.share:.4f}, which is not a fraction of a step")
        return problems


def ratio_lo(numerator: "KnobReading", denominator: "KnobReading") -> float:
    """`r`, the worst pairing the per-round samples permit.

    Section 4.2 credits the lower endpoint of a measured interval, so the
    smallest attributed cost divides the largest floor cost. That is
    deliberately the opposite reduction from the one `share` uses: a median
    share against a worst-case ratio gives the smaller gain, which is the
    conservative direction for a rule that decides what to build. The
    credited result is therefore a bound and not a predicted measurement.
    """
    if numerator.kind != denominator.kind:
        raise RunInvalid(
            f"the share is measured on a {numerator.kind} and the floor on a "
            f"{denominator.kind}, so `A` and `F` are not the same kind of "
            f"quantity and the formula they feed is not the registered one")
    smallest = min(f.slope for f in numerator.per_round) + numerator.residue
    largest = max(f.slope for f in denominator.per_round) + denominator.residue
    if largest <= 0.0:
        raise RunInvalid(
            "the floor's cost reduces to zero or less, so the ratio it would "
            "credit is not a number the gain formula accepts")
    return smallest / largest


def shape_ratio(stock_by_direction: Mapping[str, float],
                floor_by_direction: Mapping[str, float],
                counts: Mapping[str, int]) -> float:
    """One shape's ratio, as a ratio of SUMMED costs (Amendment 5 clause 8).

    Not an average of per-direction ratios, which is a different quantity.
    On equal counts with stock costs 100 and 20 against floor costs 100 and
    10, the average of ratios is 1.50 and the ratio of sums is 1.09, so the
    two land on opposite sides of the kill rule's 1.10 threshold and only the
    second answers "is stock close to the ceiling in the time it actually
    spends".
    """
    directions = sorted(counts)
    missing = [d for d in directions
               if d not in stock_by_direction or d not in floor_by_direction]
    if missing:
        raise RunInvalid(
            f"no cost measured for {missing} although the counts name them; "
            f"a direction that never ran is written as a count of zero, and "
            f"one that was never measured cannot be summed at all")
    stock = sum(counts[d] * stock_by_direction[d] for d in directions)
    floor = sum(counts[d] * floor_by_direction[d] for d in directions)
    if floor <= 0.0:
        raise RunInvalid("the floor's summed cost is zero or less")
    return stock / floor


def choose_dial(prices: Mapping[str, float],
                completeness: Sequence[str],
                settings: Mapping[str, int]) -> dict[str, object]:
    """Amendment 5 clause 15: the attention dial, chosen by a registered rule.

    `prices` are dimensionless, |scaffold slope| over |knob slope|, so the
    tie threshold is dimensionless too. The smallest price wins; prices
    within DIAL_TIE of each other are a tie, and a tie goes to the earlier
    entry in `completeness`, most complete first.

    `settings` is how many DISTINCT realisable settings each dial placed on
    the ladder. It is an argument rather than something the caller filters on
    beforehand because the eligibility rule is registered text, and a rule the
    harness can forget to apply is a rule that decides nothing.
    """
    unknown = sorted(set(prices) - set(completeness))
    if unknown:
        raise RunInvalid(
            f"{unknown} have a price but no registered completeness rank, so "
            f"a tie among them could not be broken by the registered rule")
    unmeasured = sorted(set(prices) - set(settings))
    if unmeasured:
        raise RunInvalid(
            f"{unmeasured} have a price but no count of realisable settings, "
            f"so clause 15's eligibility test cannot be applied to them")
    if not prices:
        raise RunInvalid("no dial was priced, so none can be chosen")

    refused = {name: settings[name] for name in prices
               if settings[name] < MIN_DIAL_SETTINGS}
    eligible = {name: price for name, price in prices.items()
                if name not in refused}
    if not eligible:
        raise RunInvalid(
            f"every priced dial placed fewer than {MIN_DIAL_SETTINGS} distinct "
            f"settings on the ladder ({refused}), so none is eligible and "
            f"candidate A has no dial")

    best = min(eligible.values())
    tied = [name for name, price in eligible.items() if price - best < DIAL_TIE]
    chosen = min(tied, key=completeness.index)
    return {
        "chosen": chosen,
        "tied": sorted(tied, key=completeness.index),
        "prices": dict(prices),
        "refused": refused,
        "by_completeness": chosen != min(eligible, key=lambda n: eligible[n]),
    }


# ---------------------------------------------------------------------------
# The live half. Everything below touches MLX and a device; everything above
# is arithmetic and is tested without one.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One setting of one dial, as the seams it installs while it traces."""

    label: str
    phi: float
    seams: Mapping[object, Callable] = field(default_factory=dict)


@dataclass
class CompiledArm:
    """A compiled step that has already traced, with its own trace counter.

    `traces` is a list the compiled body appends to. A compiled function's
    Python body runs only while tracing, so its length IS the trace count.
    `traced_by_warmup` is that length after the warm-ups, and any growth
    after it is the fault clause 25 exists to catch.
    """

    label: str
    phi: float
    step: Callable
    state: list
    traces: list
    traced_by_warmup: int

    def retraced(self) -> bool:
        return len(self.traces) > self.traced_by_warmup


def build_step(model, optimizer, state) -> tuple[Callable, list]:
    """A FRESH closure every call, which is what keeps arms apart.

    MLX keys its compile cache on the underlying callable, so two arms that
    share one raw function share one traced graph and the second arm's dial
    does nothing. Returning a new closure per call is the whole mechanism.

    Mirrors the step mlx-lm's own trainer compiles, including the branches
    this profile never takes, because a mirror with the unused half removed
    is a different function that happens to agree today.
    """
    from functools import partial

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.nn.utils import average_gradients
    from mlx.utils import tree_map
    from mlx_lm.tuner.trainer import default_loss

    loss_value_and_grad = nn.value_and_grad(model, default_loss)
    traces: list[int] = []

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        traces.append(1)
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)
        if do_update:
            grad = average_gradients(grad)
            optimizer.update(model, grad)
            grad = None
        return lvalue, toks, grad

    return step, traces


def settle_optimizer(model, optimizer) -> None:
    """Clause 25: give the optimizer its state before any arm is built.

    Adam allocates `m` and `v` on its first update, which grows the tree
    `mx.compile` captured and forces a second trace. If that trace lands
    after an arm's seam has been removed, the arm silently becomes stock.
    """
    import mlx.core as mx

    optimizer.init(model.trainable_parameters())
    mx.eval(optimizer.state)


def one_step(compiled: CompiledArm, batch, clear_cache_threshold: int = 0) -> float:
    """One step, timed exactly where mlx-lm's own loop times it.

    The cache clear is inside the timed region because it is inside
    mlx-lm's, and at the default threshold it runs every step. It is stock's
    real cost and the denominator has to carry it.
    """
    import mlx.core as mx
    from mlx_lm.tuner.trainer import _clear_cache

    start = time.perf_counter()
    lvalue, toks, grad = compiled.step(batch, None, True)
    mx.eval(compiled.state, lvalue, toks, grad)
    _clear_cache(clear_cache_threshold)
    return time.perf_counter() - start


def prepare_arm(model, optimizer, state, batch, arm: Arm,
                warmups: int = WARMUPS) -> CompiledArm:
    """Build one arm's compiled step, trace it UNDER its seams, then unhook.

    The ordering is the whole point and it is not an implementation detail:
    the trace happens at the first CALL, not at construction, so the seams
    have to be live across that first call. Once traced, the graph is fixed
    and the seams are irrelevant, which is why the timed rounds run with no
    installer active at all and pay none of its cost.
    """
    from metalrunner import seams

    installation = seams.Installation()
    try:
        for seam, wrap in arm.seams.items():
            installation.install(seam, wrap)
        step, traces = build_step(model, optimizer, state)
        compiled = CompiledArm(label=arm.label, phi=arm.phi, step=step,
                               state=state, traces=traces, traced_by_warmup=0)
        for _ in range(warmups):
            one_step(compiled, batch)
    finally:
        installation.remove()
    compiled.traced_by_warmup = len(traces)
    if installation.foreign_on_removal:
        raise RunInvalid(
            f"arm {arm.label}: something replaced "
            f"{installation.foreign_on_removal} while it was building, so "
            f"what traced is not what this arm describes")
    return compiled


def timed_rounds(arms: Sequence[CompiledArm], batch, *,
                 rounds: int = ROUNDS) -> dict[str, list[float]]:
    """Every arm timed inside every round, in rotation, with no seam active.

    Interleaving is load-bearing and NOT sufficient: it spreads a clock
    excursion across the arms but cannot detect one, so the caller still
    gates on the spread. Rotating the order stops any arm always running
    first, which is where allocation and cache effects land.
    """
    samples: dict[str, list[float]] = {arm.label: [] for arm in arms}
    for index in range(rounds):
        order = list(arms[index % len(arms):]) + list(arms[:index % len(arms)])
        for arm in order:
            samples[arm.label].append(one_step(arm, batch))
    retraced = [arm.label for arm in arms if arm.retraced()]
    if retraced:
        raise RunInvalid(
            f"arms {retraced} were re-traced during timed rounds; a trace "
            f"during timing is always a fault, and this one means the timed "
            f"graph is not the graph the seams produced")
    return samples


STOCK, KNOB, SCAFFOLD, ABLATION = "stock", "knob", "scaffold", "ablation"
ROLES = (STOCK, KNOB, SCAFFOLD, ABLATION)


@dataclass(frozen=True)
class ArmRole:
    """What one timed label WAS, for the reducer that reads its samples.

    Kept apart from `Arm`, which describes what to install while an arm
    traces. By the time samples exist the seams are long gone, and what the
    reducer needs is the label's role and the fraction the dial ACTUALLY
    placed, which is not always the fraction that was asked for.
    """

    label: str
    candidate: str
    role: str
    phi: float | None = None

    def __post_init__(self):
        if self.role not in ROLES:
            raise RunInvalid(
                f"arm {self.label!r} has role {self.role!r}, which is none of "
                f"{ROLES}")
        if self.role in (KNOB, SCAFFOLD) and self.phi is None:
            raise RunInvalid(
                f"arm {self.label!r} is a {self.role} arm and carries no "
                f"actual fraction, so it cannot enter a fit")


def kind_of(candidate: str) -> str:
    """Whether section 4.2's own names make this candidate a retune.

    Looked up rather than defaulted, because clause 18 credits a retune with
    its slope and a rewrite with its slope PLUS a residue, so a guess here is
    a credit nobody registered. Candidate A lives in the dial registry rather
    than in `KNOBS`, since which dial IS candidate A was a measurement.
    """
    if candidate in KNOBS:
        return KNOBS[candidate].kind
    for knob in ATTENTION_KNOBS.values():
        if knob.candidate == candidate:
            return knob.kind
    raise RunInvalid(
        f"candidate {candidate!r} is in no knob registry, so nothing says "
        f"whether section 4.2 names it a retune or a rewrite, and the two "
        f"are credited differently")


def reduce_width(samples: Mapping[str, Sequence[float]],
                 roles: Sequence[ArmRole], *,
                 resolution_floor: float,
                 rounds: int = ROUNDS) -> dict[str, KnobReading]:
    """One width's raw arm samples, reduced to one reading per candidate.

    Pure: it takes times and returns readings, so every rule downstream of it
    is testable without a device. Two quantities are computed HERE rather
    than taken from a caller, because both have a wrong version that looks
    right:

    the SPAN is the distance between the highest and lowest ACTUAL realisable
    fractions, not the nominal ladder's, since a dial that cannot place the
    setting it was asked for still gets fitted at the one it did place;

    the scaffold's observed RANGE is taken over its ARM MEDIANS, the same
    reduction the pooled fit uses, and not over the raw rounds, because four
    arm medians all equal to 100 can sit on rounds spanning 99 to 101 and the
    two readings disagree about whether the arm moved at all.
    """
    if resolution_floor is None or not resolution_floor > 0.0:
        raise RunInvalid(
            f"the resolution floor is {resolution_floor!r}; clause 26 needs a "
            f"positive one, because a floor of zero claims the machine can "
            f"resolve any difference at all")

    seen: set[str] = set()
    for role in roles:
        if role.label in seen:
            raise RunInvalid(
                f"two arms were recorded under the label {role.label!r}, so "
                f"one of them silently overwrote the other's samples")
        seen.add(role.label)
        if role.label not in samples:
            raise RunInvalid(
                f"the manifest names arm {role.label!r} and the recording "
                f"holds no samples for it, which is a hole and not an absence")
        if len(samples[role.label]) != rounds:
            raise RunInvalid(
                f"arm {role.label!r} carries {len(samples[role.label])} "
                f"samples against {rounds} rounds, so the rounds are not the "
                f"repeats the spread gate reads")
    extra = sorted(set(samples) - seen)
    if extra:
        raise RunInvalid(
            f"the recording holds samples for {extra}, which the manifest "
            f"does not name; an arm nobody registered cannot enter a fit")

    stock = [r for r in roles if r.role == STOCK]
    if len(stock) != 1:
        raise RunInvalid(
            f"a width needs exactly one stock arm and this one names "
            f"{len(stock)}; the share's denominator is stock's own time")
    stock_median = statistics.median(samples[stock[0].label])

    by_candidate: dict[str, dict[str, list[ArmRole]]] = {}
    for role in roles:
        if role.role == STOCK:
            continue
        by_candidate.setdefault(role.candidate, {}).setdefault(
            role.role, []).append(role)

    readings = {}
    for candidate in sorted(by_candidate):
        parts = by_candidate[candidate]
        knob = parts.get(KNOB, [])
        scaffold = parts.get(SCAFFOLD, [])
        ablation = parts.get(ABLATION, [])
        knob_phis = {r.label: r.phi for r in knob}
        scaffold_phis = {r.label: r.phi for r in scaffold}
        if len(set(knob_phis.values())) < MIN_DIAL_SETTINGS:
            raise RunInvalid(
                f"candidate {candidate} placed "
                f"{len(set(knob_phis.values()))} distinct actual settings and "
                f"a dial needs {MIN_DIAL_SETTINGS} to be shown linear; two "
                f"nominal settings that round to one size are one setting")
        if set(scaffold_phis.values()) != set(knob_phis.values()):
            raise RunInvalid(
                f"candidate {candidate}'s scaffold was placed at "
                f"{sorted(set(scaffold_phis.values()))} and its knob at "
                f"{sorted(set(knob_phis.values()))}; a scaffold measured at "
                f"different settings prices a different arm")
        if len(ablation) > 1:
            raise RunInvalid(
                f"candidate {candidate} carries {len(ablation)} ablated arms "
                f"and clause 18 credits one residue")

        actual = sorted(knob_phis.values())
        span = actual[-1] - actual[0]
        scaffold_medians = {label: statistics.median(samples[label])
                            for label in scaffold_phis}
        full = max(scaffold_phis, key=lambda label: scaffold_phis[label])
        readings[candidate] = KnobReading(
            candidate=candidate,
            kind=kind_of(candidate),
            stock_median=stock_median,
            per_round=tuple(per_round_fits(samples, knob_phis, rounds)),
            pooled=pooled_fit(samples, knob_phis),
            scaffold=pooled_fit(samples, scaffold_phis),
            scaffold_offset=scaffold_medians[full] - stock_median,
            ablated_median=(statistics.median(samples[ablation[0].label])
                            if ablation else None),
            resolution_floor=resolution_floor,
            span=span,
            scaffold_range=max(scaffold_medians.values())
            - min(scaffold_medians.values()),
        )
    return readings


def per_round_fits(samples: Mapping[str, Sequence[float]],
                   phis: Mapping[str, float],
                   rounds: int = ROUNDS) -> list[Fit]:
    """One line per round, across that round's settings of one dial."""
    labels = sorted(phis, key=lambda label: -phis[label])
    fits = []
    for index in range(rounds):
        fits.append(fit([phis[label] for label in labels],
                        [samples[label][index] for label in labels]))
    return fits


def pooled_fit(samples: Mapping[str, Sequence[float]],
               phis: Mapping[str, float]) -> Fit:
    """One line over the arm medians, which is what the gates read."""
    labels = sorted(phis, key=lambda label: -phis[label])
    return fit([phis[label] for label in labels],
               [statistics.median(samples[label]) for label in labels])


# ---------------------------------------------------------------------------
# The dials themselves. Each one shrinks its operation's SIZE while holding
# the call count, the output shape and the graph structure fixed.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Knob:
    """One candidate's dial, and everything needed to drive it.

    `regions` names entries in `profile_instrument.REGIONS` rather than
    repeating their seams, so where a region lives is written once.
    `prepare(model, width)` materialises every setting's operands OUTSIDE any
    timed region, so the arms differ in the work they do and not in when they
    paid for it. It takes the batch width because one dial's ladder is a
    property of the batch rather than of the model: shortening the keys
    changes the mask, and a mask built inside the step is graph every round
    pays for whose size moves with the dial.
    """

    candidate: str
    kind: str
    regions: tuple[str, ...]
    prepare: Callable
    arm: Callable
    scaffold: Callable
    ablate: Callable | None = None

    def __post_init__(self):
        if self.kind == REWRITE and self.ablate is None:
            raise RunInvalid(
                f"candidate {self.candidate} is a rewrite, so its credit "
                f"needs a residue, and no ablated arm is defined for it")


def _group_aligned(phi: float, dims: int, group: int) -> int:
    """A dialled input width that is still a valid quantized operand.

    Rounded to a whole number of quantization groups, because a partial group
    has no scale of its own.
    """
    return max(group, int(round(phi * dims / group)) * group)


def _projection_ladder(model, phis=PHIS) -> dict:
    """Every quantized projection, cut to a prefix of its input dimension."""
    import mlx.core as mx
    import mlx.nn as nn

    ladder = {phi: {} for phi in phis}
    for _name, module in model.named_modules():
        if not isinstance(module, nn.QuantizedLinear):
            continue
        weight = module["weight"]
        _out_dims, packed = weight.shape
        in_dims = packed * 32 // module.bits
        for phi in phis:
            in_k = _group_aligned(phi, in_dims, module.group_size)
            packed_k = in_k * module.bits // 32
            biases = module.get("biases")
            entry = (
                in_k,
                mx.contiguous(weight[:, :packed_k]),
                mx.contiguous(module["scales"][:, :in_k // module.group_size]),
                (mx.contiguous(biases[:, :in_k // module.group_size])
                 if biases is not None else None),
            )
            mx.eval(*[part for part in entry[1:] if part is not None])
            ladder[phi][id(module)] = entry
    return ladder


def _head_ladder(model, axis: str, phis=PHIS) -> dict:
    """The tied head, cut along one axis.

    `axis` is "input" for candidate Q, which dials the hidden dimension the
    head contracts over, and "vocabulary" for candidate L, which dials the
    rows the loss then reduces across. They are different axes of the same
    weight and the two candidates are never composed, per clause 13.
    """
    import mlx.core as mx

    embedding = model.model.embed_tokens
    weight = embedding["weight"]
    scales = embedding["scales"]
    biases = embedding.get("biases")
    rows, packed = weight.shape
    in_dims = packed * 32 // embedding.bits

    ladder = {}
    for phi in phis:
        if axis == "vocabulary":
            kept = max(1, int(round(phi * rows)))
            cut = (mx.contiguous(weight[:kept]), mx.contiguous(scales[:kept]),
                   mx.contiguous(biases[:kept]) if biases is not None else None)
        else:
            kept = _group_aligned(phi, in_dims, embedding.group_size)
            packed_k = kept * embedding.bits // 32
            groups = kept // embedding.group_size
            cut = (mx.contiguous(weight[:, :packed_k]),
                   mx.contiguous(scales[:, :groups]),
                   mx.contiguous(biases[:, :groups]) if biases is not None
                   else None)
        mx.eval(*[part for part in cut if part is not None])
        ladder[phi] = (kept, *cut)
    return ladder


def _projection_call(ladder_at_phi, *, slice_operand: bool):
    """Candidate Q's and P3's per-call replacement.

    With `slice_operand` false this is the scaffold-only arm of clause 5: the
    dial's index arithmetic runs and is discarded, and the operation runs at
    full size, so what the arm costs is the scaffold and nothing else.
    """
    import mlx.core as mx

    def wrap(original):
        def call(self, x):
            in_k, weight, scales, biases = ladder_at_phi[id(self)]
            cut = x[..., :in_k]
            if not slice_operand:
                return original(self, x)
            y = mx.quantized_matmul(
                cut, weight, scales=scales, biases=biases, transpose=True,
                group_size=self.group_size, bits=self.bits, mode=self.mode)
            return y + self["bias"] if "bias" in self else y
        return call
    return wrap


def _head_call(entry, axis: str, *, slice_operand: bool):
    """The head's per-call replacement, on whichever axis the candidate dials."""
    import mlx.core as mx

    kept, weight, scales, biases = entry

    def wrap(original):
        def call(self, x):
            cut = x[..., :kept] if axis == "input" else x
            if not slice_operand:
                return original(self, x)
            return mx.quantized_matmul(
                cut, weight, scales=scales, biases=biases, transpose=True,
                group_size=self.group_size, bits=self.bits, mode=self.mode)
        return call
    return wrap


def _loss_call(kept: int, *, apply: bool):
    """Cross-entropy over a dialled vocabulary.

    The targets are taken modulo the kept width because the loss indexes them
    into the logits and a target beyond the slice would be out of range. Its
    cost depends on the width of the logits and not on which column a target
    names, so this keeps the work honest while keeping the indices valid. The
    loss VALUE changes, which is why no arm here is compared against stock's
    loss and only times are compared.
    """
    def wrap(original):
        def call(logits, targets, *args, **kwargs):
            reduced = targets % kept
            if not apply:
                return original(logits, targets, *args, **kwargs)
            return original(logits, reduced, *args, **kwargs)
        return call
    return wrap


def _seam(region: str):
    import profile_instrument as pi

    return pi.REGIONS[region].seam


def _projection_knob(candidate: str, regions: tuple[str, ...],
                     with_head: bool) -> Knob:
    """Candidate Q, and the projections-only region P3 clause 22 needs.

    They differ only in whether the tied head moves with the projections.
    Section 3.3 puts the head inside candidate Q, so Q dials it; the
    partition region excludes it so that P1, P2 and P3 are disjoint and their
    sum means something.
    """
    def prepare(model, _width):
        prepared = {"projections": _projection_ladder(model)}
        if with_head:
            prepared["head"] = _head_ladder(model, "input")
        return prepared

    def arm(prepared, phi, *, slice_operand=True):
        seams_map = {
            _seam("qmm"): _projection_call(prepared["projections"][phi],
                                           slice_operand=slice_operand),
        }
        if with_head:
            seams_map[_seam("head-matmul")] = _head_call(
                prepared["head"][phi], "input", slice_operand=slice_operand)
        return seams_map

    return Knob(
        candidate=candidate,
        kind=RETUNE,
        regions=regions,
        prepare=prepare,
        arm=arm,
        scaffold=lambda prepared, phi: arm(prepared, phi, slice_operand=False),
    )


def _loss_knob() -> Knob:
    """Candidate L: the output head and the loss, on one vocabulary dial.

    One dial moves both of L's registered regions, because slicing the head's
    rows shrinks its matmul AND the logits the loss reduces over. It is a
    REWRITE: a streamed kernel never builds the full logits tensor at all, so
    its credit is its slope plus the residue that deletion also removes.
    """
    def prepare(model, _width):
        return {"head": _head_ladder(model, "vocabulary")}

    def arm(prepared, phi, *, slice_operand=True):
        kept = prepared["head"][phi][0]
        return {
            _seam("head-matmul"): _head_call(prepared["head"][phi],
                                             "vocabulary",
                                             slice_operand=slice_operand),
            _seam("cross-entropy"): _loss_call(kept, apply=slice_operand),
        }

    return Knob(
        candidate="L",
        kind=REWRITE,
        regions=("head-matmul", "cross-entropy"),
        prepare=prepare,
        arm=arm,
        scaffold=lambda prepared, phi: arm(prepared, phi, slice_operand=False),
        ablate=lambda _prepared: _ablate_loss(),
    )


def _ablate_loss():
    """Candidate L's two regions removed outright, with the backward kept.

    Attention's ablated arm can multiply its operands by zero because the
    queries still pass through carrying a gradient of one. That trick cannot
    be used here: the loss is the ROOT of the backward, so a zero derivative
    would make every cotangent below it zero and delete the whole step
    instead of this candidate's two regions, while still producing a number
    and a faster time.

    So the head returns a one-column slice of its own input and the loss
    reads that column. No logits tensor is built at all, which is exactly
    what a streamed mask-aware kernel also never builds, and the hidden
    states keep a gradient of one on the column that survives, so the model's
    own backward runs at full size.

    The loss VALUE is meaningless here, as it is at every dialled setting,
    and nothing compares it against stock's; only times are compared.
    """
    def wrap_head(_original):
        def call(_self, x):
            return x[..., :1]
        return call

    def wrap_loss(_original):
        def call(logits, _targets, *_args, **_kwargs):
            return logits[..., 0]
        return call

    return {_seam("head-matmul"): wrap_head,
            _seam("cross-entropy"): wrap_loss}


def _causal_mask(queries: int, keys: int):
    """The mask stock builds for itself, as an array, over a shortened key set.

    The training path passes the STRING "causal", which MLX reads as "query i
    attends key j when j <= i". That reading is only stock's own mask while
    the two lengths agree: cut the keys and the diagonal runs off the end of
    the array, leaving early rows with nothing to attend to and a softmax over
    an empty row. So every arm of the length dial hands over an array it built
    itself, in which query i attends keys 0 through the smaller of i and the
    last key kept, and no row is ever empty.

    At the full setting the array IS what the string means, checked by a test
    against stock's own value and gradients rather than by reading MLX.
    """
    import mlx.core as mx

    mask = mx.arange(keys)[None, :] <= mx.arange(queries)[:, None]
    mx.eval(mask)
    return mask


def attention_width(batch_width: int) -> int:
    """The query length attention sees, which is NOT the batch's token count.

    mlx-lm's `default_loss` trains on `batch[:, :-1]` and predicts
    `batch[:, 1:]`, so a batch 97 tokens wide runs a 96-token step. The length
    dial builds its masks against this number, and the rule lives here rather
    than at each call site because a caller who forgets it produces a mask one
    token too wide, which is a broadcast failure at best.
    """
    return batch_width - 1


def _attention_ladder(model, dial: str, batch_width: int, phis=PHIS) -> dict:
    """One attention dial's settings, with everything each arm needs ready.

    Masks are built here and not inside the step. An array built in the traced
    body is graph the arm pays for on every round, and its size moves with the
    dial, so its cost would sit inside the fitted slope where clause 5's
    scaffold gate could not see it.
    """
    if dial not in ATTENTION_DIALS:
        raise RunInvalid(
            f"{dial!r} is not one of the three dials clause 15 registers: "
            f"{ATTENTION_DIALS}")

    head_dim = model.args.head_dim
    width = attention_width(batch_width)
    ladder = {}
    for phi in phis:
        if dial == "kv-length":
            kept = max(1, int(round(phi * width)))
            ladder[phi] = {"kept": kept, "mask": _causal_mask(width, kept),
                           "full": width,
                           "full_mask": _causal_mask(width, width)}
        else:
            kept = max(1, int(round(phi * head_dim)))
            ladder[phi] = {"kept": kept, "full": head_dim}
    return ladder


def realisable_settings(ladder: Mapping[float, Mapping]) -> tuple[float, ...]:
    """The settings of a ladder that are distinct computations.

    Two dial positions that round to one kept size are ONE setting, not two.
    Their arms compute the same thing, so the fit would carry a repeated point
    that adds no evidence about linearity while raising R-squared, and clause
    15's three-setting eligibility test would pass on a ladder with two.
    """
    first_at_size: dict[int, float] = {}
    for phi, entry in ladder.items():
        first_at_size.setdefault(entry["kept"], phi)
    return tuple(sorted(first_at_size.values(), reverse=True))


def _attention_call(dial: str, entry: Mapping, *, apply: bool):
    """One attention dial's per-call replacement.

    With `apply` false this is clause 5's scaffold-only arm: the dial's own
    machinery runs, the operation runs at full size, and the value returned is
    stock's. The three dials carry different machinery, so each says here what
    its scaffold arm reproduces and what it cannot.
    """
    import mlx.core as mx

    kept, full = entry["kept"], entry["full"]

    def wrap(original):
        if dial == "kv-length":
            def call(queries, keys, values, *args, **kwargs):
                if queries.shape[2] != full:
                    raise RunInvalid(
                        f"this dial's masks were built for {full} queries and "
                        f"the step is running {queries.shape[2]}, so every arm "
                        f"would mask the wrong score matrix")
                cut_k, cut_v = keys[:, :, :kept], values[:, :, :kept]
                if not apply:
                    # This dial's machinery is the array mask, and its cost is
                    # proportional to the score matrix it covers. The scaffold
                    # arm applies it at FULL size, which puts it in the
                    # scaffold offset; the part of it that moves with the dial
                    # is inside the knob slope and clause 5's slope statistic
                    # cannot separate it. Reported, not corrected.
                    kwargs["mask"] = entry["full_mask"]
                    return original(queries, keys, values, *args, **kwargs)
                kwargs["mask"] = entry["mask"]
                return original(queries, cut_k, cut_v, *args, **kwargs)
        elif dial == "head-dim-qkv":
            def call(queries, keys, values, *args, **kwargs):
                cut = (queries[..., :kept], keys[..., :kept],
                       values[..., :kept])
                if not apply:
                    # The pad grows as the dial shrinks, so the scaffold arm
                    # writes exactly as many padded columns as the real arm
                    # does and then throws them away. At the full setting it
                    # writes none, which is the one arm of this dial that is
                    # structurally unlike the rest.
                    padded = mx.pad(original(queries, keys, values,
                                             *args, **kwargs),
                                    [(0, 0)] * 3 + [(0, full - kept)])
                    return padded[..., :full]
                return mx.pad(original(*cut, *args, **kwargs),
                              [(0, 0)] * 3 + [(0, full - kept)])
        else:
            def call(queries, keys, values, *args, **kwargs):
                cut_q, cut_k = queries[..., :kept], keys[..., :kept]
                if not apply:
                    return original(queries, keys, values, *args, **kwargs)
                return original(cut_q, cut_k, values, *args, **kwargs)
        return call
    return wrap


def _ablate_attention():
    """Attention removed outright, with keys and values still consumed.

    Clause 18 credits a rewrite with its slope plus the residue between the
    fitted intercept and this arm, so what this arm removes has to be
    attention and nothing else. Returning only the queries would let a lazy
    graph drop the key and value projections and charge them here, so both are
    consumed through a reduction multiplied by zero. That is what the ablation
    probe does, which is why its figure and this one are comparable.
    """
    def wrap(_original):
        def call(queries, keys, values, *args, **kwargs):
            return queries + (keys.sum() * 0.0 + values.sum() * 0.0)
        return call
    return wrap


def _attention_knob(dial: str) -> Knob:
    """Candidate A, dialled one of the three ways clause 15 registers.

    A REWRITE, not a retune: a tiled kernel fuses the whole region and never
    materialises the score matrix, so it deletes cost that lives in the
    intercept and a slope alone would under-credit it.
    """
    def prepare(model, width):
        return {"attention": _attention_ladder(model, dial, width)}

    def arm(prepared, phi, *, slice_operand=True):
        return {_seam("attn-core"): _attention_call(
            dial, prepared["attention"][phi], apply=slice_operand)}

    return Knob(
        candidate="A",
        kind=REWRITE,
        regions=("attn-core",),
        prepare=prepare,
        arm=arm,
        scaffold=lambda prepared, phi: arm(prepared, phi, slice_operand=False),
        ablate=lambda _prepared: {_seam("attn-core"): _ablate_attention()},
    )


# The three candidate dials, all built, none of them yet candidate A's. Clause
# 15 picks among them by a criterion registered before any of the three was
# measured, and `bench/mlx_probes/probe_attention_dials.py` applies it.
ATTENTION_KNOBS = {dial: _attention_knob(dial) for dial in ATTENTION_DIALS}

# Candidate A is still absent from this registry on purpose. The three dials
# above exist; which one IS candidate A is a measurement, and until that
# measurement is recorded a caller asking for "A" gets a KeyError rather than
# a dial somebody guessed at.
KNOBS = {
    "Q": _projection_knob("Q", ("qmm", "head-matmul"), with_head=True),
    "P3": _projection_knob("P3", ("qmm",), with_head=False),
    "L": _loss_knob(),
}
