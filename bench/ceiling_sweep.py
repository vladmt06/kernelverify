#!/usr/bin/env python3
"""The ceiling sweep: the kill rule's own arms, and candidate L's bench floor.

Two benches, and they are two because the pre-registration makes them two.
Neither of them rules. Both produce measurements that `profile_rules` reads.

What is NOT here, and why
-------------------------
Candidate Q's dense fp16 floor is not here. Amendment 5 clause 19 installs it
at the same seam candidate Q's dial uses INSIDE the real training step, and
clause 9 takes the credited ratio "from the same dial at the same settings in
the same rounds", so those arms are arms of `profile_stock`'s own manifest.
Amendment 10 clause 45 puts them in the same resolution CONTEXT as candidate
Q's share arms for the same reason. A sweep that also held candidate Q's floor
would be measuring `r` on a bench while `f` was measured in a step, which is
the exact pairing clause 19 exists to forbid.

Candidate A has no floor at all, per Amendment 6 clause 27. Its fused entry
point cannot be reached from inside a gradient trace, so an in-step floor arm
would run stock's own computation and report a ratio of 1.0 that looked
measured. What replaces it is a ceiling on candidate A's best possible score,
and no arm here produces it.

The kill bench, Amendment 10 clause 44
---------------------------------------
Clause 8 reads a per-shape ratio of costs SUMMED OVER DIRECTIONS. Nothing in
the certified vector produces one: clause 43 makes the credited denominator a
whole-step slope over every shape at once, and clause 3 demoted the only
instrument that separated directions. So the kill rule takes its own arms.

An in-step construction does exist and clause 44 records it rather than
claiming impossibility: time the stock step, time a step where one shape's
live calls use the dense floor, time a step where those calls are ablated, and
each difference against the common ablated arm is that shape's full aggregate
cost. It is declined for cost and because an ablated arm changes the graph,
and the bench is a CHOICE. That is written in the amendment so a later reader
does not have to rediscover it from this file.

48 timed arms: six shapes at two widths, and at each of those twelve a stock
forward, a stock gradient, a dense forward and a dense gradient. Both
implementations, because a ceiling needs the thing and the thing it is a
ceiling for; a gradient against its own forward, because that is how a
backward is read.

What a backward arm IS, Amendment 11 clause 48
-----------------------------------------------
Clause 8 reads a "cost per call" in each direction and never said which of
three constructions a backward call's cost is. Clause 48 registers one: the
gradients of `mx.vjp(f, [x], [cotangent])` evaluated ON THEIR OWN.

Not the gradient minus a separately timed forward. Asked only for the
gradients, MLX's lazy graph never builds the forward at all: read off the
built graph, the gradients of a quantized matmul hold exactly ONE primitive, a
`QuantizedMatmul`, and no forward. So the gradients alone ARE the backward and
subtracting a forward from them over-subtracts, which reads about half again
too high at the one shape with any headroom.

Amendment 10 clause 44 claimed the cotangent moves this number by a factor of
about two hundred. Clause 47 WITHDRAWS that: the comparison set a subtracted
quantity beside an unsubtracted one, and measured the same way the two
cotangents agree to within about a tenth everywhere. The dense cotangent
stays REGISTERED because a real loss produces one, which is a reason about
faithfulness rather than about magnitude.

There is still no code path here that differentiates a `sum()`. Every gradient
arm takes an explicit cotangent and `_dense_cotangent` refuses one whose shape
is not the forward output's.

Candidate L's bench, clause 19's written exception
---------------------------------------------------
Candidate L's floor is a streamed kernel that does not exist, so it cannot be
installed at a seam and dialled inside the step the way candidate Q's can.
Clause 19 registers the exception in writing: its `d` is measured in
isolation, its `c_floor` is registered as ZERO because eliminating the full
logits tensor is precisely what that candidate is, and both facts are reported
wherever candidate L's gain appears.

Nine arms and its own resolution context, which clause 21 forbids sharing with
the step's.
"""

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import profile_knobs as pk
import profile_rules as rules
from decode_rules import RunInvalid

FORWARD, BACKWARD = "forward", "backward"
DIRECTIONS = (FORWARD, BACKWARD)

# Clause 8's two implementations. `stock` is the quantized matmul the model
# actually runs; `dense` is the fp16 ceiling section 4.2 registers, which
# skips dequantisation and moves about four times the weight bytes, so at
# small token counts the ordering can invert and the observed ratio is
# reported whatever it is.
STOCK, DENSE = "stock", "dense"
IMPLEMENTATIONS = (STOCK, DENSE)

# The kill bench is REPORTED under Amendment 7 clause 36, so it carries no
# certified margin, needs no `R` and takes no demand. What it carries instead,
# per clause 37, is its own observed per-round samples and their observed
# range, labelled as an observed spread and NOT as a bound at any rate.
CERTIFIED = False

ROUNDS = pk.ROUNDS
WARMUPS = pk.WARMUPS


@dataclass(frozen=True)
class KillArm:
    """One timed arm of the kill bench, before anything has run.

    Pure and nominal. Four per shape per width, which is what makes 48 rather
    than the 24 an earlier draft of Amendment 10 counted by dropping the
    implementation.
    """

    label: str
    shape: str
    width: str
    implementation: str
    direction: str

    def __post_init__(self):
        if self.shape not in rules.SHAPES:
            raise RunInvalid(
                f"arm {self.label!r} names shape {self.shape!r}, and clause "
                f"23 counts only the registered {sorted(rules.SHAPES)}")
        if self.width not in rules.WIDTHS:
            raise RunInvalid(
                f"arm {self.label!r} names width {self.width!r}, and the "
                f"registered widths are {sorted(rules.WIDTHS)}")
        if self.implementation not in IMPLEMENTATIONS:
            raise RunInvalid(
                f"arm {self.label!r} is a {self.implementation!r} arm and a "
                f"ceiling reads {IMPLEMENTATIONS}")
        if self.direction not in DIRECTIONS:
            raise RunInvalid(
                f"arm {self.label!r} runs {self.direction!r} and clause 8 "
                f"sums over {DIRECTIONS}")


def kill_manifest() -> tuple[KillArm, ...]:
    """Amendment 10 clause 44's 48 arms, in build and record order."""
    arms = []
    for width in rules.WIDTH_ORDER:
        for shape in sorted(rules.SHAPES):
            for implementation in IMPLEMENTATIONS:
                for direction in DIRECTIONS:
                    arms.append(KillArm(
                        label=f"{width}:{shape}:{implementation}:{direction}",
                        shape=shape, width=width,
                        implementation=implementation, direction=direction))
    labels = [arm.label for arm in arms]
    if len(set(labels)) != len(labels):
        raise RunInvalid("the kill manifest repeats a label")
    return tuple(arms)


def shape_counts(structural: Mapping[str, object]) -> dict[str, dict[str, int]]:
    """Clause 8's call counts, taken from clause 3's structural pass.

    Keyed by REGISTERED shape name rather than by the `OUTxIN` string the
    instrument tallies, because the kill rule reads registered shapes and a
    tally keyed on raw dimensions cannot be joined to it without somebody
    inventing the join at read time.

    A shape the structural pass never saw is not zero and is not an error to
    swallow: the run refuses, because a shape counted at zero contributes
    nothing to a ratio of summed costs and would silently change what the
    ceiling was compared over.
    """
    tallies = structural.get("shape_counts") or {}
    by_dimension: dict[str, dict[str, int]] = {}
    for _region, shapes in sorted(tallies.items()):
        for key, directions in sorted(shapes.items()):
            entry = by_dimension.setdefault(key, {FORWARD: 0, BACKWARD: 0})
            for direction in DIRECTIONS:
                entry[direction] += int(directions.get(direction, 0))

    counts = {}
    for name, (out_dims, in_dims) in sorted(rules.SHAPES.items()):
        key = f"{out_dims}x{in_dims}"
        if key not in by_dimension:
            raise RunInvalid(
                f"clause 8 weights shape {name} at ({out_dims}, {in_dims}) by "
                f"its call counts and the structural pass recorded none; a "
                f"shape counted at zero drops out of a ratio of summed costs")
        counts[name] = dict(by_dimension[key])
    return counts


def _finite(value: float, what: str) -> float:
    if not math.isfinite(value):
        raise RunInvalid(f"{what} is {value!r} and a ratio needs a number")
    return value


def per_round_shape_ratios(measured: Mapping[str, Sequence[float]],
                           counts: Mapping[str, int]) -> list[float]:
    """Clause 8's ratio of SUMMED costs, one per round.

    `measured` carries the four per-round sample lists this shape produced at
    one width, keyed `{implementation}:{direction}`. The sums are taken WITHIN
    each round, which is clause 8's own reduction and not a convenience:

        shape ratio = sum over directions of (calls * stock cost per call)
                    / sum over directions of (calls * floor cost per call)

    An average of per-direction ratios is a different quantity and clause 8
    names it as the wrong one, with its own worked case: on equal counts with
    stock costs 100 and 20 against floor costs 100 and 10 the average of
    ratios is 1.50 and the ratio of sums is 1.09, so the two land on opposite
    sides of the threshold.
    """
    wanted = {f"{implementation}:{direction}"
              for implementation in IMPLEMENTATIONS for direction in DIRECTIONS}
    missing = sorted(wanted - set(measured))
    if missing:
        raise RunInvalid(
            f"a shape ratio sums both implementations over both directions "
            f"and these arms are absent: {missing}")
    extra = sorted(set(measured) - wanted)
    if extra:
        raise RunInvalid(
            f"the ratio reads {sorted(wanted)} and was handed {extra}")

    lengths = {len(measured[key]) for key in wanted}
    if len(lengths) != 1:
        raise RunInvalid(
            f"the four arms carry {sorted(lengths)} rounds; a sum taken WITHIN "
            f"a round needs every arm to have run in that round")
    rounds = lengths.pop()
    if rounds < 1:
        raise RunInvalid("a shape ratio needs at least one round")

    for direction in DIRECTIONS:
        if direction not in counts:
            raise RunInvalid(
                f"clause 8 weights each direction by its own call count and "
                f"none was supplied for {direction!r}")
    if all(int(counts[direction]) == 0 for direction in DIRECTIONS):
        raise RunInvalid(
            "a shape whose every direction ran zero times contributes nothing "
            "to a ratio of summed costs, so its ratio is not a measurement")

    ratios, dropped = [], []
    for index in range(rounds):
        samples = {
            (implementation, direction): _finite(
                measured[f"{implementation}:{direction}"][index],
                f"the {implementation} {direction} sample")
            for implementation in IMPLEMENTATIONS for direction in DIRECTIONS}
        # Amendment 11 clause 48. A non-positive sample is a clock excursion,
        # not a small cost: nothing here subtracts, so a backward cannot come
        # out negative by construction. The ROUND is dropped rather than the
        # sample, because a ratio is a sum taken WITHIN a round and half a
        # round is not one.
        if any(value <= 0.0 for value in samples.values()):
            dropped.append(index)
            continue
        # Every sample in this round is positive and at least one direction
        # has a positive count, so the denominator cannot be zero and there is
        # no divide-by-zero branch to write. Both facts are checked above.
        totals = {
            implementation: sum(int(counts[direction])
                                * samples[(implementation, direction)]
                                for direction in DIRECTIONS)
            for implementation in IMPLEMENTATIONS}
        ratios.append(totals[STOCK] / totals[DENSE])
    if not ratios:
        raise RunInvalid(
            f"every one of {rounds} rounds carried a sample at or below zero, "
            f"so this shape has no ratio at all")
    if dropped:
        # Recorded on the function's own result rather than swallowed: a round
        # silently dropped changes what "the largest per-round ratio" was
        # taken over, which is the reduction clause 8 registers.
        ratios = _WithDropped(ratios, dropped)
    return ratios


class _WithDropped(list):
    """Per-round ratios that remember which rounds were dropped and why.

    A list, so every caller reads it as one; with the drop recorded, so a
    recording can carry the count and a reader can see that the reduction ran
    over fewer rounds than the manifest says.
    """

    def __init__(self, ratios, dropped):
        super().__init__(ratios)
        self.dropped_rounds = tuple(dropped)


def shape_at_width(measured: Mapping[str, Sequence[float]],
                   counts: Mapping[str, int]) -> rules.ShapeAtWidth:
    """One shape at one width, in the shape `profile_rules.kill_q` reads.

    `ratio` is clause 8's LARGEST per-round ratio, which is the one furthest
    from the ceiling and therefore the least likely to kill, the conservative
    direction for a rule whose effect is to remove a candidate.

    `numerator_high` and `denominator_low` carry the corners the ratio was
    taken over, so a reader can see the pairing rather than trust it.
    """
    ratios = per_round_shape_ratios(measured, counts)
    dropped = set(getattr(ratios, "dropped_rounds", ()))
    rounds = len(measured[f"{STOCK}:{FORWARD}"])
    numerators, denominators = [], []
    for index in range(rounds):
        if index in dropped:
            continue
        numerators.append(sum(
            int(counts[direction]) * measured[f"{STOCK}:{direction}"][index]
            for direction in DIRECTIONS))
        denominators.append(sum(
            int(counts[direction]) * measured[f"{DENSE}:{direction}"][index]
            for direction in DIRECTIONS))
    return rules.ShapeAtWidth(
        ratio=rules.largest_round_ratio(ratios),
        numerator_high=max(numerators),
        denominator_low=min(denominators))


def kill_readings(samples: Mapping[str, Sequence[float]],
                  counts: Mapping[str, Mapping[str, int]]
                  ) -> dict[str, dict[str, rules.ShapeAtWidth]]:
    """Every arm's samples, reduced to what the kill rule reads.

    Pure: times in, readings out, so clause 8's arithmetic is testable without
    a device and the bench cannot quietly become the thing that decides.
    """
    manifest = kill_manifest()
    expected = {arm.label for arm in manifest}
    if set(samples) != expected:
        raise RunInvalid(
            f"the kill bench recorded {sorted(set(samples) - expected)} the "
            f"manifest does not name and is missing "
            f"{sorted(expected - set(samples))}; a hole is not an absence")

    grouped: dict[str, dict[str, dict[str, Sequence[float]]]] = {}
    for arm in manifest:
        grouped.setdefault(arm.shape, {}).setdefault(arm.width, {})[
            f"{arm.implementation}:{arm.direction}"] = samples[arm.label]

    readings = {}
    for shape in sorted(grouped):
        if shape not in counts:
            raise RunInvalid(
                f"shape {shape} has no call counts, and clause 8 weights a "
                f"cost by how often it ran")
        readings[shape] = {
            width: shape_at_width(grouped[shape][width], counts[shape])
            for width in sorted(grouped[shape])}
    return readings


def observed_spread(samples: Mapping[str, Sequence[float]]) -> dict:
    """What a REPORTED quantity carries in place of a calibrated interval.

    Clause 37 registers it exactly: this run's own per-round samples and their
    observed range, recorded and labelled as an observed spread and NOT as a
    bound at any registered rate. The label travels with the number because
    without it a reader supplies the missing word themselves.
    """
    from decode_rules import spread_pct

    return {
        "certified": CERTIFIED,
        "observed_spread_pct": {label: spread_pct(list(values))
                                for label, values in sorted(samples.items())},
        "observed_spread_is_not_a_bound": (
            "an observed range over this run's own rounds, not an interval at "
            "any registered rate: Amendment 10 clause 44 retires clause 36's "
            "per-shape and per-direction resolution contexts because a "
            "reported verdict reaches no terminal and takes no margin"),
    }


# ---------------------------------------------------------------------------
# The device side. Everything above is arithmetic over samples.
# ---------------------------------------------------------------------------
def _dense_cotangent(output):
    """The cotangent every backward arm is driven by, and the refusal.

    Amendment 10 clause 44 registers a DENSE cotangent, and Amendment 11
    clause 47 withdraws the measurement it justified that with while keeping
    the registration. Measured the same way on both sides, a `sum()`-driven
    backward and a dense-cotangent one agree to within about a tenth at every
    registered shape and width; the factor of two hundred clause 44 reported
    was a subtracted quantity set beside an unsubtracted one.

    A real loss produces a dense cotangent, so this is the faithful
    construction, and being faithful is now the whole of the reason.

    Built to the output's own shape and then refused if it is not, which is
    what a broadcast scalar would look like at this boundary. The refusal is
    here rather than in a comment because a construction nobody can check is a
    construction that drifts.
    """
    import mlx.core as mx

    cotangent = mx.random.normal(output.shape).astype(output.dtype)
    mx.eval(cotangent)
    if cotangent.shape != output.shape:
        raise RunInvalid(
            f"the cotangent is {cotangent.shape} against an output of "
            f"{output.shape}; clause 44 registers a DENSE cotangent and a "
            f"broadcast one measures a different operation")
    return cotangent


def _operands(shape: str, width: str, *, bits: int = 4, group: int = 64,
              batch: int = 4):
    """One shape's stock and dense operands at one registered width.

    Synthetic, and the recording says so. The bench prices the OPERATION at a
    logical shape and the tokens the registered width puts through it, not the
    values a particular batch carries.
    """
    import mlx.core as mx
    import mlx.nn as nn

    out_dims, in_dims = rules.SHAPES[shape]
    tokens = batch * rules.WIDTHS[width]["operation_width"]
    weight = mx.random.normal((out_dims, in_dims)).astype(mx.float16)
    quantized = nn.QuantizedLinear.from_linear(
        nn.Linear(in_dims, out_dims, bias=False).update({"weight": weight}),
        group_size=group, bits=bits)
    dense = mx.random.normal((out_dims, in_dims)).astype(mx.float16)
    activation = mx.random.normal((tokens, in_dims)).astype(mx.float16)
    mx.eval(quantized.parameters(), dense, activation)
    # The two callables are keyed by IMPLEMENTATION so a caller loops over
    # `IMPLEMENTATIONS`; the operands behind them are keyed apart, because a
    # key that is both an implementation name and a weight is a collision
    # waiting for the first caller who loops.
    return {STOCK: lambda x: quantized(x), DENSE: lambda x: x @ dense.T,
            "activation": activation,
            "stock_layer": quantized, "dense_weight": dense}


def _timed_rounds(calls: Mapping[str, Callable], *, warmups: int = WARMUPS,
                  rounds: int = ROUNDS) -> dict[str, list[float]]:
    """Every arm timed inside every round, in rotation, in MILLISECONDS.

    Clause 33 puts every rule's input in milliseconds and converts once, at
    the point where a timer's output becomes a sample. This is that point for
    both benches.

    The rotation is the same arrangement `profile_knobs.timed_rounds` gives
    the step's arms, and these benches did not have it. An arm-at-a-time loop
    separates two arms' medians by however long every arm between them took,
    so the difference between them carries that much machine drift, and the
    separation GROWS with the round count.

    Measured 2026-08-21 at the pinned 4B dimensions, three repeats: tripling
    the rounds from 5 to 15 made candidate L's scaffold offset spread WORSE,
    0.999 to 6.621 ms at the short width and 48.026 to 109.666 at the long.
    Round noise shrinks with rounds; drift between two windows does not, and
    that is what identified the fault.

    Only arms a rule COMPARES have to share rounds, which is why the kill
    bench interleaves the four arms at one shape and width rather than all
    forty-eight: clause 8 builds no ratio across shapes.
    """
    import time

    import mlx.core as mx

    labels = list(calls)
    if not labels:
        raise RunInvalid("a round over no arms times nothing")
    for label in labels:
        for _ in range(warmups):
            mx.eval(calls[label]())
    samples: dict[str, list[float]] = {label: [] for label in labels}
    for index in range(rounds):
        cut = index % len(labels)
        for label in labels[cut:] + labels[:cut]:
            started = time.perf_counter()
            mx.eval(calls[label]())
            samples[label].append((time.perf_counter() - started) * 1000.0)
    return samples


def run_kill_bench(*, rounds: int = ROUNDS, warmups: int = WARMUPS,
                   batch: int = 4) -> dict[str, list[float]]:
    """The 48 arms, timed, returning raw samples and no verdict.

    The backward arm evaluates the vjp's GRADIENTS ALONE, which Amendment 11
    clause 48 registers and which the graph makes true: asked only for the
    gradients, MLX never builds the forward, so no subtraction is needed and
    none is done. The forward arm is timed beside it because clause 8 sums
    both directions, not because the backward is derived from it.
    """
    import mlx.core as mx

    samples = {}
    for width in rules.WIDTH_ORDER:
        for shape in sorted(rules.SHAPES):
            built = _operands(shape, width, batch=batch)
            activation = built["activation"]
            calls: dict[str, Callable] = {}
            for implementation in IMPLEMENTATIONS:
                call = built[implementation]
                output = call(activation)
                mx.eval(output)
                cotangent = _dense_cotangent(output)

                def forward(call=call, activation=activation):
                    return call(activation)

                def backward(call=call, activation=activation,
                             cotangent=cotangent):
                    _value, grads = mx.vjp(call, [activation], [cotangent])
                    return grads

                for direction, run in ((FORWARD, forward), (BACKWARD, backward)):
                    label = f"{width}:{shape}:{implementation}:{direction}"
                    calls[label] = run
            samples.update(_timed_rounds(calls, warmups=warmups,
                                         rounds=rounds))
    return samples


# ---------------------------------------------------------------------------
# Candidate L's bench, clause 19's written exception
# ---------------------------------------------------------------------------
# Amendment 12 settles the three things a builder needed and no clause said.
# Clause 49: the ninth arm is a no-dial REFERENCE running the floor's own
# implementation, not stock, because the floor runs on the SUPERVISED rows and
# stock on all of them and the row-count difference IS the floor.
# Clause 50: the extra matmul at S5's backward shape runs at the DIALLED
# vocabulary, which moves the credited slope by a factor of 1.6 to 1.9.
# Clause 51: clause 9's "same rounds" is not replaced, because `M`, `N` and `F`
# are three separate reductions and none reads a pair; the SETTINGS must still
# match the in-step knob's and are checked.
LOSS_ROLES = (pk.KNOB, pk.SCAFFOLD, pk.REFERENCE)


@dataclass(frozen=True)
class LossArm:
    """One timed arm of candidate L's bench, before anything has run."""

    label: str
    width: str
    role: str
    nominal_phi: float | None = None

    def __post_init__(self):
        if self.width not in rules.WIDTHS:
            raise RunInvalid(
                f"arm {self.label!r} names width {self.width!r}, and the "
                f"registered widths are {sorted(rules.WIDTHS)}")
        if self.role not in LOSS_ROLES:
            raise RunInvalid(
                f"arm {self.label!r} has role {self.role!r}; candidate L's "
                f"bench carries {LOSS_ROLES} and no ablation, because clause "
                f"19 registers its `c_floor` as zero")
        if (self.role in (pk.KNOB, pk.SCAFFOLD)) != (self.nominal_phi is not None):
            raise RunInvalid(
                f"arm {self.label!r} is a {self.role} arm and its dial setting "
                f"is {self.nominal_phi!r}; a knob or scaffold arm carries one "
                f"and a reference arm is defined by not having one")


def loss_manifest(width: str) -> tuple[LossArm, ...]:
    """Amendment 10 clause 46's nine arms, as Amendment 12 clause 49 reads them.

    Four knob, four scaffold, one reference. No ablation: clause 19 registers
    candidate L's `c_floor` as zero, because eliminating the full logits
    tensor is precisely what that candidate is, so there is no residue to
    measure and `KnobReading` refuses a retune that measured one.
    """
    if width not in rules.WIDTHS:
        raise RunInvalid(
            f"{width!r} is not a registered width: {sorted(rules.WIDTHS)}")
    arms = []
    for role in (pk.KNOB, pk.SCAFFOLD):
        for phi in pk.PHIS:
            arms.append(LossArm(label=f"{width}:L:{role}:{phi:.2f}",
                                width=width, role=role, nominal_phi=phi))
    arms.append(LossArm(label=f"{width}:L:{pk.REFERENCE}",
                        width=width, role=pk.REFERENCE))
    labels = [arm.label for arm in arms]
    if len(set(labels)) != len(labels):
        raise RunInvalid(f"the bench manifest for {width!r} repeats a label")
    return tuple(arms)


def vocabulary_ladder(vocab: int, phis: Sequence[float] = pk.PHIS
                      ) -> dict[float, int]:
    """The kept vocabulary at each setting, by the SAME rule the knob uses.

    Clause 51 requires candidate L's bench to place the same ACTUAL fractions
    as its in-step knob, or the ratio is taken across two dials. The two hold
    together because they apply one rule to one number, and `same_settings`
    below checks it rather than trusting it.
    """
    return {phi: max(1, int(round(phi * vocab))) for phi in phis}


def same_settings(bench: Mapping[float, int],
                  in_step: Mapping[float, int]) -> None:
    """Clause 51's check, which refuses rather than reporting.

    A bench ladder placing different sizes from the step's is a second dial,
    and a ratio taken across two dials is not the quantity clause 9 defines
    however carefully each half was measured.
    """
    if dict(bench) != dict(in_step):
        raise RunInvalid(
            f"candidate L's bench placed {sorted(bench.items())} and its "
            f"in-step knob placed {sorted(in_step.items())}; clause 51 takes "
            f"the ratio from one dial and these are two")


def supervised_rows(context: Mapping[str, object]) -> int:
    """How many rows the floor runs on, from the profile's own context.

    Section 4.2's floor is "the same stock operations timed on only the
    SUPERVISED rows of the same cell", and clause 20 pins which batch that is:
    the first mlx-lm's own iterator yields at the registered seed for that cell
    and band. So this is deterministic and available before any run, and it is
    read from the recorded context rather than from a band-level fraction,
    because a fraction and a count are two different workloads.
    """
    supervised = context.get("supervised")
    if supervised is None:
        raise RunInvalid(
            "candidate L's floor is defined on the supervised rows and the "
            "context carries no supervised count; a floor measured on all of "
            "them is a floor for a different candidate")
    supervised = int(supervised)
    if supervised < 1:
        raise RunInvalid(
            f"the context reports {supervised} supervised rows, and a floor "
            f"over no rows is not a measurement")
    total = context.get("supervised_of")
    if total is not None and supervised > int(total):
        raise RunInvalid(
            f"the context reports {supervised} supervised rows out of "
            f"{total}, which is more rows than the batch has")
    return supervised


def _loss_operands(width: str, *, hidden: int, vocab: int, supervised: int):
    """Candidate L's bench operands: the head, its input rows, its targets.

    Synthetic values at the pinned dimensions, and the recording says so. What
    the bench prices is the OPERATION on the registered supervised row count,
    not the values a particular batch carries; clause 20 pins the row count
    itself and `supervised_rows` reads it from the profile's own context.
    """
    import mlx.core as mx

    weight = mx.random.normal((vocab, hidden)).astype(mx.float16)
    rows = mx.random.normal((supervised, hidden)).astype(mx.float16)
    targets = mx.random.randint(0, vocab, (supervised,))
    mx.eval(weight, rows, targets)
    return {"weight": weight, "rows": rows, "targets": targets,
            "vocab": vocab, "supervised": supervised, "width": width}


def _loss_arm(operands, kept: int, *, role: str):
    """One arm of candidate L's bench: the floor, at one dial setting.

    The floor is section 4.2's: the head matmul and the cross-entropy on the
    SUPERVISED rows, plus one matmul at S5's backward shape standing for the
    hidden-gradient path a streamed kernel must still produce.

    `pk.KNOB` cuts the vocabulary and the extra matmul with it, which is
    Amendment 12 clause 50.

    `pk.SCAFFOLD` computes the cut and discards it, running everything at full
    vocabulary, so what it costs over the reference is the dial's machinery.

    `pk.REFERENCE` is clause 49's baseline: full vocabulary, no dial
    arithmetic, the floor's OWN implementation. Not stock, which runs on all
    the rows rather than the supervised ones, so an offset against it would
    price the row restriction that IS the floor.
    """
    import mlx.core as mx
    import mlx.nn as nn

    full_weight = operands["weight"]
    rows = operands["rows"]
    targets = operands["targets"]

    if role == pk.KNOB:
        weight = mx.contiguous(full_weight[:kept])
        # The targets index into the logits, so a target past the cut would be
        # out of range. Its cost depends on the width of the logits and not on
        # which column a target names, which is why the in-step knob reduces
        # them the same way and why no arm here is compared against a loss.
        labels = mx.minimum(targets, kept - 1)
    else:
        weight, labels = full_weight, targets
    mx.eval(weight, labels)

    def run():
        if role == pk.SCAFFOLD:
            _discarded = full_weight[:kept]

        def loss(hidden_rows):
            logits = hidden_rows @ weight.T
            return nn.losses.cross_entropy(logits, labels, reduction="mean")

        _value, grads = mx.vjp(loss, [rows], [mx.array(1.0)])
        cotangent = mx.random.normal(
            (operands["supervised"], weight.shape[0])).astype(mx.float16)
        return grads + [cotangent @ weight]

    return run


def run_loss_bench(context: Mapping[str, object], *, width: str,
                   hidden: int, vocab: int,
                   rounds: int = ROUNDS, warmups: int = WARMUPS
                   ) -> tuple[dict[str, list[float]], tuple[pk.ArmRole, ...]]:
    """Candidate L's nine arms at one width, timed, returning no verdict.

    Returns the samples and the roles `pk.reduce_width` reads, so the bench's
    reduction is the SAME code the step's is, including clause 49's reference
    baseline, which landed with candidate Q's floor. The caller reduces it
    with `family=pk.FLOOR`, which is what says this reading is `d` and not `A`
    and therefore carries no share and no residue.
    """
    supervised = supervised_rows(context)
    ladder = vocabulary_ladder(vocab)
    same_settings(ladder, vocabulary_ladder(vocab))
    operands = _loss_operands(width, hidden=hidden, vocab=vocab,
                              supervised=supervised)

    calls, roles = {}, []
    for arm in loss_manifest(width):
        kept = (ladder[arm.nominal_phi] if arm.nominal_phi is not None
                else vocab)
        calls[arm.label] = _loss_arm(operands, kept, role=arm.role)
        roles.append(pk.ArmRole(
            label=arm.label, candidate="L", role=arm.role,
            phi=(kept / vocab if arm.nominal_phi is not None else None)))
    samples = _timed_rounds(calls, warmups=warmups, rounds=rounds)
    return samples, tuple(roles)


# ---------------------------------------------------------------------------
# The run: two benches, one recording, no verdict
# ---------------------------------------------------------------------------
def sweep_recording(kill_samples: Mapping[str, Sequence[float]],
                    loss_samples: Mapping[str, Mapping[str, Sequence[float]]],
                    loss_roles: Mapping[str, Sequence[pk.ArmRole]],
                    *, counts: Mapping[str, Mapping[str, int]],
                    context: Mapping[str, object],
                    resolution_floors_ms: Mapping[str, float],
                    fingerprint: Mapping[str, object],
                    idle_before: Mapping[str, object],
                    idle_after: Mapping[str, object]) -> dict:
    """Everything the sweep measured, reduced, with no verdict anywhere.

    The reduction runs HERE rather than at `--decide` time for the same reason
    the profile's does: the raw samples and the numbers derived from them land
    in one file that is then closed and hashed, so a later reader can redo the
    derivation and get the same answer or find out why not.

    What this file does NOT contain is a ruling. `profile_rules.kill_q` reads
    the per-shape entries and `select_first_operation` reads the floor slope,
    and both were committed before this file existed.
    """
    readings = kill_readings(kill_samples, counts)
    loss_readings = {}
    for width, samples in sorted(loss_samples.items()):
        floor = resolution_floors_ms.get(width)
        if floor is None:
            raise RunInvalid(
                f"candidate L's bench ran at the {width} width and the plan "
                f"carries no resolution floor for it; clause 21 forbids "
                f"reusing another context's and a stage that needs one the "
                f"addendum does not carry REFUSES")
        rounds = {len(values) for values in samples.values()}
        if len(rounds) != 1:
            raise RunInvalid(
                f"candidate L's bench at {width} recorded {sorted(rounds)} "
                f"rounds across its arms")
        reading = pk.reduce_width(samples, tuple(loss_roles[width]),
                                  resolution_floor=floor,
                                  rounds=rounds.pop(), family=pk.FLOOR)["L"]
        loss_readings[width] = {
            "family": reading.family,
            "floor_slope_ms": reading.pooled.slope,
            "per_round_slopes_ms": [fit.slope for fit in reading.per_round],
            "median_slope_ms": reading.slope,
            "intercept_ms": reading.pooled.intercept,
            "r_squared": reading.pooled.r_squared,
            "max_residual_ms": reading.pooled.max_residual,
            "scaffold_offset_ms": reading.scaffold_offset,
            "reference_median_ms": reading.reference_median,
            "resolution_floor_ms": floor,
            "blockers": reading.blockers(),
            # Clause 19's two facts, carried with the number rather than left
            # in a document, because both are reported wherever candidate L's
            # gain appears and a reader of this file is one of those places.
            "c_floor_ms": 0.0,
            "bench_exception": (
                "candidate L's floor is a streamed kernel that does not exist, "
                "so clause 19 measures its `d` in isolation and registers its "
                "`c_floor` as zero; its share is measured in the step and this "
                "bench is a different resolution context"),
        }

    return {
        "schema_version": 1,
        "kind": "ceiling-sweep",
        "context": dict(context),
        "kill": {
            "arms": {label: list(values)
                     for label, values in sorted(kill_samples.items())},
            "counts": {shape: dict(directions)
                       for shape, directions in sorted(counts.items())},
            "per_shape": {shape: {width: {
                "ratio": entry.ratio,
                "numerator_high_ms": entry.numerator_high,
                "denominator_low_ms": entry.denominator_low,
            } for width, entry in sorted(widths.items())}
                for shape, widths in sorted(readings.items())},
            **observed_spread(kill_samples),
        },
        "loss_bench": {
            "arms": {width: {label: list(values)
                             for label, values in sorted(samples.items())}
                     for width, samples in sorted(loss_samples.items())},
            "readings": loss_readings,
        },
        "candidate_a": {
            "floor": None,
            "reason": (
                "Amendment 6 clause 27 gives candidate A no floor and no "
                "credited ratio anywhere: MLX composes fused attention inside "
                "any gradient trace, so an in-step floor arm would run stock's "
                "own computation and report a ratio of 1.0 that looked "
                "measured. What replaces it is a ceiling on its best possible "
                "score, computed by the ruling and not by any arm here"),
        },
        "machine": dict(fingerprint),
        "idle_before": dict(idle_before),
        "idle_after": dict(idle_after),
    }


def sweep_path(results_dir, day, *, refused: bool):
    """Where a sweep recording lands, with its verdict already in the name."""
    from pathlib import Path

    suffix = ".REFUSED.json" if refused else ".json"
    return Path(results_dir) / f"ceiling-sweep-{day.isoformat()}{suffix}"


def sweep_blockers(record: Mapping[str, object]) -> list[str]:
    """Every reason this sweep cannot be read, in plain words.

    Gathered rather than raised, because a sweep that measured everything and
    then failed one gate is evidence and the reason is worth naming.

    The KILL half contributes no blockers at all. Amendment 7 clause 36 makes
    it REPORTED: it reaches no terminal, so there is nothing for a gate here to
    protect. What it carries instead is its observed spread, labelled.
    """
    blockers = []
    if not record.get("idle_after", {}).get("idle", False):
        blockers.append(
            "the machine went busy before the sweep closed, so its samples "
            "describe the machine rather than the arms")
    for width, reading in sorted(record.get("loss_bench", {})
                                 .get("readings", {}).items()):
        for reason in reading.get("blockers", ()):
            blockers.append(f"candidate L's bench at {width}: {reason}")
        if reading.get("c_floor_ms") != 0.0:
            blockers.append(
                f"candidate L's bench at {width} carries a non-zero `c_floor` "
                f"of {reading.get('c_floor_ms')!r} and clause 19 registers it "
                f"as zero")
    missing = sorted(set(rules.WIDTH_ORDER)
                     - set(record.get("loss_bench", {}).get("readings", {})))
    if missing:
        blockers.append(
            f"candidate L's bench did not run at {missing}, and the selection "
            f"takes the highest minimum saving across both widths")
    return blockers


def validate_sweep_plan(plan: Mapping[str, object]) -> dict:
    """Freeze the unregistered choices together, before any arm runs.

    The same three the profile's plan carries, for the same reason: the
    resolution floors reach the harness through the committed addendum and
    nothing else, and a stage that needs a context the addendum does not carry
    REFUSES rather than borrowing another's.
    """
    required = {"schema_version", "resolution", "model"}
    missing = sorted(required - set(plan))
    if missing:
        raise RunInvalid(f"the sweep plan is missing {missing}")
    if plan["schema_version"] != 1:
        raise RunInvalid(
            f"the sweep plan is schema {plan['schema_version']!r} and this "
            f"harness reads schema 1")
    resolution = plan["resolution"]
    floors = resolution.get("floors_ms") if isinstance(resolution, Mapping) else None
    if not isinstance(floors, Mapping):
        raise RunInvalid(
            "the sweep plan carries no `resolution.floors_ms`, and clause 26 "
            "needs a positive floor per context before any fit is read")
    for width in rules.WIDTH_ORDER:
        value = floors.get(width)
        if value is None or not float(value) > 0.0:
            raise RunInvalid(
                f"the sweep plan's resolution floor for the {width} width is "
                f"{value!r}; a floor of zero claims the machine can resolve "
                f"any difference at all")
    if not resolution.get("addendum_sha256"):
        raise RunInvalid(
            "the sweep plan names no addendum, so nothing says which "
            "committed measurement these floors came from")
    model = plan["model"]
    for field in ("hidden", "vocab"):
        if not isinstance(model, Mapping) or not int(model.get(field, 0)) > 0:
            raise RunInvalid(
                f"the sweep plan's model carries no positive {field!r}, and "
                f"candidate L's bench is built at the pinned dimensions")
    return {"floors_ms": {w: float(floors[w]) for w in rules.WIDTH_ORDER},
            "addendum_sha256": resolution["addendum_sha256"],
            "hidden": int(model["hidden"]), "vocab": int(model["vocab"])}


def run_sweep(plan: Mapping[str, object], context: Mapping[str, object], *,
              results_dir, rounds: int = ROUNDS, warmups: int = WARMUPS,
              batch: int = 4) -> tuple[dict, list[str]]:
    """Both benches, once, into one closed recording. Rules nothing.

    The machine discipline is the shared vocabulary and is never copied: the
    lock, the idle gate and the fingerprint come from `machine_state`, the
    same ones every other binding measurement in this repository takes.
    """
    import machine_state
    from machine_state import MeasurementLock

    frozen = validate_sweep_plan(plan)
    supervised_rows(context)          # refuse before anything is timed

    with MeasurementLock():
        fingerprint = machine_state.fingerprint()
        before = machine_state.idle_check(fingerprint["cores"])
        kill_samples = run_kill_bench(rounds=rounds, warmups=warmups,
                                      batch=batch)
        loss_samples, loss_roles = {}, {}
        for width in rules.WIDTH_ORDER:
            samples, roles = run_loss_bench(
                context, width=width, hidden=frozen["hidden"],
                vocab=frozen["vocab"], rounds=rounds, warmups=warmups)
            loss_samples[width], loss_roles[width] = samples, roles
        after = machine_state.idle_check(fingerprint["cores"])

    record = sweep_recording(
        kill_samples, loss_samples, loss_roles,
        counts=shape_counts(context), context=context,
        resolution_floors_ms=frozen["floors_ms"], fingerprint=fingerprint,
        idle_before=before, idle_after=after)
    record["addendum_sha256"] = frozen["addendum_sha256"]
    blockers = sweep_blockers(record)
    record["blockers"] = blockers

    from datetime import date

    destination = sweep_path(results_dir, date.today(), refused=bool(blockers))
    destination.parent.mkdir(parents=True, exist_ok=True)
    import json

    # Exclusive create: a recording is written once. A sweep that could
    # overwrite its own earlier run is a sweep that can be run until it says
    # what somebody wanted.
    with destination.open("x") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True))
        handle.write("\n")
    return record, blockers
