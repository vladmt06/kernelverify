#!/usr/bin/env python3
"""The Day 1 stock training profile: where a real QLoRA step's time goes.

What this produces, and what it refuses to produce
--------------------------------------------------
It produces one reading per candidate operation per registered width: the
fraction `f` of the training step that operation occupies, measured inside
mlx-lm's own step on the pinned model and the pinned corpus. It produces no
verdict. The gain formula, the tie rule and the selection live in
`profile_rules`, were committed before this file existed, and are applied to a
recording this harness has already closed and hashed. A harness that could
also rule would be a harness that could quietly become the thing that decides.

A share is a slope, and there is no clock inside the step
---------------------------------------------------------
Amendment 5 clause 1 replaced section 3.3's marked spans with a fitted slope,
because a mark evals and then timestamps: its fence sits inside the span that
forms a share's numerator and outside the plain step that forms its
denominator, so the error grows with how many marks a candidate carries -
which is exactly what separates the three candidates. Measured 2026-08-20, the
same three regions read 0.263, 0.193 and 1.401 with marks and 0.245, 0.045 and
0.376 without, and a share of 1.401 is not a fraction of anything.

So nothing here times a region. Every arm is a whole compiled step, and what
separates two arms is how much arithmetic one operation does inside it. A
candidate's ladder shrinks its operation's size while holding the launch
count, the output shape and the graph structure fixed; the line through those
arms has a slope, and `slope / stock` is the fraction of the step that moves
with that operation. `profile_knobs` owns the arms, the fits and the gates;
this file owns the process, the workload and the recording.

The manifest, and why candidate A is at one width only
-------------------------------------------------------
Amendment 8 registers 26 arms at the short width and 35 at the long:

    stock         1     no seam installed, the denominator of every share
    candidate L   9     four knob, four scaffold, one ablation (a rewrite)
    candidate Q   8     four knob, four scaffold (a retune, so no ablation)
    candidate P3  8     the projections without the head, clause 22's region
    candidate A   9     LONG WIDTH ONLY, on the `kv-length` dial

Candidate A is absent at the short width by design and not by a measurement
that failed: Amendment 7 clause 35 names its dial by rule and records that the
dial fits at the long width, where attention's quadratic term dominates, and
not at the short, where every scaffold price measured sat inside the machine's
own jitter. Its reading gates no terminal, so a gap there is a gap in a report
rather than a hole in a ruling.

All arms at one width are interleaved inside one set of rounds. Running one
round group per candidate would let the machine drift between two candidates
that the rule then compares.

What is checked, and what each check would catch
------------------------------------------------
  the process is stock  - before anything is timed, every seam is checked to
                      hold the object mlx-lm itself defines and metalrunner is
                      checked to certify nothing, because a profile of stock
                      taken through somebody's patch is a profile of the patch.
  exact counts      - forward, a region fires once per place it appears.
                      Backward it does NOT: mlx-lm adapts only the last
                      `num_layers` blocks, so attention has a backward once per
                      ADAPTED block and the projections `7 * adapted - 3` times,
                      the three being those that consume the lowest adapted
                      block's input, whose gradient nothing below asks for. A
                      seam installed and never reached leaves its region absent
                      from the log, and only a count tells that apart from an
                      operation that really costs nothing.
  identity          - the marked pass and the plain pass must agree on the loss
                      and on every gradient array, or the boundary instrument
                      changed the computation and its counts describe a
                      different step.
  no trace while timing - `mx.compile` traces at an object's FIRST CALL, not at
                      construction, so an arm whose graph is built after its
                      seam is gone is silently stock. A counter refuses the run
                      if any arm traces during the timed rounds, which is the
                      general guard the other two trace rules do not have to be
                      exhaustive about.
  the fits          - `profile_knobs` gates every reading: a positive slope on
                      both estimators, clause 6's two shape limits, clause 26's
                      three scaffold cases, the scaffold offset against `3R`,
                      both excursions against `10R`, clause 33's residue fault,
                      and a share strictly inside (0, 1).
  the partition     - clause 22's double-counting test over P1, P2 and P3. It
                      detects double-counting and cannot detect omission,
                      because P4 absorbs anything unmeasured by construction.

The boundary instrument is kept and demoted
--------------------------------------------
`profile_instrument` still runs, once per cell-width, in its own untimed pass.
Its counts are exact and have already caught a seam that never fired and a
test that left gradient checkpointing installed for a whole session. It
contributes no timing to anything.

Refused deliberately
--------------------
`--grad-checkpoint` is not offered. Checkpointing discards activations and
runs a block's forward AGAIN during the backward; the marks survive that with
the loss and gradients unchanged, but the replayed forward fires forward
marks, so a region's forward count absorbs work belonging to the backward.
Measured 2026-08-20: attention fires 30 times forward on a 28-block model and
the projections 210 times where an unchecked step fires 196.

Usage
-----
    bench/start_binding_run.sh bench/profile_stock.py -- --plan <plan.json>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
import types
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import machine_state
import profile_instrument as pi
import profile_knobs as pk
import profile_rules as rules
from decode_rules import RunInvalid
from harness_runner import (
    ROOT,
    ChildRefusal,
    NoDevice,
    PreconditionFailed,
    _ChildGuard,
    _atomic_json,
    _check_child_inputs,
    _file_sha256,
    _json_sha256,
    child_exit_for_parent,
    preflight_inputs,
    spawn_child,
    stack_record,
)
from machine_state import MeasurementLock
from memory_guard import (
    EXIT_BUDGET_REFUSAL,
    EXIT_LOCK_HELD,
    EXIT_LOW_MEMORY,
    EXIT_NO_DEVICE,
    EXIT_NOT_IDLE,
    EXIT_ORPHANED,
    EXIT_PRECONDITION,
    BudgetExceeded,
    BudgetGuard,
    LowMemoryRefusal,
    Orphaned,
    phys_footprint_gb,
    require_available_memory,
)

RESULTS_DIR = ROOT / "bench" / "results"

# Section 3.4, and `profile_knobs` uses the same number: five rounds, so the
# rounds are genuine repeats and the spread gate reads the machine.
ROUNDS = pk.ROUNDS

# The profile installs nothing. The shared preflight records which module a
# run could have routed through, and for this run the honest answer is the
# real installer with nothing certified in it.
STOCK_MEASUREMENT_MODULE = "metalrunner.measurement"

# Amendment 8's manifest. Candidate A is at the long width alone, per
# Amendment 7 clause 35, and P3 is at both because clause 22's REJECT branch
# reads it at both. The order is the order the arms are built and recorded in,
# so two recordings can be compared row by row.
CANDIDATES_AT_WIDTH = {
    "short": ("L", "P3", "Q", "Qfloor"),
    "long": ("A", "L", "P3", "Q", "Qfloor"),
}

# Amendment 10 clause 45. Candidate Q's dense floor runs in the SAME context
# as its share arms, so it lives in this manifest rather than in a bench of
# its own, and it carries a no-dial reference arm because clause 5's offset
# against a QUANTIZED stock would price the two implementations' speed
# difference as scaffold overhead and refuse the family at clause 21's 3R.
FLOOR_OF = {"Qfloor": "Q"}
FLOOR_FAMILY = {share: floor for floor, share in FLOOR_OF.items()}
NEEDS_REFERENCE = ("Qfloor",)

# Amendment 7 clause 35: candidate A's dial is NAMED rather than measured, by
# clause 15's own completeness tie-break, so no run reselects it.
ATTENTION_DIAL = "kv-length"

# Clause 22's partition, and which reading measures each region. P4 is the
# remainder by subtraction and is measured by nothing.
PARTITION = {"P1": "L", "P2": "A", "P3": "P3"}

# The block's shape, measured rather than read off the config. On the 0.6B
# model at 1, 2, 4 and 8 adapted blocks the backward count was exactly
# 7 * adapted - 3 every time (2026-08-20): a block holds seven quantized
# projections, and in the LOWEST adapted block the three that consume the
# block's input have no backward at all, because nothing below that block is
# trainable and so that gradient is never asked for.
PROJECTIONS_PER_BLOCK = 7
INPUT_CONSUMING_PROJECTIONS = 3


# ---------------------------------------------------------------------------
# Pure functions: no MLX, no device, no machine
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmSpec:
    """One arm of the manifest, before any ladder has said what it can place.

    Pure and nominal. The fraction a dial ACTUALLY places is a property of the
    model's dimensions, not of the manifest, and the two are kept apart
    because a fit run on the fraction that was asked for rather than the one
    that was placed is a fit on numbers nothing measured.
    """

    label: str
    candidate: str
    role: str
    nominal_phi: float | None = None


def widths_for(cell: str) -> tuple[str, ...]:
    """Which registered widths a cell is measured at.

    Only the deciding cell takes both. Section 4.3 gives cells A, C and D no
    part in the selection, so measuring them twice would double their cost to
    move no rule, and Amendment 5 clause 14's two-width comparison is a
    property of the cell that decides.
    """
    if cell not in rules.CELLS:
        raise RunInvalid(f"not a registered cell: {cell!r}")
    return (rules.WIDTH_ORDER if cell == rules.PRIMARY_CELL
            else (rules.WIDTH_ORDER[0],))


def arm_manifest(width: str) -> tuple[ArmSpec, ...]:
    """Amendment 8's arms for one width, in build and record order.

    One stock arm, then each candidate's knob ladder, its scaffold ladder at
    the same settings, and an ablation where clause 18 credits a rewrite with
    a residue. A retune carries no ablation at all rather than an unused one,
    because `KnobReading` refuses a retune that measured a residue nothing may
    consume.

    Amendment 10 clause 45 adds candidate Q's floor family here rather than in
    a bench: clause 19 puts the floor in the step at the same seam and clause
    9 puts it at the same settings in the same rounds, which makes it nine
    more arms of THIS manifest. Nine and not eight, because a family running
    an implementation stock does not run needs its own no-dial reference.
    """
    if width not in rules.WIDTHS:
        raise RunInvalid(
            f"{width!r} is not a registered width: "
            f"{sorted(rules.WIDTHS)}")
    arms = [ArmSpec(label="stock", candidate="stock", role=pk.STOCK)]
    for candidate in CANDIDATES_AT_WIDTH[width]:
        for role in (pk.KNOB, pk.SCAFFOLD):
            for phi in pk.PHIS:
                arms.append(ArmSpec(label=f"{candidate}:{role}:{phi:.2f}",
                                    candidate=candidate, role=role,
                                    nominal_phi=phi))
        if pk.kind_of(candidate) == pk.REWRITE:
            arms.append(ArmSpec(label=f"{candidate}:{pk.ABLATION}",
                                candidate=candidate, role=pk.ABLATION))
        if candidate in NEEDS_REFERENCE:
            arms.append(ArmSpec(label=f"{candidate}:{pk.REFERENCE}",
                                candidate=candidate, role=pk.REFERENCE))
    labels = [arm.label for arm in arms]
    if len(set(labels)) != len(labels):
        raise RunInvalid(
            f"the manifest for width {width!r} repeats a label, so one arm's "
            f"samples would overwrite another's")
    return tuple(arms)


def cell_context(cell: str, width: str, *, batch: int, batch_width: int,
                 model: str, adapted: int, supervised: int | None = None,
                 supervised_of: int | None = None,
                 batch_sha256: str | None = None) -> dict:
    """What identifies the workload one width of one cell measured.

    Carried into every arm's samples and checked wherever a number from one
    measurement meets a number from another: across widths inside this
    harness, and across harnesses when the floor sweep's samples meet this
    profile's shares. Two measurements with different contexts describe
    different steps, and a ratio between them is a number nothing downstream
    could tell was meaningless.

    `operation_width` is the token count the matmuls actually run at, and it
    is carried rather than left to be derived because deriving it is where a
    floor sweep would go wrong. mlx-lm pads a batch to `batch_width` and then
    `default_loss` trains on `batch[:, :-1]`, so every matmul in the step sees
    one fewer token per row. It is taken from the registered band rather than
    from the batch, and the batch is then checked against it, so a batch that
    padded to some other width refuses instead of quietly redefining the cell.

    `supervised` is the count of tokens the loss actually trains on, and it is
    here because candidate L's floor is defined as the same operations "timed
    on only the supervised rows of the same cell". A floor measured against a
    different supervised count is a floor for a different problem, and the
    credited ratio would still divide.
    """
    if width not in rules.WIDTHS:
        raise RunInvalid(f"{width!r} is not a registered width")
    registered = rules.WIDTHS[width]
    if batch_width != registered["batch_width"]:
        raise RunInvalid(
            f"cell {cell} at the {width} width padded to {batch_width} and "
            f"the registered band {registered['band']} pads to "
            f"{registered['batch_width']}: this batch is not the width the "
            f"profile registered")
    fraction = (supervised / supervised_of
                if supervised is not None and supervised_of else None)
    return {"cell": cell, "width": width, "band": registered["band"],
            "batch": batch, "batch_width": batch_width,
            "operation_width": registered["operation_width"],
            "tokens": batch * registered["operation_width"],
            "model": model, "adapted": adapted, "supervised": supervised,
            "supervised_of": supervised_of,
            "supervised_fraction": fraction, "batch_sha256": batch_sha256}


def expected_counts(depth: int, adapted: int,
                    regions: Sequence[str]) -> dict[str, dict[str, int]]:
    """How often each region must fire, per direction, on this arrangement.

    Forward is fixed by the model's geometry. Backward is fixed by the LoRA
    arrangement: mlx-lm adapts the last `adapted` blocks, so attention has a
    backward only in those, and the projections have one everywhere except the
    three that consume the lowest adapted block's input.
    """
    if depth <= 0:
        raise RunInvalid(f"a model with {depth} blocks has no step to profile")
    if not 0 < adapted <= depth:
        raise RunInvalid(
            f"{adapted} adapted blocks in a {depth}-block model is not an "
            f"arrangement mlx-lm can produce")
    table = {
        "attn-core": {pi.FORWARD: depth, pi.BACKWARD: adapted},
        "head-matmul": {pi.FORWARD: 1, pi.BACKWARD: 1},
        "cross-entropy": {pi.FORWARD: 1, pi.BACKWARD: 1},
        "qmm": {pi.FORWARD: PROJECTIONS_PER_BLOCK * depth,
                pi.BACKWARD: (PROJECTIONS_PER_BLOCK * adapted
                              - INPUT_CONSUMING_PROJECTIONS)},
    }
    unknown = sorted(set(regions) - set(table))
    if unknown:
        raise RunInvalid(f"no expected count is registered for {unknown}")
    return {name: dict(table[name]) for name in regions}


def completeness(observed: Mapping[str, Mapping[str, int]],
                 expected: Mapping[str, Mapping[str, int]]) -> dict:
    """Did every seam fire as often as the arrangement says it must?

    This is the check that catches a seam installed and never reached. Without
    it that region's share reads as zero, lands in the remainder, and looks
    exactly like an operation that costs nothing.
    """
    mismatches = []
    for region in sorted(expected):
        for direction in (pi.FORWARD, pi.BACKWARD):
            want = expected[region][direction]
            got = observed.get(region, {}).get(direction)
            if got != want:
                mismatches.append({"region": region, "direction": direction,
                                   "expected": want, "observed": got})
    return {"ok": not mismatches, "mismatches": mismatches,
            "expected": {r: dict(c) for r, c in sorted(expected.items())},
            "observed": {r: dict(c) for r, c in sorted(observed.items())}}


# Amendment 7 clause 34. Only these two name the first operation, so only
# their readings can stop a recording binding. Candidate A gates nothing by
# ruling 2 and candidate P3 gates nothing because no certified margin reads
# it; both are measured, published, and allowed to be absent.
# Clause 34's three certified sites all read candidates L and Q, and
# Amendment 10 clause 45 puts candidate Q's floor family in the certified
# vector too, because the credited saving `K = M(1 - F/N)` reads its slope as
# `F`. It is certified and it is not SCORED: no terminal ranks it, and
# `rules.CANDIDATES` is what the profile matrix is built over.
CERTIFIED_CANDIDATES = ("L", "Q", "Qfloor")


def _fit_record(one: pk.Fit) -> dict:
    """A fitted line as data, so a reader can re-derive every gate from it."""
    return {"slope_ms": one.slope, "intercept_ms": one.intercept,
            "r_squared": one.r_squared, "max_residual_ms": one.max_residual}


def _reading_record(reading: pk.KnobReading) -> dict:
    """One candidate's reading, with everything each gate rested on beside it.

    Held as the measured quantity, its threshold and its signed distance in
    units of `R` rather than as a pass or a fail, because a reader who can
    only see the verdict cannot tell a gate that barely held from one that
    held by a factor of ten, and clause 26's own margins are the evidence a
    later amendment would be argued from.
    """
    floor = reading.resolution_floor
    return {
        "type": "reading",
        "candidate": reading.candidate,
        "kind": reading.kind,
        "stock_median_ms": reading.stock_median,
        "registered_slope_ms": reading.slope,
        "pooled": _fit_record(reading.pooled),
        "per_round": [_fit_record(one) for one in reading.per_round],
        "scaffold": _fit_record(reading.scaffold),
        "scaffold_case": reading.scaffold_case,
        "scaffold_offset_ms": reading.scaffold_offset,
        "scaffold_range_ms": reading.scaffold_range,
        "effective_scaffold_slope_ms": reading.effective_scaffold_slope,
        "actual_span": reading.span,
        "ablated_median_ms": reading.ablated_median,
        "raw_residue_ms": reading.raw_residue,
        "credited_residue_ms": reading.residue,
        "residue_is_a_fault": reading.residue_is_a_fault,
        "attributed_ms": reading.attributed,
        "share": reading.share,
        # Clause 22 reads `b/T` with NO residue term, "because a partition
        # region is a region and not a candidate". It is computed here beside
        # the candidate share rather than derived by the partition, so the two
        # quantities are visibly different numbers in the record.
        "scaling_share": reading.slope / reading.stock_median,
        "resolution_floor_ms": floor,
        "margins_in_R": {
            "pooled_excursion": (abs(reading.pooled.slope) * reading.span
                                 / floor - pk.EXCURSION_R_MULTIPLE)
            if floor and reading.span is not None else None,
            "median_excursion": (abs(reading.slope) * reading.span / floor
                                 - pk.EXCURSION_R_MULTIPLE)
            if floor and reading.span is not None else None,
            "scaffold_offset": (pk.SCAFFOLD_OFFSET_R_MULTIPLE
                                - abs(reading.scaffold_offset) / floor)
            if floor else None,
            "scaffold_range": (reading.scaffold_range / floor - 1.0)
            if floor and reading.scaffold_range is not None else None,
            "raw_residue": reading.raw_residue / floor if floor else None,
        },
        "blockers": reading.blockers(),
    }


def arm_spreads(arms: Mapping[str, Mapping[str, object]], *,
                spread_limit_pct: float = machine_state.MAX_SPREAD_PCT
                ) -> dict:
    """Every arm's own spread across the rounds, and whether it is eligible.

    The old design gated on the plain arm alone, because interleaving equalises
    a clock excursion across arms but cannot detect one, so one reference arm
    disagreeing with itself was the detector. With a ladder there is no
    reference arm: every arm is a whole step and any of them can be the one the
    machine moved under, so the gate runs over all of them.
    """
    out = {}
    for label in sorted(arms):
        samples = list(arms[label]["samples_ms"])
        out[label] = {
            "median_ms": statistics.median(samples),
            "spread_pct": machine_state.spread_pct(samples),
            "rounds_eligible": rules.round_is_eligible(samples,
                                                       spread_limit_pct),
        }
    return out


def partition(entries: Mapping[str, Mapping[str, object]]) -> dict:
    """Clause 22's double-counting test, at one width.

    Three outcomes and not two. Where the MEASURED regions alone already sum
    above one the profile REJECTS, and it rejects whether or not a region is
    absent, because every share is non-negative so a missing addend can only
    raise the sum. Where they do not and a region is absent the test is NOT
    RUN with that reason, which is not a pass: recording NOT RUN in the first
    case would suppress a rejection the numbers already establish.

    It detects double-counting and cannot detect omission, because P4 is
    whatever the other three leave and absorbs anything unmeasured. And three
    separately fitted slopes are not guaranteed additive under compilation, so
    a pass is a necessary condition on the decomposition and not a proof of it.
    """
    measured, absent = {}, {}
    for region, candidate in sorted(PARTITION.items()):
        entry = entries.get(candidate)
        if entry is None or entry.get("type") != "reading":
            absent[region] = (candidate if entry is None
                              else entry.get("reason_code", "absent"))
        else:
            measured[region] = entry["scaling_share"]
    subtotal = sum(measured.values())
    if subtotal > 1.0:
        outcome, reason = "REJECT", (
            f"the measured regions {sorted(measured)} sum to {subtotal:.4f}, "
            f"above one; every share is non-negative so no absent region can "
            f"bring it back down")
    elif absent:
        outcome, reason = "NOT RUN", (
            f"regions {sorted(absent)} carry no reading, and the measured "
            f"regions sum to {subtotal:.4f}, which proves nothing")
    else:
        outcome, reason = "PASS", (
            f"the three regions sum to {subtotal:.4f}, at or below one")
    return {"outcome": outcome, "reason": reason, "measured": measured,
            "absent": absent, "subtotal": subtotal,
            "remainder_P4": None if absent else 1.0 - subtotal,
            "reads": "b/T, the scaling share with no residue term, because a "
                     "partition region is a region and not a candidate"}


def width_reading(measured: Mapping[str, object], *,
                  resolution_floor_ms: float,
                  spread_limit_pct: float = machine_state.MAX_SPREAD_PCT
                  ) -> dict:
    """One width of one cell, reduced from raw arm samples to typed entries.

    Every candidate resolves to exactly one of two things and never to a
    placeholder: a `reading`, or a `missing_share` naming why. An absence is a
    measurement that legitimately does not exist and the rules carry it; it is
    not a zero, and it is not a null anything downstream could divide by.
    """
    width = measured["width"]
    arms = measured["arms"]
    manifest = arm_manifest(width)
    expected = {arm.label for arm in manifest}
    if set(arms) != expected:
        raise RunInvalid(
            f"width {width!r} recorded arms {sorted(set(arms) - expected)} "
            f"the manifest does not name and is missing "
            f"{sorted(expected - set(arms))}; a hole is not an absence")
    context = measured["context"]
    for label in sorted(arms):
        if arms[label]["context"] != context:
            raise RunInvalid(
                f"arm {label!r} carries {arms[label]['context']!r} and the "
                f"width carries {context!r}: two arms in one fit measured "
                f"different workloads")

    samples = {label: list(arms[label]["samples_ms"]) for label in arms}
    counted = {len(values) for values in samples.values()}
    if len(counted) != 1:
        raise RunInvalid(
            f"width {width!r} recorded {sorted(counted)} rounds across its "
            f"arms; arms that ran a different number of times are not the "
            f"repeats the spread gate reads")
    roles = tuple(pk.ArmRole(label=label, candidate=arms[label]["candidate"],
                             role=arms[label]["role"],
                             phi=arms[label]["actual_phi"])
                  for label in sorted(arms))
    readings = pk.reduce_width(samples, roles,
                               resolution_floor=resolution_floor_ms,
                               rounds=counted.pop())

    spreads = arm_spreads(arms, spread_limit_pct=spread_limit_pct)
    entries = {}
    for candidate in CANDIDATES_AT_WIDTH[width]:
        reading = readings[candidate]
        record = _reading_record(reading)
        # Clause 34 gives a reported quantity an interval, and candidate A is
        # in neither the pilot nor the fresh set, so there is no calibrated
        # interval for it to carry. What it carries instead is its own arms'
        # observed range, labelled as an observed spread and NOT as a bound at
        # any registered rate, which is the difference the label exists to
        # keep visible.
        record["certified"] = candidate in CERTIFIED_CANDIDATES
        if not record["certified"]:
            record["observed_spread_pct"] = {
                label: spreads[label]["spread_pct"]
                for label in sorted(spreads)
                if arms[label]["candidate"] == candidate}
            record["observed_spread_is_not_a_bound"] = (
                "an observed range over this run's own rounds, not an "
                "interval at any registered rate: clause 37 keeps candidate "
                "A and candidate P3 out of the calibration entirely")
        if record["blockers"]:
            entries[candidate] = {
                "type": "missing_share",
                "candidate": candidate,
                "certified": record["certified"],
                "reason_code": "fit_gates_failed",
                "blockers": record["blockers"],
                "evidence": record,
            }
        else:
            entries[candidate] = record

    return {
        "width": width,
        "context": context,
        "resolution_floor_ms": resolution_floor_ms,
        "spread_limit_pct": spread_limit_pct,
        "arms": spreads,
        "peak_gb": measured["peak_gb"],
        "entries": entries,
        "partition": partition(entries),
        "actual_settings": {
            label: {"nominal_phi": arms[label]["nominal_phi"],
                    "actual_phi": arms[label]["actual_phi"],
                    "sites": arms[label].get("sites")}
            for label in sorted(arms) if arms[label]["actual_phi"] is not None
        },
    }


# Clause 11's context travels with every measurement, and the two widths are
# SUPPOSED to differ in some of it. These are the fields that must agree
# across the columns anyway, because a disagreement in one of them means the
# two columns measured two arrangements and reported it as width.
INVARIANT_CONTEXT = ("cell", "batch", "model", "adapted")

# The fields the two columns are allowed to differ in, named rather than left
# as "everything else", so a field added later has to be classified.
VARIANT_CONTEXT = ("width", "band", "batch_width", "operation_width", "tokens",
                   "supervised", "supervised_of", "supervised_fraction",
                   "batch_sha256")


def profile_matrix(readings: Mapping[str, Mapping[str, object]]) -> dict:
    """The rule-facing table: every candidate at every width, typed, in order.

    Serialised in the registered order so two recordings can be compared row
    by row, and complete by construction: a candidate a width does not measure
    gets a typed absence naming why, never an omission and never a null. The
    absences here are legal outcomes rather than failures - Amendment 7 clause
    37 registers that candidate A absent at either width or at both is never
    incompleteness, which is REVERSED from Amendment 6.

    Every entry in a column carries that column's context, which `width_reading`
    already enforces arm by arm. What is checked HERE is the part it cannot
    see: that the two columns agree about the arrangement they measured, and
    differ only in the fields width is allowed to move.
    """
    missing = sorted(set(rules.WIDTH_ORDER) - set(readings))
    if missing:
        raise RunInvalid(
            f"the deciding cell carries no reading for {missing}, and the "
            f"selection takes the highest minimum gain across both widths")
    contexts = {width: readings[width]["context"] for width in readings}
    reference = contexts[rules.WIDTH_ORDER[0]]
    for width in rules.WIDTH_ORDER[1:]:
        for field in INVARIANT_CONTEXT:
            if contexts[width][field] != reference[field]:
                raise RunInvalid(
                    f"the {width} column measured {field}="
                    f"{contexts[width][field]!r} and the "
                    f"{rules.WIDTH_ORDER[0]} column measured "
                    f"{reference[field]!r}: these two columns describe two "
                    f"arrangements and the comparison would report it as width")
        unclassified = sorted(set(contexts[width])
                              - set(INVARIANT_CONTEXT) - set(VARIANT_CONTEXT))
        if unclassified:
            raise RunInvalid(
                f"the context carries {unclassified}, which is neither "
                f"registered as invariant across widths nor as something "
                f"width is allowed to move")

    entries = {}
    for candidate in rules.CANDIDATES:
        column = {}
        for width in rules.WIDTH_ORDER:
            if candidate not in CANDIDATES_AT_WIDTH[width]:
                column[width] = {
                    "type": "missing_share", "candidate": candidate,
                    "certified": candidate in CERTIFIED_CANDIDATES,
                    "reason_code": "not_measured_at_this_width",
                    "reason": (
                        f"candidate {candidate} is measured at the long width "
                        f"alone, per Amendment 7 clause 35, because that is "
                        f"where clause 26 records its dial fits"),
                }
                continue
            entry = readings[width]["entries"][candidate]
            if entry.get("type") not in ("reading", "missing_share"):
                raise RunInvalid(
                    f"candidate {candidate} at the {width} width is typed "
                    f"{entry.get('type')!r}, and the matrix carries readings "
                    f"and typed absences and nothing else")
            floor = FLOOR_FAMILY.get(candidate)
            if floor is not None:
                # Amendment 10 clause 45. The credited saving reads `F` off
                # the floor's own fitted slope, so the floor travels in the
                # SAME column as the share it divides, at the same width. A
                # copy, because the width record owns the entry it came from.
                entry = dict(entry)
                entry["floor"] = readings[width]["entries"][floor]
                entry["floor_candidate"] = floor
            column[width] = entry
        entries[candidate] = column
    return {
        "candidate_order": list(rules.CANDIDATES),
        "width_order": list(rules.WIDTH_ORDER),
        "entries": entries,
        "context_by_width": contexts,
    }


def binding_blockers(record: Mapping[str, object]) -> list[str]:
    """Every reason this recording cannot bind, in plain words.

    Gathered rather than raised, because a run that measured everything and
    then failed one gate is evidence: the samples are worth keeping and the
    reason is worth naming.

    Every gate the mode-based version held has a named replacement or a
    written reason it is gone, so nothing was lost by not being carried
    forward:

      closing idle           survives unchanged.
      the plain arm's spread  becomes every arm's spread, because with a
                             ladder there is no reference arm and any of them
                             can be the one the machine moved under.
      the compile transfer    gone by Amendment 5 clause 4: every timed arm is
                             compiled, so there is no uncompiled total to
                             transfer to.
      the reconciliation      gone by clause 2, which voids it outright, and
                             its named weaker replacement is clause 22's
                             partition, which is checked below.
      a share is a fraction   survives, inside `KnobReading.blockers`, along
                             with every fit gate the marks never had.
      the instrument band     gone: there is no timing device inside the step,
                             so there is no perturbation left to price. What
                             replaces it is the scaffold price, which is the
                             honest measure of what a dial costs and is
                             checked at every setting rather than at one.
      exact region counts     survives, in the untimed structural pass.
      marked/plain identity   survives, in the same untimed structural pass.

    A typed absence is not a blocker. Candidate A gates no terminal by
    Amendment 7 ruling 2, and candidate P3 is read only by a fault check that
    takes no margin, so their absence costs a report a row and a check a
    width. Candidates L and Q are the two the profile certifies, and a gate
    they fail is a gate the ruling would have rested on.
    """
    blockers = []
    if not record.get("closing_idle", {}).get("idle"):
        blockers.append("the machine was not idle at the closing gate")
    for cell, cell_record in sorted(record.get("cells", {}).items()):
        for width, reading in sorted(cell_record.get("readings", {}).items()):
            for label, arm in sorted(reading["arms"].items()):
                if not arm["rounds_eligible"]:
                    blockers.append(
                        f"cell {cell} {width} arm {label}: its own spread is "
                        f"{arm['spread_pct']:.1f}%, past the "
                        f"{reading['spread_limit_pct']:.1f}% limit, so its "
                        f"rounds describe the machine rather than the step")
            for candidate, entry in sorted(reading["entries"].items()):
                evidence = (entry if entry["type"] == "reading"
                            else entry["evidence"])
                # Clause 33: a residue below `-R` is a FAULT and not an
                # absence, whatever the candidate. It says the step ran SLOWER
                # with the operation removed than the fit predicts without its
                # scaling part, which is not a measurement that can be true,
                # so no rule carries it and no demotion excuses it.
                if evidence.get("residue_is_a_fault"):
                    blockers.append(
                        f"cell {cell} {width} candidate {candidate}: the raw "
                        f"residue is {evidence['raw_residue_ms']:.6f} ms, "
                        f"negative by more than the resolution floor of "
                        f"{evidence['resolution_floor_ms']:.6f} ms")
                if entry["type"] == "reading":
                    continue
                if candidate not in CERTIFIED_CANDIDATES:
                    continue
                for reason in entry["blockers"]:
                    blockers.append(f"cell {cell} {width}: {reason}")
            if reading["partition"]["outcome"] == "REJECT":
                blockers.append(
                    f"cell {cell} {width}: clause 22 REJECTS the profile - "
                    f"{reading['partition']['reason']}")
        # Amendment 7 clause 37 and clause 11: the deciding cell's certified
        # entries must be complete at BOTH widths, because the selection takes
        # the highest minimum gain across them and a candidate measured at one
        # width has no minimum to take. Candidate A absent at either width is
        # never incompleteness, which is REVERSED from Amendment 6.
        if cell == rules.PRIMARY_CELL:
            for candidate in CERTIFIED_CANDIDATES:
                for width in widths_for(cell):
                    entry = cell_record.get("readings", {}).get(
                        width, {}).get("entries", {}).get(candidate)
                    if entry is None:
                        blockers.append(
                            f"cell {cell} {width}: candidate {candidate} was "
                            f"not measured, and the selection takes the "
                            f"highest minimum gain across both widths")
        # Per width, because the widths run different batches through the
        # same seams and a fault at the second one would otherwise be recorded
        # and never read.
        for width, checked in sorted(cell_record.get("structural", {}).items()):
            if not checked["completeness"]["ok"]:
                blockers.append(
                    f"cell {cell} {width}: region counts differ from the "
                    f"arrangement: {checked['completeness']['mismatches']}")
            if checked.get("unregistered_shapes"):
                blockers.append(
                    f"cell {cell} {width}: this model runs quantized shapes "
                    f"{checked['unregistered_shapes']} that Amendment 5 never "
                    f"registered, so candidate Q's floor covers a site "
                    f"clause 23's kill rule never priced")
            if not checked["loss_equal"] or checked["gradients_differing"]:
                blockers.append(
                    f"cell {cell} {width}: the marked pass and the plain pass "
                    f"do not compute the same thing, so the counts beside it "
                    f"describe a different step from the one that was timed")
            if checked["foreign_on_removal"]:
                blockers.append(
                    f"cell {cell} {width}: {checked['foreign_on_removal']} "
                    f"was replaced while the structural pass ran, so what it "
                    f"counted is not what this profile installed")
    return blockers


def recording_path(results_dir: str | Path, day: date, *, kind: str,
                   binding: bool) -> Path:
    """Where a recording lands, with its verdict already in the name.

    Keyed on the final verdict rather than on the closing idle gate alone.
    The old name marked a recording REFUSED only when the machine went busy,
    so a run that measured everything cleanly on a quiet machine and then
    failed a fit gate landed under the same name as one that bound, and the
    difference was visible only inside the file.
    """
    suffix = ".json" if binding else ".REFUSED.json"
    return Path(results_dir) / f"profile-stock-{kind}-{day.isoformat()}{suffix}"


def validate_plan(plan: Mapping[str, object]) -> dict:
    """Freeze the unregistered choices together, before any model is loaded."""
    required = {"schema_version", "bands", "seed", "optimizer",
                "optimizer_config", "learning_rate", "memory_ceiling_gb",
                "wall_cap_seconds", "cells", "resolution"}
    missing = sorted(required - set(plan))
    if missing:
        raise RunInvalid(f"run plan is missing {missing}")
    if plan["schema_version"] != 2:
        raise RunInvalid(
            f"unknown plan schema {plan['schema_version']!r}; schema 1 named "
            f"marked modes and one dataset, and this harness measures knob "
            f"arms at two registered widths")
    # mlx-lm seeds its batch permutation behind `if seed:`, so a seed of 0
    # leaves numpy unseeded and draws a fresh permutation on every call.
    # Measured 2026-08-21 on the short band: six draws at seed 0 gave 116,
    # 109, 114, 130, 114 and 121 supervised rows, and six at seed 7 gave 91
    # every time. The fixed batch is what makes five rounds repeats of one
    # measurement, and candidate L's floor is DEFINED on the supervised rows,
    # so an unseeded draw silently moves both.
    seed = plan["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0:
        raise RunInvalid(
            f"the plan registers seed {seed!r}; mlx-lm's own iterator seeds "
            f"itself only when the seed is truthy, so a seed of 0 draws a "
            f"different batch every call and the width's fixed batch is not "
            f"fixed at all. A positive integer is required")

    # A stock profile installs nothing. A plan that could name a kept kernel
    # could produce a profile of somebody's kernel labelled as stock, so the
    # key is refused outright rather than defaulted to empty.
    for banned in ("kept_candidates", "measurement_module"):
        if banned in plan:
            raise RunInvalid(
                f"a stock profile installs nothing, so a plan may not carry "
                f"{banned!r}")
    cells = list(plan["cells"])
    unknown = sorted(set(cells) - set(rules.CELLS))
    if unknown:
        raise RunInvalid(f"not registered cells: {unknown}")
    if not cells:
        raise RunInvalid("a profile with no cells measures nothing")

    # One corpus, two bands, per Amendment 5 clause 20: two datasets have two
    # supervised fractions, which IS candidate L's floor, so a width
    # comparison drawn from two corpora would carry a second variable and
    # report it as width.
    bands = dict(plan["bands"])
    needed_bands = sorted({width for cell in cells
                           for width in widths_for(cell)})
    if sorted(bands) != needed_bands:
        raise RunInvalid(
            f"the plan pins bands {sorted(bands)} and the requested cells "
            f"need exactly {needed_bands}")
    for width, directory in sorted(bands.items()):
        if Path(directory).name != rules.WIDTHS[width]["data"]:
            raise RunInvalid(
                f"the {width} width is registered at band "
                f"{rules.WIDTHS[width]['band']}, whose slice is "
                f"{rules.WIDTHS[width]['data']!r}, and the plan points it at "
                f"{Path(directory).name!r}")

    # Clause 21 forbids reusing one resolution floor across contexts, and
    # clause 26 refuses a floor of zero, so both floors enter as data the plan
    # copies out of the committed addendum and the artifact that supplied them
    # is named. The harness reads `R` and never measures it: a run that could
    # set its own floor could set it after seeing what it needed to clear.
    resolution = plan["resolution"]
    floors = dict(resolution.get("floors_ms", {}))
    needed = sorted({width for cell in cells for width in widths_for(cell)})
    for width in needed:
        value = floors.get(width)
        if not isinstance(value, (int, float)) or not value > 0.0:
            raise RunInvalid(
                f"the plan carries no positive resolution floor for the "
                f"{width} width, and every gate clause 26 registers is a "
                f"multiple of one")
    if not resolution.get("addendum_sha256"):
        raise RunInvalid(
            "the plan names no addendum for its resolution floors, so a "
            "reading could not be traced to the measurement that set them")

    fixed = dict(plan)
    fixed["cells"] = cells
    fixed["kept_candidates"] = []
    fixed["measurement_module"] = STOCK_MEASUREMENT_MODULE
    fixed["rounds"] = int(plan.get("rounds", ROUNDS))
    if fixed["rounds"] < 1:
        raise RunInvalid("a round count below one measures nothing")
    fixed["bands"] = bands
    fixed["widths"] = {cell: list(widths_for(cell)) for cell in cells}
    fixed["resolution"] = dict(resolution, floors_ms=floors)
    return fixed


def decide(profile: Mapping[str, object],
           sweep: Mapping[str, object]) -> dict:
    """Step 9's floors, refused here rather than half-done.

    The selection rule is rewritten and lives in `profile_rules`, and it reads
    a CREDITED SAVING per candidate per width. A saving is `M * (1 - F/N)`,
    and `F` is the floor's credited denominator: candidate Q's dense fp16
    comparison installed at the same seam as its dial, and candidate L's bench
    arrangement under clause 19's written exception. Neither exists until the
    ceiling sweep is built.

    So this refuses on the input rather than on the arithmetic. A `--decide`
    that supplied its own `F` would be inventing the one measurement the whole
    credit rests on.
    """
    del profile, sweep
    raise RunInvalid(
        "no ceiling sweep exists yet, so no candidate has a credited "
        "denominator and no saving can be computed; the recording is closed "
        "and hashed and loses nothing by waiting for it")


# ---------------------------------------------------------------------------
# The child: everything below here touches MLX and needs the device
# ---------------------------------------------------------------------------
def stock_process() -> dict:
    """Refuse unless this process is the one the profile claims to measure."""
    from metalrunner import routing, seams

    observed = {}
    for name in sorted(pi.REGIONS):
        seam = pi.REGIONS[name].seam
        held = seams.current(seam)
        module = getattr(held, "__module__", None)
        if module != seam.origin:
            raise RunInvalid(
                f"{seam} holds an object defined in {module!r}, not "
                f"{seam.origin!r}: this process is not stock, and a profile "
                f"taken through it would describe somebody else's patch")
        observed[str(seam)] = module
    if routing.CERTIFIED:
        raise RunInvalid(
            f"metalrunner certifies {len(routing.CERTIFIED)} kernel(s), and "
            f"the profile measures stock with none of ours installed")
    return {"seams": observed, "certified": len(routing.CERTIFIED),
            "forced_to_stock": routing.forced_to_stock()}


def _dataset_args(plan: Mapping[str, object], data_dir: str,
                  mask_prompt: bool) -> types.SimpleNamespace:
    """The subset of mlx-lm's config that `load_dataset` actually reads."""
    return types.SimpleNamespace(
        data=data_dir, train=True, test=False, hf_dataset=False,
        mask_prompt=mask_prompt, prompt_feature="prompt",
        completion_feature="completion", text_feature="text",
        chat_feature="messages")


def _load_model(plan: Mapping[str, object], provenance: Mapping[str, object]):
    """The model, the adapters and the optimizer, built mlx-lm's way, once.

    The optimizer's state is initialised and evaluated HERE, before any arm is
    built, which is clause 25's second requirement: Adam allocates `m` and `v`
    on its first update, which grows the tree `mx.compile` captured and forces
    a second trace. If that trace lands after an arm's seam is gone, the arm
    silently becomes stock and nothing downstream can tell.
    """
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers
    from mlx_lm.utils import load

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    else:
        raise NoDevice("the profile measures a GPU step and there is no GPU")

    model, tokenizer = load(provenance["base_model"]["directory"])
    mx.random.seed(plan["seed"])
    model.freeze()
    linear_to_lora_layers(model, rules.LORA_LAYERS,
                          {"rank": rules.LORA_RANK, "scale": 20.0,
                           "dropout": 0.0})
    model.train()

    schedule = plan.get("schedule")
    rate = build_schedule(schedule) if schedule else plan["learning_rate"]
    name = str(plan["optimizer"]).lower()
    classes = {"adam": optim.Adam, "adamw": optim.AdamW, "sgd": optim.SGD}
    if name not in classes:
        raise RunInvalid(f"the profile does not register optimizer {name!r}")
    optimizer = classes[name](learning_rate=rate,
                              **dict(plan["optimizer_config"]))
    pk.settle_optimizer(model, optimizer)

    return types.SimpleNamespace(
        model=model, tokenizer=tokenizer, optimizer=optimizer,
        state=[model.state, optimizer.state, mx.random.state],
        depth=len(model.model.layers), adapted=rules.LORA_LAYERS)


def _fixed_batch(plan: Mapping[str, object], tokenizer, data_dir: str, *,
                 batch: int, mask_prompt: bool):
    """ONE batch out of mlx-lm's own iterator, then frozen for the width.

    Taken from the iterator rather than assembled here, so its width, its
    padding and its prompt mask are the ones a real fine-tune would produce.
    Frozen for the whole width, which is what makes five rounds repeats of one
    measurement rather than five different workloads.
    """
    import mlx.core as mx
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import iterate_batches

    train_set, _, _ = load_dataset(
        _dataset_args(plan, data_dir, mask_prompt), tokenizer)
    batches = iterate_batches(dataset=CacheDataset(train_set),
                              batch_size=batch,
                              max_seq_length=rules.SEQ_LEN,
                              loop=False, seed=plan["seed"],
                              comm_group=mx.distributed.init())
    tokens, lengths = next(batches)
    mx.eval(tokens, lengths)
    # The supervised count is derived from the batch by mlx-lm's own rule
    # rather than read off a step's return value. The step is compiled and
    # every arm's loss is meaningless at a dialled setting, so a count taken
    # from one would be a count of whatever that arm happened to compute; the
    # rule below is `default_loss`'s own mask, applied to the same tensors.
    targets = tokens[:, 1:]
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    supervised = int(mask.sum().item())
    digest = _json_sha256({"tokens": tokens.tolist(),
                           "lengths": lengths.tolist()})
    return types.SimpleNamespace(batch=(tokens, lengths), digest=digest,
                                 rows=int(tokens.shape[0]),
                                 width=int(tokens.shape[1]),
                                 supervised=supervised,
                                 supervised_of=int(targets.size))


def _structural_pass(model, batch) -> dict:
    """The boundary instrument, once, untimed, contributing no number.

    Two things only, and both have caught a real fault. The COUNTS say every
    seam fired as often as the arrangement demands, which is what tells a seam
    that never fired apart from an operation that costs nothing. The IDENTITY
    says the marked pass computes what the plain pass computes, on the same
    weights and over every gradient array rather than a summary, because a
    summary can agree while the arrays beneath it do not.

    Uncompiled, because a mark evals and MLX refuses an eval inside a compiled
    step, which is the whole of Amendment 4. Nothing timed runs in this state.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.trainer import default_loss

    value_and_grad = nn.value_and_grad(model, default_loss)

    def once():
        (loss, _toks), grad = value_and_grad(model, *batch)
        mx.eval(loss, grad)
        return loss, dict(tree_flatten(grad))

    mx.disable_compile()
    try:
        once()
        plain_loss, plain_grad = once()
        if not plain_grad:
            raise RunInvalid(
                "the model produced no gradients, so nothing was compared "
                "and the identity claim is empty")
        recorder = pi.Recorder()
        installation = pi.install(sorted(pi.REGIONS), recorder)
        try:
            marked_loss, marked_grad = once()
        finally:
            installation.remove()
    finally:
        mx.enable_compile()

    return {
        "gradients_compared": len(plain_grad),
        "loss_equal": bool(mx.array_equal(plain_loss, marked_loss)),
        "gradients_differing": sorted(
            name for name in plain_grad
            if not bool(mx.array_equal(plain_grad[name], marked_grad[name]))),
        "counts": pi.counts(recorder.entries),
        # Clause 8's shape ratio is a sum over directions of `calls * cost per
        # call`, "with the call counts supplied by the structural pass of
        # clause 3", and this is that supply. Per shape AND per direction,
        # because under LoRA the two genuinely differ: the three projections
        # consuming the lowest adapted block's input have no backward at all.
        "shape_counts": {region: {shape: dict(directions)
                                  for shape, directions in sorted(shapes.items())}
                         for region, shapes in sorted(recorder.shapes.items())},
        "foreign_on_removal": list(installation.foreign_on_removal),
        # Amendment 10 clause 45's floor holds one dense weight per SHAPE, so
        # it covers whatever shapes this model runs. Clause 23's kill prices
        # exactly the six Amendment 5 registers, so a seventh would be a site
        # the floor credits and the kill never looked at.
        "unregistered_shapes": [list(shape)
                                for shape in pk.unregistered_shapes(model)],
    }


def _knob_for(candidate: str) -> pk.Knob:
    """The dial one candidate is measured on, looked up and never chosen.

    Candidate A's entry is `ATTENTION_KNOBS[ATTENTION_DIAL]` rather than a
    price comparison run at measurement time: Amendment 7 clause 35 names the
    dial by clause 15's own completeness tie-break, so no run reselects it and
    a recording cannot depend on which dial happened to price best that night.
    """
    if candidate == "A":
        return pk.ATTENTION_KNOBS[ATTENTION_DIAL]
    if candidate in pk.KNOBS:
        return pk.KNOBS[candidate]
    raise RunInvalid(f"no dial is registered for candidate {candidate!r}")


def _kept_sizes(prepared: Mapping[str, object], phi: float) -> dict:
    """Every dialled site's kept size at one setting, keyed by the site.

    The key is internal and is whatever identifies a site inside one prepared
    ladder; `_actual_settings` relabels it by the site's FULL dimension before
    anything is recorded, because a module's `id()` is a memory address that
    means nothing in a recording while "the 2560-wide projections" is a thing
    a reader can check against the model card.
    """
    sizes = {}
    for name, ladder in sorted(prepared.items()):
        entry = ladder[phi]
        if name == "projections":
            for site, value in entry.items():
                sizes[(name, site)] = value[0]
        elif name == "dense":
            for shape, value in sorted(entry.items()):
                sizes[(name, shape)] = value[0]
        elif name == "attention":
            sizes[(name, None)] = entry["kept"]
        else:
            sizes[(name, None)] = entry[0]
    return sizes


def _actual_settings(candidate: str, prepared: Mapping[str, object]) -> dict:
    """What fraction each nominal setting ACTUALLY placed, and where.

    A dial asked for 0.75 places a whole number of quantization groups or a
    whole number of rows, and the fraction it lands on is a property of the
    model's dimensions. Fitting the fraction that was asked for rather than
    the one that was placed is a fit on numbers nothing measured.

    Two settings that place ONE size are one setting, not two: their arms
    compute the same thing, so the fit would carry a repeated point that adds
    no evidence about linearity while raising R-squared. That refuses here
    rather than being silently deduplicated, because Amendment 7 registers
    four knob arms per candidate and dropping one changes a registered count.

    Sites that disagree about the fraction they placed also refuse. Nothing
    registers how to reduce several per-site fractions to the one scalar a fit
    reads, and inventing that reduction here would be a rule made to fit the
    model in front of it.
    """
    full = _kept_sizes(prepared, pk.PHIS[0])
    labelled = {key: f"{key[0]}:{size}" for key, size in full.items()}
    settings = {}
    for phi in pk.PHIS:
        sizes = _kept_sizes(prepared, phi)
        if set(sizes) != set(full):
            raise RunInvalid(
                f"candidate {candidate} dialled {len(sizes)} sites at {phi} "
                f"and {len(full)} at {pk.PHIS[0]}, so the ladder is not one "
                f"dial moving one set of operations")
        fractions = {key: sizes[key] / full[key] for key in sizes}
        distinct = sorted(set(fractions.values()))
        if len(distinct) != 1:
            placed = sorted({(labelled[key], value)
                             for key, value in fractions.items()})
            raise RunInvalid(
                f"candidate {candidate} at nominal {phi} placed fractions "
                f"{placed}, and nothing registers how several per-site "
                f"fractions reduce to the one scalar a fit reads")
        settings[phi] = {
            "actual_phi": distinct[0],
            "sites": {labelled[key]: {"kept": sizes[key], "full": full[key],
                                      "fraction": fractions[key]}
                      for key in sorted(sizes, key=lambda one: labelled[one])},
        }
    placed = [settings[phi]["actual_phi"] for phi in pk.PHIS]
    if len(set(placed)) != len(placed):
        raise RunInvalid(
            f"candidate {candidate}'s four nominal settings placed only "
            f"{sorted(set(placed))}; two arms computing the same thing are a "
            f"repeated point in the fit, and the manifest registers four")
    return settings


def _build_width(target, batch, width: str, guard: _ChildGuard) -> dict:
    """Every arm of one width, prepared, built, traced and warmed.

    The ordering is the whole point and it is not an implementation detail.
    Every dial's operands are materialised OUTSIDE any timed region, so arms
    differ in the work they do and not in when they paid for it. Each arm then
    gets its own freshly built closure, because MLX keys its compile cache on
    the underlying callable and two arms sharing one raw function share one
    traced graph, which would make the second arm's dial do nothing. And the
    trace happens at an object's FIRST CALL rather than at construction, so
    the seams are live across the warm-ups and gone before anything is timed.
    """
    prepared = {}
    settings = {}
    for candidate in CANDIDATES_AT_WIDTH[width]:
        knob = _knob_for(candidate)
        prepared[candidate] = knob.prepare(target.model, batch.width)
        settings[candidate] = _actual_settings(candidate, prepared[candidate])

    compiled, roles = [], {}
    for spec in arm_manifest(width):
        guard.check(f"{width}: build {spec.label}")
        if spec.role == pk.STOCK:
            seams_map, actual = {}, None
        else:
            knob = _knob_for(spec.candidate)
            if spec.role == pk.KNOB:
                seams_map = knob.arm(prepared[spec.candidate],
                                     spec.nominal_phi)
            elif spec.role == pk.SCAFFOLD:
                seams_map = knob.scaffold(prepared[spec.candidate],
                                          spec.nominal_phi)
            elif spec.role == pk.REFERENCE:
                seams_map = knob.reference(prepared[spec.candidate])
            else:
                seams_map = knob.ablate(prepared[spec.candidate])
            actual = (settings[spec.candidate][spec.nominal_phi]["actual_phi"]
                      if spec.nominal_phi is not None else None)
        compiled.append(pk.prepare_arm(
            target.model, target.optimizer, target.state, batch.batch,
            pk.Arm(label=spec.label, phi=actual, seams=seams_map)))
        roles[spec.label] = {
            "candidate": spec.candidate, "role": spec.role,
            "nominal_phi": spec.nominal_phi, "actual_phi": actual,
            "sites": (settings[spec.candidate][spec.nominal_phi]["sites"]
                      if spec.nominal_phi is not None else None),
        }
    return {"compiled": compiled, "roles": roles}


def _profile_child(task: Mapping[str, object], guard: _ChildGuard) -> dict:
    """One cell: one model, one batch per width, every arm in rotation."""
    plan = task["plan"]
    provenance = task["provenance"]
    cell = task["cell_name"]
    rounds = int(plan["rounds"])
    registered = rules.CELLS[cell]
    mask_prompt = registered["supervision"] == "masked"

    stock = stock_process()
    guard.check(f"{cell}: load")
    target = _load_model(plan, provenance)
    model_name = Path(provenance["base_model"]["directory"]).name

    widths, structural = {}, {}
    for width in widths_for(cell):
        guard.check(f"{cell} {width}: batch")
        batch = _fixed_batch(plan, target.tokenizer,
                             plan["bands"][width],
                             batch=registered["batch"],
                             mask_prompt=mask_prompt)
        # Expected over EVERY region the instrument marks, not over the
        # regions that reported. A seam installed and never reached leaves its
        # region absent from the log entirely, so checking only what appeared
        # would let exactly the failure this check exists for pass silently.
        checked = _structural_pass(target.model, batch.batch)
        checked["completeness"] = completeness(
            checked["counts"],
            expected_counts(target.depth, target.adapted, sorted(pi.REGIONS)))
        structural[width] = checked
        built = _build_width(target, batch, width, guard)
        guard.check(f"{cell} {width}: {len(built['compiled'])} arms timed")
        # Seconds become milliseconds HERE and nowhere else. Amendment 6
        # clause 33 puts every rule's input in milliseconds, and one artifact
        # carrying two unit conventions is exactly what that discipline exists
        # to prevent, so the conversion happens at the single point where a
        # timer's output becomes a recorded sample.
        samples = pk.timed_rounds(built["compiled"], batch.batch,
                                  rounds=rounds)
        context = cell_context(
            cell, width, batch=batch.rows, batch_width=batch.width,
            model=model_name, adapted=target.adapted,
            supervised=batch.supervised, supervised_of=batch.supervised_of,
            batch_sha256=batch.digest)
        widths[width] = {
            "width": width,
            "context": context,
            # The process peak is monotonic, so it is a property of the width
            # rather than of any one arm; recording it per arm would imply a
            # per-arm measurement nothing took.
            "peak_gb": phys_footprint_gb()[1],
            "arms": {label: dict(built["roles"][label],
                                 samples_ms=[value * 1000.0
                                             for value in samples[label]],
                                 context=context)
                     for label in samples},
            # Both counts, because `timed_rounds` proves they are equal by
            # refusing the run otherwise, and a record that carried only one
            # could not show that the proof had anything to prove.
            "traces": {arm.label: {"after_warmup": arm.traced_by_warmup,
                                   "after_timing": len(arm.traces)}
                       for arm in built["compiled"]},
        }

    return {
        "cell": cell,
        "depth": target.depth,
        "adapted": target.adapted,
        "widths": widths,
        "structural": structural,
        "stock": stock,
        "rounds": rounds,
    }


def child_main(task_path: Path, result_path: Path) -> int:
    """Run one profile cell, or refuse with a number the runner understands."""
    if os.environ.get("KV_FORCE_NO_METAL") == "1":
        print("REFUSED: KV_FORCE_NO_METAL disables the measurement child")
        return EXIT_NO_DEVICE
    try:
        task = json.loads(task_path.read_text())
        if not isinstance(task, Mapping):
            raise PreconditionFailed("child task is not a JSON object")
        budget = task.get("budget_gb")
        parent_pid = task.get("parent_pid")
        if not isinstance(budget, (int, float)) or not isinstance(parent_pid, int):
            raise PreconditionFailed("child task has no budget or parent pid")
        guard = _ChildGuard(float(budget), parent_pid)
        guard.check(f"{task.get('cell', 'child')}: startup")
        _check_child_inputs(task, stack_record())
        if task.get("kind") != "profile":
            raise PreconditionFailed(f"unknown child kind {task.get('kind')!r}")
        result = _profile_child(task, guard)
        guard.check(f"{task.get('cell', 'child')}: publish")
        current, peak = phys_footprint_gb()
        if peak > budget:
            raise BudgetExceeded(str(task.get("cell", "child")), peak, budget)
        result["footprint_gb"] = {"current": current, "peak": peak}
        _atomic_json(result_path, result)
        return 0
    except BudgetExceeded as error:
        print(f"REFUSED (exit {EXIT_BUDGET_REFUSAL}): {error}")
        return EXIT_BUDGET_REFUSAL
    except LowMemoryRefusal as error:
        print(f"REFUSED (exit {EXIT_LOW_MEMORY}): {error}")
        return EXIT_LOW_MEMORY
    except Orphaned as error:
        print(f"REFUSED (exit {EXIT_ORPHANED}): {error}")
        return EXIT_ORPHANED
    except NoDevice as error:
        print(f"REFUSED (exit {EXIT_NO_DEVICE}): {error}")
        return EXIT_NO_DEVICE
    except (PreconditionFailed, RunInvalid, OSError,
            json.JSONDecodeError) as error:
        print(f"REFUSED (exit {EXIT_PRECONDITION}): {error}")
        return EXIT_PRECONDITION


# ---------------------------------------------------------------------------
# The parent
# ---------------------------------------------------------------------------
class SystemRuntime:
    """The concrete machine, guard, child and recording seams."""

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.lock = MeasurementLock("profile_stock")
        self.machine = None
        self.guard = None
        self._temporary = None

    def acquire_lock(self) -> tuple[bool, str]:
        acquired, detail = self.lock.acquire()
        if acquired:
            try:
                self._temporary = tempfile.TemporaryDirectory(
                    prefix="kernelverify-profile-stock-")
            except Exception:
                self.lock.release()
                raise
        return acquired, detail

    def release_lock(self) -> None:
        try:
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
        finally:
            self.lock.release()

    def fingerprint(self) -> dict:
        self.machine = machine_state.fingerprint()
        return self.machine

    def idle_check(self, cores: int) -> dict:
        return machine_state.idle_check(cores)

    def require_memory(self, budget_gb: float, cell: str) -> None:
        if self.guard is None:
            self.guard = BudgetGuard(budget_gb)
        elif self.guard.budget_gb != budget_gb:
            raise PreconditionFailed("memory ceiling changed during the run")
        current = self.guard.check(cell)
        require_available_memory(max(budget_gb - current, 0.0), cell)

    def preflight(self, plan: Mapping[str, object]) -> dict:
        if self.machine is None:
            raise PreconditionFailed("machine fingerprint was not sampled")
        return preflight_inputs(plan, root=self.root, machine=self.machine)

    @property
    def work_dir(self) -> Path:
        if self._temporary is None:
            raise PreconditionFailed(
                "the child workspace exists only under the lock")
        return Path(self._temporary.name)

    def run_cell(self, plan: Mapping[str, object],
                 provenance: Mapping[str, object], cell: str) -> dict:
        return spawn_child(
            {"kind": "profile", "cell": f"cell {cell}", "cell_name": cell,
             "plan": plan, "provenance": provenance,
             "budget_gb": plan["memory_ceiling_gb"]},
            self.work_dir,
            wall_cap_s=plan["wall_cap_seconds"],
            child_entrypoint=Path(__file__).resolve())

    def write_record(self, path: Path, record: Mapping[str, object]) -> None:
        """Exclusive create: a recording is never overwritten.

        A run that could overwrite its own recording could also be re-run
        until it produced a number somebody liked, and nothing in the file
        would show it. The ruling is written by `--decide` into a separate
        artifact for the same reason.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x") as handle:
                handle.write(json.dumps(record, indent=2, sort_keys=True))
                handle.write("\n")
        except FileExistsError as error:
            raise PreconditionFailed(
                f"{path} already exists and a recording is never overwritten"
            ) from error

    def today(self) -> date:
        return date.today()


def _print_refusal(reason: object) -> None:
    print(json.dumps({"refused": reason if isinstance(reason, (str, Mapping))
                      else str(reason)}, default=str), flush=True)


def run_profile(plan: Mapping[str, object], runtime, *,
                results_dir: str | Path = RESULTS_DIR) -> int:
    """Measure every requested cell, then let the gates decide.

    There is no `--calibrate` mode. The one that existed priced the boundary
    instrument, and Amendment 5 clause 1 removed the instrument from the
    timing path, so its band has no referent. What replaces it is a
    calibration STAGE of its own, which runs identical and scaffold-only arms
    with no candidate dialled and no share computed, and whose output enters
    this harness as the registered resolution floor the plan carries.
    """
    try:
        fixed = validate_plan(plan)
    except RunInvalid as error:
        _print_refusal(error)
        return EXIT_PRECONDITION

    acquired, lock_reason = runtime.acquire_lock()
    if not acquired:
        _print_refusal(lock_reason)
        return EXIT_LOCK_HELD
    try:
        machine = runtime.fingerprint()
        opening_idle = runtime.idle_check(machine["cores"])
        if not opening_idle["idle"]:
            _print_refusal(opening_idle)
            return EXIT_NOT_IDLE
        try:
            runtime.require_memory(fixed["memory_ceiling_gb"], "startup")
            provenance = runtime.preflight(fixed)
        except BudgetExceeded as error:
            _print_refusal(error)
            return EXIT_BUDGET_REFUSAL
        except LowMemoryRefusal as error:
            _print_refusal(error)
            return EXIT_LOW_MEMORY
        except PreconditionFailed as error:
            _print_refusal(error)
            return EXIT_PRECONDITION

        cells = {}
        try:
            for cell in fixed["cells"]:
                runtime.require_memory(fixed["memory_ceiling_gb"], f"cell {cell}")
                cells[cell] = runtime.run_cell(fixed, provenance, cell)
                print(json.dumps({"cell": cell, "measured": True}), flush=True)
        except BudgetExceeded as error:
            _print_refusal(error)
            return EXIT_BUDGET_REFUSAL
        except LowMemoryRefusal as error:
            _print_refusal(error)
            return EXIT_LOW_MEMORY
        except ChildRefusal as error:
            _print_refusal(error)
            return child_exit_for_parent(error.returncode)
        except PreconditionFailed as error:
            _print_refusal(error)
            return EXIT_PRECONDITION

        closing_idle = runtime.idle_check(machine["cores"])
        floors = fixed["resolution"]["floors_ms"]
        try:
            for name, one in sorted(cells.items()):
                one["readings"] = {
                    width: width_reading(measured,
                                         resolution_floor_ms=floors[width])
                    for width, measured in sorted(one["widths"].items())}
        except (RunInvalid, KeyError) as error:
            _print_refusal(error)
            return EXIT_PRECONDITION
        record = {
            "schema_version": 2,
            "kind": "binding",
            "units": {"time": "ms", "fraction": "dimensionless"},
            "plan": fixed,
            "provenance": provenance,
            "machine": machine,
            "opening_idle": opening_idle,
            "closing_idle": closing_idle,
            "cells": cells,
        }
        # The matrix is built only where the ruling is computed. The other
        # cells are reported in full under `cells` and decide nothing under
        # section 4.3, so giving them a rule-facing table would invite one.
        if rules.PRIMARY_CELL in cells:
            try:
                record["profile_matrix"] = profile_matrix(
                    cells[rules.PRIMARY_CELL]["readings"])
            except RunInvalid as error:
                _print_refusal(error)
                return EXIT_PRECONDITION
        record["binding_blockers"] = binding_blockers(record)
        record["binding"] = not record["binding_blockers"]
        path = recording_path(results_dir, runtime.today(), kind="binding",
                              binding=record["binding"])
        try:
            runtime.write_record(path, record)
        except PreconditionFailed as error:
            _print_refusal(error)
            return EXIT_PRECONDITION
        print(json.dumps({"recording": str(path),
                          "binding": record["binding"],
                          "blockers": record["binding_blockers"]},
                         indent=2), flush=True)
        return 0 if record["binding"] else 1
    finally:
        runtime.release_lock()


def _load_json(path: Path) -> dict:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PreconditionFailed(f"cannot read {path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise PreconditionFailed(f"{path} is not a JSON object")
    return dict(raw)


def run_decide(profile_path: Path, sweep_path: Path,
               out_path: Path | None) -> int:
    """Apply the registered rule to two closed recordings, once."""
    try:
        artifact = decide(_load_json(profile_path), _load_json(sweep_path))
    except (PreconditionFailed, RunInvalid) as error:
        _print_refusal(error)
        return EXIT_PRECONDITION
    destination = (Path(out_path) if out_path is not None else
                   Path(profile_path).with_name(
                       f"profile-decision-{date.today().isoformat()}.json"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x") as handle:
            handle.write(json.dumps(artifact, indent=2, sort_keys=True))
            handle.write("\n")
    except FileExistsError:
        _print_refusal(f"{destination} already exists; a ruling is written once")
        return EXIT_PRECONDITION
    print(json.dumps({"decision": str(destination),
                      "selected": artifact["ruling"]["selected"],
                      "verdict": artifact["ruling"]["verdict"]}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--decide", action="store_true",
                        help="apply the registered rule to closed recordings")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--sweep", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--child-task", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child_task is not None:
        if args.child_out is None:
            parser.error("--child-task requires --child-out")
        return child_main(args.child_task, args.child_out)
    if args.decide:
        if args.profile is None or args.sweep is None:
            print(f"REFUSED (exit {EXIT_PRECONDITION}): --decide needs "
                  f"--profile and --sweep")
            return EXIT_PRECONDITION
        return run_decide(args.profile, args.sweep, args.out)
    if args.plan is None:
        print(f"REFUSED (exit {EXIT_PRECONDITION}): --plan is required")
        return EXIT_PRECONDITION
    try:
        plan = _load_json(args.plan)
    except PreconditionFailed as error:
        _print_refusal(error)
        return EXIT_PRECONDITION
    return run_profile(plan, SystemRuntime(), results_dir=RESULTS_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
