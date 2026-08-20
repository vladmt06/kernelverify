#!/usr/bin/env python3
"""The Day 1 stock training profile: where a real QLoRA step's time goes.

What this produces, and what it refuses to produce
--------------------------------------------------
It produces one number per candidate operation: the share f of the training
step that operation occupies, forward and backward, measured inside mlx-lm's
own step on the pinned model and the pinned corpus. It produces no verdict.
The gain formula, the tie rule and the selection live in `profile_rules`, were
committed before this file existed, and are applied by `--decide` to a
recording this harness has already closed and hashed. A harness that could
also rule would be a harness that could quietly become the thing that decides.

One child per cell, one batch, every mode round-robin
-----------------------------------------------------
Amendment 4 requires the step total both compiled and uncompiled "in the same
run, on the same batch". A fresh process per mode cannot satisfy that, so a
cell is one child: it loads the model once, pulls ONE batch out of mlx-lm's
own iterator, and then drives the trainer's step through every mode in
rotation, five rounds. The rotation is why no mode always runs warm, and the
single batch is why five rounds are genuine repeats rather than five different
workloads.

Why the deciding cell splits its instrumented passes
-----------------------------------------------------
Every mark costs time, and that cost lands inside the marked region's
numerator while the step total it is divided by carries every region's cost.
Attention is marked once per layer, the output head once per step, the
quantized projections seven times per layer. Marking all of them in one pass
therefore inflates the heavily marked region's share and deflates the others,
which is precisely the three-way comparison the selection rule makes. So cell
B runs one instrumented pass per candidate, and each share takes its
denominator from the plain pass in the same child. Cells A, C and D decide
nothing, keep one combined pass, and their shares are labelled as carrying
cross-region bias rather than quietly presented as clean.

What is checked, and against what
---------------------------------
Section 3.3 registered "named regions plus one remainder must equal the step
within 2%", and which step that is decides whether the check exists at all.
Against the marked pass's own elapsed it cannot fail, because the remainder is
defined as whatever the spans leave on the same timeline. Amendment 4 does not
say that: it compares against the step "taken without the interior
boundaries", which is the plain pass. Read as written the check is therefore
the instrument's own cost measured against a 2% limit, so section 3.3's
reconciliation and Amendment 5's instrument-cost band are one quantity at two
limits rather than two checks. The 2% is what is registered today, and no
instrument that can put a clock inside an MLX backward meets it, so a
recording says so and does not bind.

Three other things are checked and each can fail:

  exact counts      - forward, a region fires once per place it appears.
                      Backward it does NOT: mlx-lm adapts only the last
                      `num_layers` blocks, so attention has a backward once
                      per ADAPTED block and the projections
                      `7 * adapted - 3` times, the three being those that
                      consume the lowest adapted block's input, whose gradient
                      nothing below asks for. A seam installed and never
                      reached leaves its region absent from the log, so its
                      share reads as zero and its time lands silently in the
                      remainder, and only a count tells that apart from an
                      operation that really costs nothing.
  identity          - the marked pass and the plain pass must agree on the
                      loss and on every gradient array, or the custom gradient
                      rule changed the computation and the two modes describe
                      different work.
  a share is a fraction - a candidate whose share does not land in (0, 1) has
                      not measured a fraction of a step. That happens: the
                      marks sit inside the spans that form the numerator and
                      outside the plain step that forms the denominator, so
                      every share is an upper bound and the overstatement
                      grows with how many marks the candidate carries.

And the process must be stock. Before anything is timed, every seam is checked
to hold the object mlx-lm itself defines and metalrunner is checked to certify
nothing, because a profile of stock taken through somebody's patch is a
profile of the patch.

Refused deliberately
--------------------
`--grad-checkpoint` is not offered. Checkpointing discards activations and
runs a block's forward AGAIN during the backward; the marks survive that with
the loss and gradients unchanged, but the replayed forward fires forward
marks, so a region's forward count and forward span absorb work belonging to
the backward. Measured 2026-08-20: attention fires 30 times forward on a
28-block model and the projections 210 times where an unchecked step fires
196. Nothing downstream would notice, and the collapse rule weights a credited
ratio by exactly those counts.

Usage
-----
    python bench/profile_stock.py --plan <plan.json> --calibrate
    bench/start_binding_run.sh bench/profile_stock.py -- --plan <plan.json>
    python bench/profile_stock.py --decide --profile <rec.json> --sweep <rec.json>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
import types
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import machine_state
import profile_instrument as pi
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

# Section 3.4. Five rounds, and the calibration run takes fewer because it
# binds nothing and exists only to price the instrument.
ROUNDS = 5
CALIBRATION_ROUNDS = 3

# The profile installs nothing. The shared preflight records which module a
# run could have routed through, and for this run the honest answer is the
# real installer with nothing certified in it.
STOCK_MEASUREMENT_MODULE = "metalrunner.measurement"

# Modes at the cell that decides, and at the cells that only report. The
# deciding cell splits one instrumented pass per candidate; see the module
# docstring for why the combined pass biases exactly the comparison the rule
# makes.
MODES_DECIDING = ("compiled", "plain", "instr-A", "instr-L", "instr-Q")
# `compiled` is here too, and not as symmetry. Amendment 4 requires the
# compiled-to-uncompiled ratio "beside every share it publishes", and the
# cells that decide nothing still publish shares - section 4.3 requires them
# reported, and requires cell C's ordering compared against cell B's. A cell
# with no compiled total could not carry the bound its own shares are read
# under.
MODES_REPORTING = ("compiled", "plain", "instr-all")

MODE_REGIONS = {
    "compiled": (),
    "plain": (),
    "instr-all": tuple(sorted(pi.REGIONS)),
    **{f"instr-{candidate}": regions
       for candidate, regions in pi.CANDIDATE_REGIONS.items()},
}

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
def rotated(items: Sequence[str], round_index: int) -> tuple[str, ...]:
    """Rotate the first slot while preserving the registered order.

    Without it the first mode of every round always runs on a machine that has
    just been idle and every later mode always runs warm, so the mode order
    would be part of what the profile measures.
    """
    items = tuple(items)
    offset = round_index % len(items)
    return items[offset:] + items[:offset]


def modes_for(cell: str) -> tuple[str, ...]:
    """Which modes a cell runs. Only the primary cell splits its passes."""
    if cell not in rules.CELLS:
        raise RunInvalid(f"not a registered cell: {cell!r}")
    return (MODES_DECIDING if cell == rules.PRIMARY_CELL
            else MODES_REPORTING)


def candidates_from(mode: str) -> tuple[str, ...]:
    """Which candidates a mode's log can score.

    A split pass scores exactly one candidate; the combined pass scores all of
    them and each share it produces carries cross-region bias.
    """
    if mode == "instr-all":
        return tuple(rules.CANDIDATES)
    if mode.startswith("instr-"):
        return (mode.split("-", 1)[1],)
    return ()


def cell_context(cell: str, *, batch: int, width: int, model: str,
                 adapted: int, supervised: int | None = None) -> dict:
    """What identifies the workload a pass measured.

    Carried into every decomposition, and checked wherever a numerator from
    one pass is divided by a denominator from another: across modes inside
    this harness, and across harnesses when the floor sweep's samples meet
    this profile's shares. Two passes with different contexts describe
    different steps, and a ratio between them is a number nothing downstream
    could tell was meaningless.

    `tokens` is the token count the projections actually run at, and it is
    carried rather than left to be derived because deriving it is where a
    floor sweep would go wrong. mlx-lm pads a batch to `width` and then
    `default_loss` trains on `batch[:, :-1]`, so every matmul in the step sees
    `width - 1` tokens per row. At the widest band this corpus can fill that
    is 160 rather than 161, and a sweep that measured its floor at the padded
    width would be measuring a shape the step never produces while agreeing
    with the profile about everything else.

    `supervised` is the count of tokens the loss actually trains on, and it is
    in the context because candidate L's floor is defined as the same
    operations "timed on only the supervised rows of the same cell". A floor
    measured against a different supervised count is a floor for a different
    problem, and the credited ratio would still divide.
    """
    return {"cell": cell, "batch": batch, "width": width,
            "tokens": batch * (width - 1), "model": model, "adapted": adapted,
            "supervised": supervised}


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


def median_decomposition(passes: Sequence[Mapping[str, object]]) -> dict:
    """One decomposition standing for several rounds of the same mode.

    Section 3.3 takes both sides of a share as medians over the eligible
    rounds, so this is a ratio of medians rather than a median of ratios: the
    per-region medians go in the numerator and the plain step's median goes in
    the denominator. The result is not a decomposition any single round
    produced, and it is not presented as one.

    Refuses when the rounds disagree about what they measured. Two rounds of
    one mode that ran different contexts, or that reached different regions,
    or that fired a region a different number of times, are not repeats, and a
    median across them describes nothing.
    """
    if not passes:
        raise RunInvalid("a median needs at least one pass")
    context = passes[0]["context"]
    counts = passes[0]["counts"]
    regions = set(passes[0]["totals"])
    for index, one in enumerate(passes[1:], start=1):
        if one["context"] != context:
            raise RunInvalid(
                f"pass {index} ran in {one['context']!r} and pass 0 ran in "
                f"{context!r}: these are not repeats of one measurement")
        if set(one["totals"]) != regions:
            raise RunInvalid(
                f"pass {index} measured {sorted(one['totals'])} and pass 0 "
                f"measured {sorted(regions)}: a region present in one round "
                f"and absent in another cannot be reduced to a median")
        if one["counts"] != counts:
            raise RunInvalid(
                f"pass {index} fired {one['counts']!r} and pass 0 fired "
                f"{counts!r}: the workload changed between rounds")
    return {
        "totals": {region: statistics.median([one["totals"][region]
                                              for one in passes])
                   for region in sorted(regions)},
        "elapsed": statistics.median([one["elapsed"] for one in passes]),
        "covered": statistics.median([one["covered"] for one in passes]),
        "remainder": statistics.median([one["remainder"] for one in passes]),
        "counts": counts,
        "context": context,
        "passes": len(passes),
    }


def cell_reading(record: Mapping[str, object], *,
                 spread_limit_pct: float = machine_state.MAX_SPREAD_PCT
                 ) -> dict:
    """Every derived number for one cell, and what each one rests on.

    The plain pass is the reference arm and its own spread across rounds is
    the detector, exactly as `interleave` uses its canary: interleaving the
    modes equalises a clock excursion across them but cannot detect one, so a
    plain arm that disagreed with itself between rounds describes the machine
    and the cell binds nothing.
    """
    modes = record["modes"]
    plain = list(modes["plain"]["totals_s"])
    plain_total = statistics.median(plain)
    context = record["context"]
    denominator = {"total": plain_total, "context": context}

    reading = {
        "cell": record["cell"],
        "context": context,
        "plain_total_s": plain_total,
        "plain_samples_s": plain,
        "plain_spread_pct": machine_state.spread_pct(plain),
        "spread_limit_pct": spread_limit_pct,
        "rounds_eligible": rules.round_is_eligible(plain, spread_limit_pct),
        "peak_gb": {mode: max(modes[mode]["peak_gb"]) for mode in sorted(modes)},
        "shares": {},
        "instrument_cost": {},
    }

    if "compiled" in modes:
        reading["compile_transfer"] = rules.compile_transfer(
            plain_total, statistics.median(modes["compiled"]["totals_s"]))

    for mode in sorted(modes):
        scored = candidates_from(mode)
        if not scored:
            continue
        combined = median_decomposition(modes[mode]["passes"])
        instrumented_total = statistics.median(modes[mode]["totals_s"])
        reading["instrument_cost"][mode] = rules.instrument_cost(
            instrumented_total, plain_total)
        # How much longer the marked pass ran than the unmarked one. Recorded
        # because it is the size of the perturbation sitting inside this
        # candidate's numerator, and it is NOT subtracted: see the note on
        # `caveat` below for the two attempts that showed it cannot be.
        excess = instrumented_total - plain_total
        split = len(scored) == 1
        # Section 3.3 as Amendment 4 restates it: the regions and the
        # remainder are measured INSIDE the marked step and compared against
        # "that same uncompiled step's own end-to-end time, taken without the
        # interior boundaries" - which is the plain pass, not this one.
        #
        # Compared against this pass's own elapsed it could not fail, because
        # the remainder is defined as whatever the spans leave on the same
        # timeline. Compared against the plain pass, as registered, it is the
        # instrument's cost measured against a 2% limit, so the registered
        # reconciliation and the instrument-cost band are one check with two
        # limits rather than two checks. Both are reported; the 2% is what is
        # registered today and it is the one that binds.
        reading.setdefault("reconciles", {})[mode] = dict(
            rules.reconciles(combined["totals"], combined["remainder"],
                             plain_total),
            same_check_as="instrument_cost, at section 3.3's 2% limit")
        for candidate in scored:
            regions = list(pi.CANDIDATE_REGIONS[candidate])
            marked_total = sum(combined["totals"][name] for name in regions)
            reading["shares"][candidate] = {
                "share": pi.candidate_share(combined, candidate, denominator),
                "from_mode": mode,
                "regions": regions,
                "region_totals_s": {name: combined["totals"][name]
                                    for name in regions},
                "cross_region_bias": mode == "instr-all",
                "marked_total_s": marked_total,
                "instrument_excess_s": excess if split else None,
                "caveat": (
                    "this share carries its own marks: they sit inside the "
                    "spans that form the numerator and outside the plain step "
                    "that forms the denominator, so it is an upper bound on "
                    "the candidate's true share, and the overstatement grows "
                    "with how many marks the candidate's regions carry. No "
                    "correction is applied because none is available: "
                    "subtracting the whole excess produced negative shares "
                    "for two candidates, and a doubled fence measured nothing "
                    "because a second eval of an already-materialised tensor "
                    "is free (both measured 2026-08-20)."),
            }
    return reading


def binding_blockers(record: Mapping[str, object]) -> list[str]:
    """Every reason this recording cannot bind, in plain words.

    Gathered rather than raised, because a run that measured everything and
    then failed one gate is evidence: the samples are worth keeping and the
    reason is worth naming.
    """
    blockers = []
    if not record.get("closing_idle", {}).get("idle"):
        blockers.append("the machine was not idle at the closing gate")
    for cell, reading in sorted(record.get("readings", {}).items()):
        if not reading["rounds_eligible"]:
            blockers.append(
                f"cell {cell}: the plain arm's own spread is "
                f"{reading['plain_spread_pct']:.1f}%, past the "
                f"{reading['spread_limit_pct']:.1f}% limit, so the rounds "
                f"describe the machine rather than the step")
        transfer = reading.get("compile_transfer")
        if transfer is not None and not transfer["ok"]:
            blockers.append(
                f"cell {cell}: the compiled-to-uncompiled ratio is "
                f"{transfer['ratio']:.4f}, outside {transfer['band']}")
        for mode, report in sorted(reading.get("reconciles", {}).items()):
            if not report["ok"]:
                blockers.append(
                    f"cell {cell} {mode}: the marked regions and the "
                    f"remainder account for {report['accounted']:.4f}s against "
                    f"{report['step_total']:.4f}s for the same step without "
                    f"them, a gap of {report['gap_pct']:.1f}% past section "
                    f"3.3's {report['limit_pct']:.1f}% limit")
        for candidate, share in sorted(reading["shares"].items()):
            if not 0.0 < share["share"] < 1.0:
                blockers.append(
                    f"cell {cell}: candidate {candidate}'s registered share is "
                    f"{share['share']:.4f}, which is not a fraction of a step; "
                    f"the marks' own cost sits inside the numerator and "
                    f"outside the denominator, and no correction is registered")
        for mode, cost in sorted(reading["instrument_cost"].items()):
            if cost["ok"] is None:
                blockers.append(
                    f"cell {cell} {mode}: the instrument cost is "
                    f"{cost['ratio']:.4f} and no band is registered for it")
            elif not cost["ok"]:
                blockers.append(
                    f"cell {cell} {mode}: the instrument cost is "
                    f"{cost['ratio']:.4f}, outside {cost['band']}")
    for cell, cell_record in sorted(record.get("cells", {}).items()):
        if not cell_record["completeness"]["ok"]:
            blockers.append(
                f"cell {cell}: region counts differ from the arrangement: "
                f"{cell_record['completeness']['mismatches']}")
        for mode, verdict in sorted(cell_record["identity"]["modes"].items()):
            if not verdict["loss_equal"] or verdict["gradients_differing"]:
                blockers.append(
                    f"cell {cell} {mode}: the marked pass and the plain pass "
                    f"do not compute the same thing")
    return blockers


def recording_path(results_dir: str | Path, day: date, *, kind: str,
                   closing_idle: bool) -> Path:
    """Where a recording lands, with its verdict already in the name."""
    suffix = ".json" if closing_idle else ".REFUSED.json"
    return Path(results_dir) / f"profile-stock-{kind}-{day.isoformat()}{suffix}"


def validate_plan(plan: Mapping[str, object]) -> dict:
    """Freeze the unregistered choices together, before any model is loaded."""
    required = {"schema_version", "data", "seed", "optimizer",
                "optimizer_config", "learning_rate", "memory_ceiling_gb",
                "wall_cap_seconds", "cells"}
    missing = sorted(required - set(plan))
    if missing:
        raise RunInvalid(f"run plan is missing {missing}")
    if plan["schema_version"] != 1:
        raise RunInvalid(f"unknown plan schema {plan['schema_version']!r}")
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
    fixed = dict(plan)
    fixed["cells"] = cells
    fixed["kept_candidates"] = []
    fixed["measurement_module"] = STOCK_MEASUREMENT_MODULE
    fixed["rounds"] = int(plan.get("rounds", ROUNDS))
    if fixed["rounds"] < 1:
        raise RunInvalid("a round count below one measures nothing")
    return fixed


def decide(profile: Mapping[str, object],
           sweep: Mapping[str, object]) -> dict:
    """Section 4.3 applied to two closed recordings, and nothing else.

    The shares come from the profile, the credited ratios from the floor
    sweep, and the two are checked to describe the same workload before either
    divides the other. That check is the whole reason a context travels with
    every measurement: the profile and the sweep are separate runs of separate
    harnesses, and nothing else in the pipeline would notice if one of them
    had measured a different batch.
    """
    for name, record in (("profile", profile), ("sweep", sweep)):
        if not record.get("binding"):
            raise RunInvalid(
                f"the {name} recording does not bind: "
                f"{record.get('binding_blockers')}")
    cell = rules.PRIMARY_CELL
    reading = profile.get("readings", {}).get(cell)
    floors = sweep.get("cells", {}).get(cell)
    if reading is None or floors is None:
        raise RunInvalid(
            f"the ruling is computed at cell {cell} and one of the two "
            f"recordings does not carry it")
    if reading["context"] != floors["context"]:
        raise RunInvalid(
            f"the profile measured {reading['context']!r} and the floor sweep "
            f"measured {floors['context']!r}: a share from one over a ratio "
            f"from the other is a comparison of two workloads")

    missing = sorted(set(rules.CANDIDATES) - set(reading["shares"]))
    if missing:
        raise RunInvalid(
            f"the profile carries no share for {missing}, and the rule ranks "
            f"every registered candidate; a candidate absent from the table "
            f"is not a candidate that scored zero")
    absent = sorted(set(rules.CANDIDATES) - set(floors.get("candidates", {})))
    if absent:
        raise RunInvalid(
            f"the floor sweep carries no credited ratio for {absent}, so "
            f"their gain cannot be computed")
    kill = rules.kill_q(floors["ceiling_ratios"])
    readings, ratios = [], {}
    for candidate in rules.CANDIDATES:
        if candidate == "Q" and kill["killed"]:
            continue
        floor = floors["candidates"][candidate]
        if candidate == "Q":
            # The share counts every quantized linear the step ran, so the
            # floor has to cover every shape the share counted. A floor over a
            # subset would credit Q with a ratio measured on part of the
            # operation and divided into a share measured on all of it.
            registered = set(rules.SHAPES)
            observed = set(floor["per_shape"])
            if observed != registered:
                raise RunInvalid(
                    f"Q's floor covers {sorted(observed)} and the registered "
                    f"shapes are {sorted(registered)}: missing "
                    f"{sorted(registered - observed)}, unregistered "
                    f"{sorted(observed - registered)}")
            collapsed = rules.collapse_ratio_lo(floor["per_shape"])
            ratios[candidate] = collapsed
            credited = collapsed["ratio_lo"]
        else:
            credited = rules.ratio_lo(floor["numerator"], floor["denominator"])
            ratios[candidate] = {"ratio_lo": credited}
        readings.append(rules.Reading(
            candidate=candidate,
            share=reading["shares"][candidate]["share"],
            ratio_lo=credited,
            footprint_delta=floor.get("footprint_delta_bytes")))

    ruling = rules.select_first_operation(readings)
    return {
        "schema_version": 1,
        "cell": cell,
        "ruling": ruling,
        "kill_q": kill,
        "ratios": ratios,
        "shares": reading["shares"],
        "context": reading["context"],
        "assumptions": {c: floors["candidates"][c].get("assumption")
                        for c in rules.CANDIDATES},
        "footprint_delta_source": (
            "the floor arms' own measured peak deltas, which is a statement "
            "about the floor rather than about a kernel that does not exist "
            "yet (Amendment 5)"),
        "gain_is_an_upper_bound": (
            "each floor measures what the operation could become, so every "
            "gain here is an upper bound on that candidate's end-to-end "
            "effect, per section 4.2"),
        "rested_on": {
            "profile_sha256": _json_sha256(profile),
            "sweep_sha256": _json_sha256(sweep),
            "profile_rules_sha256": _file_sha256(
                Path(rules.__file__).resolve()),
        },
        "reported_only": {
            name: other["shares"]
            for name, other in sorted(profile.get("readings", {}).items())
            if name != cell
        },
    }


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


def _load_target(plan: Mapping[str, object], provenance: Mapping[str, object],
                 *, batch: int, mask_prompt: bool):
    """The model, the optimizer and ONE fixed batch, all built mlx-lm's way.

    The batch comes out of mlx-lm's own iterator rather than being assembled
    here, so its width, its padding and its prompt mask are the ones a real
    fine-tune would produce. It is then frozen for the whole cell, which is
    what makes five rounds repeats of one measurement.
    """
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import iterate_batches
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

    train_set, _, _ = load_dataset(
        _dataset_args(plan, provenance["data"]["directory"], mask_prompt),
        tokenizer)
    batches = iterate_batches(dataset=CacheDataset(train_set),
                              batch_size=batch,
                              max_seq_length=rules.SEQ_LEN,
                              loop=False, seed=plan["seed"],
                              comm_group=mx.distributed.init())
    tokens, lengths = next(batches)
    mx.eval(tokens, lengths)

    schedule = plan.get("schedule")
    rate = build_schedule(schedule) if schedule else plan["learning_rate"]
    name = str(plan["optimizer"]).lower()
    classes = {"adam": optim.Adam, "adamw": optim.AdamW, "sgd": optim.SGD}
    if name not in classes:
        raise RunInvalid(f"the profile does not register optimizer {name!r}")
    optimizer = classes[name](learning_rate=rate,
                              **dict(plan["optimizer_config"]))

    return types.SimpleNamespace(
        model=model, optimizer=optimizer, batch=(tokens, lengths),
        depth=len(model.model.layers), adapted=rules.LORA_LAYERS,
        width=int(tokens.shape[1]), rows=int(tokens.shape[0]))


def _build_step(model, optimizer):
    """A mirror of the step mlx-lm's own trainer compiles and runs.

    Mirrored rather than called, because the closure is local to
    `mlx_lm.tuner.trainer.train` and there is no way to reach it from outside;
    what keeps the mirror honest is not care but the pin, since the harness
    refuses to start unless mlx-lm's installed files hash to the recorded
    value, so an upstream edit to that closure stops the run rather than
    quietly changing what "the step" means.

    The accumulation and averaging branches are kept even though this profile
    always updates and never accumulates. They are part of the step being
    measured, and a mirror with the unused half removed is a different
    function that happens to agree today.
    """
    from functools import partial

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.nn.utils import average_gradients
    from mlx.utils import tree_map
    from mlx_lm.tuner.trainer import default_loss

    loss_value_and_grad = nn.value_and_grad(model, default_loss)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)
        if do_update:
            grad = average_gradients(grad)
            optimizer.update(model, grad)
            grad = None
        return lvalue, toks, grad

    return step, state


def _timed_step(step, state, batch, clear_cache_threshold: int):
    """One step, timed exactly where mlx-lm's own loop times it.

    The cache clear is inside the timed region because it is inside mlx-lm's,
    and with the default threshold of zero it runs every single step. It is
    stock's real cost and the denominator has to carry it.
    """
    import mlx.core as mx
    from mlx_lm.tuner.trainer import _clear_cache

    start = time.perf_counter()
    lvalue, toks, grad = step(batch, None, True)
    mx.eval(state, lvalue, toks, grad)
    _clear_cache(clear_cache_threshold)
    return start, time.perf_counter(), lvalue, toks


def timed_mode(step, state, batch, mode: str, *, context: Mapping,
               clear_cache_threshold: int = 0) -> dict:
    """One step in one mode: install, time, decompose, and put the names back.

    Compilation is toggled here rather than once per run because MLX refuses
    an eval inside a compiled step - which is Amendment 4's whole reason for
    existing - while the compiled total that amendment prices has to come from
    the same child on the same batch. Measured 2026-08-20 on the 0.6B model:
    the switch takes fifteen alternations in one process without a fault, the
    marks fire with exactly the expected counts every time, and the loss falls
    monotonically across them, so the step is really training in both states.

    Compilation is left ENABLED on the way out, in every path including a
    raise, because enabled is the state a process starts in and a harness that
    left it off would silently change what runs next.
    """
    import mlx.core as mx

    regions = MODE_REGIONS[mode]
    if mode == "compiled":
        mx.enable_compile()
    else:
        mx.disable_compile()
    recorder = pi.Recorder()
    installation = pi.install(list(regions), recorder) if regions else None
    try:
        start, end, _lvalue, toks = _timed_step(step, state, batch,
                                                clear_cache_threshold)
    finally:
        if installation is not None:
            installation.remove()
        mx.enable_compile()
    return {
        "mode": mode,
        "elapsed_s": end - start,
        "supervised_tokens": int(toks.item()),
        "decomposed": (pi.decompose(recorder.entries, start, end,
                                    context=dict(context))
                       if regions else None),
        "shapes": {region: dict(counts)
                   for region, counts in recorder.shapes.items()},
        "foreign_on_removal": (list(installation.foreign_on_removal)
                               if installation is not None else []),
    }


def _identity(model, batch, modes: Sequence[str]) -> dict:
    """Do the marked passes compute what the plain pass computes?

    Compared on the same weights, before any optimizer update has moved them,
    and over every gradient array rather than a summary, because a summary can
    agree while the arrays beneath it do not. A disagreement here means the
    custom gradient rule changed the computation, and then the marked pass and
    the plain pass are timing two different pieces of work.
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

    once()
    plain_loss, plain_grad = once()
    if not plain_grad:
        raise RunInvalid("the model produced no gradients, so nothing was "
                         "compared and the identity claim is empty")

    report = {"gradients_compared": len(plain_grad), "modes": {}}
    for mode in modes:
        regions = MODE_REGIONS[mode]
        if not regions:
            continue
        recorder = pi.Recorder()
        installation = pi.install(list(regions), recorder)
        try:
            marked_loss, marked_grad = once()
        finally:
            installation.remove()
        report["modes"][mode] = {
            "loss_equal": bool(mx.array_equal(plain_loss, marked_loss)),
            "gradients_differing": sorted(
                name for name in plain_grad
                if not bool(mx.array_equal(plain_grad[name],
                                           marked_grad[name]))),
            "foreign_on_removal": list(installation.foreign_on_removal),
        }
    return report


def _profile_child(task: Mapping[str, object], guard: _ChildGuard) -> dict:
    """One cell: one model, one batch, every mode in rotation."""
    import mlx.core as mx

    plan = task["plan"]
    provenance = task["provenance"]
    cell = task["cell_name"]
    modes = modes_for(cell)
    rounds = int(plan["rounds"])
    registered = rules.CELLS[cell]
    mask_prompt = registered["supervision"] == "masked"

    stock = stock_process()
    guard.check(f"{cell}: load")
    target = _load_target(plan, provenance, batch=registered["batch"],
                          mask_prompt=mask_prompt)

    mx.disable_compile()
    try:
        identity = _identity(target.model, target.batch, modes)
    finally:
        mx.enable_compile()

    step, state = _build_step(target.model, target.optimizer)
    threshold = int(plan.get("clear_cache_threshold", 0))
    records = {mode: {"totals_s": [], "peak_gb": [], "passes": []}
               for mode in modes}

    def run(mode: str, context) -> dict:
        return timed_mode(step, state, target.batch, mode, context=context,
                          clear_cache_threshold=threshold)

    def described(supervised) -> dict:
        return cell_context(
            cell, batch=target.rows, width=target.width, adapted=target.adapted,
            model=Path(provenance["base_model"]["directory"]).name,
            supervised=supervised)

    # Warm every mode once, untimed: the first compiled call pays for
    # compilation and the first of any mode pays for first-touch allocation,
    # and neither is a property of the step. The warm-up is also where the
    # supervised token count comes from, which is why the context the timed
    # rounds carry is built after it rather than before.
    supervised = None
    for mode in modes:
        supervised = run(mode, described(None))["supervised_tokens"]
    context = described(supervised)

    observed_counts = {}
    for round_index in range(rounds):
        guard.check(f"{cell}: round {round_index + 1}/{rounds}")
        for mode in rotated(modes, round_index):
            measured = run(mode, context)
            records[mode]["totals_s"].append(measured["elapsed_s"])
            records[mode]["peak_gb"].append(phys_footprint_gb()[1])
            if measured["decomposed"] is not None:
                records[mode]["passes"].append(measured["decomposed"])
                observed_counts.update(measured["decomposed"]["counts"])
                records[mode].setdefault("shapes", {}).update(
                    measured["shapes"])

    # Expected over the regions the modes were told to mark, NOT over the
    # regions that reported. A seam installed and never reached leaves its
    # region absent from the log entirely, so checking only what appeared
    # would let exactly the failure this check exists for pass silently.
    marked = sorted({region for mode in modes for region in MODE_REGIONS[mode]})
    checked = completeness(
        observed_counts,
        expected_counts(target.depth, target.adapted, marked))

    return {
        "cell": cell,
        "context": context,
        "depth": target.depth,
        "adapted": target.adapted,
        "batch": {"rows": target.rows, "width": target.width,
                  "tokens": target.rows * (target.width - 1),
                  "supervised_tokens": supervised,
                  "mask_prompt": mask_prompt},
        "modes": records,
        "identity": identity,
        "completeness": checked,
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
                results_dir: str | Path = RESULTS_DIR,
                calibrate: bool = False) -> int:
    """Measure every requested cell, then let the closing gate decide."""
    try:
        fixed = validate_plan(plan)
    except RunInvalid as error:
        _print_refusal(error)
        return EXIT_PRECONDITION

    if calibrate:
        fixed = dict(fixed, cells=[rules.PRIMARY_CELL],
                     rounds=int(plan.get("rounds", CALIBRATION_ROUNDS)))
    elif rules.INSTRUMENT_COST_BAND is None:
        _print_refusal(
            "no instrument-cost band is registered, so a binding profile "
            "cannot say whether the marked pass described the same step. Run "
            "--calibrate, then write the band into profile_rules by "
            "Amendment 5, then run this.")
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
        record = {
            "schema_version": 1,
            "kind": "calibration" if calibrate else "binding",
            "plan": fixed,
            "provenance": provenance,
            "machine": machine,
            "opening_idle": opening_idle,
            "closing_idle": closing_idle,
            "cells": cells,
            "readings": {name: cell_reading(one)
                         for name, one in sorted(cells.items())},
        }
        record["binding_blockers"] = binding_blockers(record)
        # A calibration run binds nothing by construction: it exists to
        # measure the cost the band is written from, and a run that set its
        # own limit would be setting it after seeing the number.
        record["binding"] = bool(not calibrate and not record["binding_blockers"])
        path = recording_path(results_dir, runtime.today(),
                              kind="calibration" if calibrate else "binding",
                              closing_idle=bool(closing_idle["idle"]))
        try:
            runtime.write_record(path, record)
        except PreconditionFailed as error:
            _print_refusal(error)
            return EXIT_PRECONDITION
        print(json.dumps({"recording": str(path),
                          "binding": record["binding"],
                          "blockers": record["binding_blockers"]},
                         indent=2), flush=True)
        return 0 if record["binding"] or calibrate else 1
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
    parser.add_argument("--calibrate", action="store_true",
                        help="price the instrument at the primary cell; binds "
                             "nothing")
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
    return run_profile(plan, SystemRuntime(), results_dir=RESULTS_DIR,
                       calibrate=args.calibrate)


if __name__ == "__main__":
    raise SystemExit(main())
