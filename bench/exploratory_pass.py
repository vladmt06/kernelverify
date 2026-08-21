#!/usr/bin/env python3
"""The exploratory pass: three repetitions, and the branch they size.

Amendment 13 clause 57 registers a pass that runs before the calibration and
decides what schedule the calibration takes. Amendment 14 registers how its
two terms are built, because clause 57 named both and constructed neither.

This module holds the RUNNER, which drives three passes and their sweeps, and
the READER, which reduces the six recordings they leave to one branch reading.

What this is NOT
----------------
It is not a decision. Clause 57 registers in its own text that a margin
clearing any multiple of an observed pair contrast is a REPORTED fact about
cost and never a resolution, because the multiplier of 8 rests on a simulated
ideal null while clause 33's demand deliberately assumes nothing about the
null's shape. So nothing here names an operation, nothing here scores a
candidate, and the only thing the branch selects is a SCHEDULE.

It is also not a stage anything downstream may read as evidence. Every
recording it produces is exploratory, which clause 57 gives an unclearable
blocker, and this reader REFUSES a recording that is not.

The two terms, and where they come from
---------------------------------------
The margin is `|K_L - K_Q|` per width, which `profile_stock.exploratory_margins`
computes from one profile recording and one sweep recording. Clause 16's time
conversion equals it exactly at a shared width, which is why no score is
computed anywhere in this path.

The null is the identical-arm pairs. Clause 58 registers that every contrast
the manifest itself produces sits INSIDE one pass, because all four pair arms
are arms of a manifest `profile_knobs.timed_rounds` interleaves inside every
round, so the window-separated draw the branch reads has to be built across
passes. It is the maximum over the four labels, for each of the three pairs of
passes, and the maximum rather than one designated label because clearing the
bar takes the cheaper schedule, so a larger denominator is the conservative
direction.

Clause 59 forms the ratio INSIDE a width. Clause 21 makes the two widths two
contexts and clause 31 says why a difference is only comparable to a null
converted with its own step total, so a long-width margin against a
short-width null is the fault those clauses exist to prevent, in a place no
gate looks.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import profile_rules as rules
import profile_stock as ps
from ceiling_sweep import sweep_path
from decode_rules import RunInvalid

# Amendment 13 clause 57: three passes, and three because a margin read once
# is a margin whose own stability is unknown. Three passes are also what makes
# the window-separated family exactly three, which is the count clause 57's
# branch rule already names.
PASSES = 3

# Clause 57's multiplier, fixed by simulation before the pass ran: three
# window-separated draws at 8x cover the demand 97.2 percent of the time under
# an ideal null. It sizes a schedule and may never substitute for a calibrated
# demand, which is clause 57's own most important sentence.
MULTIPLIER = 8.0

# Clause 57's two branches. Neither names an operation and neither moves the
# registered rate: 19 fresh blocks attain alpha in both, and what the schedule
# moves is the demand's variance.
LEAN = ("the LEAN schedule: a 5-block pilot at two rounds and 19 fresh blocks "
        "at the binding pass's own round count, 43.23 hours")
RULED = ("the schedule is ruled with the measured margin in hand, and that "
         "ruling is written into the pre-registration before the calibration "
         "runs")


def _pair_labels() -> tuple[str, ...]:
    """The four identical-arm labels, from the manifest that builds them."""
    return tuple(f"{ps.PAIR_PREFIX}{index}:{side}"
                 for index in range(ps.EXPLORATORY_PAIRS)
                 for side in ("a", "b"))


def _exploratory(record: Mapping[str, object], what: str) -> Mapping[str, object]:
    """Refuse anything that is not an exploratory recording, and say why.

    A binding recording read through this path would have a cost branch
    applied to numbers that decide an operation, and clause 57 registers that
    the branch decides a schedule and nothing else. The refusal is on the
    recording's own declared kind rather than on its filename, because a name
    is not evidence.
    """
    kind = record.get("kind")
    if kind != "exploratory":
        raise RunInvalid(
            f"the {what} recording declares kind {kind!r} and this reader "
            f"applies clause 57's branch, which sizes a schedule from a pass "
            f"that binds nothing; a recording that binds is read by the "
            f"selection rule and never by this one")
    return record


def pair_medians(record: Mapping[str, object], width: str) -> dict[str, float]:
    """One pass's median for each identical-arm label, at one width.

    Taken from the RAW samples the recording keeps beside its reduction, not
    from the reduced `pair_contrasts_ms`, because a within-pass contrast is
    already a difference and a difference cannot be re-differenced across
    passes without losing which arm moved.
    """
    cells = record.get("cells", {})
    cell = cells.get(rules.PRIMARY_CELL)
    if cell is None:
        raise RunInvalid(
            f"a pass recorded no {rules.PRIMARY_CELL} cell, and the deciding "
            f"cell is the only one that carries a margin")
    measured = cell.get("widths", {}).get(width)
    if measured is None:
        raise RunInvalid(f"a pass recorded no {width!r} width")
    arms = measured.get("arms", {})
    medians = {}
    for label in _pair_labels():
        arm = arms.get(label)
        if arm is None:
            raise RunInvalid(
                f"a pass carries no identical-arm {label!r} at the {width} "
                f"width; clause 57 gives every exploratory pass two pairs and "
                f"a pass missing one produces no null at all")
        samples = list(arm.get("samples_ms", ()))
        if not samples:
            raise RunInvalid(f"identical arm {label!r} recorded no samples")
        medians[label] = statistics.median(samples)
    return medians


def within_pass_contrasts(profiles: Sequence[Mapping[str, object]],
                          width: str) -> list[dict]:
    """Clause 57's six: two pairs inside each of three passes.

    Reported and never the branch's denominator. Clause 57 registers that two
    pairs inside ONE pass share a time window, so they are evidence about leaf
    spread rather than two blocks, and clause 58 registers that all six of
    them sit inside a pass because the manifest interleaves all four arms
    inside every round.
    """
    out = []
    for index, record in enumerate(profiles, start=1):
        readings = (record["cells"][rules.PRIMARY_CELL]
                    .get("readings", {}).get(width, {}))
        contrasts = readings.get("pair_contrasts_ms")
        if not contrasts:
            raise RunInvalid(
                f"pass {index} carries no pair contrasts at the {width} "
                f"width, so its reduction never saw the identical arms")
        for pair, value in sorted(contrasts.items()):
            out.append({"pass": index, "pair": pair, "contrast_ms": value,
                        "separated_by_a_pass": False})
    return out


def window_separated_contrasts(profiles: Sequence[Mapping[str, object]],
                               width: str) -> list[dict]:
    """Amendment 14 clause 58's three, built across passes.

    For each of the three unordered pairs of passes, the MAXIMUM over the four
    identical-arm labels of the absolute difference between that label's
    median in one pass and its median in the other.

    The maximum rather than one designated label, because the branch's
    denominator is what a margin has to clear, clearing it takes the lean
    schedule, and the lean schedule is the cheaper one. Designating one label
    of four would leave the branch movable by which label was designated, and
    that choice would be made after the numbers exist.
    """
    medians = [pair_medians(record, width) for record in profiles]
    out = []
    for left, right in itertools.combinations(range(len(medians)), 2):
        by_label = {label: abs(medians[left][label] - medians[right][label])
                    for label in _pair_labels()}
        out.append({"passes": [left + 1, right + 1],
                    "contrast_ms": max(by_label.values()),
                    "by_label_ms": by_label,
                    "separated_by_a_pass": True})
    return out


def branch_reading(margins: Sequence[Mapping[str, object]],
                   separated: Mapping[str, Sequence[Mapping[str, object]]]
                   ) -> dict:
    """Amendment 14 clause 59's ratio, formed inside a width.

    Per width: that width's margin over the largest of that width's own three
    window-separated contrasts. The branch clears when the SMALLER of the two
    per-width ratios is at least clause 57's multiplier, which is clause 57's
    "the smaller of the two widths' margins" read as the tighter width
    governing.

    A margin the pass could not compute at a width is not a zero and not a
    failure: it means the branch cannot be read at all, and the caller is told
    that rather than handed a number.
    """
    per_width, ratios = {}, []
    for width in rules.WIDTH_ORDER:
        null = max(entry["contrast_ms"] for entry in separated[width])
        readings = [one["widths"][width].get("margin_ms") for one in margins]
        absent = [index for index, value in enumerate(readings, start=1)
                  if value is None]
        row = {"margins_ms": readings,
               "window_separated_null_ms": null,
               "passes_without_a_margin": absent}
        if absent or null <= 0.0:
            row["ratio"] = None
            row["unreadable_because"] = (
                f"passes {absent} produced no margin" if absent else
                "the window-separated null is zero, and a ratio against it "
                "would divide by a machine that never moved")
        else:
            # Amendment 15 clause 62: the smallest of the three. Taking the
            # largest would let one pass that happened to read generously buy
            # the cheaper schedule and a median would let two outvote the one
            # that disagreed, and the smallest is the same direction clause 58
            # takes over the four labels, so every reduction on this branch
            # resolves against clearing it.
            row["margin_read_ms"] = min(readings)
            row["ratio"] = row["margin_read_ms"] / null
            ratios.append(row["ratio"])
        per_width[width] = row

    if len(ratios) != len(rules.WIDTH_ORDER):
        return {"per_width": per_width, "clears": None,
                "multiplier": MULTIPLIER,
                "follows": "neither branch: the pass did not produce a "
                           "readable ratio at both widths, and clause 57's "
                           "branch reads both",
                "binds": False}
    smallest = min(ratios)
    clears = smallest >= MULTIPLIER
    return {"per_width": per_width,
            "smallest_ratio": smallest,
            "multiplier": MULTIPLIER,
            "clears": clears,
            "follows": LEAN if clears else RULED,
            "binds": False,
            "is_not_a_resolution": (
                "clause 57 registers that a margin clearing any multiple of "
                "an observed pair contrast is a REPORTED fact about cost and "
                "never a resolution: the multiplier rests on a simulated "
                "ideal null while clause 33's demand assumes nothing about "
                "the null's shape")}


def aggregate(passes: Sequence[tuple[Mapping[str, object],
                                     Mapping[str, object]]]) -> dict:
    """Every number the three passes leave, reduced to one report.

    Refuses a pass count clause 57 does not register, because the window
    separated family is three exactly when the passes are three, and a branch
    read off two or four draws is a branch read at a coverage nothing
    simulated.
    """
    if len(passes) != PASSES:
        raise RunInvalid(
            f"clause 57 registers {PASSES} passes and this reader was given "
            f"{len(passes)}; the branch reads three window-separated "
            f"contrasts and three passes is what makes exactly three")
    profiles = []
    margins = []
    for index, (profile, sweep) in enumerate(passes, start=1):
        _exploratory(profile, f"pass {index} profile")
        _exploratory(sweep, f"pass {index} sweep")
        profiles.append(profile)
        margins.append(ps.exploratory_margins(profile, sweep))

    separated = {width: window_separated_contrasts(profiles, width)
                 for width in rules.WIDTH_ORDER}
    within = {width: within_pass_contrasts(profiles, width)
              for width in rules.WIDTH_ORDER}
    return {
        "kind": "exploratory",
        "cell": rules.PRIMARY_CELL,
        "passes": PASSES,
        "units": {"time": "ms"},
        "margins": margins,
        "window_separated_contrasts": separated,
        "within_pass_contrasts": within,
        "branch": branch_reading(margins, separated),
        "names_no_operation": (
            "clause 57 registers that this pass dials nothing the binding "
            "manifest does not already dial, computes no score that binds, "
            "and names no candidate"),
    }


# ---------------------------------------------------------------------------
# The runner: three passes, six stages, one process each
# ---------------------------------------------------------------------------
PROFILE = Path(__file__).resolve().parent / "profile_stock.py"
SWEEP = Path(__file__).resolve().parent / "ceiling_sweep.py"


def _subprocess(argv: Sequence[str]) -> int:
    return subprocess.run([sys.executable, *argv], check=False).returncode


def run_passes(*, plan: Path, sweep_plan: Path, results_dir: Path,
               passes: int = PASSES, runner=_subprocess,
               today=None) -> list[dict]:
    """Each stage in its own process, in order, with the earlier ones kept.

    A process per stage rather than one long one, for the reason the profile
    already spawns a child per cell: five hours of a single-GPU machine is the
    cost of a fault that only appears at run time, and a pass that dies half
    way must leave the passes before it readable and the passes after it
    runnable.

    A pass whose recordings already exist is SKIPPED rather than re-run.
    Recordings are created exclusively, so a re-run would refuse at the write
    anyway, after paying for the arms; skipping is that refusal moved to
    before the machine is spent.

    The sweep runs against ITS OWN pass's profile recording, which is what
    binds the bench's supervised counts and its cell to the step the same pass
    measured. Clause 60 registers three sweeps rather than one for the reason
    clause 57 reads the margin three times.
    """
    day = (today or _today)()
    outcomes = []
    for index in range(1, passes + 1):
        profile_out = ps.recording_path(results_dir, day, kind="exploratory",
                                        binding=False, pass_index=index)
        sweep_out = sweep_path(results_dir, day, refused=True,
                               pass_index=index)
        outcome = {"pass": index, "profile": str(profile_out),
                   "sweep": str(sweep_out)}
        if profile_out.exists() and sweep_out.exists():
            outcome["skipped"] = "both recordings already exist"
            outcomes.append(outcome)
            continue
        if not profile_out.exists():
            outcome["profile_exit"] = runner(
                [str(PROFILE), "--plan", str(plan),
                 "--pass-index", str(index)])
        if not profile_out.exists():
            outcome["sweep_exit"] = None
            outcome["stopped"] = (
                "the profile left no recording, so the sweep has no cell to "
                "measure the same workload against")
            outcomes.append(outcome)
            continue
        outcome["sweep_exit"] = runner(
            [str(SWEEP), "--plan", str(sweep_plan),
             "--profile", str(profile_out), "--pass-index", str(index)])
        outcomes.append(outcome)
    return outcomes


def _today():
    from datetime import date

    return date.today()


def _load(path: Path) -> dict:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RunInvalid(f"cannot read {path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise RunInvalid(f"{path} is not a JSON object")
    return dict(raw)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path,
                        help="the exploratory profile plan")
    parser.add_argument("--sweep-plan", type=Path,
                        help="the exploratory sweep plan")
    parser.add_argument("--results", type=Path, default=ps.RESULTS_DIR)
    parser.add_argument("--read", action="store_true",
                        help="reduce recordings that already exist")
    parser.add_argument("--pass-recording", action="append", nargs=2,
                        metavar=("PROFILE", "SWEEP"), default=[],
                        help="one pass's two recordings; give it three times")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.read:
        try:
            passes = [(_load(Path(profile)), _load(Path(sweep)))
                      for profile, sweep in args.pass_recording]
            report = aggregate(passes)
        except RunInvalid as error:
            print(f"REFUSED: {error}")
            return ps.EXIT_PRECONDITION
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.out is not None:
            args.out.write_text(text + "\n")
        print(text)
        return 0

    if args.plan is None or args.sweep_plan is None:
        print("REFUSED: --plan and --sweep-plan are both required to run")
        return ps.EXIT_PRECONDITION
    outcomes = run_passes(plan=args.plan, sweep_plan=args.sweep_plan,
                          results_dir=args.results)
    print(json.dumps({"passes": outcomes}, indent=2))
    ran = [one for one in outcomes if "skipped" not in one]
    return 0 if all(one.get("sweep_exit") == 0 for one in ran) else 1


if __name__ == "__main__":
    raise SystemExit(main())
