"""Measure QLoRA training under the sprint-1 pre-registration.

The pure functions at the top of this module own every scientific reading.
The process driver below them only gathers records and may never turn a
missing check into a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import types
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import machine_state
from machine_state import MeasurementLock
from memory_guard import (
    BudgetGuard,
    BudgetExceeded,
    EXIT_BUDGET_REFUSAL,
    EXIT_CHILD_DEATH,
    EXIT_LOCK_HELD,
    EXIT_LOW_MEMORY,
    EXIT_NO_DEVICE,
    EXIT_NOT_IDLE,
    EXIT_ORPHANED,
    EXIT_PRECONDITION,
    LowMemoryRefusal,
    Orphaned,
    phys_footprint_gb,
    require_available_memory,
)


ARMS = ("ours", "stock", "control")
# Amendment 3. A comparison whose REFERENCE arm's samples spread wider than
# this describes the machine, not the arms, so it claims no direction and its
# cell does not bind. Inherited from the decode lane's kernel arms and NOT
# calibrated for full-job wall time, which is a far steadier quantity: this is
# a loose upper bound that catches gross excursions, to be tightened once the
# Day 1 profile has measured what round-to-round variation is normal.
MAX_REFERENCE_SPREAD = 1.5
ROUNDS = 5
MLX_VERSION = "0.32.0"
MLX_LM_VERSION = "0.31.3"
ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "qwen3-4b-4bit-g64"
MODEL_DIR = ROOT / "bench" / ".models" / MODEL_NAME
MODEL_FILE = MODEL_DIR / "model.safetensors"
PIN_MANIFEST = MODEL_DIR.parent / "PINNED-HASHES.txt"
RESULTS_DIR = ROOT / "bench" / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Cell:
    name: str
    batch_size: int
    mask_prompt: bool


CELLS = (
    Cell("A", 1, True),
    Cell("B", 4, True),
    Cell("C", 1, False),
    Cell("D", 4, False),
)
CELL_BY_NAME = {cell.name: cell for cell in CELLS}


class RunInvalid(RuntimeError):
    """The records cannot support the registered comparison."""


class PreconditionFailed(RuntimeError):
    """A permanent prerequisite failed before training could be measured."""


class ChildRefusal(RuntimeError):
    """A fresh arm process stopped with a shared refusal code."""

    def __init__(self, cell: str, returncode: int, reason: str):
        super().__init__(f"{cell}: {reason}")
        self.cell = cell
        self.returncode = returncode
        self.reason = reason


class NoDevice(RuntimeError):
    """No usable Metal device exists in the measurement child."""


def rotated_arms(round_index: int) -> tuple[str, ...]:
    """Rotate the first slot while preserving the registered arm order."""
    offset = round_index % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def _deduplicated(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _record_reasons(cell: Cell, round_number: int, arm: str,
                    record: Mapping[str, object]) -> list[str]:
    reasons = []
    prefix = f"{cell.name} round {round_number} {arm}"
    if record.get("cell") != cell.name:
        reasons.append(f"{prefix}: wrong cell {record.get('cell')!r}")
    if record.get("round") != round_number:
        reasons.append(f"{prefix}: wrong round {record.get('round')!r}")
    if record.get("arm") != arm:
        reasons.append(f"{prefix}: wrong arm {record.get('arm')!r}")
    if record.get("completed") is not True:
        reasons.append(f"{prefix}: completed is not true")
    if record.get("adapter_written") is not True:
        reasons.append(f"{prefix}: adapter_written is not true")

    fairness = record.get("fairness")
    if not isinstance(fairness, Mapping):
        reasons.append(f"{prefix}: fairness record is missing")
        return reasons
    base_model = fairness.get("base_model")
    if (not isinstance(base_model, Mapping)
            or not base_model.get("path") or not base_model.get("sha256")):
        reasons.append(f"{prefix}: base model path or sha256 is missing")
    if not fairness.get("adapter_init_sha256"):
        reasons.append(f"{prefix}: adapter initialisation hash is missing")
    steps = None
    optimization = fairness.get("optimization")
    if not isinstance(optimization, Mapping):
        reasons.append(f"{prefix}: optimization record is missing")
    else:
        expected = {
            "batch_size": cell.batch_size,
            "mask_prompt": cell.mask_prompt,
            "max_seq_length": 2048,
            "lora_rank": 8,
            "num_layers": 16,
        }
        wrong = {
            key: optimization.get(key)
            for key, value in expected.items()
            if optimization.get(key) != value
        }
        if wrong:
            reasons.append(
                f"{prefix}: configuration does not match cell {cell.name}: "
                f"{wrong}"
            )
        steps = optimization.get("steps")
        warmup_steps = optimization.get("warmup_steps")
        data = fairness.get("data")
        batches = data.get("batches") if isinstance(data, Mapping) else None
        if not isinstance(data, Mapping) or not data.get("source_sha256"):
            reasons.append(f"{prefix}: data source hash is missing")
        if not isinstance(steps, int) or steps < 2:
            reasons.append(f"{prefix}: fixed step count must be at least 2")
        elif (not isinstance(warmup_steps, int)
              or isinstance(warmup_steps, bool)
              or not 1 <= warmup_steps < steps):
            reasons.append(f"{prefix}: warmup step count is invalid")
        elif not isinstance(batches, Sequence) or len(batches) != steps:
            reasons.append(
                f"{prefix}: data trace does not contain one entry per step"
            )
        else:
            for expected_step, batch in enumerate(batches, 1):
                if (not isinstance(batch, Mapping)
                        or batch.get("step") != expected_step
                        or any(not batch.get(field) for field in (
                            "tokens_sha256", "lengths_sha256", "mask_sha256"
                        ))
                        or not isinstance(batch.get("supervised_tokens"), int)
                        or isinstance(batch.get("supervised_tokens"), bool)
                        or batch["supervised_tokens"] <= 0):
                    reasons.append(
                        f"{prefix}: data trace step {expected_step} is incomplete"
                    )
                    break

    ceiling = fairness.get("memory_ceiling_gb")
    if not _finite_positive(ceiling):
        reasons.append(f"{prefix}: declared memory ceiling is invalid")

    stack = fairness.get("stack")
    expected_versions = {"mlx": MLX_VERSION, "mlx_lm": MLX_LM_VERSION}
    if not isinstance(stack, Mapping):
        reasons.append(f"{prefix}: verified stack record is missing")
    else:
        for package, expected_version in expected_versions.items():
            package_record = stack.get(package)
            if (not isinstance(package_record, Mapping)
                    or package_record.get("version") != expected_version
                    or not package_record.get("package_sha256")):
                reasons.append(
                    f"{prefix}: verified stack does not pin {package} "
                    f"{expected_version} by hash"
                )

    for field in ("full_job_wall_s", "peak_footprint_gb"):
        value = record.get(field)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            reasons.append(f"{prefix}: {field} must be positive and finite")
    peak = record.get("peak_footprint_gb")
    if (_finite_positive(peak) and _finite_positive(ceiling)
            and peak > ceiling):
        reasons.append(f"{prefix}: peak exceeded the declared memory ceiling")
    step_seconds = record.get("step_seconds")
    if (not isinstance(step_seconds, Sequence) or len(step_seconds) != steps
            or any(not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0
                   for value in step_seconds)):
        reasons.append(
            f"{prefix}: step_seconds must contain one sample per step"
        )
    curve = record.get("loss_curve")
    if (not isinstance(curve, Sequence) or not curve
            or len(curve) != steps
            or any(not isinstance(point, Mapping)
                   or point.get("step") != expected_step
                   or not isinstance(point.get("loss"), (int, float))
                   or isinstance(point.get("loss"), bool)
                   or not math.isfinite(point["loss"])
                   for expected_step, point in enumerate(curve, 1))):
        reasons.append(f"{prefix}: loss curve must contain one point per step")
    return reasons


def _routing_reasons(records: Mapping[str, Mapping[str, object]]) -> list[str]:
    reasons = []
    routing = {
        arm: records[arm].get("routing")
        for arm in ARMS
    }
    if any(not isinstance(value, Mapping) for value in routing.values()):
        return ["one or more arms did not report routing evidence"]

    ours = routing["ours"]
    stock = routing["stock"]
    control = routing["control"]
    if ours.get("wrapper_installed") is not True:
        reasons.append("ours wrapper was not installed")
    if not ours.get("wrapper_sha256") or not control.get("wrapper_sha256"):
        reasons.append("wrapper hash is missing")
    if ours.get("forced_stock") is not False:
        reasons.append("ours was forced to stock")
    ours_calls = ours.get("routed_calls")
    ours_decisions = ours.get("routing_decisions")
    if (not isinstance(ours.get("routed_candidates"), list)
            or not ours.get("routed_candidates")
            or not isinstance(ours_calls, int) or isinstance(ours_calls, bool)
            or ours_calls <= 0):
        reasons.append("ours routed no verified call")
    if (not isinstance(ours_decisions, int)
            or isinstance(ours_decisions, bool) or ours_decisions <= 0):
        reasons.append("ours observed no routing decision")
    if (stock.get("wrapper_installed") is not False
            or stock.get("forced_stock") is not False
            or stock.get("routed_candidates")
            or stock.get("routed_calls", 0) != 0
            or stock.get("routing_decisions", 0) != 0):
        reasons.append("stock was not untouched")
    if control.get("wrapper_installed") is not True:
        reasons.append("control wrapper was not installed")
    if control.get("forced_stock") is not True:
        reasons.append("control did not force stock")
    if (not isinstance(control.get("routing_decisions"), int)
            or isinstance(control.get("routing_decisions"), bool)
            or control.get("routing_decisions") <= 0):
        reasons.append("control observed no routing decision")
    if control.get("routed_candidates") or control.get("routed_calls", 0) != 0:
        reasons.append("control routed a kernel")
    if ours.get("wrapper_sha256") != control.get("wrapper_sha256"):
        reasons.append("ours and control installed different wrappers")
    return reasons


def round_validity(cell_name: str, round_number: int,
                   records: Mapping[str, Mapping[str, object]], *,
                   expected: Mapping[str, object] | None = None,
                   kept_candidates: Sequence[str] | None = None) -> dict:
    """Check all six fairness conditions and the three observed arm identities."""
    if cell_name not in CELL_BY_NAME:
        return {"valid": False, "reasons": [f"unknown cell {cell_name!r}"]}
    got = [arm for arm in ARMS if arm in records]
    extras = [arm for arm in records if arm not in ARMS]
    if got != list(ARMS) or extras:
        return {
            "valid": False,
            "reasons": [f"expected arms {list(ARMS)}, got {got + extras}"],
        }

    cell = CELL_BY_NAME[cell_name]
    reasons = []
    for arm in ARMS:
        reasons.extend(_record_reasons(cell, round_number, arm, records[arm]))

    if expected is not None:
        for arm in ARMS:
            fairness = records[arm].get("fairness", {})
            actual = {
                "base_model": fairness.get("base_model"),
                "data_source_sha256": fairness.get("data", {}).get(
                    "source_sha256"
                ),
                "optimization": fairness.get("optimization"),
                "memory_ceiling_gb": fairness.get("memory_ceiling_gb"),
                "stack": fairness.get("stack"),
            }
            if actual != expected:
                reasons.append(
                    f"{cell_name} round {round_number} {arm}: "
                    "fairness evidence differs from the frozen plan"
                )
    if kept_candidates is not None:
        routed = records["ours"].get("routing", {}).get(
            "routed_candidates", []
        )
        unkept = sorted(set(routed) - set(kept_candidates))
        if unkept:
            reasons.append(f"ours routed unkept candidates: {unkept}")

    reference = records["stock"].get("fairness")
    for field in (
        "base_model",
        "adapter_init_sha256",
        "data",
        "optimization",
        "memory_ceiling_gb",
        "stack",
    ):
        values = [records[arm].get("fairness", {}).get(field) for arm in ARMS]
        if any(value != values[0] for value in values[1:]):
            reasons.append(f"fairness mismatch: {field}")
    if not isinstance(reference, Mapping):
        reasons.append("stock fairness record is missing")
    reasons.extend(_routing_reasons(records))
    reasons = _deduplicated(reasons)
    return {"valid": not reasons, "reasons": reasons}


def _positive_samples(samples: Sequence[float], name: str) -> list[float]:
    values = list(samples)
    if not values:
        raise RunInvalid(f"{name} has no sample")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value <= 0 for value in values):
        raise RunInvalid(f"{name} samples must be positive and finite")
    return values


def _spread_pct(samples: Sequence[float]) -> float:
    return (max(samples) - min(samples)) / statistics.median(samples) * 100


def time_comparison(first: Sequence[float], second: Sequence[float]) -> dict:
    """The interval verdict for elapsed-time samples, floor rule included.

    The endpoint formulas are the pricing probe's, kept here rather than
    imported because importing that probe pulls in MLX, and these decision
    rules must stay testable on a machine with no device.

    Two things beyond the probe's endpoints, both from amendment 3, and both
    living HERE rather than in the outcomes so that one rule serves every
    outcome instead of each one deciding again:

    REJECTED when the reference arm - the second of the two - spreads wider
    than the class limit. Interleaving would equalise a clock excursion across
    arms but cannot detect one, and these arms are not even interleaved: they
    are separate processes in rotated slots. The reference arm's own spread is
    the only registered signal that the machine moved, and an excursion that
    lifts the reference arm's minimum moves the ratio faster than it widens
    the floor, so the floor alone does not see it.

    And a direction is claimed only when the delta also clears its own noise
    floor. Section 8.3 states that rule for every comparison; an outcome that
    read the endpoints directly would be deciding it a second time, which is
    how O1 and O3 came to disagree about the same samples.
    """
    first_values = _positive_samples(first, "first arm")
    second_values = _positive_samples(second, "second arm")
    median_first = statistics.median(first_values)
    median_second = statistics.median(second_values)
    ratio = median_second / median_first
    ratio_lo = min(second_values) / max(first_values)
    ratio_hi = max(second_values) / min(first_values)
    spread_first = _spread_pct(first_values)
    spread_second = _spread_pct(second_values)
    floor = max(spread_first, spread_second)
    delta_pct = 100 * (ratio - 1)
    reference_spread = max(second_values) / min(second_values)
    if reference_spread > MAX_REFERENCE_SPREAD:
        verdict = "REJECTED"
    elif ratio_lo > 1.0 and clears_floor(abs(delta_pct), floor):
        verdict = "WIN"
    elif ratio_hi < 1.0 and clears_floor(abs(delta_pct), floor):
        verdict = "LOSS"
    else:
        verdict = "REFUSED"
    return {
        "median_first_s": median_first,
        "median_second_s": median_second,
        "ratio": ratio,
        "ratio_lo": ratio_lo,
        "ratio_hi": ratio_hi,
        "delta_pct": delta_pct,
        "spread_first_pct": spread_first,
        "spread_second_pct": spread_second,
        "noise_floor_pct": floor,
        "reference_spread": reference_spread,
        "clears_noise_floor": clears_floor(abs(delta_pct), floor),
        "verdict": verdict,
        "samples_first_s": first_values,
        "samples_second_s": second_values,
    }


def clears_floor(delta_pct: float, floor_pct: float) -> bool:
    """Equality is noise, not evidence."""
    return delta_pct > floor_pct


def _curve_values(record: Mapping[str, object]) -> tuple[tuple[int, float], ...]:
    curve = record.get("loss_curve")
    if not isinstance(curve, Sequence) or not curve:
        raise RunInvalid("loss curve is missing")
    values = []
    for point in curve:
        if (not isinstance(point, Mapping)
                or not isinstance(point.get("step"), int)
                or not isinstance(point.get("loss"), (int, float))
                or not math.isfinite(point["loss"])):
            raise RunInvalid("loss curve contains an invalid point")
        values.append((point["step"], point["loss"]))
    return tuple(values)


def loss_curve_gate(rounds: Sequence[Mapping[str, Mapping[str, object]]]) -> dict:
    """Keep each arm inside the per-step range stock reproduced across rounds."""
    try:
        stock_curves = [_curve_values(records["stock"]) for records in rounds]
        expected_steps = tuple(step for step, _ in stock_curves[0])
        for records in rounds:
            for arm in ARMS:
                steps = tuple(step for step, _ in _curve_values(records[arm]))
                if steps != expected_steps:
                    return {
                        "valid": False,
                        "reasons": [
                            f"loss steps for {arm} do not match stock"
                        ],
                        "divergences": [],
                    }
    except (KeyError, IndexError, RunInvalid) as error:
        return {"valid": False, "reasons": [str(error)], "divergences": []}

    stock_ranges = []
    for position, step in enumerate(expected_steps):
        values = [curve[position][1] for curve in stock_curves]
        stock_ranges.append((step, min(values), max(values)))

    divergences = []
    for arm in ("ours", "control"):
        for records in rounds:
            round_number = records[arm]["round"]
            curve = _curve_values(records[arm])
            for position, (step, loss) in enumerate(curve):
                _, low, high = stock_ranges[position]
                if loss < low or loss > high:
                    divergences.append({
                        "arm": arm,
                        "round": round_number,
                        "step": step,
                        "loss": loss,
                        "stock_range": [low, high],
                    })
    return {
        "valid": not divergences,
        "reasons": [] if not divergences else [
            "one or more loss points exceeded stock's same-seed range"
        ],
        "divergences": divergences,
    }


def _loss_summary(rounds: Sequence[Mapping[str, Mapping[str, object]]],
                  arm: str) -> dict:
    curves = [_curve_values(records[arm]) for records in rounds]
    steps = [step for step, _ in curves[0]]
    median_curve = []
    for position, step in enumerate(steps):
        values = [curve[position][1] for curve in curves]
        median_curve.append({
            "step": step,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        })
    return {
        "steps": len(steps),
        "first_median": median_curve[0]["median"],
        "last_median": median_curve[-1]["median"],
        "minimum": min(point["min"] for point in median_curve),
        "maximum": max(point["max"] for point in median_curve),
        "median_curve": median_curve,
    }


def arm_metrics(rounds: Sequence[Mapping[str, Mapping[str, object]]],
                arm: str) -> dict:
    """Primary uses eligible rounds; the secondary also excludes round 1."""
    if arm not in ARMS:
        raise RunInvalid(f"unknown arm {arm!r}")
    if len(rounds) < 2:
        raise RunInvalid("at least two eligible rounds are required")
    full = [records[arm]["full_job_wall_s"] for records in rounds]
    warmup_steps = rounds[0][arm]["fairness"]["optimization"][
        "warmup_steps"
    ]
    warmed = [
        statistics.median(records[arm]["step_seconds"][warmup_steps:])
        for records in rounds if records[arm]["round"] > 1
    ]
    if len(warmed) < 2:
        # Section 8.3 reads the secondary metric as medians over roundS after
        # the first, and takes the noise floor from a spread over rounds. One
        # sample has a spread of zero, so a single warmed round would hand the
        # comparison a zero-width interval and a 0.0% floor, and any direction
        # at all would then read as a WIN that cleared its floor.
        raise RunInvalid(
            "at least two eligible rounds after the first are required "
            "for the warmed comparison")
    peaks = [records[arm]["peak_footprint_gb"] for records in rounds]
    return {
        "full_job_samples_s": full,
        "full_job_median_s": statistics.median(full),
        "full_job_spread_pct": _spread_pct(full),
        "warmed_step_samples_s": warmed,
        "warmed_step_median_s": statistics.median(warmed),
        "warmed_step_spread_pct": _spread_pct(warmed),
        "peak_footprint_gb": max(peaks),
        "loss_curve": _loss_summary(rounds, arm),
    }


def _comparison_or_void(cell: Mapping[str, object], key: str) -> dict:
    """The named comparison, or the void reading a cell that did not bind has.

    A non-binding cell has no eligible samples, so there is nothing to compare
    and the honest reading is REFUSED: undecided, which is what this
    repository's interval vocabulary already calls a comparison whose
    direction cannot be claimed. It is NOT a LOSS, because nothing was
    measured, and it must not be an exception either: section 9 requires O4 at
    every cell and O5 whatever the kernel did, so one void cell may not take
    the other outcomes down with it.
    """
    comparison = cell.get(key)
    if isinstance(comparison, Mapping):
        return dict(comparison)
    return {
        "verdict": "REFUSED",
        "void": True,
        "reason": cell.get("verdict") or "cell did not bind",
    }


def decide_o1(comparison: Mapping[str, object], *, ours_peak_gb: float,
              stock_peak_gb: float) -> dict:
    if comparison.get("void"):
        # Section 9: GO requires a measured interval clear of 1.10. Anything
        # else is NO-GO, and an unmeasured cell is the plainest anything else.
        return {**comparison, "memory_no_worse": None,
                "clears_noise_floor": False,
                "ours_peak_footprint_gb": ours_peak_gb,
                "stock_peak_footprint_gb": stock_peak_gb,
                "verdict": "NO-GO"}
    memory_no_worse = ours_peak_gb <= stock_peak_gb
    return {
        **comparison,
        "memory_no_worse": memory_no_worse,
        "ours_peak_footprint_gb": ours_peak_gb,
        "stock_peak_footprint_gb": stock_peak_gb,
        # Section 9's two conjuncts, on a comparison whose verdict already
        # carries amendment 3's floor and reference-spread rules. Requiring
        # the WIN is what makes those rules reach the shipping question: a
        # decider reading the endpoints alone would ship a claim the
        # comparison had already declined to make.
        "verdict": (
            "GO"
            if (comparison["verdict"] == "WIN"
                and comparison["ratio_lo"] > 1.10
                and memory_no_worse)
            else "NO-GO"
        ),
    }


def decide_o2(comparison: Mapping[str, object]) -> dict:
    if comparison.get("void"):
        return {**comparison, "clears_noise_floor": False,
                "verdict": "BELOW-TARGET"}
    return {
        **comparison,
        "verdict": (
            "AT-TARGET"
            if (comparison["verdict"] == "WIN"
                and comparison["ratio_lo"] >= 1.30)
            else "BELOW-TARGET"
        ),
    }


def decide_o3(comparison: Mapping[str, object]) -> dict:
    if comparison.get("void"):
        return {**comparison, "interval_verdict": "REFUSED",
                "clears_noise_floor": False, "verdict": "REFUSED"}
    # The floor is already in the verdict, so O3 reports it rather than
    # applying a second rule of its own. That is what "the same floor rule"
    # as O1 now means: there is one rule, and it lives upstream of both.
    return {
        **comparison,
        "interval_verdict": comparison["verdict"],
        "verdict": comparison["verdict"],
    }


def decide_o4(comparisons: Mapping[str, Mapping[str, object]]) -> dict:
    result = {}
    for cell in CELLS:
        comparison = comparisons[cell.name]
        if comparison.get("void"):
            # No samples, so no wrapper cost to read. An unmeasured wrapper
            # cost is not a finding against the interface; claiming one would
            # be reading a verdict out of an absence.
            result[cell.name] = {**comparison, "wrapper_slowdown_pct": None,
                                 "interface_finding": False}
            continue
        # Two different quantities, and only one of them is registered.
        # `wrapper_slowdown_pct` is how much longer the wrapper takes, which
        # is what a reader wants to see. The floor comparison must use how far
        # the interval sits BELOW 1.0, which is 100 * (1 - ratio) and is
        # exactly abs(delta_pct), the quantity O1, O2 and O3 all pass to
        # clears_floor. The two differ, and 1/ratio - 1 is always the larger,
        # so using it here reports a finding against the interface on a
        # comparison the registered floor rule calls noise.
        slowdown = 100 * (1 / comparison["ratio"] - 1)
        result[cell.name] = {
            **comparison,
            # How much longer the wrapper takes, which is what a reader wants
            # to see, beside how far the interval sits below 1.0, which is
            # what section 9 registers. They are different numbers and the
            # first is always the larger, so only the second may gate.
            "wrapper_slowdown_pct": slowdown,
            "wrapper_below_one_pct": abs(comparison["delta_pct"]),
            "interface_finding": comparison["verdict"] == "LOSS",
        }
    return result


def validate_loop_outcome(record: Mapping[str, object]) -> dict:
    """O5 is accepted only as a complete, internally consistent census."""
    required = {
        "generated", "died_by_stage", "priced", "kept", "wall_seconds"
    }
    missing = required - set(record)
    if missing:
        raise RunInvalid(f"O5 is missing {sorted(missing)}")
    counts = {name: record[name] for name in ("generated", "priced", "kept")}
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in counts.values()):
        raise RunInvalid("O5 counts must be non-negative integers")
    deaths = record["died_by_stage"]
    if (not isinstance(deaths, Mapping)
            or any(not isinstance(name, str) or not name
                   or not isinstance(value, int) or isinstance(value, bool)
                   or value < 0 for name, value in deaths.items())):
        raise RunInvalid("O5 stage deaths must be named non-negative counts")
    wall = record["wall_seconds"]
    if (not isinstance(wall, (int, float)) or isinstance(wall, bool)
            or not math.isfinite(wall) or wall < 0):
        raise RunInvalid("O5 wall_seconds must be non-negative and finite")
    if counts["kept"] > counts["priced"] or counts["priced"] > counts["generated"]:
        raise RunInvalid("O5 must satisfy kept <= priced <= generated")
    if sum(deaths.values()) + counts["kept"] != counts["generated"]:
        raise RunInvalid("O5 deaths plus kept must account for every candidate")
    return {
        "generated": counts["generated"],
        "died_by_stage": dict(deaths),
        "priced": counts["priced"],
        "kept": counts["kept"],
        "wall_seconds": wall,
    }


def pre_registered_outcomes(cells: Mapping[str, Mapping[str, object]],
                            loop_outcome: Mapping[str, object]) -> dict:
    """Read O4 first, then O1-O3 and O5 in the registered order."""
    missing = [cell.name for cell in CELLS if cell.name not in cells]
    if missing:
        raise RunInvalid(f"outcomes are missing cells {missing}")
    o4 = decide_o4({
        cell.name: _comparison_or_void(cells[cell.name], "control_vs_stock")
        for cell in CELLS
    })
    primary = cells["B"]
    peaks = primary.get("peaks_gb") or {}
    o1 = decide_o1(
        _comparison_or_void(primary, "ours_vs_stock"),
        ours_peak_gb=peaks.get("ours"),
        stock_peak_gb=peaks.get("stock"),
    )
    o2 = decide_o2(_comparison_or_void(primary, "ours_vs_stock"))
    o3 = decide_o3(_comparison_or_void(cells["C"], "ours_vs_stock"))
    scope = (
        "prompt-masked instruction tuning"
        if o1["verdict"] == "GO" and o3["verdict"] == "REFUSED"
        else None
    )
    return {
        "O4": o4,
        "O1": o1,
        "O2": o2,
        "O3": o3,
        "O5": validate_loop_outcome(loop_outcome),
        "claim_scope": scope,
    }


def equal_memory_row(*, ours: Mapping[str, object],
                     stock: Mapping[str, object]) -> dict:
    """A different batch is a throughput comparison and nothing else."""
    if ours["batch_size"] <= stock["batch_size"]:
        raise RunInvalid("ours must use a larger batch for an equal-memory row")
    if ours["memory_ceiling_gb"] != stock["memory_ceiling_gb"]:
        raise RunInvalid("equal-memory arms declared different ceilings")
    ceiling = ours["memory_ceiling_gb"]
    if (ours["peak_footprint_gb"] > ceiling
            or stock["peak_footprint_gb"] > ceiling):
        raise RunInvalid("an equal-memory arm exceeded the declared ceiling")
    ours_rate = ours["supervised_tokens"] / ours["wall_seconds"]
    stock_rate = stock["supervised_tokens"] / stock["wall_seconds"]
    return {
        "label": "throughput at equal memory",
        "same_batch_speed": False,
        "memory_ceiling_gb": ceiling,
        "ours_batch_size": ours["batch_size"],
        "stock_batch_size": stock["batch_size"],
        "ours_tokens_per_s": ours_rate,
        "stock_tokens_per_s": stock_rate,
        "ratio": ours_rate / stock_rate,
    }


def child_exit_for_parent(returncode: int) -> int:
    """Preserve known shared refusals and classify every other failure."""
    if returncode == 0:
        return 0
    shared = {
        EXIT_BUDGET_REFUSAL,
        EXIT_LOW_MEMORY,
        EXIT_NO_DEVICE,
        EXIT_PRECONDITION,
        EXIT_ORPHANED,
    }
    return returncode if returncode in shared else EXIT_CHILD_DEATH


def recording_path(results_dir: str | Path, day: date,
                   *, closing_idle: bool) -> Path:
    suffix = ".json" if closing_idle else ".REFUSED.json"
    return Path(results_dir) / f"train-lora-e2e-{day.isoformat()}{suffix}"


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def validate_plan(plan: Mapping[str, object]) -> dict:
    """Unregistered choices must be frozen together before model work starts."""
    required = {
        "schema_version",
        "data",
        "steps",
        "warmup_steps",
        "optimizer",
        "optimizer_config",
        "learning_rate",
        "schedule",
        "seed",
        "memory_ceiling_gb",
        "wall_cap_seconds",
        "measurement_module",
        "kept_candidates",
        "loop_outcome",
        "equal_memory",
    }
    if set(plan) != required:
        missing = sorted(required - set(plan))
        extras = sorted(set(plan) - required)
        raise RunInvalid(f"run plan keys differ: missing={missing}, extra={extras}")
    if plan["schema_version"] != 1:
        raise RunInvalid("run plan schema_version must be 1")
    if not isinstance(plan["data"], str) or not plan["data"]:
        raise RunInvalid("run plan data path is missing")
    if (not isinstance(plan["steps"], int) or isinstance(plan["steps"], bool)
            or plan["steps"] < 2):
        raise RunInvalid("run plan needs at least two fixed steps")
    if (not isinstance(plan["warmup_steps"], int)
            or isinstance(plan["warmup_steps"], bool)
            or not 1 <= plan["warmup_steps"] < plan["steps"]):
        raise RunInvalid("run plan warmup_steps must be inside the fixed steps")
    if not isinstance(plan["optimizer"], str) or not plan["optimizer"]:
        raise RunInvalid("run plan optimizer is missing")
    if not isinstance(plan["optimizer_config"], Mapping):
        raise RunInvalid("run plan optimizer_config must be an object")
    if not _finite_positive(plan["learning_rate"]):
        raise RunInvalid("run plan learning_rate must be positive and finite")
    if (not isinstance(plan["seed"], int) or isinstance(plan["seed"], bool)
            or plan["seed"] < 0):
        raise RunInvalid("run plan seed must be a non-negative integer")
    for field in ("memory_ceiling_gb", "wall_cap_seconds"):
        if not _finite_positive(plan[field]):
            raise RunInvalid(f"run plan {field} must be positive and finite")
    if (not isinstance(plan["measurement_module"], str)
            or not plan["measurement_module"]):
        raise RunInvalid("run plan measurement_module is missing")
    candidates = plan["kept_candidates"]
    if (not isinstance(candidates, list)
            or any(not isinstance(value, str) or not value
                   for value in candidates)
            or len(set(candidates)) != len(candidates)):
        raise RunInvalid("run plan kept_candidates must be unique hashes")
    loop = validate_loop_outcome(plan["loop_outcome"])
    if len(candidates) != loop["kept"]:
        raise RunInvalid("run plan candidates do not match O5 kept count")
    if plan["equal_memory"] is not None:
        raise RunInvalid(
            "equal-memory batch search has no pre-registered grid or fit rule"
        )
    return dict(plan)


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def trace_batch(tokens: Sequence[Sequence[int]],
                lengths: Sequence[Sequence[int]], *, step: int,
                split: str) -> dict:
    """Hash the post-truncation batch and the exact loss mask MLX-LM applies."""
    token_rows = [list(row) for row in tokens]
    length_rows = [list(row) for row in lengths]
    if (not token_rows or len(token_rows) != len(length_rows)
            or any(len(pair) != 2 for pair in length_rows)):
        raise RunInvalid("batch tokens and length pairs do not align")
    width = len(token_rows[0])
    if width < 2 or any(len(row) != width for row in token_rows):
        raise RunInvalid("batch token rows must share a width of at least two")
    mask = []
    for start, end in length_rows:
        if (not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool)
                or start < 0 or end < start or end > width):
            raise RunInvalid("batch length pair is outside the token row")
        mask.append([start <= position <= end for position in range(1, width)])
    return {
        "split": split,
        "step": step,
        "tokens_sha256": _json_sha256(token_rows),
        "lengths_sha256": _json_sha256(length_rows),
        "mask_sha256": _json_sha256(mask),
        "supervised_tokens": sum(sum(row) for row in mask),
    }


def build_cell_reading(
    cell_name: str,
    rounds: Sequence[Mapping[str, Mapping[str, object]]],
    *,
    expected: Mapping[str, object] | None = None,
    kept_candidates: Sequence[str] | None = None,
) -> dict:
    """Invalid aligned rounds remain raw evidence but are never samples."""
    if len(rounds) != ROUNDS:
        return {
            "cell": cell_name,
            "binding": False,
            "verdict": "REFUSED",
            "invalid_rounds": list(range(len(rounds) + 1, ROUNDS + 1)),
            "round_validity": [],
            "raw_rounds": list(rounds),
        }
    validity = [
        round_validity(
            cell_name,
            number,
            records,
            expected=expected,
            kept_candidates=kept_candidates,
        )
        for number, records in enumerate(rounds, 1)
    ]
    baseline = rounds[0]["stock"].get("fairness")
    for round_index, records in enumerate(rounds):
        if any(records[arm].get("fairness") != baseline for arm in ARMS):
            validity[round_index]["valid"] = False
            validity[round_index]["reasons"].append(
                "fairness evidence drifted across rounds"
            )
    invalid = [
        number for number, result in enumerate(validity, 1)
        if not result["valid"]
    ]
    eligible = [
        records for records, result in zip(rounds, validity)
        if result["valid"]
    ]
    eligible_numbers = [records["stock"]["round"] for records in eligible]
    if len(eligible) < 2:
        return {
            "cell": cell_name,
            "binding": False,
            "verdict": "REFUSED",
            "invalid_rounds": invalid,
            "eligible_rounds": eligible_numbers,
            "reasons": ["fewer than two eligible aligned rounds"],
            "round_validity": validity,
            "raw_rounds": list(rounds),
        }
    warmed_rounds = [
        records for records in eligible if records["stock"]["round"] > 1
    ]
    if len(warmed_rounds) < 2:
        # Refused here rather than raised out of arm_metrics, so a cell too
        # thin for the secondary metric becomes one non-binding cell instead
        # of an exception that takes the other cells' outcomes with it.
        return {
            "cell": cell_name,
            "binding": False,
            "verdict": "REFUSED",
            "invalid_rounds": invalid,
            "eligible_rounds": eligible_numbers,
            "reasons": ["fewer than two eligible rounds after the first"],
            "round_validity": validity,
            "raw_rounds": list(rounds),
        }
    loss_gate = loss_curve_gate(eligible)
    arms = {arm: arm_metrics(eligible, arm) for arm in ARMS}
    if not loss_gate["valid"]:
        return {
            "cell": cell_name,
            "binding": False,
            "verdict": "LOSS-DIVERGED",
            "invalid_rounds": invalid,
            "eligible_rounds": eligible_numbers,
            "round_validity": validity,
            "loss_gate": loss_gate,
            "arms": arms,
            "peaks_gb": {
                arm: arms[arm]["peak_footprint_gb"] for arm in ARMS
            },
            "raw_rounds": list(rounds),
        }

    def compare(first: str, second: str, field: str) -> dict:
        return time_comparison(arms[first][field], arms[second][field])

    comparisons = {
        "ours_vs_stock": compare("ours", "stock", "full_job_samples_s"),
        "ours_vs_control": compare("ours", "control", "full_job_samples_s"),
        "control_vs_stock": compare("control", "stock", "full_job_samples_s"),
        "warmed_ours_vs_stock": compare(
            "ours", "stock", "warmed_step_samples_s"
        ),
        "warmed_ours_vs_control": compare(
            "ours", "control", "warmed_step_samples_s"
        ),
        "warmed_control_vs_stock": compare(
            "control", "stock", "warmed_step_samples_s"
        ),
    }
    reading = {
        "cell": cell_name,
        "binding": True,
        "invalid_rounds": invalid,
        "eligible_rounds": eligible_numbers,
        "round_validity": validity,
        "loss_gate": loss_gate,
        "arms": arms,
        "peaks_gb": {
            arm: arms[arm]["peak_footprint_gb"] for arm in ARMS
        },
        **comparisons,
        "raw_rounds": list(rounds),
    }
    # Amendment 3. A reference arm that spread past the class limit describes
    # the machine, so the comparison claims nothing. The whole cell stops
    # binding rather than only that comparison, because every comparison here
    # reads the same rounds: a clock that moved under one of them moved under
    # all of them.
    rejected = sorted(name for name, comparison in comparisons.items()
                      if comparison["verdict"] == "REJECTED")
    if rejected:
        reading["binding"] = False
        reading["verdict"] = "REJECTED"
        reading["reasons"] = [
            f"reference arm spread past {MAX_REFERENCE_SPREAD}x in "
            f"{', '.join(rejected)}"
        ]
    return reading


def _print_refusal(reason: object) -> None:
    print(f"REFUSED: {reason}", flush=True)


def run_binding(plan: Mapping[str, object], runtime,
                *, results_dir: str | Path = "bench/results") -> int:
    """Only a clean closing gate can turn gathered samples into a binding record."""
    try:
        fixed_plan = validate_plan(plan)
    except RunInvalid as error:
        _print_refusal(error)
        return EXIT_PRECONDITION
    loop = validate_loop_outcome(fixed_plan["loop_outcome"])
    print(json.dumps({"O5": loop}), flush=True)

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
        if not fixed_plan["kept_candidates"]:
            closing_idle = runtime.idle_check(machine["cores"])
            record = {
                "schema_version": 1,
                "binding": bool(closing_idle["idle"]),
                "reason": "O5 kept no kernel, so no speed claim exists",
                "plan": fixed_plan,
                "provenance": {"plan_sha256": _json_sha256(fixed_plan)},
                "machine": machine,
                "opening_idle": opening_idle,
                "closing_idle": closing_idle,
                "cells": {},
                "outcomes": {"O5": loop},
            }
            runtime.write_record(
                recording_path(
                    results_dir, runtime.today(),
                    closing_idle=closing_idle["idle"],
                ),
                record,
            )
            return 0 if closing_idle["idle"] else 1
        try:
            runtime.require_memory(
                fixed_plan["memory_ceiling_gb"], "startup"
            )
            provenance = runtime.preflight(fixed_plan)
        except BudgetExceeded as error:
            _print_refusal(error)
            return EXIT_BUDGET_REFUSAL
        except LowMemoryRefusal as error:
            _print_refusal(error)
            return EXIT_LOW_MEMORY
        except PreconditionFailed as error:
            _print_refusal(error)
            return EXIT_PRECONDITION

        try:
            backward = runtime.verify_backward(fixed_plan, provenance)
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
        verified_candidates = backward.get("candidate_sha256s")
        if verified_candidates is None:
            verified_candidates = [backward.get("candidate_sha256")]
        if (backward.get("exact") is not True
                or set(verified_candidates)
                != set(fixed_plan["kept_candidates"])):
            _print_refusal("kept-kernel backward comparison was not exact")
            return 1

        gathered = {cell.name: [] for cell in CELLS}
        try:
            for cell in CELLS:
                for round_index in range(ROUNDS):
                    round_number = round_index + 1
                    records = {}
                    for arm in rotated_arms(round_index):
                        label = f"{cell.name} round {round_number} {arm}"
                        runtime.require_memory(
                            fixed_plan["memory_ceiling_gb"], label
                        )
                        records[arm] = runtime.run_arm(
                            fixed_plan, provenance, cell, round_number, arm
                        )
                    gathered[cell.name].append(records)
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

        cells = {
            cell.name: build_cell_reading(
                cell.name,
                gathered[cell.name],
                expected=(
                    runtime.expected_fairness(fixed_plan, provenance, cell)
                    if hasattr(runtime, "expected_fairness") else None
                ),
                kept_candidates=fixed_plan["kept_candidates"],
            )
            for cell in CELLS
        }
        scientific_binding = all(cell["binding"] for cell in cells.values())
        # Every outcome is written even when a cell did not bind. Section 9
        # requires O4 at every cell whatever O1 says, O3 never omitted from a
        # quotation of O1, and O5 whatever the kernel did; reporting only O5
        # because one cell voided would delete three registered readings that
        # the surviving cells did measure. The record still says binding is
        # false, which is what stops any of it being quoted as a claim.
        outcomes = pre_registered_outcomes(cells, loop)
        closing_idle = runtime.idle_check(machine["cores"])
        binding = scientific_binding and closing_idle["idle"]
        record = {
            "schema_version": 1,
            "binding": binding,
            "plan": fixed_plan,
            "provenance": provenance,
            "machine": machine,
            "opening_idle": opening_idle,
            "closing_idle": closing_idle,
            "backward": backward,
            "cells": cells,
            "outcomes": outcomes,
        }
        runtime.write_record(
            recording_path(
                results_dir, runtime.today(), closing_idle=binding
            ),
            record,
        )
        if not binding:
            _print_refusal(
                closing_idle if not closing_idle["idle"] else "invalid cell"
            )
            return 1
        print(json.dumps(outcomes), flush=True)
        return 0
    finally:
        runtime.release_lock()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path, files: Sequence[Path] | None = None) -> str:
    selected = (
        sorted(files, key=lambda path: str(path.relative_to(root)))
        if files is not None else sorted(
            path for path in root.rglob("*")
            if (path.is_file() and "__pycache__" not in path.parts
                and path.suffix != ".pyc" and not path.name.startswith("."))
        )
    )
    if not selected:
        raise PreconditionFailed(f"nothing hashable under {root}")
    digest = hashlib.sha256()
    for path in selected:
        relative = str(path.relative_to(root)).encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def package_sha256(distribution_name: str) -> str:
    """Hash every installed distribution file except interpreter caches."""
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as error:
        raise PreconditionFailed(
            f"required distribution {distribution_name!r} is not installed"
        ) from error
    files = []
    for entry in distribution.files or ():
        path = Path(distribution.locate_file(entry))
        if (path.is_file() and "__pycache__" not in path.parts
                and path.suffix != ".pyc"):
            files.append((str(entry), path))
    if not files:
        raise PreconditionFailed(
            f"distribution {distribution_name!r} has no hashable files"
        )
    digest = hashlib.sha256()
    for relative, path in sorted(files):
        name = relative.encode()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def stack_record() -> dict:
    records = {}
    for key, distribution_name, expected in (
        ("mlx", "mlx", MLX_VERSION),
        ("mlx_lm", "mlx-lm", MLX_LM_VERSION),
    ):
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError as error:
            raise PreconditionFailed(
                f"required distribution {distribution_name!r} is not installed"
            ) from error
        if version != expected:
            raise PreconditionFailed(
                f"{distribution_name} is {version}, required {expected}"
            )
        records[key] = {
            "version": version,
            "package_sha256": package_sha256(distribution_name),
        }
    return records


def _verified_model_manifest(model_dir: Path, manifest: Path) -> dict:
    if not manifest.is_file():
        raise PreconditionFailed(f"pin manifest is missing: {manifest}")
    entries = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, path = line.split(maxsplit=1)
        relative = path.removeprefix("bench/.models/")
        if relative.startswith(f"{MODEL_NAME}/"):
            entries[relative.removeprefix(f"{MODEL_NAME}/")] = digest
    actual = {
        str(path.relative_to(model_dir))
        for path in model_dir.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    }
    if actual != set(entries):
        raise PreconditionFailed(
            "model files differ from the pin manifest: "
            f"unlisted={sorted(actual - set(entries))}, "
            f"missing={sorted(set(entries) - actual)}"
        )
    for relative, expected in entries.items():
        observed = _file_sha256(model_dir / relative)
        if observed != expected:
            raise PreconditionFailed(
                f"pin mismatch for {MODEL_NAME}/{relative}"
            )
    return entries


def _data_record(path: Path) -> dict:
    directory = path.resolve()
    if not directory.is_dir():
        raise PreconditionFailed(
            f"training data must be a local directory: {directory}"
        )
    files = [
        directory / f"{split}.jsonl"
        for split in ("train", "valid", "test")
        if (directory / f"{split}.jsonl").is_file()
    ]
    if not files or files[0].name != "train.jsonl":
        raise PreconditionFailed(f"training data has no train.jsonl: {directory}")
    for file in files:
        rows = [line for line in file.read_text().splitlines() if line.strip()]
        if not rows:
            raise PreconditionFailed(f"dataset split is empty: {file}")
        try:
            for row in rows:
                if not isinstance(json.loads(row), Mapping):
                    raise ValueError("row is not an object")
        except (json.JSONDecodeError, ValueError) as error:
            raise PreconditionFailed(f"invalid JSONL in {file}: {error}") from error
    return {
        "directory": str(directory),
        "files": [file.name for file in files],
        "sha256": _tree_sha256(directory, files),
    }


def _absent_module(name: str, find_module) -> bool:
    """Is the named measurement module missing, however it is missing?

    `find_spec` returns None for an absent top-level name and RAISES for a
    dotted name whose parent package does not resolve, so a plan naming
    "metalrunnr.lora" would leave a bare traceback and exit 1 rather than a
    numbered refusal. The detached runner reads exit 1 with a traceback as a
    crash and spends a retry on it, when the answer for a misspelled plan is a
    permanent refusal that should not be retried at all.
    """
    try:
        return find_module(name) is None
    except (ImportError, ValueError, TypeError):
        return True


def preflight_inputs(
    plan: Mapping[str, object],
    *,
    root: Path = ROOT,
    machine: Mapping[str, object],
    package_records: Mapping[str, object] | None = None,
    find_module=importlib.util.find_spec,
) -> dict:
    """Bind every permanent input before a child may load the model."""
    if machine.get("chip") != "Apple M3 Pro":
        raise PreconditionFailed(
            f"sprint-1 is registered for Apple M3 Pro, got {machine.get('chip')!r}"
        )
    memory_bytes = machine.get("memory_bytes")
    if (not isinstance(memory_bytes, int)
            or round(memory_bytes / 2**30) != 36):
        raise PreconditionFailed(
            "sprint-1 is registered for the 36 GB machine"
        )
    model_dir = root / "bench" / ".models" / MODEL_NAME
    model_file = model_dir / "model.safetensors"
    manifest = model_dir.parent / "PINNED-HASHES.txt"
    if not model_file.is_file():
        raise PreconditionFailed(f"pinned model file is missing: {model_file}")
    model_manifest = _verified_model_manifest(model_dir, manifest)
    observed_model = model_manifest.get("model.safetensors")
    if observed_model is None:
        raise PreconditionFailed("pin manifest omits model.safetensors")
    stack = dict(package_records if package_records is not None else stack_record())
    expected_versions = {"mlx": MLX_VERSION, "mlx_lm": MLX_LM_VERSION}
    for package, expected_version in expected_versions.items():
        record = stack.get(package)
        if (not isinstance(record, Mapping)
                or record.get("version") != expected_version
                or not record.get("package_sha256")):
            raise PreconditionFailed(
                f"stack record does not bind {package} {expected_version}"
            )
    if plan["kept_candidates"] and _absent_module(
            plan["measurement_module"], find_module):
        raise PreconditionFailed(
            f"measurement module {plan['measurement_module']!r} is unavailable"
        )
    wrapper = root / "metalrunner"
    if not wrapper.is_dir():
        raise PreconditionFailed(f"metalrunner wrapper is missing: {wrapper}")
    return {
        "base_model": {
            "directory": str(model_dir.resolve()),
            "file": str(model_file.resolve()),
            "sha256": observed_model,
        },
        "model_manifest": model_manifest,
        "data": _data_record(Path(plan["data"])),
        "stack": stack,
        "wrapper_sha256": _tree_sha256(wrapper),
        "plan_sha256": _json_sha256(plan),
        "measurement_module": plan["measurement_module"],
    }


def build_training_args(plan: Mapping[str, object],
                        provenance: Mapping[str, object], cell: Cell,
                        adapter_path: str | Path) -> types.SimpleNamespace:
    """Construct the one MLX-LM configuration every arm receives."""
    optimizer = plan["optimizer"]
    return types.SimpleNamespace(
        model=provenance["base_model"]["directory"],
        train=True,
        fine_tune_type="lora",
        optimizer=optimizer,
        optimizer_config={optimizer: dict(plan["optimizer_config"])},
        data=provenance["data"]["directory"],
        seed=plan["seed"],
        num_layers=16,
        batch_size=cell.batch_size,
        iters=plan["steps"],
        val_batches=25,
        learning_rate=plan["learning_rate"],
        steps_per_report=1,
        steps_per_eval=200,
        resume_adapter_file=None,
        adapter_path=str(adapter_path),
        save_every=plan["steps"] + 1,
        test=False,
        test_batches=500,
        max_seq_length=2048,
        config=None,
        grad_checkpoint=False,
        grad_accumulation_steps=1,
        clear_cache_threshold=0,
        lr_schedule=plan["schedule"],
        lora_parameters={"rank": 8, "dropout": 0.0, "scale": 20.0},
        mask_prompt=cell.mask_prompt,
        report_to=None,
        project_name=None,
    )


def _signal_child(proc, signum: int) -> None:
    try:
        own_group = os.getpgid(proc.pid) == proc.pid
    except (ProcessLookupError, PermissionError):
        return
    try:
        if own_group:
            os.killpg(proc.pid, signum)
        else:
            os.kill(proc.pid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def _reap(proc) -> None:
    if proc.poll() is not None:
        return
    _signal_child(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        _signal_child(proc, signal.SIGKILL)


def spawn_child(task: Mapping[str, object], work_dir: str | Path, *,
                wall_cap_s: float, popen_factory=None, clock_ns=None,
                reap=None) -> dict:
    """A process-start timestamp and child adapter timestamp bound job wall."""
    popen = subprocess.Popen if popen_factory is None else popen_factory
    clock = time.perf_counter_ns if clock_ns is None else clock_ns
    reap_process = _reap if reap is None else reap
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    task_path = work / f"task-{nonce}.json"
    result_path = work / f"result-{nonce}.json"
    stamped = {**task, "parent_pid": os.getpid()}
    task_path.write_text(json.dumps(stamped, sort_keys=True))
    argv = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--child-task",
        str(task_path),
        "--child-out",
        str(result_path),
    ]
    launched_ns = clock()
    proc = popen(argv, start_new_session=True)
    timed_out = False
    try:
        try:
            proc.wait(timeout=wall_cap_s)
        except subprocess.TimeoutExpired:
            timed_out = True
    finally:
        reap_process(proc)
        task_path.unlink(missing_ok=True)
    completed_ns = clock()
    cell = str(task.get("cell", task.get("kind", "child")))
    if timed_out:
        raise ChildRefusal(
            cell, EXIT_CHILD_DEATH, f"hit the {wall_cap_s:.0f}s wall cap"
        )
    if proc.returncode != 0:
        raise ChildRefusal(cell, proc.returncode, f"child exited {proc.returncode}")
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ChildRefusal(
            cell, EXIT_CHILD_DEATH, f"child left no valid result: {error}"
        ) from error
    finally:
        result_path.unlink(missing_ok=True)
    if task.get("kind") == "train":
        written_ns = result.get("adapter_written_monotonic_ns")
        if (not isinstance(written_ns, int)
                or not launched_ns <= written_ns <= completed_ns):
            raise ChildRefusal(
                cell, EXIT_CHILD_DEATH,
                "adapter completion timestamp is outside child lifetime",
            )
        result["process_launch_monotonic_ns"] = launched_ns
        result["full_job_wall_s"] = (written_ns - launched_ns) / 1e9
    return result


class SystemRuntime:
    """Concrete machine, guard, pin, child and recording seams."""

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.lock = MeasurementLock("train_lora_e2e")
        self.machine = None
        self.guard = None
        self._temporary = None

    def acquire_lock(self) -> tuple[bool, str]:
        acquired, detail = self.lock.acquire()
        if acquired:
            try:
                self._temporary = tempfile.TemporaryDirectory(
                    prefix="kernelverify-train-lora-"
                )
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

    def expected_fairness(self, plan: Mapping[str, object],
                          provenance: Mapping[str, object], cell: Cell) -> dict:
        return {
            "base_model": {
                "path": provenance["base_model"]["file"],
                "sha256": provenance["base_model"]["sha256"],
            },
            "data_source_sha256": provenance["data"]["sha256"],
            "optimization": {
                "optimizer": plan["optimizer"],
                "optimizer_config": dict(plan["optimizer_config"]),
                "learning_rate": plan["learning_rate"],
                "schedule": plan["schedule"],
                "seeds": {
                    "python": plan["seed"],
                    "numpy": plan["seed"],
                    "mlx": plan["seed"],
                },
                "steps": plan["steps"],
                "warmup_steps": plan["warmup_steps"],
                "batch_size": cell.batch_size,
                "mask_prompt": cell.mask_prompt,
                "max_seq_length": 2048,
                "lora_rank": 8,
                "num_layers": 16,
            },
            "memory_ceiling_gb": plan["memory_ceiling_gb"],
            "stack": provenance["stack"],
        }

    @property
    def work_dir(self) -> Path:
        if self._temporary is None:
            raise PreconditionFailed("child workspace exists only under the lock")
        return Path(self._temporary.name)

    def verify_backward(self, plan: Mapping[str, object],
                        provenance: Mapping[str, object]) -> dict:
        return spawn_child(
            {
                "kind": "backward",
                "cell": "backward exactness",
                "plan": plan,
                "provenance": provenance,
                "budget_gb": plan["memory_ceiling_gb"],
            },
            self.work_dir,
            wall_cap_s=plan["wall_cap_seconds"],
        )

    def run_arm(self, plan: Mapping[str, object],
                provenance: Mapping[str, object], cell: Cell,
                round_number: int, arm: str) -> dict:
        label = f"{cell.name} round {round_number} {arm}"
        return spawn_child(
            {
                "kind": "train",
                "cell": label,
                "cell_name": cell.name,
                "round": round_number,
                "arm": arm,
                "plan": plan,
                "provenance": provenance,
                "budget_gb": plan["memory_ceiling_gb"],
                "adapter_path": str(self.work_dir / (
                    f"adapter-{cell.name}-{round_number}-{arm}"
                )),
            },
            self.work_dir,
            wall_cap_s=plan["wall_cap_seconds"],
        )

    def write_record(self, path: Path, record: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(temporary, path)

    def today(self) -> date:
        return date.today()


def _load_plan(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PreconditionFailed(f"cannot read run plan {path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise PreconditionFailed("run plan must be a JSON object")
    try:
        return validate_plan(raw)
    except RunInvalid as error:
        raise PreconditionFailed(str(error)) from error


def _atomic_json(path: Path, record: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True))
    os.replace(temporary, path)


class _ChildGuard:
    def __init__(self, budget_gb: float, parent_pid: int):
        self.budget_gb = budget_gb
        self._budget = BudgetGuard(budget_gb, parent_pid=parent_pid)

    def check(self, cell: str) -> float:
        current = self._budget.check(cell)
        require_available_memory(max(self.budget_gb - current, 0.0), cell)
        return current


def _check_child_inputs(task: Mapping[str, object], observed_stack: dict) -> None:
    plan = task.get("plan")
    provenance = task.get("provenance")
    if not isinstance(plan, Mapping) or not isinstance(provenance, Mapping):
        raise PreconditionFailed("child task has no plan or provenance")
    if _json_sha256(plan) != provenance.get("plan_sha256"):
        raise PreconditionFailed("child run plan differs from parent preflight")
    if observed_stack != provenance.get("stack"):
        raise PreconditionFailed("child package hashes differ from preflight")
    base = provenance.get("base_model")
    if not isinstance(base, Mapping) or not Path(base.get("file", "")).is_file():
        raise PreconditionFailed("child base model evidence is missing")
    model_dir = Path(base["directory"])
    model_manifest = _verified_model_manifest(
        model_dir, model_dir.parent / "PINNED-HASHES.txt"
    )
    if model_manifest != provenance.get("model_manifest"):
        raise PreconditionFailed("child model artifact manifest changed")
    if model_manifest.get("model.safetensors") != base.get("sha256"):
        raise PreconditionFailed("child base model sha256 differs from preflight")
    observed_data = _data_record(Path(plan["data"]))
    if observed_data != provenance.get("data"):
        raise PreconditionFailed("child training data differs from preflight")
    wrapper = ROOT / "metalrunner"
    if _tree_sha256(wrapper) != provenance.get("wrapper_sha256"):
        raise PreconditionFailed("child wrapper hash differs from preflight")


def _load_measurement_module(plan: Mapping[str, object]):
    try:
        module = importlib.import_module(plan["measurement_module"])
    except (ImportError, AttributeError) as error:
        raise PreconditionFailed(
            f"cannot import measurement module {plan['measurement_module']!r}: "
            f"{error}"
        ) from error
    return module


def _backward_child(task: Mapping[str, object], guard: BudgetGuard) -> dict:
    plan = task["plan"]
    module = _load_measurement_module(plan)
    verify = getattr(module, "verify_backward_exact", None)
    if not callable(verify):
        raise PreconditionFailed(
            "measurement module must expose verify_backward_exact"
        )
    result = verify(
        model_path=task["provenance"]["base_model"]["directory"],
        candidate_sha256s=tuple(plan["kept_candidates"]),
        guard=guard,
    )
    if not isinstance(result, Mapping):
        raise PreconditionFailed("backward verifier returned no evidence object")
    candidates = result.get("candidate_sha256s")
    if (not isinstance(candidates, Sequence) or isinstance(candidates, str)
            or set(candidates) != set(plan["kept_candidates"])):
        raise PreconditionFailed("backward evidence names different candidates")
    if (not isinstance(result.get("cases"), int)
            or result["cases"] <= 0 or not result.get("cases_sha256")):
        raise PreconditionFailed("backward evidence does not bind fixed cases")
    if result.get("exact") not in (True, False):
        raise PreconditionFailed("backward evidence has no exact-array verdict")
    return dict(result)


class _TrainingCapture:
    def __init__(self, guard: BudgetGuard, cell: str):
        self.guard = guard
        self.cell = cell
        self.train = []
        self.validation = []
        self.batches = []
        self.adapter_init_sha256 = None

    def on_train_loss_report(self, info: Mapping[str, object]) -> None:
        self.train.append(dict(info))
        self.guard.check(f"{self.cell}: step {info.get('iteration')}")

    def on_val_loss_report(self, info: Mapping[str, object]) -> None:
        self.validation.append(dict(info))
        self.guard.check(f"{self.cell}: validation")


def _adapter_initial_sha256(model, mx, np, tree_flatten) -> str:
    parameters = tree_flatten(model.trainable_parameters())
    if not parameters:
        raise PreconditionFailed("LoRA conversion produced no trainable weights")
    mx.eval(*(array for _, array in parameters))
    digest = hashlib.sha256()
    for name, array in sorted(parameters, key=lambda item: item[0]):
        identity = json.dumps({
            "name": name,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }, sort_keys=True, separators=(",", ":")).encode()
        digest.update(len(identity).to_bytes(4, "big"))
        digest.update(identity)
        digest.update(np.asarray(array.astype(mx.float32)).tobytes())
    return digest.hexdigest()


def _routing_evidence(patch, arm: str,
                      provenance: Mapping[str, object]) -> dict:
    evidence_fn = getattr(patch, "evidence", None)
    if not callable(evidence_fn):
        raise PreconditionFailed(
            "installed measurement wrapper must expose evidence()"
        )
    evidence = evidence_fn()
    if not isinstance(evidence, Mapping):
        raise PreconditionFailed("measurement wrapper returned no evidence object")
    required = {
        "wrapper_installed",
        "forced_stock",
        "wrapper_sha256",
        "routed_candidates",
        "routed_calls",
        "routing_decisions",
    }
    if not required <= set(evidence):
        raise PreconditionFailed(
            f"measurement wrapper evidence is missing {sorted(required - set(evidence))}"
        )
    if evidence["wrapper_sha256"] != provenance["wrapper_sha256"]:
        raise PreconditionFailed("installed wrapper hash differs from preflight")
    if evidence["wrapper_installed"] is not True:
        raise PreconditionFailed("measurement wrapper did not attest installation")
    if evidence["forced_stock"] is not (arm == "control"):
        raise PreconditionFailed("measurement wrapper attested the wrong arm mode")
    return dict(evidence)


def _training_child(task: Mapping[str, object], guard: BudgetGuard,
                    observed_stack: Mapping[str, object]) -> dict:
    import numpy as np
    import mlx.core as mx
    from mlx.utils import tree_flatten
    import mlx_lm.lora as lora
    from mlx_lm.tuner import trainer

    if not mx.metal.is_available():
        raise NoDevice("MLX reports no Metal device")

    plan = task["plan"]
    provenance = task["provenance"]
    cell = CELL_BY_NAME.get(task.get("cell_name"))
    arm = task.get("arm")
    if cell is None or arm not in ARMS:
        raise PreconditionFailed("training child names an unknown cell or arm")
    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    adapter_path = Path(task["adapter_path"])
    adapter_path.mkdir(parents=True, exist_ok=False)
    args = build_training_args(plan, provenance, cell, adapter_path)
    capture = _TrainingCapture(guard, task["cell"])
    original_train = lora.train
    original_callbacks = lora.get_reporting_callbacks
    original_batches = trainer.iterate_batches

    def measured_batches(*batch_args, **batch_kwargs):
        training = batch_kwargs.get("loop", False)
        for batch, lengths in original_batches(*batch_args, **batch_kwargs):
            guard.check(f"{task['cell']}: batch")
            if training and len(capture.batches) < plan["steps"]:
                capture.batches.append(trace_batch(
                    batch.tolist(),
                    lengths.tolist(),
                    step=len(capture.batches) + 1,
                    split="train",
                ))
            yield batch, lengths

    def measured_train(*train_args, **train_kwargs):
        model = train_kwargs.get("model")
        if model is None and train_args:
            model = train_args[0]
        capture.adapter_init_sha256 = _adapter_initial_sha256(
            model, mx, np, tree_flatten
        )
        guard.check(f"{task['cell']}: adapter initialised")
        train_kwargs["iterate_batches"] = measured_batches
        return original_train(*train_args, **train_kwargs)

    lora.train = measured_train
    lora.get_reporting_callbacks = lambda *args, **kwargs: capture
    patch = None
    try:
        if arm != "stock":
            module = _load_measurement_module(plan)
            install = getattr(module, "install", None)
            if not callable(install):
                raise PreconditionFailed(
                    "measurement module must expose install"
                )
            installed = install(
                candidate_sha256s=tuple(plan["kept_candidates"]),
                force_stock=(arm == "control"),
                guard=guard,
            )
            if not callable(getattr(installed, "uninstall", None)):
                raise PreconditionFailed(
                    "installed measurement wrapper must expose uninstall()"
                )
            patch = installed
        lora.run(args)
        adapter_written_ns = time.perf_counter_ns()
        routing = (
            {
                "wrapper_installed": False,
                "forced_stock": False,
                "wrapper_sha256": None,
                "routed_candidates": [],
                "routed_calls": 0,
                "routing_decisions": 0,
            }
            if arm == "stock" else _routing_evidence(patch, arm, provenance)
        )
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            if patch is not None:
                patch.uninstall()
        except Exception as cleanup_error:
            if not active_error:
                raise
            print(
                f"wrapper cleanup also failed: {cleanup_error}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            lora.train = original_train
            lora.get_reporting_callbacks = original_callbacks

    adapter_file = adapter_path / "adapters.safetensors"
    if not adapter_file.is_file():
        raise PreconditionFailed("mlx_lm.lora returned without writing an adapter")
    if capture.adapter_init_sha256 is None:
        raise PreconditionFailed("initial adapter weights were not captured")
    if len(capture.train) != plan["steps"]:
        raise PreconditionFailed("trainer did not report exactly one loss per step")
    if len(capture.batches) != plan["steps"]:
        raise PreconditionFailed("trainer did not expose exactly one batch per step")
    trained_before = 0
    step_seconds = []
    loss_curve = []
    for expected_step, (info, batch) in enumerate(
        zip(capture.train, capture.batches), 1
    ):
        if info.get("iteration") != expected_step:
            raise PreconditionFailed("trainer loss reports are not consecutive")
        rate = info.get("iterations_per_second")
        trained = info.get("trained_tokens")
        if not _finite_positive(rate) or not isinstance(trained, (int, float)):
            raise PreconditionFailed("trainer reported invalid step metrics")
        token_delta = trained - trained_before
        if not float(token_delta).is_integer():
            raise PreconditionFailed("trainer reported a fractional token count")
        step_tokens = int(token_delta)
        if step_tokens != batch["supervised_tokens"]:
            raise PreconditionFailed(
                "trainer token count differs from the captured loss mask"
            )
        trained_before = trained
        step_seconds.append(1.0 / rate)
        loss_curve.append({
            "step": expected_step,
            "loss": float(info["train_loss"]),
        })
    optimization = {
        "optimizer": plan["optimizer"],
        "optimizer_config": dict(plan["optimizer_config"]),
        "learning_rate": plan["learning_rate"],
        "schedule": plan["schedule"],
        "seeds": {
            "python": plan["seed"],
            "numpy": plan["seed"],
            "mlx": plan["seed"],
        },
        "steps": plan["steps"],
        "warmup_steps": plan["warmup_steps"],
        "batch_size": cell.batch_size,
        "mask_prompt": cell.mask_prompt,
        "max_seq_length": 2048,
        "lora_rank": 8,
        "num_layers": 16,
    }
    return {
        "cell": cell.name,
        "arm": arm,
        "round": task["round"],
        "completed": True,
        "adapter_written": True,
        "adapter_written_monotonic_ns": adapter_written_ns,
        "adapter_final_sha256": _file_sha256(adapter_file),
        "step_seconds": step_seconds,
        "loss_curve": loss_curve,
        "validation_loss_curve": capture.validation,
        "mlx_peak_memory_gb": mx.get_peak_memory() / 1e9,
        "fairness": {
            "base_model": {
                "path": provenance["base_model"]["file"],
                "sha256": provenance["base_model"]["sha256"],
            },
            "adapter_init_sha256": capture.adapter_init_sha256,
            "data": {
                "source_sha256": provenance["data"]["sha256"],
                "batches": capture.batches,
            },
            "optimization": optimization,
            "memory_ceiling_gb": plan["memory_ceiling_gb"],
            "stack": dict(observed_stack),
        },
        "routing": routing,
    }


def child_main(task_path: Path, result_path: Path) -> int:
    """Run one guarded backward check or one fresh training arm."""
    if os.environ.get("KV_FORCE_NO_METAL") == "1":
        print("REFUSED: KV_FORCE_NO_METAL disables the measurement child")
        return EXIT_NO_DEVICE
    try:
        task = json.loads(task_path.read_text())
        if not isinstance(task, Mapping):
            raise PreconditionFailed("child task is not a JSON object")
        budget = task.get("budget_gb")
        parent_pid = task.get("parent_pid")
        if not _finite_positive(budget) or not isinstance(parent_pid, int):
            raise PreconditionFailed("child task has no budget or parent pid")
        guard = _ChildGuard(budget, parent_pid)
        guard.check(f"{task.get('cell', 'child')}: startup")
        observed_stack = stack_record()
        _check_child_inputs(task, observed_stack)
        if task.get("kind") == "backward":
            module = _load_measurement_module(task["plan"])
            if not callable(getattr(module, "verify_backward_exact", None)):
                raise PreconditionFailed(
                    "measurement module must expose verify_backward_exact"
                )
            result = _backward_child(task, guard)
        elif task.get("kind") == "train":
            if task.get("arm") != "stock":
                module = _load_measurement_module(task["plan"])
                if not callable(getattr(module, "install", None)):
                    raise PreconditionFailed(
                        "measurement module must expose install"
                    )
            result = _training_child(task, guard, observed_stack)
        else:
            raise PreconditionFailed(f"unknown child kind {task.get('kind')!r}")
        guard.check(f"{task.get('cell', 'child')}: publish")
        current, peak = phys_footprint_gb()
        if peak > budget:
            raise BudgetExceeded(str(task.get("cell", "child")), peak, budget)
        result["footprint_gb"] = {"current": current, "peak": peak}
        if task.get("kind") == "train":
            result["peak_footprint_gb"] = peak
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
    except (PreconditionFailed, RunInvalid, OSError, json.JSONDecodeError) as error:
        print(f"REFUSED (exit {EXIT_PRECONDITION}): {error}")
        return EXIT_PRECONDITION


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--child-task", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_task is not None:
        if args.child_out is None:
            parser.error("--child-task requires --child-out")
        return child_main(args.child_task, args.child_out)
    if args.plan is None:
        print(f"REFUSED (exit {EXIT_PRECONDITION}): --plan is required")
        return EXIT_PRECONDITION
    try:
        plan = _load_plan(args.plan)
    except PreconditionFailed as error:
        print(f"REFUSED (exit {EXIT_PRECONDITION}): {error}")
        return EXIT_PRECONDITION
    return run_binding(plan, SystemRuntime(), results_dir=RESULTS_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
