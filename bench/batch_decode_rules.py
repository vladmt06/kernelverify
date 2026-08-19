"""Pure decision arithmetic for the batched-decode end-to-end run."""

from __future__ import annotations

import math
import statistics

from attribution import composed_attribution
from decode_rules import (
    Comparison,
    RoundSample,
    RunInvalid,
    classify_passes,
    clears,
    comparison,
    delta_pct,
    eligible_rounds,
    first_mismatch,
    is_decider,
    spread_pct,
)
from typing import Mapping, Sequence


# docs/research/2026-08-18-batch-decode-e2e.md, section 3.
B_GRID = (1, 4, 5, 6, 7, 8, 9, 11, 12, 16)

# docs/research/2026-08-18-batch-decode-e2e.md, section 3.
IN_ZONE = frozenset({5, 6, 7, 8, 9})

# docs/research/2026-08-18-batch-decode-e2e.md, section 5.
CEILING_PCT = {5: 5.70, 6: 14.69, 7: 16.44, 8: 15.65, 9: 15.65}

# docs/research/2026-08-15-sub4bit-serve-findings.md, section 9.
LOOP_KERNEL_RATIO = {5: 1.0570, 6: 1.1469, 8: 1.1565}

# docs/research/2026-08-15-sub4bit-serve-findings.md, section 10.
LOOP_KERNEL_FLOOR_PCT = {5: 0.05, 6: 0.10, 8: 0.07}

# docs/research/2026-08-15-sub4bit-serve-findings.md, section 10.
LOOP_COMPOSED_RATIO = {5: 1.0438, 6: 1.0947, 8: 1.1114}

# docs/research/2026-08-15-sub4bit-serve-findings.md, section 10.
PERPLEXITY_PAIR = (22.7069, 15.2355)


_KERNEL_EXCLUSIONS = frozenset({"kernel-diverged"})
_MISSING_STREAM = object()


def decode_passes(
    shapes: Sequence[Sequence[int]], *, b: int, prompt_t: int
) -> tuple[tuple[int, ...], ...]:
    prefill_width = b * (prompt_t - 1)
    passes, prefill = classify_passes(
        shapes,
        is_pass=lambda width: width == b,
        prefill_width=prefill_width,
    )
    if not prefill:
        raise RunInvalid(
            "observed no prefill call; the registered configuration makes "
            f"exactly one, of width {prefill_width}, so the counted seam did "
            "not see it"
        )
    return passes


def expected_routed_calls(
    decode_shapes: Sequence[Sequence[int]],
    routed_sites_at: Mapping[int, int],
) -> int:
    return sum(routed_sites_at[math.prod(shape)] for shape in decode_shapes)


def per_stream_tps(aggregate_tps: float, b: int) -> float:
    return aggregate_tps / b


def ceiling_for(b: int) -> float:
    return CEILING_PCT[b] if b in IN_ZONE else 0.0


def _stream_mismatch(left: object, right: object) -> int | None:
    if left is _MISSING_STREAM or right is _MISSING_STREAM:
        return 0
    return first_mismatch(left, right)


def stream_identity_labels(
    *,
    a1: Sequence[Sequence[object]],
    a2: Sequence[Sequence[object]],
    a4: Sequence[Sequence[object]],
) -> dict[str, dict[int, int]]:
    """Name per-stream kernel mismatches and reject a bad control."""
    stream_count = max(len(a1), len(a2), len(a4))
    kernel_mismatches = {}
    for stream_index in range(stream_count):
        stream_a1 = a1[stream_index] if stream_index < len(a1) else _MISSING_STREAM
        stream_a2 = a2[stream_index] if stream_index < len(a2) else _MISSING_STREAM
        stream_a4 = a4[stream_index] if stream_index < len(a4) else _MISSING_STREAM
        control_position = _stream_mismatch(stream_a4, stream_a2)
        if control_position is not None:
            raise RunInvalid(
                f"arm 4 differs from arm 2 at stream {stream_index}, "
                f"position {control_position}"
            )
        kernel_position = _stream_mismatch(stream_a1, stream_a2)
        if kernel_position is not None:
            kernel_mismatches[stream_index] = kernel_position
    if kernel_mismatches:
        return {"kernel-diverged": kernel_mismatches}
    return {}


def validate_arm_record(
    record: Mapping[str, object], *, b: int, gen_tokens: int
) -> None:
    expected_tokens = gen_tokens * b
    if record["generation_tokens"] != expected_tokens:
        raise RunInvalid(
            f"generation_tokens is {record['generation_tokens']}, expected "
            f"{expected_tokens}"
        )
    for stream_index, tokens in enumerate(record["tokens"]):
        if len(tokens) != gen_tokens:
            raise RunInvalid(
                f"stream {stream_index} returned {len(tokens)} tokens, "
                f"expected {gen_tokens}"
            )
    for stream_index, reason in enumerate(record["finish_reasons"]):
        if reason != "length":
            raise RunInvalid(
                f"finish reason for stream {stream_index} is {reason!r}, "
                "expected 'length'"
            )


def excluded_round_cap(rounds: Sequence[RoundSample]) -> str | None:
    excluded = sum("kernel-diverged" in sample.identity for sample in rounds)
    return "identity-unstable" if excluded > 1 else None


def arm_summary(
    b: int, rounds: Sequence[RoundSample], arm: int
) -> dict[str, object]:
    """One arm's median and spread over this cell's eligible rounds."""
    eligible = eligible_rounds(
        rounds, _KERNEL_EXCLUSIONS, what=f"arm {arm} B={b}"
    )
    samples = [sample.generation_tps[arm] for sample in eligible]
    median = statistics.median(samples)
    return {
        "median_tps": median,
        "per_stream_tps": per_stream_tps(median, b),
        "spread_pct": spread_pct(samples),
        "eligible_rounds": len(eligible),
    }


def _compare(
    b: int,
    rounds: Sequence[RoundSample],
    numerator: int,
    denominator: int,
    outcome: str,
) -> Comparison:
    return comparison(
        rounds,
        numerator,
        denominator,
        excluded_labels=_KERNEL_EXCLUSIONS,
        what=f"{outcome} B={b}",
    )


def decide_ob1(
    b: int,
    rounds: Sequence[RoundSample],
    *,
    routed_calls: int,
    routed_sites_at_width: int,
) -> dict[str, object]:
    kernel = _compare(b, rounds, 1, 2, "OB1")
    control = _compare(b, rounds, 1, 4, "OB1 control")
    host_cost = _compare(b, rounds, 4, 2, "OB1 host cost")
    result = {
        "b": b,
        "arm1_tps": kernel.numerator_tps,
        "arm2_tps": kernel.denominator_tps,
        "arm4_tps": control.denominator_tps,
        "ratio_12": kernel.numerator_tps / kernel.denominator_tps,
        "delta_pct": kernel.delta_pct,
        "noise_floor_pct": kernel.noise_floor_pct,
        "control_delta_pct": control.delta_pct,
        "control_floor_pct": control.noise_floor_pct,
        "host_cost_delta_pct": host_cost.delta_pct,
        "host_cost_floor_pct": host_cost.noise_floor_pct,
        "ceiling_pct": ceiling_for(b),
        "eligible_rounds": len(kernel.rounds),
        "routed_calls": routed_calls,
        "routed_sites_at_width": routed_sites_at_width,
    }
    if b not in IN_ZONE:
        controlled = (
            routed_calls == 0
            and routed_sites_at_width == 0
            and not clears(abs(control.delta_pct), control.noise_floor_pct)
        )
        return {
            **result,
            "decider": False,
            "verdict": "NULL" if controlled else "NULL-uncontrolled",
        }

    decider = is_decider(ceiling_for(b), kernel.noise_floor_pct)
    if not decider:
        return {**result, "decider": False, "verdict": "not-a-decider"}
    if clears(kernel.delta_pct, kernel.noise_floor_pct):
        verdict = "WIN"
    elif clears(-kernel.delta_pct, kernel.noise_floor_pct):
        verdict = "LOSS"
    else:
        verdict = "NULL"
    return {**result, "decider": True, "verdict": verdict}


def decide_ob2(b: int, rounds: Sequence[RoundSample]) -> dict[str, object]:
    capability = _compare(b, rounds, 1, 3, "OB2")
    arm2_tps = statistics.median(
        sample.generation_tps[2] for sample in capability.rounds
    )
    base_pct = delta_pct(arm2_tps, capability.denominator_tps)
    return {
        "b": b,
        "arm1_tps": capability.numerator_tps,
        "arm2_tps": arm2_tps,
        "arm3_tps": capability.denominator_tps,
        "ratio_13": capability.numerator_tps / capability.denominator_tps,
        "delta_pct": capability.delta_pct,
        "base_pct": base_pct,
        "noise_floor_pct": capability.noise_floor_pct,
        "clears_floor_upward": clears(
            capability.delta_pct, capability.noise_floor_pct
        ),
        "eligible_rounds": len(capability.rounds),
        "attribution": composed_attribution(
            capability.delta_pct,
            base_pct,
            capability.noise_floor_pct,
        ),
        "perplexity_pair": PERPLEXITY_PAIR,
    }


def decide_ob3(b: int, rounds: Sequence[RoundSample]) -> dict[str, object]:
    kernel = _compare(b, rounds, 1, 2, "OB3")
    arm3_tps = statistics.median(
        sample.generation_tps[3] for sample in kernel.rounds
    )
    engine_kernel_ratio = kernel.numerator_tps / kernel.denominator_tps
    engine_composed_ratio = kernel.numerator_tps / arm3_tps
    result = {
        "b": b,
        "engine_kernel_ratio": engine_kernel_ratio,
        "engine_composed_ratio": engine_composed_ratio,
        "engine_floor_pct": kernel.noise_floor_pct,
        "eligible_rounds": len(kernel.rounds),
    }
    if b in {7, 9}:
        return {
            **result,
            "loop_kernel_ratio": None,
            "loop_composed_ratio": None,
            "loop_floor_pct": None,
            "ratio_delta_pct": None,
            "threshold_pct": None,
            "verdict": "no-loop-number",
        }
    if b not in LOOP_KERNEL_RATIO:
        raise RunInvalid(f"OB3 has no registered in-zone cell B={b}")
    ratio_delta = delta_pct(engine_kernel_ratio, LOOP_KERNEL_RATIO[b])
    threshold = max(kernel.noise_floor_pct, LOOP_KERNEL_FLOOR_PCT[b])
    verdict = (
        "ENGINE-ABOVE-LOOP"
        if clears(ratio_delta, threshold)
        else "AT-OR-BELOW-LOOP"
    )
    return {
        **result,
        "loop_kernel_ratio": LOOP_KERNEL_RATIO[b],
        "loop_composed_ratio": LOOP_COMPOSED_RATIO[b],
        "loop_floor_pct": LOOP_KERNEL_FLOOR_PCT[b],
        "ratio_delta_pct": ratio_delta,
        "threshold_pct": threshold,
        "verdict": verdict,
    }


def _range_with_ties(values: Mapping[int, float]) -> dict[str, object]:
    minimum = min(values.values())
    maximum = max(values.values())
    return {
        "min": minimum,
        "min_bs": tuple(b for b in sorted(values) if values[b] == minimum),
        "max": maximum,
        "max_bs": tuple(b for b in sorted(values) if values[b] == maximum),
    }


def decide_zone(
    cells: Mapping[int, Sequence[RoundSample]],
    *,
    routed_calls_by_b: Mapping[int, int],
    routed_sites_by_b: Mapping[int, int],
) -> dict[str, object]:
    for b in sorted(IN_ZONE):
        if b not in cells:
            raise RunInvalid(f"ZONE is missing the registered cell B={b}")

    cell_results = {}
    ob1_deltas = {}
    ob2_deltas = {}
    for b in sorted(IN_ZONE):
        rounds = cells[b]
        instability = excluded_round_cap(rounds)
        if instability:
            cell_results[b] = instability
            continue
        ob1 = decide_ob1(
            b,
            rounds,
            routed_calls=routed_calls_by_b[b],
            routed_sites_at_width=routed_sites_by_b[b],
        )
        ob2 = decide_ob2(b, rounds)
        ob1_deltas[b] = ob1["delta_pct"]
        ob2_deltas[b] = ob2["delta_pct"]
        if ob1["verdict"] != "WIN":
            cell_results[b] = ob1["verdict"]
        elif not ob2["clears_floor_upward"]:
            cell_results[b] = "OB2-inside-floor"
        else:
            cell_results[b] = "pass"

    all_cells_read = len(ob1_deltas) == len(IN_ZONE)
    return {
        "verdict": (
            "GO"
            if all_cells_read
            and all(result == "pass" for result in cell_results.values())
            else "NO-GO"
        ),
        "cells": cell_results,
        "ob1_delta_range_pct": (
            _range_with_ties(ob1_deltas) if all_cells_read else None
        ),
        "ob2_delta_range_pct": (
            _range_with_ties(ob2_deltas) if all_cells_read else None
        ),
    }
