"""Shared CPU-only decision arithmetic for decode experiments."""

from __future__ import annotations

import json
import math
import statistics

from dataclasses import dataclass, field
from itertools import zip_longest
from machine_state import spread_pct as machine_spread_pct
from typing import Callable, Collection, Mapping, Sequence


class RunInvalid(RuntimeError):
    """The registered run cannot produce a decision from these records."""


class ExactCountMismatch(RunInvalid):
    """Observed routed calls differ from the per-site expected count."""


@dataclass(frozen=True)
class RoundSample:
    """The decision inputs from one aligned multi-arm round."""

    generation_tps: Mapping[int, float]
    identity: Mapping[str, object] = field(default_factory=dict)
    valid: bool = True


@dataclass(frozen=True)
class Comparison:
    """One median throughput comparison and its per-comparison floor."""

    numerator_tps: float
    denominator_tps: float
    delta_pct: float
    noise_floor_pct: float
    rounds: tuple[RoundSample, ...]


def _token_width(shape: Sequence[int]) -> int:
    if len(shape) != 2:
        raise RunInvalid(
            f"observed pass shape {tuple(shape)} is rank {len(shape)}; the "
            "counted seam takes rank-2 token identifiers"
        )
    return math.prod(shape)


def classify_passes(
    shapes: Sequence[Sequence[int]],
    *,
    is_pass: Callable[[int], bool],
    prefill_width: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Separate registered decode passes from at most one exact prefill."""
    widths = [(tuple(shape), _token_width(shape)) for shape in shapes]
    prefill = [(shape, width) for shape, width in widths if not is_pass(width)]
    if len(prefill) > 1:
        raise RunInvalid(
            f"observed {len(prefill)} candidate prefill calls, expected at "
            f"most one; widths {[width for _, width in prefill]} (two or more)"
        )
    if prefill and prefill[0][1] != prefill_width:
        raise RunInvalid(
            f"the one candidate prefill call has width {prefill[0][1]}, "
            f"expected the registered prefill width {prefill_width}"
        )
    passes = tuple(shape for shape, width in widths if is_pass(width))
    return passes, tuple(shape for shape, _ in prefill)


def assert_exact_count(observed: int, expected: int, cell: str) -> None:
    if observed != expected:
        raise ExactCountMismatch(
            f"{cell}: observed {observed} routed calls, expected {expected}"
        )


def delta_pct(t_a: float, t_b: float) -> float:
    return 100 * (t_a / t_b - 1)


def is_decider(ceiling: float, floor: float) -> bool:
    return ceiling > floor


def clears(delta: float, floor: float) -> bool:
    """Whether a delta beats its floor upward.

    One spelling of the rule every registered outcome shares: a delta equal to
    its floor does NOT clear it. A downward comparison passes the negated
    delta, so the two directions cannot drift apart.
    """
    return delta > floor


def spread_pct(samples: Sequence[float]) -> float:
    if not samples:
        raise RunInvalid("no eligible paired round")
    return machine_spread_pct(samples)


def noise_floor(samples_a: Sequence[float], samples_b: Sequence[float]) -> float:
    return max(spread_pct(samples_a), spread_pct(samples_b))


_MISSING = object()


def first_mismatch(
    left: Sequence[object], right: Sequence[object]
) -> int | None:
    for position, (a, b) in enumerate(
        zip_longest(left, right, fillvalue=_MISSING)
    ):
        if a is _MISSING or b is _MISSING or a != b:
            return position
    return None


def eligible_rounds(
    rounds: Sequence[RoundSample],
    excluded_labels: Collection[str],
    *,
    what: str,
) -> tuple[RoundSample, ...]:
    """Return valid rounds that carry none of this comparison's exclusions."""
    excluded = frozenset(excluded_labels)
    eligible = tuple(
        round_sample
        for round_sample in rounds
        if round_sample.valid and excluded.isdisjoint(round_sample.identity)
    )
    if not eligible:
        raise RunInvalid(f"{what} has no eligible paired round")
    return eligible


def comparison(
    rounds: Sequence[RoundSample],
    numerator: int,
    denominator: int,
    *,
    excluded_labels: Collection[str],
    what: str,
) -> Comparison:
    eligible = eligible_rounds(rounds, excluded_labels, what=what)
    samples_a = [r.generation_tps[numerator] for r in eligible]
    samples_b = [r.generation_tps[denominator] for r in eligible]
    t_a = statistics.median(samples_a)
    t_b = statistics.median(samples_b)
    return Comparison(
        numerator_tps=t_a,
        denominator_tps=t_b,
        delta_pct=delta_pct(t_a, t_b),
        noise_floor_pct=noise_floor(samples_a, samples_b),
        rounds=eligible,
    )


def result_line(fields: dict, outcome: str, result: dict) -> str:
    return "RESULT: " + json.dumps({**fields, "outcome": outcome, **result})
