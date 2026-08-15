"""Test policies: how a budget of kernel evaluations is spent.

Promoted from bench/score_oracles.py (eng review decision 4A). The measured
decision trail lives in ADR 0002 (boundary singles) and ADR 0003 (pairwise
upgrade); nothing in any policy references a known fault.
"""

from __future__ import annotations

import itertools
import random

from kernelverify.battery.core import (
    CORPUS_MODE,
    IID_MODES,
    SWEEP_SEED,
    Case,
    benchmark_dims,
)


def policy_single_shape(cases: list[Case], meta: dict, budget: int, rng: random.Random):
    """What a fixed benchmark does: one shape, one dtype, repeated random draws."""
    bench = tuple(sorted(benchmark_dims(meta).items()))
    pool = [c for c in cases
            if c.dims == bench and c.dtype == "float32" and c.distribution == CORPUS_MODE]
    pool.sort(key=lambda c: c.seed)
    return [pool[i % len(pool)] for i in range(budget)]


def policy_random_schema(cases: list[Case], meta: dict, budget: int, rng: random.Random):
    """What the corpus fuzzer does: uniform random pick per dimension, one distribution."""
    pool = [c for c in cases if c.seed == SWEEP_SEED and c.distribution == CORPUS_MODE]
    return [rng.choice(pool) for _ in range(budget)]


def policy_random_both_scales(cases: list[Case], meta: dict, budget: int, rng: random.Random):
    """Random over the schema, allowed to vary the input scale as well."""
    pool = [c for c in cases if c.seed == SWEEP_SEED and c.distribution in IID_MODES]
    return [rng.choice(pool) for _ in range(budget)]


def policy_random_all_modes(cases: list[Case], meta: dict, budget: int, rng: random.Random):
    """Random over everything, structured modes included.

    This is the fair baseline for our policy: it draws from the identical
    enlarged space, so any gap between the two is attributable to the boundary
    core rather than to access to modes the others lack.
    """
    pool = [c for c in cases if c.seed == SWEEP_SEED]
    return [rng.choice(pool) for _ in range(budget)]


def _is_pow2(n: int) -> bool:
    return n >= 2 and (n & (n - 1)) == 0


def _dim_bounds(meta: dict) -> tuple[dict, dict]:
    dims_spec = meta["op_schema"]["dims"]
    dim_min = {d["name"]: min(d["candidates"]) for d in dims_spec}
    dim_max = {d["name"]: max(d["candidates"]) for d in dims_spec}
    return dim_min, dim_max


def _boundary_features(case: Case, dim_min: dict, dim_max: dict) -> set:
    """The boundary vocabulary from ADR 0002: extremes, padding, dtype, mode."""
    feats = {("dtype", case.dtype), ("dist", case.distribution)}
    for name, value in case.dims:
        if value == dim_min[name]:
            feats.add((name, "min"))
        if value == dim_max[name]:
            feats.add((name, "max"))
        if not _is_pow2(value):
            feats.add((name, "nonpow2"))
    return feats


def _tie_break(case: Case, dim_min: dict, dim_max: dict):
    extremes = sum(1 for name, value in case.dims
                   if value in (dim_min[name], dim_max[name]))
    return (-extremes, case.dims, case.dtype, case.distribution)


# The features and pairs are pure functions of (pool, bounds), and the scoring
# harness calls each policy dozens of times per operator over the identical
# pool; the memo makes that rebuild once per operator instead of once per call.
_FEATURE_MEMO: dict = {}


def _op_feature_sets(pool: list[Case], dim_min: dict, dim_max: dict):
    """Per-case boundary features and feature pairs, memoized per operator."""
    key = (tuple(pool),
           tuple(sorted(dim_min.items())), tuple(sorted(dim_max.items())))
    hit = _FEATURE_MEMO.get(key)
    if hit is None:
        feature_sets = {c: _boundary_features(c, dim_min, dim_max) for c in pool}
        pair_sets = {c: {frozenset(p) for p in itertools.combinations(sorted(f), 2)}
                     for c, f in feature_sets.items()}
        hit = _FEATURE_MEMO[key] = (feature_sets, pair_sets)
    return hit


def _greedy_pick(pool: list[Case], gain_sets: dict, covered: set,
                 dim_min: dict, dim_max: dict):
    """One greedy set-cover step: the case adding the most uncovered items,
    ties broken toward cases touching more extremes; None when nothing gains."""
    best, best_gain = None, 0
    for case in pool:
        gain = len(gain_sets[case] - covered)
        if gain > best_gain or (gain == best_gain and gain > 0
                                and _tie_break(case, dim_min, dim_max)
                                < _tie_break(best, dim_min, dim_max)):
            best, best_gain = case, gain
    return best


def policy_boundary_hybrid(cases: list[Case], meta: dict, budget: int, rng: random.Random):
    """Deterministic boundary coverage first, random exploration with the rest.

    The deterministic core targets the boundaries where fault mechanisms live,
    on both ends at once:

    - each dimension's *smallest* candidate, because dilution means a small
      reduction carries the strongest absolute error signal, and degenerate
      sizes like K=1 are where accumulator faults hide;
    - each dimension's *largest* candidate, because tiling faults such as the
      flash-attention rescale need a sequence spanning several tiles before
      they express at all;
    - a non-power-of-two value per dimension, because block padding does not
      exist without one;
    - both dtypes and both input scales, because tolerance width and
      saturation each flipped verdicts in the escape measurement.

    Greedy set cover over those features, ties broken toward cases touching
    more extremes. The core stops when every feature is covered, typically
    after 4 to 6 cases; it never repeats itself. The remaining budget is spent
    on uniform random samples of the whole space, because the first run of
    this harness measured that rare-combination faults reward exploration and
    punish any fixed list. Nothing here refers to any known fault.
    """
    pool = [c for c in cases if c.seed == SWEEP_SEED]
    dim_min, dim_max = _dim_bounds(meta)
    feature_sets, _ = _op_feature_sets(pool, dim_min, dim_max)

    covered: set = set()
    chosen: list[Case] = []
    while len(chosen) < budget:
        best = _greedy_pick(pool, feature_sets, covered, dim_min, dim_max)
        if best is None:  # every feature covered: switch to exploration
            break
        chosen.append(best)
        covered |= feature_sets[best]

    while len(chosen) < budget:
        chosen.append(rng.choice(pool))
    return chosen


def policy_boundary_pairwise(cases: list[Case], meta: dict, budget: int, rng: random.Random):
    """Boundary coverage upgraded from single features to co-occurring pairs.

    ADR 0002 adopted single-feature coverage and pre-registered pairwise as the
    upgrade to take only when a measured miss demanded it. The structured input
    modes produced that miss, twice: the flash-attention running-max fault
    expresses almost only where the adversarial sign pattern meets the largest
    head dimension, and the l2norm epsilon fault where near-zero inputs meet
    the smallest row. Single-feature cover guarantees each feature appears
    somewhere, but nothing puts the two halves of such a pair in the *same*
    case, and at ~11% background density random exploration misses them at
    small budgets.

    The ordering inside the core was itself decided by measurement, because
    pure pair-greedy picks pair-dense cases before basic diversity and fell to
    67% at B=4 with no exploration left. Of the candidate orderings (pure
    pair-greedy; singles then pairs; singles then pairs alternated with random)
    the winner at every budget was: cover single features first, which
    reproduces the ADR 0002 core exactly, then extend to uncovered pairs, and
    only then explore at random. The feature vocabulary is unchanged, so
    nothing here refers to any known fault.
    """
    pool = [c for c in cases if c.seed == SWEEP_SEED]
    dim_min, dim_max = _dim_bounds(meta)
    feature_sets, pair_sets = _op_feature_sets(pool, dim_min, dim_max)

    chosen: list[Case] = []
    covered_singles: set = set()
    while len(chosen) < budget:
        best = _greedy_pick(pool, feature_sets, covered_singles, dim_min, dim_max)
        if best is None:  # every single feature covered: move on to pairs
            break
        chosen.append(best)
        covered_singles |= feature_sets[best]

    covered_pairs: set = set()
    for case in chosen:
        covered_pairs |= pair_sets[case]
    while len(chosen) < budget:
        best = _greedy_pick(pool, pair_sets, covered_pairs, dim_min, dim_max)
        if best is None:  # every achievable pair covered: switch to exploration
            break
        chosen.append(best)
        covered_pairs |= pair_sets[best]

    while len(chosen) < budget:
        chosen.append(rng.choice(pool))
    return chosen


POLICIES = {
    "single shape, fp32 (benchmark)": (policy_single_shape, False),
    "random schema (corpus fuzzer)": (policy_random_schema, True),
    "random schema + both scales": (policy_random_both_scales, True),
    "random schema + all input modes": (policy_random_all_modes, True),
    "boundary singles + random (ADR 0002)": (policy_boundary_hybrid, True),
    "boundary pairs + random (ours)": (policy_boundary_pairwise, True),
}
