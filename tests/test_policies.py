"""Invariants of the battery policies, pinned so a regression is loud.

The properties tested here are the ones ADR 0002/0003 paid for with
measurement: exact budget accounting, a deterministic non-repeating core,
singles covered before pairs, full pair coverage when budget allows, and
reproducibility under a fixed seed.
"""

import random

import pytest

import score_oracles
from measure_escape import load_meta
from score_oracles import (
    POLICIES,
    SWEEP_SEED,
    _boundary_features,
    _dim_bounds,
    case_space,
    policy_boundary_hybrid,
    policy_boundary_pairwise,
)

OPS = ["attention_triton", "softmax_triton", "matmul_triton"]
METAS = {op: load_meta(op) for op in OPS}
SPACES = {op: case_space(METAS[op]) for op in OPS}


def pairs_of(feats):
    feats = sorted(feats)
    return {frozenset((a, b)) for i, a in enumerate(feats) for b in feats[i + 1:]}


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("budget", [1, 4, 16, 33])
@pytest.mark.parametrize("name", list(POLICIES))
def test_budget_is_spent_exactly(op, budget, name):
    fn, _ = POLICIES[name]
    chosen = fn(SPACES[op], METAS[op], budget, random.Random(0))
    assert len(chosen) == budget


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("name", list(POLICIES))
def test_policies_are_reproducible_under_a_seed(op, name):
    fn, _ = POLICIES[name]
    a = fn(SPACES[op], METAS[op], 16, random.Random(5))
    b = fn(SPACES[op], METAS[op], 16, random.Random(5))
    assert a == b


@pytest.mark.parametrize("op", OPS)
def test_sweep_policies_stay_inside_the_sweep_pool(op):
    pool = {c for c in SPACES[op] if c.seed == SWEEP_SEED}
    for name, (fn, _) in POLICIES.items():
        if name.startswith("single shape"):
            continue
        assert set(fn(SPACES[op], METAS[op], 16, random.Random(1))) <= pool


@pytest.mark.parametrize("op", OPS)
def test_deterministic_core_never_repeats(op):
    chosen = policy_boundary_pairwise(SPACES[op], METAS[op], 8, random.Random(0))
    assert len(set(chosen)) == 8


@pytest.mark.parametrize("op", OPS)
def test_pairwise_covers_singles_first_and_matches_the_adr0002_core(op):
    """ADR 0003: the pairwise core's prefix must BE the ADR 0002 singles core."""
    cases, meta = SPACES[op], METAS[op]
    dim_min, dim_max = _dim_bounds(meta)
    pool = [c for c in cases if c.seed == SWEEP_SEED]
    all_feats = set().union(*(_boundary_features(c, dim_min, dim_max) for c in pool))

    chosen = policy_boundary_pairwise(cases, meta, 16, random.Random(0))
    covered = set()
    core_len = None
    for i, case in enumerate(chosen):
        covered |= _boundary_features(case, dim_min, dim_max)
        if covered >= all_feats:
            core_len = i + 1
            break
    assert core_len is not None and core_len <= 6, "singles cover missing or bloated"
    singles = policy_boundary_hybrid(cases, meta, core_len, random.Random(0))
    assert chosen[:core_len] == singles


@pytest.mark.parametrize("op", OPS)
def test_pairwise_reaches_full_pair_coverage_given_budget(op):
    cases, meta = SPACES[op], METAS[op]
    dim_min, dim_max = _dim_bounds(meta)
    pool = [c for c in cases if c.seed == SWEEP_SEED]
    achievable = set().union(
        *(pairs_of(_boundary_features(c, dim_min, dim_max)) for c in pool)
    )
    chosen = policy_boundary_pairwise(cases, meta, len(pool), random.Random(0))
    seen = set()
    prefix = []
    for case in chosen:
        if case in seen:
            break  # first repeat: the deterministic core has certainly ended
        seen.add(case)
        prefix.append(case)
    covered = set().union(
        *(pairs_of(_boundary_features(c, dim_min, dim_max)) for c in prefix)
    )
    assert covered == achievable


# --- the exact-miss report names its policy by lookup ----------------------
#
# AGENTS.md: any 100% claim must be backed by exact miss counts. The report
# under the 100.0% cells counts misses for one policy, so it has to be able to
# find that policy; an inline `name == "..."` comparison against a renamed
# policy evaluates False, leaves the miss table empty, and prints "none, every
# viable fault caught in every run" for a policy it never scored.


def test_ours_names_a_registered_policy():
    assert score_oracles.OURS in POLICIES
    assert score_oracles.ours_policy() == POLICIES[score_oracles.OURS]


def test_a_renamed_policy_raises_rather_than_reporting_zero_misses(monkeypatch):
    renamed = {("boundary pairs + random" if name == score_oracles.OURS else name): entry
               for name, entry in POLICIES.items()}
    monkeypatch.setattr(score_oracles, "POLICIES", renamed)
    with pytest.raises(KeyError, match="boundary pairs"):
        score_oracles.ours_policy()
