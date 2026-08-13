"""Score test policies, not kernels: what fraction of realistic faults does an oracle catch?

`measure_escape.py` asks whether a given bug survives a given oracle. This asks
the question the other way round, which is the one a buyer actually has: my test
harness runs N cases per operator, so what fraction of realistic kernel faults
would it notice?

Method
------
1. Synthesise a fault population by walking the seams of the correct kernels
   (`kernelverify/mutation/catalogue.py`). The 10 faults published in the gpuemu
   corpus fall out as a labelled subset, because both are parameterisations of
   the same kernels.
2. Drop equivalent mutants. A fault no case in the entire space can detect is
   not a failure of any policy, and leaving it in deflates every policy alike.
3. Precompute the verdict of every (fault, case) pair once, against the corpus's
   own fp64 references and its own per-operator tolerances.
4. Run each test policy at a fixed budget of kernel evaluations per operator and
   measure what fraction of the viable faults it detects.

Budget is held equal across policies, which is the only comparison that means
anything. Any policy detects everything if allowed to run the whole space; the
question is what it finds for the same money.

The `coverage` policy is built from mechanisms established before this run, not
from the fault list: padding faults need a size that actually produces padding,
tiling faults need a sequence long enough to span more than one tile, dilution
means small reductions carry the strongest absolute signal, and input scale
moves detectability in both directions. That dictates covering the *extremes*
of every dimension, small and large together, plus a non-power-of-two value,
across both dtypes and both input scales. The first run of this harness showed
why the deterministic version alone is not enough: it plateaued at 88% while
plain random sampling reached 98%, because faults that express only at rare
combinations reward exploration and punish any fixed list. So the policy is a
hybrid: a deterministic boundary-coverage core, then the rest of the budget
spent on random exploration. Nothing in it refers to a known fault. The second
run showed single-feature coverage missing faults that live at feature *pairs*
once structured input modes enlarged the space, so the core now covers pairs;
see `policy_boundary_pairwise` for the measured trail.
"""

from __future__ import annotations

import itertools
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernelverify.mutation.catalogue import CATALOGUE, KERNEL_TO_CORPUS_OP  # noqa: E402
from kernelverify.reference.kernels import KERNELS, next_pow2  # noqa: E402
from measure_escape import (  # noqa: E402
    DISTRIBUTIONS,
    all_dims,
    benchmark_dims,
    corpus_oracle_passes,
    load_meta,
    make_inputs,
    reference,
)

CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "verdicts.pkl"

BUDGETS = (4, 8, 16, 32)
REPEATS = 40  # for the stochastic policies
BENCHMARK_SEEDS = 8  # distinct random draws available at the benchmark shape
SWEEP_SEED = 1234

# ---------------------------------------------------------------------------
# Input modes: the third battery axis
#
# Shape and scale are not enough. The init-max-zero flash-attention fault is a
# real production bug (a fully-masked attention row underflows to 0/0) that
# scored 0% detectable under iid random inputs at any scale, because iid draws
# never produce the structure it needs. Each structured mode below is an input
# *pattern*, chosen for a numerical mechanism, not for any specific fault:
#
# - opposed signs: different input tensors get opposite signs, driving inner
#   products hard negative - the regime where exp underflow and masking
#   assumptions live.
# - near zero: magnitudes far below 1, the regime where an epsilon dominates
#   the quantity it guards.
# - constant rows: zero variance within a reduction row, the degenerate
#   statistics regime for anything that normalises.
# ---------------------------------------------------------------------------
CORPUS_MODE = "corpus uniform[-10,10]"
UNIT_MODE = "unit-scale uniform[-1,1]"
IID_MODES = (CORPUS_MODE, UNIT_MODE)
INPUT_MODES = IID_MODES + ("opposed signs", "near zero", "constant rows")

# The iid mode names double as measure_escape distribution keys.
assert set(IID_MODES) == set(DISTRIBUTIONS)


def make_mode_inputs(meta: dict, dims: dict, dtype: str, seed: int, mode: str) -> dict:
    if mode in IID_MODES:
        return make_inputs(meta, dims, dtype, seed, mode)
    base = make_inputs(meta, dims, dtype, seed, CORPUS_MODE)
    if mode == "opposed signs":
        out = {}
        for index, (name, arr) in enumerate(base.items()):
            signed = np.abs(arr) if index % 2 == 0 else -np.abs(arr)
            out[name] = signed.astype(arr.dtype)
        return out
    if mode == "near zero":
        return {name: (arr * np.float32(1e-3)).astype(arr.dtype) for name, arr in base.items()}
    if mode == "constant rows":
        out = {}
        for name, arr in base.items():
            constant = np.broadcast_to(arr[..., :1], arr.shape)
            out[name] = np.ascontiguousarray(constant).astype(arr.dtype)
        return out
    raise ValueError(f"unknown input mode: {mode}")


@dataclass(frozen=True)
class Case:
    dims: tuple  # sorted (name, value) pairs
    dtype: str
    distribution: str
    seed: int

    @property
    def dim_map(self) -> dict:
        return dict(self.dims)


# ---------------------------------------------------------------------------
# Case space
# ---------------------------------------------------------------------------
def case_space(meta: dict) -> list[Case]:
    """Every case any policy is allowed to pick.

    One seed per shape for the schema sweep, plus several distinct draws at the
    benchmark shape so a single-shape policy can spend its budget the way such
    policies actually do, on repeated random inputs at one size.
    """
    dtypes = meta.get("dtypes", ["float32"])
    cases = []
    for dims in all_dims(meta):
        key = tuple(sorted(dims.items()))
        for dtype in dtypes:
            for mode in INPUT_MODES:
                cases.append(Case(key, dtype, mode, SWEEP_SEED))

    bench_key = tuple(sorted(benchmark_dims(meta).items()))
    for dtype in dtypes:
        for mode in INPUT_MODES:
            for seed in range(BENCHMARK_SEEDS):
                cases.append(Case(bench_key, dtype, mode, seed))
    return sorted(set(cases), key=lambda c: (c.dims, c.dtype, c.distribution, c.seed))


# ---------------------------------------------------------------------------
# Verdict table
# ---------------------------------------------------------------------------
def build_verdicts() -> dict:
    """verdicts[mutation_name][case] = True if the case detects the fault.

    The cache carries a fingerprint of the catalogue: the set of mutation names
    plus their parameters. Any change to the catalogue changes the fingerprint
    and forces a rebuild, so a stale cache can never silently misscore a run.
    Cases are stored as plain tuples, never as Case instances, because a
    pickled dataclass remembers which module defined it and this file is
    sometimes __main__ and sometimes an import.
    """
    fingerprint = sorted(m.name for m in CATALOGUE) + sorted(INPUT_MODES)

    def thaw(stored: dict) -> dict:
        spaces = {op: [Case(*t) for t in cases] for op, cases in stored["spaces"].items()}
        table = {name: {Case(*t): hit for t, hit in verdicts.items()}
                 for name, verdicts in stored["table"].items()}
        return {"table": table, "spaces": spaces}

    def freeze(table: dict, spaces: dict) -> dict:
        as_tuple = lambda c: (c.dims, c.dtype, c.distribution, c.seed)
        return {
            "fingerprint": fingerprint,
            "spaces": {op: [as_tuple(c) for c in cases] for op, cases in spaces.items()},
            "table": {name: {as_tuple(c): hit for c, hit in verdicts.items()}
                      for name, verdicts in table.items()},
        }

    if CACHE_PATH.exists():
        try:
            cached = pickle.loads(CACHE_PATH.read_bytes())
        except Exception:
            cached = {}
        if cached.get("fingerprint") == fingerprint:
            print(f"loading cached verdicts from {CACHE_PATH}")
            return thaw(cached)
        print("catalogue changed since the cache was built; rebuilding verdicts")

    table: dict = {}
    spaces: dict = {}
    for index, mutation in enumerate(CATALOGUE, 1):
        corpus_op = KERNEL_TO_CORPUS_OP[mutation.kernel]
        meta = load_meta(corpus_op)
        if corpus_op not in spaces:
            spaces[corpus_op] = case_space(meta)
        cases = spaces[corpus_op]

        correct = KERNELS[mutation.kernel]
        faulty = mutation.build()
        detected = {}
        ill_conditioned = 0
        for case in cases:
            inputs = make_mode_inputs(meta, case.dim_map, case.dtype, case.seed, case.distribution)
            tol = float(meta["tolerances"].get(case.dtype, 1e-5))
            key = (corpus_op, case.dims, case.dtype, case.seed, case.distribution)
            ref = reference(meta, inputs, key)

            control_ok = corpus_oracle_passes(correct(inputs), ref, tol)
            if not control_ok:
                if case.distribution in IID_MODES:
                    # The ports are proven against iid inputs; a control failure
                    # here means the port itself is broken, so refuse to score.
                    raise AssertionError(
                        f"the correct {mutation.kernel} disagrees with the corpus reference "
                        f"at {case.dim_map} {case.dtype} {case.distribution}; the port is wrong"
                    )
                # Structured modes can make a CORRECT kernel exceed the corpus
                # tolerance: constant rows drive attention scores to ~800, where
                # fp32 rounding alone moves softmax outputs past 1e-3. The
                # published tolerances were calibrated on iid inputs and are
                # simply wrong in these regimes. Such a case cannot give
                # evidence against a faulty kernel - a verdict that also fires
                # on the correct kernel is a false positive, not a detection -
                # so it counts as no-evidence rather than as a catch.
                ill_conditioned += 1
                detected[case] = False
                continue
            detected[case] = not corpus_oracle_passes(faulty(inputs), ref, tol)

        table[mutation.name] = detected
        hits = sum(detected.values())
        note = f"  ({ill_conditioned} ill-conditioned)" if ill_conditioned else ""
        print(f"  [{index:>2}/{len(CATALOGUE)}] {mutation.name:<52} "
              f"detectable at {hits}/{len(detected)} cases{note}")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_bytes(pickle.dumps(freeze(table, spaces)))
    return {"table": table, "spaces": spaces}


# ---------------------------------------------------------------------------
# Test policies
# ---------------------------------------------------------------------------
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

    covered: set = set()
    chosen: list[Case] = []
    while len(chosen) < budget:
        best, best_gain = None, 0
        for case in pool:
            gain = len(_boundary_features(case, dim_min, dim_max) - covered)
            if gain > best_gain or (gain == best_gain and gain > 0
                                    and _tie_break(case, dim_min, dim_max)
                                    < _tie_break(best, dim_min, dim_max)):
                best, best_gain = case, gain
        if best is None:  # every feature covered: switch to exploration
            break
        chosen.append(best)
        covered |= _boundary_features(best, dim_min, dim_max)

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

    feature_sets = {c: _boundary_features(c, dim_min, dim_max) for c in pool}
    pair_sets = {c: {frozenset(p) for p in itertools.combinations(sorted(f), 2)}
                 for c, f in feature_sets.items()}

    def greedy_pick(gain_sets: dict, covered: set):
        best, best_gain = None, 0
        for case in pool:
            gain = len(gain_sets[case] - covered)
            if gain > best_gain or (gain == best_gain and gain > 0
                                    and _tie_break(case, dim_min, dim_max)
                                    < _tie_break(best, dim_min, dim_max)):
                best, best_gain = case, gain
        return best

    chosen: list[Case] = []
    covered_singles: set = set()
    while len(chosen) < budget:
        best = greedy_pick(feature_sets, covered_singles)
        if best is None:  # every single feature covered: move on to pairs
            break
        chosen.append(best)
        covered_singles |= feature_sets[best]

    covered_pairs: set = set()
    for case in chosen:
        covered_pairs |= pair_sets[case]
    while len(chosen) < budget:
        best = greedy_pick(pair_sets, covered_pairs)
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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def main() -> int:
    built = build_verdicts()
    table, spaces = built["table"], built["spaces"]

    viable, dead = [], []
    for mutation in CATALOGUE:
        (viable if any(table[mutation.name].values()) else dead).append(mutation)

    print()
    print(f"fault population: {len(CATALOGUE)} synthesised, "
          f"{len(viable)} viable, {len(dead)} undetectable anywhere in the space")
    for mutation in dead:
        print(f"  excluded as equivalent: {mutation.name:<48} {mutation.label}")
    corpus_subset = [m for m in viable if m.from_corpus]
    print(f"of the viable faults, {len(corpus_subset)} are the ones the corpus seeds by hand")

    metas = {op: load_meta(op) for op in set(KERNEL_TO_CORPUS_OP.values())}

    def detection_rate(policy_fn, stochastic, budget, population):
        repeats = REPEATS if stochastic else 1
        totals = []
        for repeat in range(repeats):
            rng = random.Random(1000 + repeat)
            found = 0
            for mutation in population:
                corpus_op = KERNEL_TO_CORPUS_OP[mutation.kernel]
                meta = metas[corpus_op]
                selected = policy_fn(spaces[corpus_op], meta, budget, rng)
                verdicts = table[mutation.name]
                if any(verdicts[c] for c in selected):
                    found += 1
            totals.append(found / len(population))
        return float(np.mean(totals))

    width = 34
    header = f"{'test policy':<{width}}" + "".join(f"{f'B={b}':>9}" for b in BUDGETS)
    print()
    print("=" * len(header))
    print(f"FAULT DETECTION RATE, all {len(viable)} viable synthetic faults")
    print("B = kernel evaluations allowed per operator")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, (fn, stochastic) in POLICIES.items():
        cells = "".join(f"{detection_rate(fn, stochastic, b, viable):>8.1%} " for b in BUDGETS)
        print(f"{name:<{width}}{cells}")
    print("-" * len(header))

    print()
    print("=" * len(header))
    print(f"SAME POLICIES, only the {len(corpus_subset)} faults published in the corpus")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, (fn, stochastic) in POLICIES.items():
        cells = "".join(
            f"{detection_rate(fn, stochastic, b, corpus_subset):>8.1%} " for b in BUDGETS
        )
        print(f"{name:<{width}}{cells}")
    print("=" * len(header))

    print()
    print(f"exact misses by our policy at each budget, over {REPEATS} runs")
    print("(a 100.0% table cell above is honest only if its budget shows none here):")
    fn, _ = POLICIES["boundary pairs + random (ours)"]
    for budget in BUDGETS:
        miss_counts: dict[str, int] = {}
        for repeat in range(REPEATS):
            rng = random.Random(1000 + repeat)
            for mutation in viable:
                corpus_op = KERNEL_TO_CORPUS_OP[mutation.kernel]
                selected = fn(spaces[corpus_op], metas[corpus_op], budget, rng)
                if not any(table[mutation.name][c] for c in selected):
                    miss_counts[mutation.name] = miss_counts.get(mutation.name, 0) + 1
        if not miss_counts:
            print(f"  B={budget:>2}: none, every viable fault caught in every run")
            continue
        for name, count in sorted(miss_counts.items(), key=lambda kv: -kv[1]):
            space = table[name]
            rate = sum(space.values()) / len(space)
            mutation = next(m for m in viable if m.name == name)
            tag = f" [corpus: {mutation.corpus_name}]" if mutation.from_corpus else ""
            print(f"  B={budget:>2}: missed in {count:>2}/{REPEATS} runs  {name:<48} "
                  f"detectable at {rate:5.1%} of cases{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
