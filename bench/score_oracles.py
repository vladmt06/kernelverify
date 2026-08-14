"""Score test policies, not kernels: what fraction of realistic faults does an oracle catch?

`measure_escape.py` asks whether a given bug survives a given oracle. This asks
the question the other way round, which is the one a buyer actually has: my test
harness runs N cases per operator, so what fraction of realistic kernel faults
would it notice?

The machinery lives in `kernelverify/battery/` (core: input modes, case space,
verdict table; policies: how a budget is spent) - promoted out of bench/ so the
certificate emitter and pack pipeline import product code, not a bench script.
This file is the thin caller that runs the scoring report, and it re-exports
the battery names so older imports keep working.

Method
------
1. Synthesise a fault population by walking the seams of the correct kernels
   (`kernelverify/mutation/catalogue.py`). The 10 faults published in the gpuemu
   corpus fall out as a labelled subset, because both are parameterisations of
   the same kernels.
2. Drop equivalent mutants. A fault no case in the entire space can detect is
   not a failure of any policy, and leaving it in deflates every policy alike.
3. Precompute the verdict of every (fault, case) pair once, against the corpus's
   own fp64 references and the shipped conditioning-aware tolerance (ADR 0004).
4. Run each test policy at a fixed budget of kernel evaluations per operator and
   measure what fraction of the viable faults it detects.

Budget is held equal across policies, which is the only comparison that means
anything. The policy decision trail is ADR 0002 (boundary singles), ADR 0003
(pairwise upgrade), ADR 0004 (the oracle the verdicts use).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernelverify.battery import (  # noqa: E402,F401 - re-exported for callers
    BENCHMARK_SEEDS,
    BUDGETS,
    CACHE_PATH,
    CORPUS_MODE,
    IID_MODES,
    INPUT_MODES,
    POLICIES,
    REPEATS,
    SWEEP_SEED,
    UNIT_MODE,
    Case,
    build_verdicts,
    case_space,
    make_mode_inputs,
    policy_boundary_hybrid,
    policy_boundary_pairwise,
    policy_random_all_modes,
    policy_random_both_scales,
    policy_random_schema,
    policy_single_shape,
)
from kernelverify.battery.policies import (  # noqa: E402,F401 - test helpers
    _boundary_features,
    _dim_bounds,
    _is_pow2,
    _tie_break,
)
from kernelverify.mutation.catalogue import CATALOGUE, KERNEL_TO_OP  # noqa: E402
from kernelverify.schemas.native_ops import NATIVE_OPS  # noqa: E402
from measure_escape import load_meta  # noqa: E402


def _meta_for(op: str) -> dict:
    """Schema for an operator, from its native registry or the corpus."""
    native = NATIVE_OPS.get(op)
    return native.meta if native else load_meta(op)


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

    metas = {op: _meta_for(op) for op in set(KERNEL_TO_OP.values())}

    def detection_rate(policy_fn, stochastic, budget, population):
        repeats = REPEATS if stochastic else 1
        totals = []
        for repeat in range(repeats):
            rng = random.Random(1000 + repeat)
            found = 0
            for mutation in population:
                corpus_op = KERNEL_TO_OP[mutation.kernel]
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
                corpus_op = KERNEL_TO_OP[mutation.kernel]
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
