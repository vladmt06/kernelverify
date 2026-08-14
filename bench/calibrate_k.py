"""Anchor K to the admissible-implementation contract, instead of fitting it.

What this measures and why
--------------------------
The shipped oracle (ADR 0004) is

    tolerance(case) = max(base_tol, K * ensemble_floor(case)),  K = 1.5

where the floor is the worst deviation of three hand-written correct
implementations. K = 1.5 was the smallest value on a grid that produced no
false positive over a handful of held-out variants. Nothing in that procedure
says what K would have to be to be safe; it says only that 1.5 was not observed
to be unsafe against the few implementations we happened to write.

`kernelverify/tolerance/contract.py` writes the promise down instead: the class
of implementations the verifier undertakes never to flag. This script measures
that class directly. For every case in the full battery it computes

    floor_E = worst deviation over the shipped 3-member ensemble
    floor_C = worst deviation over a sampled population of the contract class
    K_req   = the smallest K for which max(base_tol, K * floor_E) >= floor_C

and reports K_anchored = max K_req over the battery. That number is a measured
property of a stated promise rather than a grid fit, and it is falsifiable in
one direction that matters: if K_anchored > 1.5 then the shipped verifier
rejects an implementation it promised to accept, and the false positive is
exhibited, not hypothesised.

Soundness alone is not the decision rule. Raising K buys freedom from false
positives with detection, so the run also sweeps K against the fault catalogue
and reports what each candidate K absolves - the fp16-score canary above all,
which ADR 0004 pins as the fault no tolerance change may ever excuse.

Running
-------
    .venv/bin/python bench/calibrate_k.py                # ~? min cold, instant warm
    .venv/bin/python bench/calibrate_k.py --n-random 24  # a denser contract sample
    .venv/bin/python bench/calibrate_k.py --rebuild      # ignore the cache

Measurements are cached at `bench/.cache/contract_k.pkl`, fingerprinted by the
contract version, the sample size, the catalogue and the input modes, so any
change to the contract or the fault population forces a rebuild.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernelverify.mutation.catalogue import CATALOGUE, KERNEL_TO_CORPUS_OP  # noqa: E402
from kernelverify.reference.kernels import KERNELS  # noqa: E402
from kernelverify.tolerance.contract import (  # noqa: E402
    CONTRACT_VERSION,
    contract_implementations,
)
from kernelverify.tolerance.floor import (  # noqa: E402
    ENSEMBLE_MEMBERS,
    ENSEMBLES,
    K_ENSEMBLE,
    ensemble_labels,
)
from measure_escape import load_meta, reference  # noqa: E402
from score_oracles import INPUT_MODES, Case, case_space, make_mode_inputs  # noqa: E402

CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "contract_k.pkl"

# Candidate values of K the two-sided sweep reports. 1.5 is what ships.
K_GRID = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0)

# Ensemble sizes the budget sweep reports. 3 is what ships today.
ENSEMBLE_BUDGETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)

# Two independent draws of the contract population. The ensemble is built from
# the first and scored against the second, so no ensemble is ever credited for
# covering the very samples it was drawn from.
SEED_BUILD = 20260814
SEED_HOLDOUT = 71072026

CANARY = "attention[scores_dtype=float16]"

# Structural families, used only to group the report. The contract is stated
# per operator; the families are how ADR 0004 grouped its own transfer factors.
FAMILY = {
    "gelu_triton": "elementwise",
    "silu_triton": "elementwise",
    "leaky_relu_triton": "elementwise",
    "softmax_triton": "row-reduction",
    "rmsnorm_triton": "row-reduction",
    "l2norm_triton": "row-reduction",
    "matmul_triton": "bilinear",
    "attention_triton": "bilinear",
    "flash_attention_triton": "bilinear",
}


def oracle_error(candidate: np.ndarray, ref: np.ndarray) -> float:
    """Max absolute error, or inf where the corpus oracle rejects outright.

    `corpus_oracle_passes` fails a wrong shape or a non-finite output before it
    ever looks at the tolerance, so those cases are recorded as infinite error
    and stay detected at every K.
    """
    if candidate.shape != ref.shape:
        return float("inf")
    a = candidate.astype(np.float64)
    if not np.all(np.isfinite(a)):
        return float("inf")
    diff = np.abs(a - ref.astype(np.float64))
    return float(np.max(diff)) if diff.size else 0.0


def faults_for(op: str) -> list:
    return [m for m in CATALOGUE if KERNEL_TO_CORPUS_OP[m.kernel] == op]


# The ADR 0004 ensemble membership, named through contract members. Verified
# to reproduce the hand-written originals bit for bit, so the baseline row of
# the ADR 0005 tables stays reproducible after floor.py stopped shipping it.
ADR0004_ENSEMBLE = {
    "gelu_triton": ["reference"],
    "silu_triton": ["reference"],
    "leaky_relu_triton": ["reference"],
    "softmax_triton": ["pow2-pad,sum:pairwise", "pow2-pad,sum:sequential"],
    "rmsnorm_triton": ["sum:pairwise", "sum:sequential"],
    "l2norm_triton": ["sum:pairwise", "sum:sequential"],
    "matmul_triton": ["blas", "k-sum:sequential"],
    "attention_triton": ["blas", "sum:sequential", "flash-16"],
    "flash_attention_triton": ["flash-32", "sum:sequential", "flash-16"],
}


def ensemble_tag() -> list[str]:
    """A stable name for every shipped ensemble member.

    ADR 0004 put K into the verdict-cache fingerprint but not the ensemble
    membership, so changing who is in the ensemble changes the floor while
    leaving the tag identical. This calibration fingerprints the membership as
    well, because the whole measurement is about how well that membership
    represents the contract.
    """
    return [f"{op}:{'+'.join(ensemble_labels(op))}" for op in sorted(ENSEMBLE_MEMBERS)]


def measure(n_random: int, rebuild: bool) -> dict:
    """Per-case floors and fault errors for the whole battery.

    Errors are stored, not verdicts, so the K sweep afterwards is pure
    arithmetic: a candidate K never costs another reference call.
    """
    fingerprint = [
        CONTRACT_VERSION,
        f"n_random={n_random}",
        *ensemble_tag(),
        *sorted(m.name for m in CATALOGUE),
        *sorted(INPUT_MODES),
    ]
    if CACHE_PATH.exists() and not rebuild:
        try:
            cached = pickle.loads(CACHE_PATH.read_bytes())
        except Exception:
            cached = {}
        if cached.get("fingerprint") == fingerprint:
            print(f"loading cached measurements from {CACHE_PATH}")
            return cached
        print("contract or catalogue changed since the cache was built; rebuilding")

    records: dict[str, list] = {}
    for op in sorted(ENSEMBLES):
        meta = load_meta(op)
        cases = case_space(meta)
        build_pop = contract_implementations(op, n_random=n_random, seed=SEED_BUILD)
        holdout_pop = contract_implementations(op, n_random=n_random, seed=SEED_HOLDOUT)
        by_label = dict(build_pop)
        legacy = [by_label[label] for label in ADR0004_ENSEMBLE[op]]
        ensemble = ENSEMBLES[op]
        control = ensemble[0]
        faults = faults_for(op)
        rows = []

        def sweep(population, inputs, ref, case):
            """Running max over the population, in ensemble-priority order."""
            running, worst_label, curve = 0.0, "", []
            for label, fn in population:
                err = oracle_error(fn(inputs), ref)
                if not np.isfinite(err):
                    raise AssertionError(
                        f"contract member {label} produced a non-finite output for "
                        f"{op} at {case.dim_map} {case.dtype} {case.distribution}; "
                        "it is not a correct implementation and does not belong "
                        "in the population"
                    )
                if err > running:
                    running, worst_label = err, label
                curve.append(running)
            return running, worst_label, curve

        for case in cases:
            inputs = make_mode_inputs(
                meta, case.dim_map, case.dtype, case.seed, case.distribution
            )
            base_tol = float(meta["tolerances"].get(case.dtype, 1e-5))
            key = (op, case.dims, case.dtype, case.seed, case.distribution)
            ref = reference(meta, inputs, key)

            floor_e = max(oracle_error(fn(inputs), ref) for fn in ensemble)
            floor_legacy = max(oracle_error(fn(inputs), ref) for fn in legacy)
            floor_build, build_label, curve = sweep(build_pop, inputs, ref, case)
            floor_holdout, holdout_label, _ = sweep(holdout_pop, inputs, ref, case)
            rows.append({
                "case": (case.dims, case.dtype, case.distribution, case.seed),
                "base_tol": base_tol,
                "floor_e": floor_e,
                "floor_legacy": floor_legacy,
                "floor_c": max(floor_build, floor_holdout),
                "worst_label": build_label if floor_build >= floor_holdout else holdout_label,
                "floor_holdout": floor_holdout,
                "floor_at": {b: curve[min(b, len(curve)) - 1] for b in ENSEMBLE_BUDGETS},
                "control_err": oracle_error(control(inputs), ref),
                "faults": {m.name: oracle_error(m.build()(inputs), ref) for m in faults},
            })
        records[op] = rows
        binding = sum(1 for r in rows if r["floor_c"] > r["base_tol"])
        print(f"  {op:<24} {len(cases):>4} cases  "
              f"{len(build_pop):>3}+{len(holdout_pop):<3} contract members  "
              f"{binding:>4} cases where the contract floor exceeds base tolerance")

    out = {"fingerprint": fingerprint, "n_random": n_random, "records": records}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_bytes(pickle.dumps(out))
    return out


def k_required(row: dict) -> float:
    """Smallest K with max(base_tol, K * floor_E) >= floor_C for this case.

    Zero when the base tolerance already covers the whole contract class, and
    infinite in the pathological case of a zero ensemble floor under a positive
    contract floor, where no multiple of the floor can ever reach.
    """
    if row["floor_c"] <= row["base_tol"]:
        return 0.0
    if row["floor_e"] <= 0.0:
        return float("inf")
    return row["floor_c"] / row["floor_e"]


def describe(op: str, row: dict) -> str:
    dims, dtype, mode, seed = row["case"]
    shape = " ".join(f"{n}={v}" for n, v in dims)
    return f"{op} {shape} {dtype} {mode} seed={seed}"


# A case counts as near-binding when the contract floor reaches a tenth of the
# base tolerance. Below that the floor is arithmetically irrelevant - the ratio
# between two quantities four orders under the tolerance says nothing about the
# oracle - and ratios there are dominated by cases where the shipped ensemble
# happens to be bit-exact, which makes the ratio infinite and meaningless.
NEAR_BINDING = 0.1


def under_coverage(row: dict) -> float | None:
    """floor_C / floor_E on cases where the floor is close to mattering.

    This is a property of the ensemble membership alone: K_req is the same
    ratio charged only where it costs tolerance, while this is the ratio
    wherever the floor is within an order of magnitude of binding. It is the
    number that says whether K is a safety margin or a patch over an ensemble
    that does not represent the class it stands for. `None` where the case is
    too far under the base tolerance for the ratio to mean anything.
    """
    if row["floor_c"] < NEAR_BINDING * row["base_tol"] or row["floor_e"] <= 0.0:
        return None
    return row["floor_c"] / row["floor_e"]


def report_gate_a(records: dict) -> float:
    """Soundness: does the shipped K cover everything the contract admits?"""
    header = (f"{'operator':<24}{'family':<15}{'cases':>7}{'binding':>9}"
              f"{'max K_req':>11}{'max C/E':>10}{'FP at K=1.5':>13}")
    print("=" * len(header))
    print("GATE A - SOUNDNESS: the K the contract class demands")
    print("binding = cases where the contract floor exceeds the base tolerance,")
    print("          so the value of K decides whether a legal kernel passes")
    print(f"max C/E = worst ratio of contract floor to ensemble floor over the")
    print(f"          near-binding cases (contract floor >= {NEAR_BINDING:g} x base")
    print("          tolerance): how far the shipped ensemble underestimates the")
    print("          class where that underestimate could matter")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    anchored = 0.0
    worst_rows = []
    coverage_rows = []
    total_fp = 0
    for op in sorted(records):
        rows = records[op]
        reqs = [(k_required(r), r) for r in rows]
        binding = [(k, r) for k, r in reqs if k > 0.0]
        top_k, top_row = max(reqs, key=lambda kr: kr[0])
        near = [(under_coverage(r), r) for r in rows if under_coverage(r) is not None]
        fp = sum(1 for r in rows
                 if r["floor_c"] > max(r["base_tol"], K_ENSEMBLE * r["floor_e"]))
        total_fp += fp
        anchored = max(anchored, top_k)
        if top_k > 0.0:
            worst_rows.append((top_k, op, top_row))
        coverage_rows.extend((c, op, r) for c, r in near)
        cell = "-" if top_k == 0.0 else f"{top_k:.3f}"
        c_cell = "-" if not near else f"{max(c for c, _ in near):.3f}"
        print(f"{op:<24}{FAMILY[op]:<15}{len(rows):>7}{len(binding):>9}"
              f"{cell:>11}{c_cell:>10}{fp:>13}")
    print("-" * len(header))
    all_c = "-" if not coverage_rows else f"{max(c for c, _, _ in coverage_rows):.3f}"
    print(f"{'ALL':<24}{'':<15}"
          f"{sum(len(r) for r in records.values()):>7}"
          f"{sum(1 for rs in records.values() for r in rs if k_required(r) > 0):>9}"
          f"{anchored:>11.3f}{all_c:>10}{total_fp:>13}")
    print("=" * len(header))

    exact = [(r["floor_c"] / r["base_tol"], op, r)
             for op, rows in records.items() for r in rows
             if r["floor_e"] <= 0.0 < r["floor_c"]]
    print()
    print(f"{len(exact)} cases where the shipped ensemble is bit-exact against the "
          f"fp64 reference while some legal")
    print("implementation is not, so no multiple of the ensemble floor could ever "
          "cover the class.")
    if exact:
        worst_ratio, worst_op, worst_case = max(exact, key=lambda t: t[0])
        print(f"They are harmless only because the base tolerance dominates: the "
              f"worst sits at {worst_ratio:.2g}x")
        print(f"the base tolerance ({describe(worst_op, worst_case)}).")

    if worst_rows:
        print()
        print("the cases that set each operator's K_req (the binding cases):")
        for k, op, row in sorted(worst_rows, reverse=True):
            print(f"  K_req={k:6.3f}  floor_E={row['floor_e']:9.3g} "
                  f"floor_C={row['floor_c']:9.3g} base={row['base_tol']:8.3g}")
            print(f"                worst member: {row['worst_label']}")
            print(f"                {describe(op, row)}")

    if coverage_rows:
        print()
        print("the near-binding cases where the ensemble most underestimates the class:")
        for c, op, row in sorted(coverage_rows, key=lambda t: -t[0])[:5]:
            covered = "binding" if row["floor_c"] > row["base_tol"] else "base tolerance covers it"
            print(f"  C/E={c:6.3f}  floor_E={row['floor_e']:9.3g} "
                  f"floor_C={row['floor_c']:9.3g} base={row['base_tol']:8.3g} ({covered})")
            print(f"              worst member: {row['worst_label']}")
            print(f"              {describe(op, row)}")
    return anchored


def shipped_floor(row: dict) -> float:
    return row["floor_e"]


def detection_at(records: dict, k: float, floor_of=shipped_floor) -> dict:
    """Per-fault detection counts under tolerance max(base_tol, k * floor)."""
    counts: dict[str, list[int]] = {}
    for rows in records.values():
        for row in rows:
            tol = max(row["base_tol"], k * floor_of(row))
            for name, err in row["faults"].items():
                hit, total = counts.setdefault(name, [0, 0])
                counts[name] = [hit + (err > tol), total + 1]
    return counts


def control_failures_at(records: dict, k: float, floor_of=shipped_floor) -> int:
    return sum(
        1
        for rows in records.values()
        for row in rows
        if row["control_err"] > max(row["base_tol"], k * floor_of(row))
    )


def report_gate_b(records: dict, anchored: float) -> None:
    """Detection: what does each candidate K absolve?"""
    grid = set(K_GRID)
    if anchored > 0:
        grid.add(round(anchored, 3))
    grid = sorted(grid)
    header = (f"{'K':>7}{'viable faults':>16}{'(case,fault) hits':>20}"
              f"{'canary hits':>14}{'control fails':>15}")
    print()
    print("=" * len(header))
    print("GATE B - DETECTION: what raising K costs")
    print(f"canary = {CANARY}, the precision fault ADR 0004 forbids absolving")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for k in grid:
        counts = detection_at(records, k)
        viable = sum(1 for hits, _ in counts.values() if hits)
        hits = sum(h for h, _ in counts.values())
        canary_hits, canary_total = counts[CANARY]
        mark = "  <- shipped" if k == K_ENSEMBLE else (
            "  <- contract anchor" if abs(k - anchored) < 1e-9 else "")
        print(f"{k:>7.3f}{viable:>16}{hits:>20}"
              f"{f'{canary_hits}/{canary_total}':>14}"
              f"{control_failures_at(records, k):>15}{mark}")
    print("=" * len(header))

    shipped = detection_at(records, K_ENSEMBLE)
    target = detection_at(records, max(anchored, K_ENSEMBLE))
    changed = {n: (shipped[n][0], target[n][0]) for n in shipped
               if shipped[n][0] != target[n][0]}
    print()
    if not changed:
        print(f"no fault loses a single detectable case between K={K_ENSEMBLE} "
              f"and K={max(anchored, K_ENSEMBLE):.3f}")
        return
    print(f"faults that lose detectable cases between K={K_ENSEMBLE} and "
          f"K={max(anchored, K_ENSEMBLE):.3f}:")
    for name, (before, after) in sorted(changed.items(), key=lambda kv: kv[1][0] - kv[1][1]):
        note = "  BECOMES UNDETECTABLE" if after == 0 else ""
        print(f"  {name:<52} {before:>5} -> {after:<5}{note}")


def report_gate_c(records: dict) -> None:
    """How big does the ensemble have to be to stand for the class?

    Every row builds the floor from a prefix of the population drawn with
    SEED_BUILD and scores it against the independent SEED_HOLDOUT draw, so an
    ensemble is never credited for covering the samples it was made of. The
    shipped 3-member ensemble is reported on the same held-out basis.
    """
    header = (f"{'ensemble':<18}{'members':>9}{'max K_req':>11}{'max C/E':>10}"
              f"{'FP at K=1.5':>13}{'viable':>8}{'hits':>8}{'canary':>10}"
              f"{'verdicts moved':>16}")
    print()
    print("=" * len(header))
    print("GATE C - ENSEMBLE BUDGET: floors built from one draw, scored on another")
    print("K_req and C/E are measured against the held-out population only.")
    print("verdicts moved = (case, fault) pairs whose detect/miss verdict differs")
    print("                 from the ADR 0004 ensemble's, the exact count that says")
    print("                 whether swapping the ensemble disturbs any published table")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for budget in ("legacy", "shipped") + ENSEMBLE_BUDGETS:
        def floor_of(row):
            if budget == "legacy":
                return row["floor_legacy"]
            if budget == "shipped":
                return row["floor_e"]
            return row["floor_at"][budget]

        reqs, ratios, fp, moved = [], [], 0, 0
        for rows in records.values():
            for row in rows:
                f_e, target = floor_of(row), row["floor_holdout"]
                if target > row["base_tol"]:
                    reqs.append(float("inf") if f_e <= 0 else target / f_e)
                if target >= NEAR_BINDING * row["base_tol"] and f_e > 0:
                    ratios.append(target / f_e)
                fp += target > max(row["base_tol"], K_ENSEMBLE * f_e)
                tol = max(row["base_tol"], K_ENSEMBLE * f_e)
                legacy_tol = max(row["base_tol"], K_ENSEMBLE * row["floor_legacy"])
                moved += sum(1 for err in row["faults"].values()
                             if (err > tol) != (err > legacy_tol))
        counts = detection_at(records, K_ENSEMBLE, floor_of)
        viable = sum(1 for hits, _ in counts.values() if hits)
        hits = sum(h for h, _ in counts.values())
        canary_hits, canary_total = counts[CANARY]
        label = {"legacy": "ADR 0004 ensemble", "shipped": "shipped now"}.get(
            budget, "contract prefix")
        size = {"legacy": f"<={max(len(v) for v in ADR0004_ENSEMBLE.values())}",
                "shipped": f"<={max(len(v) for v in ENSEMBLE_MEMBERS.values())}"}.get(
            budget, str(budget))
        print(f"{label:<18}{size:>9}{max(reqs, default=0.0):>11.3f}"
              f"{max(ratios, default=0.0):>10.3f}{fp:>13}{viable:>8}{hits:>8}"
              f"{f'{canary_hits}/{canary_total}':>10}{moved:>16}")
    print("=" * len(header))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-random", type=int, default=6,
                        help="random contract members per reduction order family")
    parser.add_argument("--rebuild", action="store_true",
                        help="ignore any cached measurements")
    args = parser.parse_args()

    print(f"contract: {CONTRACT_VERSION}, {args.n_random} random members per family")
    print(f"shipped oracle: max(base_tol, {K_ENSEMBLE} * ensemble_floor)")
    print()
    measured = measure(args.n_random, args.rebuild)
    records = measured["records"]
    print()

    anchored = report_gate_a(records)
    report_gate_b(records, anchored)
    report_gate_c(records)

    print()
    if anchored == 0.0:
        print("VERDICT: no case in the battery binds; the base tolerances already "
              "cover the whole contract class, and K is unconstrained by it.")
    elif anchored <= K_ENSEMBLE:
        print(f"VERDICT: the contract class demands K >= {anchored:.3f}; the shipped "
              f"{K_ENSEMBLE} covers it with {K_ENSEMBLE / anchored:.2f}x headroom.")
    else:
        print(f"VERDICT: the contract class demands K >= {anchored:.3f}, above the "
              f"shipped {K_ENSEMBLE}. The shipped verifier rejects implementations "
              f"it promised to accept; the binding cases are listed above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
