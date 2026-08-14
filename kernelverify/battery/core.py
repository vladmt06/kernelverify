"""The battery core: input modes, the case space, and the verdict table.

Promoted from bench/score_oracles.py (eng review decision 4A) because the
certificate emitter and the pack pipeline depend on this code; bench/ remains
a thin caller. The frozen escape bench does not import anything here.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bench"))

from kernelverify.mutation.catalogue import (  # noqa: E402
    CATALOGUE,
    KERNEL_TO_CORPUS_OP,
    KERNEL_TO_OP,
)
from kernelverify.reference.kernels import KERNELS  # noqa: E402
from kernelverify.schemas.native_ops import K_NATIVE, K_QUANT, NATIVE_OPS  # noqa: E402
from kernelverify.tolerance.contract import CONTRACT_VERSION  # noqa: E402
from kernelverify.tolerance.floor import K_ENSEMBLE, conditioned_tolerance  # noqa: E402
from measure_escape import (  # noqa: E402
    DISTRIBUTIONS,
    all_dims,
    benchmark_dims,
    corpus_oracle_passes,
    load_meta,
    make_inputs,
    reference,
)

# The verdict cache stays under bench/.cache: it is measurement data, and
# "deleting bench/.cache/ is the safe full reset" stays true.
CACHE_PATH = _REPO_ROOT / "bench" / ".cache" / "verdicts.pkl"


def _oracle_member_labels() -> list:
    """Every ensemble member's identity, for the verdict-cache fingerprint."""
    from kernelverify.schemas.quant_contract import ENSEMBLE as QUANT_ENSEMBLE
    from kernelverify.tolerance.floor import ENSEMBLES

    labels = []
    for op, members in sorted(ENSEMBLES.items()):
        for fn in members:
            name = getattr(fn, "__name__", None)
            if name is None:  # functools.partial: name the base and the pin
                name = f"{fn.func.__name__}#{sorted(fn.keywords.items())}"
            labels.append(f"floor:{op}:{name}")
    labels += [f"quant:{name}" for name in sorted(QUANT_ENSEMBLE)]
    labels += ["moe:default", "moe:reversed-slots"]
    return labels

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
    # The oracle tag must name the ensemble MEMBERSHIP, not just K: K is the
    # smaller half of the tolerance, and swapping who is in an ensemble moves
    # every floor while a K-only tag stays byte-identical. Found on the phase0
    # worktree (commit f47dc39 there); re-applied here at the function's new
    # home per that branch's merge note.
    fingerprint = (sorted(m.name for m in CATALOGUE) + sorted(INPUT_MODES)
                   + [f"oracle=ensemble-floor-k{K_ENSEMBLE}"]
                   + _oracle_member_labels()
                   + sorted(NATIVE_OPS)
                   + [f"native-k=quant{K_QUANT},native{K_NATIVE}"]
                   # Member labels alone cannot see a semantic change that
                   # keeps every label the same (e.g. re-seeded permutations);
                   # the contract version is bumped exactly for those.
                   + [f"contract={CONTRACT_VERSION}"])

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
    tol_cache: dict = {}  # (op, case) -> the shipped per-case tolerance
    ref_cache: dict = {}  # (op, case) -> fp64 reference, native ops only
    for index, mutation in enumerate(CATALOGUE, 1):
        op = KERNEL_TO_OP[mutation.kernel]
        native = NATIVE_OPS.get(op)
        meta = native.meta if native else load_meta(op)
        if op not in spaces:
            spaces[op] = case_space(meta)
        cases = spaces[op]

        correct = KERNELS[mutation.kernel]
        faulty = mutation.build()
        detected = {}
        for case in cases:
            inputs = make_mode_inputs(meta, case.dim_map, case.dtype, case.seed, case.distribution)
            base_tol = float(meta["tolerances"].get(case.dtype, 1e-5))
            key = (op, case.dims, case.dtype, case.seed, case.distribution)

            if native is not None:
                # Native operators (Tranche 1.5) carry their own ground truth
                # in process: augmented inputs, an fp64 reference, and a
                # tolerance whose K is calibrated per operator family. The
                # reference is cached because it is otherwise recomputed once
                # per mutation over the same case.
                if native.augment is not None:
                    inputs = native.augment(case, inputs)
                if key not in ref_cache:
                    ref_cache[key] = native.reference(inputs)
                ref = ref_cache[key]
            else:
                ref = reference(meta, inputs, key)

            # The shipped oracle (ADR 0004): tolerance floored by what a
            # provably-correct working-precision implementation can deviate on
            # this exact case. It must clear every correct kernel on every
            # case, structured modes included; there is no no-evidence discard
            # any more, so a control failure means the oracle or the port is
            # broken and scoring against it would be meaningless.
            tol_key = (op, case)
            if tol_key not in tol_cache:
                tol_cache[tol_key] = (native.tolerance(case, inputs, ref) if native
                                      else conditioned_tolerance(op, inputs, ref, base_tol))
            tol = tol_cache[tol_key]
            if not corpus_oracle_passes(correct(inputs), ref, tol):
                raise AssertionError(
                    f"the correct {mutation.kernel} fails the shipped tolerance "
                    f"at {case.dim_map} {case.dtype} {case.distribution}"
                )
            detected[case] = not corpus_oracle_passes(faulty(inputs), ref, tol)

        table[mutation.name] = detected
        hits = sum(detected.values())
        print(f"  [{index:>2}/{len(CATALOGUE)}] {mutation.name:<52} "
              f"detectable at {hits}/{len(detected)} cases")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_bytes(pickle.dumps(freeze(table, spaces)))
    return {"table": table, "spaces": spaces}
