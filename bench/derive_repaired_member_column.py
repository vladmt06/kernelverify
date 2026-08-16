"""ADR 0016's derived reading: the `factored-groups` column under the repaired member.

ADR 0016 changed one ensemble member - `factored-groups` now sums each group as
an explicit fp32 chain instead of a pairwise reduction - and every number the ADR
reports about the serving grid follows from re-reading the committed ADR 0013
records with that column recomputed: the ten closed false positives, the
admissible demand of 3.120 that keeps K at 4, the held-out worst of 0.328, and
ADR 0014's four-cell residual collapsing to about 1.

Those numbers had no committed artifact. The device grid's did (`a6b6fa7`, under
this lane's own rule that an ADR's records get preserved), and the merge review
caught the asymmetry. This script closes it.

WHAT IS DERIVED, NOT MEASURED. Nothing here touches the GPU or the measured
records. The evidence is `bench/results/quant_serving_adequacy.json`, hash-checked
on read; the inputs are rebuilt from the harness's own pinned rng stream; the
output holds ONE column plus the identity of each record it belongs to.

EXACTLY WHAT THE CONTINUITY CHECK COVERS, because the distinction is the whole
value of the artifact. The five untouched CPU members are re-derived here and
asserted bit-identical against the evidence, on all 1,536 records rather than on
the 64-record sample `tests/test_quant_contract_members.py` can afford. The three
DEVICE members are NOT re-derived and cannot be: they need a Metal GPU, and
nothing here touches one. Their values are carried through from the evidence
unchanged, which preserves them but verifies nothing about them. A claim that
this file re-derives them would be false, and would be the same shape of hollow
assurance the artifact exists to replace.

MEMORY, and the mistake this docstring exists to stop anyone repeating. The
first version called `ENSEMBLE[name](x, artefact)` per member per record. Each
of those members dequantizes the whole weight matrix itself, so at lm_head one
record cost five independent 1.5 GB dequantizations, and 32 records of churn per
block took the footprint to 29 GB - killed by its own watchdog at block 9 of 48.
The weights are hoisted ONCE per block instead, and the members are evaluated
through the harness's own hoisted entry points (`eval_pairwise`, `eval_lut`,
`eval_serial_chunked`, `eval_factored_serial`, `eval_factored_groups`), which is
exactly why the harness fits its own budget. The fp64 reference at lm_head is a
further 3.1 GB, so the grid is walked one (shape, draw, seed) block at a time and
freed between blocks, with the phys_footprint printed per block. Run it the way
every heavy harness runs: from the coordinator, `python -u`, output to a durable
log, under a footprint watchdog.

    .venv/bin/python -u bench/derive_repaired_member_column.py

Takes about 70 minutes, almost all of it the eight lm_head blocks of the 48, and
peaked at 24.3 GB on the run that produced the committed artifact.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from kernelverify.schemas.quant_contract import (  # noqa: E402
    ENSEMBLE,
    QuantContract,
    canonical_quantize,
    dequantize,
)
from machine_state import MeasurementLock  # noqa: E402
from memory_guard import EXIT_LOCK_HELD  # noqa: E402
from calibrate_quant_serving import (  # noqa: E402
    dequant_chunk_rows,
    dequant_chunked,
    eval_factored_groups,
    eval_factored_serial,
    eval_lut,
    eval_pairwise,
    eval_serial_chunked,
    lut_gather_chunked,
    phys_footprint_gb,
)
from phase0_contract_k import err, make_w, make_x  # noqa: E402

EVIDENCE_PATH = ROOT / "bench" / "results" / "quant_serving_adequacy.json"
EVIDENCE_SHA256 = ("54d0ad5ca4f28f00a401f0481a8b1c69c"
                   "42cadcc01e3c4a21cca7728ced01833")
OUT_PATH = ROOT / "bench" / "results" / "quant_serving_repaired_member.json"

# The grid's pinned axes, in the order calibrate_quant_serving draws them:
# make_w first, then make_x per (batch, mode, dtype) with batch outermost.
BITS = 3
GROUP = 64
GRID_BATCHES = (1, 2, 8, 16)
GRID_MODES = ("unit", "corpus-scale", "near-zero", "constant-rows")
GRID_DTYPES = ("float32", "float16")
REPAIRED = "factored-groups"
# Present in the evidence and copied into nothing: these three are measured on a
# Metal GPU, so this CPU-only derivation can preserve their values but cannot
# check them. Named here so the artifact can say so rather than imply otherwise.
DEVICE_CARRIED = ("device-dequant-loop", "device-dequant-simd",
                  "device-factored-simd")

# The modules a derived value passes through. A change to any of them changes
# the column, so the artifact names their identity the way
# reinterpret_serving_adequacy names the interpretation's.
CODE_PATHS = (
    ROOT / "kernelverify" / "schemas" / "quant_contract.py",
    ROOT / "bench" / "calibrate_quant_serving.py",
    ROOT / "bench" / "phase0_contract_k.py",
    Path(__file__),
)


def code_identity() -> str:
    h = hashlib.sha256()
    for path in CODE_PATHS:
        h.update(path.read_bytes())
    return h.hexdigest()


def load_evidence() -> list:
    raw = EVIDENCE_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EVIDENCE_SHA256:
        raise ValueError(
            f"{EVIDENCE_PATH.name} has changed (sha256 {digest}, pinned "
            f"{EVIDENCE_SHA256}): this column is a reading OF those records")
    return json.loads(raw)["records"]


def block_inputs(shape: str, draw: str, seed: int):
    """The block's artefact and every input the grid draws from it.

    Self-checking by construction: if the rng order were wrong, the untouched
    members below would not come back bit-identical.
    """
    d_out, d_in = (int(v) for v in shape.split("x"))
    rng = np.random.default_rng(10_000 + seed)
    w = make_w(d_out, d_in, draw, rng)
    artefact = canonical_quantize(
        w, QuantContract(scheme="mlx-affine", bits=BITS, group_size=GROUP))
    del w
    xs = {}
    for batch in GRID_BATCHES:
        for mode in GRID_MODES:
            for name in GRID_DTYPES:
                dtype = np.float32 if name == "float32" else np.float16
                xs[(batch, mode, name)] = make_x(batch, d_in, dtype, mode, rng)
    return artefact, xs


def derive(records: list) -> dict:
    blocks: dict[tuple, list[int]] = {}
    for i, r in enumerate(records):
        blocks.setdefault((r["shape"], r["draw"], r["seed"]), []).append(i)

    column = [None] * len(records)
    for n, ((shape, draw, seed), idxs) in enumerate(sorted(blocks.items()), 1):
        artefact, xs = block_inputs(shape, draw, seed)
        # Hoisted once per block, never per record: the whole reason this fits.
        # Built here rather than with the harness's `ArtefactHoists` because that
        # class also builds `w16_32` eagerly for the fp16 boundary, which nothing
        # here evaluates - a spare 1.5 GB at lm_head, against a run whose first
        # version died on exactly that kind of surplus. The evaluators below are
        # the harness's own, so only the allocation list differs.
        w32 = dequant_chunked(artefact, np.float32)
        w_lut32 = lut_gather_chunked(artefact)
        rows, cols = artefact.q.shape
        qg32 = artefact.q.reshape(rows, cols // GROUP, GROUP).astype(np.float32)
        scales32 = artefact.scales.astype(np.float32)
        biases32 = artefact.biases.astype(np.float32)
        chunk_rows = dequant_chunk_rows(cols, w32.itemsize)
        w64 = dequantize(artefact, np.float64)
        # name -> the harness's own hoisted evaluator for that member
        untouched = {
            "dequant-pairwise": lambda x: eval_pairwise(x, w32),
            "lut-gather": lambda x: eval_lut(x, w_lut32),
            "dequant-serial": lambda x: eval_serial_chunked(x, w32, chunk_rows),
            "dequant-reversed": lambda x: eval_serial_chunked(x, w32, chunk_rows,
                                                              reverse=True),
            "factored-serial": lambda x: eval_factored_serial(x, qg32, scales32,
                                                              biases32),
        }
        for i in idxs:
            r = records[i]
            x = xs[(r["batch"], r["mode"], r["dtype"])]
            ref = x.astype(np.float64) @ w64.T
            for name, evaluate in untouched.items():
                if err(evaluate(x), ref) != r["members"][name]:
                    raise ValueError(
                        f"continuity broken: member {name} moved at {shape} "
                        f"B{r['batch']} {draw} s{seed} {r['mode']} {r['dtype']}")
            # the harness's blocked evaluator: bit-identical to the member at
            # every block size, 4x faster at lm_head (ADR 0016)
            column[i] = err(eval_factored_groups(x, qg32, scales32, biases32), ref)
            del ref
        del w64, artefact, xs, w32, w_lut32, qg32, scales32, biases32, untouched
        gc.collect()
        now, peak = phys_footprint_gb()
        print(f"[{n:2d}/{len(blocks)}] {shape:<12} {draw:<13} s{seed:<4} "
              f"footprint {now:5.2f} GB (peak {peak:5.2f})", flush=True)

    assert all(v is not None for v in column)
    return {
        "derived_from": {
            "evidence_path": "bench/results/quant_serving_adequacy.json",
            "evidence_sha256": EVIDENCE_SHA256,
            "code_sha256": code_identity(),
            "adr": "ADR 0016",
            "member": REPAIRED,
            "cpu_members_rederived": sorted(m for m in ENSEMBLE if m != REPAIRED),
            "device_members_carried_through_unverified": sorted(DEVICE_CARRIED),
            "continuity": ("the five untouched CPU members re-derived "
                           "bit-identical on all 1,536 records; the three "
                           "device members carried through from the evidence, "
                           "not re-derived (they need a GPU)"),
        },
        "records": [
            [r["shape"], r["batch"], r["draw"], r["seed"], r["mode"], r["dtype"], v]
            for r, v in zip(records, column)
        ],
    }


def main() -> int:
    # 70 minutes at ~16 GB steady and 24.3 GB peak is a heavy measurement by any
    # reading, so it takes the machine lock like every other one: co-firing with
    # the serving harness is what put this machine into a Jetsam kill three times.
    lock = MeasurementLock("derive_repaired_member_column")
    acquired, detail = lock.acquire()
    if not acquired:
        print(f"REFUSAL (exit {EXIT_LOCK_HELD}): machine measurement lock "
              f"{detail}; one heavy run per machine, and a refused run "
              f"touches nothing")
        return EXIT_LOCK_HELD
    try:
        records = load_evidence()
        print(f"deriving the {REPAIRED} column over {len(records)} committed records")
        derived = derive(records)
        OUT_PATH.write_text(json.dumps(derived))
        print(f"\nderived artifact: {OUT_PATH}")
        return 0
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
