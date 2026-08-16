"""ADR 0016's detection-price "before" column: the device grid under the PRE-repair member.

ADR 0016 reports what the `factored-groups` repair cost in detection: three
fp16-dequant boundary cells out of 384 flip from caught to not-caught, and the
boundary counts move from 39 / 96 / 96 / 96 to 38 / 95 / 95 / 96 at bits
2 / 3 / 4 / 8. The "after" halves come from the committed
`bench/results/quant_device_adequacy.json` and recompute exactly. The "before"
halves did not: they came from a verbatim pre-repair re-run whose records were
never committed, so they rested on a run that no longer existed. The merge
review of 2026-08-16 caught that, and this script closes it.

WHAT IT DOES. It runs the device harness's own `measure()` unmodified, with one
thing swapped: `ENSEMBLE["factored-groups"]` is replaced by the implementation
from before commit `d3ab30a`, which formed both per-group sums with numpy's
pairwise reduction. Nothing on disk is patched - the swap is a dict entry, made
in this process only - so there is no reverted member to forget to restore, and
the shipped module is never edited. The harness reads `ENSEMBLE.items()` at call
time, which is what makes that sufficient and faithful.

THE CONTINUITY CHECK THAT MAKES THE COMPARISON MEAN ANYTHING. A "before" run is
only comparable to the committed "after" run if nothing else moved between them.
So every record is matched against its committed counterpart and every member
except `factored-groups` must come back bit-identical - the five untouched CPU
members AND the three device members, which are genuinely re-measured on the GPU
here rather than carried through. If any of them moved, this run is measuring
two changes at once and it stops instead of writing.

Needs the Metal GPU, takes the machine lock, about 7 minutes:

    .venv/bin/python -u bench/derive_prerepair_device_records.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from kernelverify.schemas.quant_contract import ENSEMBLE, _f32  # noqa: E402
from machine_state import MeasurementLock  # noqa: E402

AFTER_PATH = ROOT / "bench" / "results" / "quant_device_adequacy.json"
OUT_PATH = ROOT / "bench" / "results" / "quant_device_adequacy_prerepair.json"
REPAIRED = "factored-groups"
# The commit that replaced the pairwise reductions below with explicit chains.
REPAIR_COMMIT = "d3ab30a"
EXIT_LOCK_HELD = 4


def member_factored_groups_prerepair(x: np.ndarray, a) -> np.ndarray:
    """`member_factored_groups` exactly as it stood before `d3ab30a`.

    Both per-group sums are numpy pairwise reductions. On a constant row the
    activation sum is then EXACT - 64 identical fp32 values halve down a
    power-of-two tree with no rounding - which is the over-accuracy the repair
    removed. Kept here verbatim rather than reconstructed, so the "before"
    column measures the member that actually shipped.
    """
    g = a.contract.group_size
    rows, cols = a.q.shape
    xg = _f32(x).reshape(x.shape[0], cols // g, g)
    qg = a.q.reshape(rows, cols // g, g).astype(np.float32)
    xq = np.einsum("bgk,rgk->brg", xg, qg, optimize=True)
    xs = xg.sum(axis=2)
    out = (xq * a.scales.astype(np.float32)[None, :, :]).sum(axis=2)
    out += xs @ a.biases.astype(np.float32).T
    return out.astype(x.dtype)


def identity(record: dict) -> tuple:
    return (record["bits"], record["shape"], record["draw"], record["seed"],
            record["mode"], record["dtype"])


def check_continuity(fresh: list, committed: dict) -> None:
    """Every member but the repaired one must match the committed run exactly.

    This is what licenses reading the two runs as one comparison. The device
    members are included: they are re-measured on the GPU here, so a driver
    change, a compiler change or a different device would show up as a mismatch
    rather than as a moved detection price.
    """
    by_id = {identity(r): r for v in committed["records"].values() for r in v}
    if len(by_id) != len(fresh):
        raise ValueError(f"grid size moved: {len(fresh)} fresh, {len(by_id)} committed")
    for record in fresh:
        other = by_id.get(identity(record))
        if other is None:
            raise ValueError(f"no committed counterpart for {identity(record)}")
        for name, value in record["members"].items():
            if name == REPAIRED:
                continue
            if value != other["members"][name]:
                raise ValueError(
                    f"continuity broken: member {name} moved at {identity(record)} "
                    f"({value} fresh, {other['members'][name]} committed); this run "
                    f"would be measuring two changes at once")
        if record["boundary_fp16_dequant"] != other["boundary_fp16_dequant"]:
            raise ValueError(f"the boundary case itself moved at {identity(record)}")
        if record["base_tol"] != other["base_tol"]:
            raise ValueError(f"base_tol moved at {identity(record)}")


def main() -> int:
    lock = MeasurementLock("derive_prerepair_device_records")
    acquired, detail = lock.acquire()
    if not acquired:
        print(f"REFUSAL (exit {EXIT_LOCK_HELD}): machine measurement lock {detail}")
        return EXIT_LOCK_HELD
    try:
        ENSEMBLE[REPAIRED] = member_factored_groups_prerepair
        import calibrate_quant_device as dev

        committed = json.loads(AFTER_PATH.read_text())
        session = dev.DeviceMemberSession()
        fresh, per_bits = [], {}
        for bits in dev.BITS:
            print(f"  measuring bits={bits} under the pre-repair member ...", flush=True)
            measured = dev.measure(bits, dev.SHAPES, session, False)
            if not measured["bit_exact"]:
                print(f"GATE 0 FAILED at bits={bits}: canonical_quantize is not "
                      f"bit-exact against mx.quantize, so nothing here describes MLX")
                return 1
            per_bits[str(bits)] = measured["records"]
            fresh += measured["records"]

        print(f"\nchecking continuity against {AFTER_PATH.name} "
              f"({len(fresh)} records) ...", flush=True)
        check_continuity(fresh, committed)
        print("every member but the repaired one is bit-identical", flush=True)

        OUT_PATH.write_text(json.dumps({
            "derived_from": {
                "what": "the device grid under the PRE-repair factored-groups member",
                "repair_commit": REPAIR_COMMIT,
                "after_path": "bench/results/quant_device_adequacy.json",
                "after_sha256": hashlib.sha256(AFTER_PATH.read_bytes()).hexdigest(),
                "adr": "ADR 0016",
                "member_swapped": REPAIRED,
                "continuity": ("every other CPU member and every device member "
                               "re-derived bit-identical against the committed "
                               "post-repair run, on all 768 records"),
            },
            "records": per_bits,
        }))
        print(f"\nderived artifact: {OUT_PATH}")
        return 0
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
