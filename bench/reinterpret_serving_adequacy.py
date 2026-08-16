"""Derive the ADR 0014 reading of the committed ADR 0013 serving evidence.

Pure CPU re-read, no GPU, no measurement lock: the records were measured
once (ADR 0013, exit 1 through the pre-registered DEMAND MISS branch) and
are committed at ``bench/results/quant_serving_adequacy.json``. This script
re-runs the STEP 4/5 interpretation over those records under the ruled
per-cell held-out eligibility (``kernelverify/schemas/heldout_eligibility``)
and the both-readings demand table, and writes a SEPARATE derived artifact
beside the evidence. It never writes into the measured artifact.

The derived artifact's header carries the evidence sha256, a sha256 over the
interpretation code itself, and the eligibility version, so a reader can
tell exactly which code read which evidence under which ruling - the same
identity discipline the harness's checkpoint fingerprint carries for
measured values.

Invariance is checked, not assumed: every in-contract quantity is compared
bit-identically against the tables the measuring run itself computed (the
``demands``, ``cells`` and ``pooled_gates`` blocks inside the evidence).
Only labelling may differ - withheld out-of-contract contributions, the
false-positive/flag partition, and the verdicts that follow from it - and
any numeric drift stops the write with a named difference.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate_quant_serving import (  # noqa: E402
    K_SHIP,
    cell_reports,
    demand_table,
    eligibility_gates,
)
from kernelverify.schemas.heldout_eligibility import (  # noqa: E402
    ELIGIBILITY_VERSION,
    MLX_QUANTIZED_KERNELS_SHA256,
    MLX_VERSION_RULED,
    verify_ruling_basis,
)

EVIDENCE_PATH = Path(__file__).with_name("results") / "quant_serving_adequacy.json"
EVIDENCE_SHA256 = (
    "54d0ad5ca4f28f00a401f0481a8b1c69c42cadcc01e3c4a21cca7728ced01833")
OUT_PATH = Path(__file__).with_name("results") / "quant_serving_reinterpretation.json"

# The modules a derived number passes through; a change to any of them is a
# different interpretation and must read as one in the artifact header.
INTERPRETATION_PATHS = (
    Path(__file__).resolve(),
    Path(__file__).with_name("calibrate_quant_serving.py"),
    Path(__file__).with_name("calibrate_quant_device.py"),
    Path(__file__).with_name("calibrate_quant_bits.py"),
    Path(__file__).resolve().parents[1] / "kernelverify" / "schemas" / "heldout_eligibility.py",
    Path(__file__).resolve().parents[1] / "kernelverify" / "schemas" / "quant_contract.py",
)

# Keys whose whole point is to differ from the pre-ruling tables; everything
# else must be bit-identical to what the measuring run computed.
RELABELLED_DEMAND_KEYS = {"covered"}
RELABELLED_GATE_KEYS = {"false_positives", "gates", "verdict"}


def interpretation_identity() -> str:
    digest = hashlib.sha256()
    for path in INTERPRETATION_PATHS:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_evidence(path: Path = EVIDENCE_PATH) -> dict:
    """The committed record, byte-verified: a derived artifact must never
    claim descent from evidence it did not actually read."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EVIDENCE_SHA256:
        raise RuntimeError(
            f"{path} hashes {digest}, not the recorded {EVIDENCE_SHA256}: "
            f"this is not the ADR 0013 evidence, refusing to reinterpret it")
    return json.loads(data)


def _cell_key(cell: tuple) -> str:
    shape, batch = cell
    return f"{shape}|B{batch}"


def _check_demands(old: dict, new: dict, where: str, problems: list) -> None:
    for key in old:
        if key in RELABELLED_DEMAND_KEYS or key not in new:
            continue
        if new[key] != old[key]:
            problems.append(f"{where}.{key}: {new[key]!r} != recorded {old[key]!r}")
    if not new["out_of_contract"] and new["covered"] != old["covered"]:
        problems.append(f"{where}.covered changed with nothing withheld")


def _check_gates(old: dict, new: dict, where: str, problems: list) -> None:
    for key in old:
        if key in RELABELLED_GATE_KEYS:
            continue
        if new.get(key) != old[key]:
            problems.append(f"{where}.{key} drifted")
    for gate in ("G2_faults_caught", "G3_class_coverage"):
        if new["gates"][gate] != old["gates"][gate]:
            problems.append(f"{where}.gates.{gate} drifted")
    flags = new["out_of_contract_flags"]
    if len(old["false_positives"]) != len(new["false_positives"]) + len(flags):
        problems.append(f"{where}: a held-out verdict was dropped, not relabelled")
    if not flags:
        for key in ("false_positives", "verdict"):
            if new[key] != old[key]:
                problems.append(f"{where}.{key} changed without a flag")
        if new["gates"]["G1_no_false_positives"] != old["gates"]["G1_no_false_positives"]:
            problems.append(f"{where}.gates.G1 changed without a flag")


def reinterpret(evidence: dict) -> dict:
    verify_ruling_basis()  # the exclusions must match the installed MLX
    records = evidence["records"]
    demands_raw = demand_table(records)
    demands = {"cells": {_cell_key(c): v for c, v in demands_raw["cells"].items()},
               "pooled": demands_raw["pooled"]}
    cells = {_cell_key(c): v for c, v in cell_reports(records, K_SHIP).items()}
    pooled_gates = eligibility_gates(records, K_SHIP)
    misses = sorted(key for key, cell in demands["cells"].items()
                    if not cell["covered"])

    problems: list = []
    for key, old in evidence["demands"]["cells"].items():
        _check_demands(old, demands["cells"][key], f"demands.{key}", problems)
    _check_demands(evidence["demands"]["pooled"], demands["pooled"],
                   "demands.pooled", problems)
    for key, old in evidence["cells"].items():
        _check_gates(old, cells[key], f"cells.{key}", problems)
    _check_gates(evidence["pooled_gates"], pooled_gates, "pooled_gates",
                 problems)
    if problems:
        raise RuntimeError(
            "in-contract quantities drifted from the recorded tables; a "
            "reinterpretation may relabel, never renumber:\n  "
            + "\n  ".join(problems))

    relabelled = sorted(key for key, cell in cells.items()
                        if cell["out_of_contract_flags"])
    attested = (not misses
                and all(cell["verdict"] == "ADEQUATE" for cell in cells.values())
                and pooled_gates["verdict"] == "ADEQUATE")
    if attested:
        verdict = ("NO IN-CONTRACT DEMAND MISS: every withheld contribution "
                   "sits in an out-of-contract-labelled cell (ADR 0014) and "
                   "every cell is ADEQUATE under eligibility-aware G1")
    else:
        verdict = (f"IN-CONTRACT DEMAND ABOVE K = {K_SHIP:g} remains at "
                   f"{misses} after the out-of-contract contributions are "
                   f"withheld (each cell's k_eligible_binding names it): "
                   f"the pre-registered membership-before-K renegotiation "
                   f"belongs to the coordinator; nothing is resolved here")
    return {
        "derived_from": {
            "evidence_path": "bench/results/quant_serving_adequacy.json",
            "evidence_sha256": EVIDENCE_SHA256,
            "interpretation_code_sha256": interpretation_identity(),
            "eligibility_version": ELIGIBILITY_VERSION,
            "adr": "ADR 0014",
            "mlx_version_ruled": MLX_VERSION_RULED,
            "mlx_kernel_source_sha256": MLX_QUANTIZED_KERNELS_SHA256,
        },
        "k_ship": K_SHIP,
        "misses": misses,
        "demands": demands,
        "cells": cells,
        "pooled_gates": pooled_gates,
        "invariance": {"ok": True, "relabelled_cells": relabelled,
                       "cells_checked": len(evidence["cells"]) + 1,
                       "demand_cells_checked":
                           len(evidence["demands"]["cells"]) + 1},
        "verdict": verdict,
    }


def main(out_path: Path = OUT_PATH) -> int:
    evidence = load_evidence()
    derived = reinterpret(evidence)
    print(f"evidence: {EVIDENCE_PATH} (sha256 {EVIDENCE_SHA256[:16]}...)")
    print(f"eligibility v{ELIGIBILITY_VERSION}, ruled on mlx "
          f"{MLX_VERSION_RULED}; interpretation code "
          f"{derived['derived_from']['interpretation_code_sha256'][:16]}...")
    print("\nper-cell demand (k_demand | overshoot of shipped tol | "
          "admissible; calibration / independent):")
    for key, d in derived["demands"]["cells"].items():
        if d["out_of_contract"]:
            label = (" [OUT OF CONTRACT: numbers withheld from coverage, "
                     "ADR 0014]")
        elif d["excluded_heldouts"]:
            label = " [fp16 heldout excluded, ADR 0014]"
        else:
            label = ""
        print(f"  {key:>17} {d['k_needed_calibration']:8.3f} / "
              f"{d['k_needed_independent']:8.3f} | "
              f"{d['overshoot_calibration']:6.2f}x / "
              f"{d['overshoot_independent']:6.2f}x | "
              f"{d['k_eligible_calibration']:6.3f} / "
              f"{d['k_eligible_independent']:6.3f}{label}")
    p = derived["demands"]["pooled"]
    print(f"  {'pooled':>17} {p['k_needed_calibration']:8.3f} / "
          f"{p['k_needed_independent']:8.3f} | "
          f"{p['overshoot_calibration']:6.2f}x / "
          f"{p['overshoot_independent']:6.2f}x | "
          f"{p['k_eligible_calibration']:6.3f} / "
          f"{p['k_eligible_independent']:6.3f}")
    flags = derived["pooled_gates"]["out_of_contract_flags"]
    print(f"\nG1: {len(derived['pooled_gates']['false_positives'])} false "
          f"positives; {len(flags)} out-of-contract flags (recorded, "
          f"not gate failures)")
    print(f"relabelled cells: {derived['invariance']['relabelled_cells']}")
    print(f"\nVERDICT: {derived['verdict']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(derived, indent=1, default=float))
    print(f"\nderived artifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
