"""Interpretation under ADR 0014: per-cell held-out eligibility, the
both-readings demand table, and the derived reinterpretation of the
committed ADR 0013 evidence.

The measured records never change here; what is under test is what the
interpretation is allowed to conclude from them: an out-of-contract cell is
labelled with its numbers instead of firing the DEMAND MISS branch, an
in-contract miss still fires it, `k_demand` keeps its definition, and the
tolerance-overshoot reading is a second quantity beside it, never a changed
divisor.
"""

import json

import pytest

from calibrate_quant_device import ALL_MEMBERS, HELDOUTS, k_demand
from calibrate_quant_serving import (
    K_SHIP,
    cell_reports,
    demand_table,
    eligibility_gates,
    k_demand_eligible,
    tolerance_overshoot,
)
from kernelverify.schemas.quant_contract import FAULTS

# Exact-arithmetic fixture values: powers of two, so every ratio the tests
# pin is exact rather than approximately equal.
MEMBER_ERR = 2.0 ** -18   # every member; the floor
BASE_TOL = 2.0 ** -20
SHIPPED_TOL = K_SHIP * MEMBER_ERR          # max(base, K*floor) = 2**-16
BT_ERR = 2.0 ** -17       # block-tiled: above base, demand 2, under K_SHIP


def _rec(shape, batch, dtype, mlx=BT_ERR, bt=BT_ERR, seed=0,
         mode="unit", heldout_draw=False, distinct_members=False):
    members = ({name: 1e-5 * (i + 1) for i, name in enumerate(ALL_MEMBERS)}
               if distinct_members
               else {name: MEMBER_ERR for name in ALL_MEMBERS})
    return {
        "shape": shape, "batch": batch, "draw": "normal-0.02", "seed": seed,
        "heldout_draw": heldout_draw, "mode": mode, "dtype": dtype,
        "ref_scale": 1.0, "base_tol": BASE_TOL,
        "members": members,
        "heldout": {"block-tiled": bt, "mlx-on-device": mlx},
        "faults": {name: 1.0 for name in FAULTS},
        "boundary_fp16_dequant": 0.0,
    }


# ---------------------------------------------------------------------------
# k_demand_eligible: the standing quantity wherever nothing is excluded
# ---------------------------------------------------------------------------
def test_eligible_demand_is_bit_identical_to_the_standing_one_in_contract():
    records = [_rec("512x512", 2, "float16", mlx=1.0),
               _rec("512x512", 8, "float32", mlx=1.0, seed=1)]
    needed, binding, withheld = k_demand_eligible(records)
    standing_needed, standing_binding = k_demand(records)
    assert needed == standing_needed
    assert binding == standing_binding
    assert withheld == []


def test_excluded_contribution_is_withheld_with_its_numbers():
    records = [_rec("2560x4096", 1, "float16", mlx=1.0)]
    needed, binding, withheld = k_demand_eligible(records)
    assert needed == BT_ERR / MEMBER_ERR == 2.0
    assert binding.startswith("heldout block-tiled")
    [entry] = withheld
    assert entry["heldout"] == "mlx-on-device"
    assert entry["error"] == 1.0
    assert entry["k_reading"] == 1.0 / MEMBER_ERR == 2.0 ** 18
    assert entry["overshoot"] == 1.0 / SHIPPED_TOL == 2.0 ** 16
    assert entry["adr"] == "ADR 0014"
    assert "affine_qmv" in entry["kernel"]


# ---------------------------------------------------------------------------
# The both-readings quantity: overshoot of the SHIPPED tolerance
# ---------------------------------------------------------------------------
def test_overshoot_divides_by_the_shipped_tolerance_not_the_floor():
    records = [_rec("512x512", 2, "float16", mlx=1.0)]
    worst, binding = tolerance_overshoot(records)
    assert worst == 1.0 / SHIPPED_TOL == 2.0 ** 16
    assert binding.startswith("heldout mlx-on-device")
    needed, _ = k_demand(records)
    assert needed == 1.0 / MEMBER_ERR == 2.0 ** 18
    assert worst == needed / K_SHIP  # same case, two readings, one gap


# ---------------------------------------------------------------------------
# demand_table: an excluded cell is labelled, never a miss; an in-contract
# miss still fires
# ---------------------------------------------------------------------------
def test_excluded_cell_cannot_produce_a_miss():
    records = [_rec("2560x4096", 1, "float16", mlx=1.0),
               _rec("2560x4096", 1, "float16", mlx=1.0, seed=100,
                    heldout_draw=True)]
    table = demand_table(records)
    cell = table["cells"][("2560x4096", 1)]
    assert cell["covered"]
    assert cell["k_needed_calibration"] == 2.0 ** 18  # raw reading preserved
    assert cell["k_eligible_calibration"] == 2.0
    assert len(cell["out_of_contract"]) == 2
    assert any("float16" in label and "mlx-on-device" in label
               for label in cell["excluded_heldouts"])
    assert table["pooled"]["covered"]


def test_in_contract_miss_still_fires_on_the_same_numbers():
    # same magnitude, batch 2: no exclusion applies, the miss is real
    at_b2 = demand_table([_rec("2560x4096", 2, "float16", mlx=1.0)])
    assert not at_b2["cells"][("2560x4096", 2)]["covered"]
    # same cell as the exclusion, but the ADMISSIBLE held-out misses
    bt_miss = demand_table([_rec("2560x4096", 1, "float16", bt=1.0)])
    cell = bt_miss["cells"][("2560x4096", 1)]
    assert not cell["covered"]
    assert cell["k_eligible_binding"].startswith("heldout block-tiled")


def test_b16_cell_is_labelled_even_when_nothing_is_withheld():
    """At B16 the excluded kernel stays under base_tol, so nothing is
    withheld - but the cell still carries the exclusion label, because its
    fp16 half has no admissible device held-out evidence either way."""
    records = [_rec("2560x4096", 16, "float16", mlx=BASE_TOL / 2)]
    cell = demand_table(records)["cells"][("2560x4096", 16)]
    assert cell["covered"]
    assert cell["out_of_contract"] == []
    assert any("affine_qmm_t" in label for label in cell["excluded_heldouts"])


# ---------------------------------------------------------------------------
# G1 under eligibility: a flagged out-of-contract implementation is the
# verifier doing its job, not a false positive
# ---------------------------------------------------------------------------
def test_gates_partition_flags_from_false_positives():
    excluded = [_rec("512x512", 1, "float16", mlx=1.0, distinct_members=True)
                for _ in range(2)]
    report = eligibility_gates(excluded, K_SHIP)
    assert report["false_positives"] == []
    assert len(report["out_of_contract_flags"]) == 2
    assert all("ADR 0014" in line for line in report["out_of_contract_flags"])
    assert report["gates"]["G1_no_false_positives"]
    assert report["verdict"] == "ADEQUATE"

    admissible = [_rec("512x512", 2, "float16", mlx=1.0,
                       distinct_members=True) for _ in range(2)]
    report = eligibility_gates(admissible, K_SHIP)
    assert len(report["false_positives"]) == 2
    assert report["out_of_contract_flags"] == []
    assert not report["gates"]["G1_no_false_positives"]
    assert report["verdict"] == "INADEQUATE"


def test_cell_reports_carry_the_partition_per_cell():
    records = ([_rec("512x512", 1, "float16", mlx=1.0, distinct_members=True)
                for _ in range(2)]
               + [_rec("512x512", 2, "float16", mlx=1.0,
                       distinct_members=True) for _ in range(2)])
    reports = cell_reports(records, K_SHIP)
    b1, b2 = reports[("512x512", 1)], reports[("512x512", 2)]
    assert b1["false_positives"] == [] and len(b1["out_of_contract_flags"]) == 2
    assert b1["verdict"] == "ADEQUATE"
    assert len(b2["false_positives"]) == 2 and b2["out_of_contract_flags"] == []
    assert b2["verdict"] == "INADEQUATE"


# ---------------------------------------------------------------------------
# The derived reinterpretation of the committed evidence (pure CPU re-read)
# ---------------------------------------------------------------------------
from reinterpret_serving_adequacy import (  # noqa: E402
    EVIDENCE_PATH,
    EVIDENCE_SHA256,
    OUT_PATH,
    interpretation_identity,
    load_evidence,
    main,
    reinterpret,
)


@pytest.fixture(scope="module")
def derived():
    return reinterpret(load_evidence())


def test_evidence_is_the_committed_record_and_hash_checked():
    assert EVIDENCE_PATH.name == "quant_serving_adequacy.json"
    assert EVIDENCE_SHA256 == ("54d0ad5ca4f28f00a401f0481a8b1c69c"
                               "42cadcc01e3c4a21cca7728ced01833")
    load_evidence()  # raises on any byte drift


def test_derived_header_names_its_source_code_and_ruling(derived):
    header = derived["derived_from"]
    assert header["evidence_sha256"] == EVIDENCE_SHA256
    assert header["interpretation_code_sha256"] == interpretation_identity()
    assert header["eligibility_version"] == 1
    assert header["adr"] == "ADR 0014"
    assert header["mlx_version_ruled"] == "0.32.0"
    assert len(header["mlx_kernel_source_sha256"]) == 64


def test_no_demand_miss_outside_excluded_labelled_cells(derived):
    """The acceptance's honest-report branch: withholding the out-of-contract
    held-out collapses the B1 demands from 78-170 to 2.2-6.97, and what
    remains above K = 4 is a REAL in-contract demand - the ensemble member
    device-factored-simd at fp32 constant-rows, previously shadowed by the
    out-of-contract readings - reported, never absorbed into the label."""
    # the six B1 cells hold every withheld contribution, with numbers
    withheld_cells = {key for key, cell in derived["demands"]["cells"].items()
                      if cell["out_of_contract"]}
    assert withheld_cells == {f"{s}|B1" for s in (
        "1024x2560", "151936x2560", "2560x4096", "2560x9728",
        "4096x2560", "9728x2560")}
    for key in withheld_cells:
        for entry in derived["demands"]["cells"][key]["out_of_contract"]:
            assert entry["heldout"] == "mlx-on-device"
            assert entry["dtype"] == "float16"
            assert entry["error"] > 0 and entry["overshoot"] > 0
    assert sum(len(derived["demands"]["cells"][key]["out_of_contract"])
               for key in withheld_cells) == 28
    # the residual misses are in-contract member demands, all inside
    # labelled cells; no miss exists outside a labelled cell
    assert derived["misses"] == ["1024x2560|B1", "2560x4096|B1",
                                 "4096x2560|B1", "9728x2560|B1"]
    assert set(derived["misses"]) <= withheld_cells
    for key in derived["misses"]:
        cell = derived["demands"]["cells"][key]
        assert not cell["covered"]
        worst = max(cell["k_eligible_calibration"],
                    cell["k_eligible_independent"])
        binding = (cell["k_eligible_binding"]
                   if cell["k_eligible_calibration"] >= worst
                   else cell["k_eligible_binding_independent"])
        assert binding.startswith("member device-factored-simd")
        assert "float32" in binding
    for key, cell in derived["demands"]["cells"].items():
        if key not in derived["misses"]:
            assert cell["covered"], key


def test_raw_k_demand_still_reads_exactly_the_adr_0013_numbers(derived):
    pooled = derived["demands"]["pooled"]
    assert round(pooled["k_needed_calibration"], 3) == 170.510
    assert round(pooled["k_needed_independent"], 3) == 195.198
    assert pooled["k_binding_independent"].startswith(
        "heldout mlx-on-device @ 2560x4096")


def test_overshoot_reads_the_true_multiple_of_the_shipped_tolerance(derived):
    """The investigation's decomposition: the 195.2 demand at the pooled
    binding case is a 12.43x overshoot of the tolerance actually shipped."""
    pooled = derived["demands"]["pooled"]
    assert round(pooled["overshoot_independent"], 2) == 12.43
    assert pooled["overshoot_independent"] < pooled["k_needed_independent"] / 10


def test_g1_flags_are_relabelled_not_erased(derived):
    expected_flags = {"1024x2560": 2, "151936x2560": 4, "2560x4096": 8,
                      "2560x9728": 6, "4096x2560": 2, "9728x2560": 6}
    for shape, count in expected_flags.items():
        cell = derived["cells"][f"{shape}|B1"]
        assert cell["false_positives"] == []
        assert len(cell["out_of_contract_flags"]) == count
        assert cell["verdict"] == "ADEQUATE"
    for key, cell in derived["cells"].items():
        assert cell["false_positives"] == [], key
        assert cell["verdict"] == "ADEQUATE", key
    assert derived["pooled_gates"]["verdict"] == "ADEQUATE"
    assert len(derived["pooled_gates"]["out_of_contract_flags"]) == 28


def test_in_contract_cells_are_bit_identical_to_the_original_tables(derived):
    """The re-interpretation invariance rule: only labelling differs."""
    inv = derived["invariance"]
    assert inv["ok"], inv
    assert inv["relabelled_cells"] == [f"{s}|B1" for s in (
        "1024x2560", "151936x2560", "2560x4096", "2560x9728",
        "4096x2560", "9728x2560")]


def test_derived_artifact_is_written_beside_the_evidence_never_into_it(
        tmp_path, derived):
    out = tmp_path / "derived.json"
    assert main(out_path=out) == 0
    on_disk = json.loads(out.read_text())
    assert on_disk["derived_from"]["evidence_sha256"] == EVIDENCE_SHA256
    assert on_disk["misses"] == derived["misses"]
    assert "IN-CONTRACT DEMAND" in on_disk["verdict"]
    assert OUT_PATH != EVIDENCE_PATH and OUT_PATH.parent.name == "results"
