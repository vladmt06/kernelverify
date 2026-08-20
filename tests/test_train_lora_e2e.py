"""CPU-only tests for the QLoRA end-to-end measurement rules."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import types
from datetime import date

import pytest

import harness_runner as runner
import train_lora_e2e as harness
from memory_guard import (
    EXIT_BUDGET_REFUSAL,
    EXIT_CHILD_DEATH,
    EXIT_LOCK_HELD,
    EXIT_LOW_MEMORY,
    EXIT_NO_DEVICE,
    EXIT_NOT_IDLE,
    EXIT_ORPHANED,
    EXIT_PRECONDITION,
)
from train_lora_e2e import (
    ARMS,
    CELL_BY_NAME,
    ROUNDS,
    ChildRefusal,
    PreconditionFailed,
    RunInvalid,
    build_training_args,
    arm_metrics,
    build_cell_reading,
    child_exit_for_parent,
    clears_floor,
    decide_o1,
    decide_o2,
    decide_o3,
    decide_o4,
    equal_memory_row,
    loss_curve_gate,
    main,
    preflight_inputs,
    pre_registered_outcomes,
    recording_path,
    rotated_arms,
    round_validity,
    run_binding,
    spawn_child,
    time_comparison,
    trace_batch,
    validate_loop_outcome,
    validate_plan,
)


def _batch_trace(step: int, supervised: int = 8) -> dict:
    return {
        "step": step,
        "tokens_sha256": f"tokens-{step}",
        "lengths_sha256": f"lengths-{step}",
        "mask_sha256": f"mask-{step}",
        "supervised_tokens": supervised,
    }


def _fairness(cell: str = "B", *, steps: int = 3) -> dict:
    spec = CELL_BY_NAME[cell]
    return {
        "base_model": {
            "path": "/pinned/qwen3-4b-4bit-g64/model.safetensors",
            "sha256": "model-sha",
        },
        "adapter_init_sha256": "adapter-sha",
        "data": {
            "source_sha256": "examples-sha",
            "batches": [_batch_trace(step) for step in range(1, steps + 1)],
        },
        "optimization": {
            "optimizer": "adam",
            "optimizer_config": {},
            "learning_rate": 1e-5,
            "schedule": None,
            "seeds": {"python": 0, "numpy": 0, "mlx": 0},
            "steps": steps,
            "warmup_steps": 1,
            "batch_size": spec.batch_size,
            "mask_prompt": spec.mask_prompt,
            "max_seq_length": 2048,
            "lora_rank": 8,
            "num_layers": 16,
        },
        "memory_ceiling_gb": 24.0,
        "stack": {
            "mlx": {"version": "0.32.0", "package_sha256": "mlx-sha"},
            "mlx_lm": {
                "version": "0.31.3",
                "package_sha256": "mlx-lm-sha",
            },
        },
    }


def _routing(arm: str) -> dict:
    if arm == "ours":
        return {
            "wrapper_installed": True,
            "forced_stock": False,
            "wrapper_sha256": "wrapper-sha",
            "routed_candidates": ["candidate-sha"],
            "routed_calls": 9,
            "routing_decisions": 9,
        }
    if arm == "control":
        return {
            "wrapper_installed": True,
            "forced_stock": True,
            "wrapper_sha256": "wrapper-sha",
            "routed_candidates": [],
            "routed_calls": 0,
            "routing_decisions": 9,
        }
    return {
        "wrapper_installed": False,
        "forced_stock": False,
        "wrapper_sha256": None,
        "routed_candidates": [],
        "routed_calls": 0,
        "routing_decisions": 0,
    }


def _record(
    arm: str,
    round_number: int,
    *,
    cell: str = "B",
    wall_s: float | None = None,
    step_seconds: tuple[float, ...] = (3.0, 2.0, 2.0),
    peak_gb: float | None = None,
    losses: tuple[float, ...] = (2.0, 1.5, 1.2),
) -> dict:
    arm_wall = {"ours": 9.0, "stock": 10.0, "control": 10.1}
    arm_peak = {"ours": 20.0, "stock": 21.0, "control": 21.0}
    return {
        "cell": cell,
        "arm": arm,
        "round": round_number,
        "completed": True,
        "adapter_written": True,
        "full_job_wall_s": arm_wall[arm] if wall_s is None else wall_s,
        "step_seconds": list(step_seconds),
        "peak_footprint_gb": arm_peak[arm] if peak_gb is None else peak_gb,
        "loss_curve": [
            {"step": step, "loss": loss}
            for step, loss in enumerate(losses, 1)
        ],
        "fairness": _fairness(cell, steps=len(step_seconds)),
        "routing": _routing(arm),
    }


def _round(round_number: int, *, cell: str = "B") -> dict[str, dict]:
    return {
        arm: _record(arm, round_number, cell=cell)
        for arm in ARMS
    }


def _rounds(*, cell: str = "B") -> list[dict[str, dict]]:
    return [_round(number, cell=cell) for number in range(1, ROUNDS + 1)]


def _set_path(record: dict, path: tuple[str, ...], value) -> None:
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


# A fixed starting arm would give one implementation every coolest slot.
def test_arm_rotation_covers_the_registered_five_rounds():
    assert [rotated_arms(index) for index in range(ROUNDS)] == [
        ("ours", "stock", "control"),
        ("stock", "control", "ours"),
        ("control", "ours", "stock"),
        ("ours", "stock", "control"),
        ("stock", "control", "ours"),
    ]


# An incomplete arm set cannot establish any aligned fairness comparison.
def test_round_validity_refuses_a_missing_arm():
    records = _round(1)
    records.pop("control")
    result = round_validity("B", 1, records)
    assert result == {
        "valid": False,
        "reasons": ["expected arms ['ours', 'stock', 'control'], got ['ours', 'stock']"],
    }


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("base_model", "sha256"), "other", "base_model"),
        (("adapter_init_sha256",), "other", "adapter_init_sha256"),
        (("data", "source_sha256"), "other", "data"),
        (("data", "batches"), [_batch_trace(1, 9), _batch_trace(2, 7),
                                _batch_trace(3, 8)], "data"),
        (("optimization", "optimizer"), "sgd", "optimization"),
        (("optimization", "schedule"), {"name": "cosine"}, "optimization"),
        (("optimization", "seeds"), {"python": 1, "numpy": 1, "mlx": 1},
         "optimization"),
        (("optimization", "steps"), 4, "optimization"),
        (("memory_ceiling_gb",), 23.0, "memory_ceiling_gb"),
        (("stack", "mlx", "package_sha256"), "other", "stack"),
        (("stack", "mlx_lm", "package_sha256"), "other", "stack"),
    ],
)
def test_every_fairness_mismatch_invalidates_the_aligned_round(
    path, value, reason
):
    records = _round(1)
    _set_path(records["ours"]["fairness"], path, value)
    result = round_validity("B", 1, records)
    assert result["valid"] is False
    assert any(reason in item for item in result["reasons"])


# Equal totals can hide different masks, so the per-step sequence must match.
def test_supervised_token_order_is_compared_not_only_its_total():
    records = _round(1)
    batches = records["ours"]["fairness"]["data"]["batches"]
    batches[0]["supervised_tokens"] = 7
    batches[1]["supervised_tokens"] = 9
    assert sum(batch["supervised_tokens"] for batch in batches) == 24
    result = round_validity("B", 1, records)
    assert result["valid"] is False
    assert any("data" in item for item in result["reasons"])


@pytest.mark.parametrize(
    ("package", "version"),
    [("mlx", "0.31.0"), ("mlx_lm", "0.31.2")],
)
def test_the_pinned_stack_is_checked_even_when_all_arms_share_the_wrong_version(
    package, version
):
    records = _round(1)
    for record in records.values():
        record["fairness"]["stack"][package]["version"] = version
    result = round_validity("B", 1, records)
    assert result["valid"] is False
    assert any("verified stack" in item for item in result["reasons"])


@pytest.mark.parametrize(
    ("arm", "field", "value", "reason"),
    [
        ("ours", "wrapper_installed", False, "ours wrapper was not installed"),
        ("ours", "forced_stock", True, "ours was forced to stock"),
        ("ours", "routed_calls", 0, "ours routed no verified call"),
        ("stock", "wrapper_installed", True, "stock was not untouched"),
        ("control", "wrapper_installed", False, "control wrapper was not installed"),
        ("control", "forced_stock", False, "control did not force stock"),
        ("control", "routed_calls", 1, "control routed a kernel"),
        ("control", "routing_decisions", 0,
         "control observed no routing decision"),
    ],
)
def test_arm_identity_is_observed_not_inferred(arm, field, value, reason):
    records = _round(1)
    records[arm]["routing"][field] = value
    result = round_validity("B", 1, records)
    assert result["valid"] is False
    assert reason in result["reasons"]


# A different wrapper in arm 3 would not isolate the wrapper's forced-stock cost.
def test_ours_and_control_must_install_the_same_wrapper():
    records = _round(1)
    records["control"]["routing"]["wrapper_sha256"] = "other-wrapper"
    result = round_validity("B", 1, records)
    assert result["valid"] is False
    assert "ours and control installed different wrappers" in result["reasons"]


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        (("base_model", "path"), "base model"),
        (("base_model", "sha256"), "base model"),
        (("adapter_init_sha256",), "adapter"),
        (("data", "source_sha256"), "data source"),
        (("stack", "mlx", "package_sha256"), "verified stack"),
    ],
)
def test_identically_missing_fairness_evidence_never_passes(path, reason):
    records = _round(1)
    for record in records.values():
        _set_path(record["fairness"], path, "")
    result = round_validity("B", 1, records)
    assert result["valid"] is False
    assert any(reason in item for item in result["reasons"])


def test_wrapper_identity_requires_a_hash_not_only_a_boolean():
    records = _round(1)
    records["ours"]["routing"]["wrapper_sha256"] = None
    records["control"]["routing"]["wrapper_sha256"] = None
    result = round_validity("B", 1, records)
    assert result["valid"] is False
    assert "wrapper hash is missing" in result["reasons"]


@pytest.mark.parametrize("field", ["completed", "adapter_written"])
def test_an_incomplete_training_job_invalidates_its_round(field):
    records = _round(1)
    records["stock"][field] = False
    result = round_validity("B", 1, records)
    assert result["valid"] is False
    assert any(field in item for item in result["reasons"])


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("step_seconds", [2.0, 2.0], "step_seconds"),
        ("loss_curve", [], "loss curve"),
        ("peak_footprint_gb", 24.1, "memory ceiling"),
    ],
)
def test_round_refuses_incomplete_steps_or_an_over_budget_peak(
    field, value, reason
):
    records = _round(1)
    records["ours"][field] = value
    result = round_validity("B", 1, records)
    assert result["valid"] is False
    assert any(reason in item for item in result["reasons"])


# Changing a cell's masking mode would compare different training work.
def test_round_validity_checks_the_registered_cell_configuration():
    records = _round(1, cell="C")
    for record in records.values():
        record["fairness"]["optimization"]["mask_prompt"] = True
    result = round_validity("C", 1, records)
    assert result["valid"] is False
    assert any("cell C" in item for item in result["reasons"])


# Price-style time ratios put the second arm over the first, so above one wins.
def test_interval_verdicts_use_the_pricing_probe_orientation():
    win = time_comparison([9.0, 10.0], [11.0, 12.0])
    loss = time_comparison([11.0, 12.0], [9.0, 10.0])
    refused = time_comparison([9.0, 11.0], [10.0, 10.0])
    assert (win["ratio_lo"], win["ratio_hi"], win["verdict"]) == (
        1.1,
        pytest.approx(4 / 3),
        "WIN",
    )
    assert loss["verdict"] == "LOSS"
    assert refused["verdict"] == "REFUSED"


# Equality to either interval boundary cannot claim a direction.
def test_interval_boundary_equal_to_one_is_refused():
    result = time_comparison([10.0, 11.0], [10.0, 11.0])
    assert result["ratio_lo"] < 1.0 < result["ratio_hi"]
    assert result["verdict"] == "REFUSED"


# The noisier arm sets the comparison floor, and equality does not clear it.
def test_noise_floor_is_the_larger_spread_and_is_strict():
    result = time_comparison([100.0, 110.0, 120.0], [100.0, 101.0, 102.0])
    assert result["noise_floor_pct"] == pytest.approx(2000 / 110)
    assert clears_floor(result["noise_floor_pct"], result["noise_floor_pct"]) is False
    assert clears_floor(result["noise_floor_pct"] + 1e-9,
                        result["noise_floor_pct"]) is True


# Full-job timing uses every round, while the secondary excludes absolute round 1.
def test_arm_metrics_drop_round_one_only_from_the_warmed_secondary():
    rounds = _rounds()
    rounds[0]["ours"]["full_job_wall_s"] = 50.0
    rounds[0]["ours"]["step_seconds"] = [30.0, 20.0, 20.0]
    summary = arm_metrics(rounds, "ours")
    assert summary["full_job_samples_s"] == [50.0, 9.0, 9.0, 9.0, 9.0]
    assert summary["full_job_median_s"] == 9.0
    assert summary["warmed_step_samples_s"] == [2.0, 2.0, 2.0, 2.0]
    assert summary["warmed_step_median_s"] == 2.0
    assert summary["peak_footprint_gb"] == 20.0


# A loss outside the stock per-step reproduction envelope voids that cell.
def test_loss_curve_gate_uses_same_seed_stock_reproduction_as_its_envelope():
    rounds = _rounds()
    for index, round_records in enumerate(rounds):
        stock_losses = (2.0, 1.5 + index * 0.01, 1.2)
        round_records["stock"]["loss_curve"] = [
            {"step": step, "loss": loss}
            for step, loss in enumerate(stock_losses, 1)
        ]
        round_records["ours"]["loss_curve"] = copy.deepcopy(
            round_records["stock"]["loss_curve"]
        )
        round_records["control"]["loss_curve"] = copy.deepcopy(
            round_records["stock"]["loss_curve"]
        )
    assert loss_curve_gate(rounds)["valid"] is True

    rounds[2]["ours"]["loss_curve"][1]["loss"] = 1.6
    result = loss_curve_gate(rounds)
    assert result["valid"] is False
    assert result["divergences"] == [
        {"arm": "ours", "round": 3, "step": 2, "loss": 1.6,
         "stock_range": [1.5, 1.54]}
    ]


# Missing loss points make the curves incomparable rather than silently shorter.
def test_loss_curve_gate_refuses_misaligned_steps():
    rounds = _rounds()
    rounds[0]["ours"]["loss_curve"].pop()
    result = loss_curve_gate(rounds)
    assert result["valid"] is False
    assert "loss steps" in result["reasons"][0]


# The shipping floor is strict and memory can independently force NO-GO.
def test_o1_requires_more_than_1_10_and_no_peak_regression():
    boundary = time_comparison([10.0] * ROUNDS, [11.0] * ROUNDS)
    assert decide_o1(boundary, ours_peak_gb=20.0,
                     stock_peak_gb=20.0)["verdict"] == "NO-GO"

    clear = time_comparison([10.0] * ROUNDS, [11.1] * ROUNDS)
    assert decide_o1(clear, ours_peak_gb=20.0,
                     stock_peak_gb=20.0)["verdict"] == "GO"
    memory_loss = decide_o1(clear, ours_peak_gb=20.1, stock_peak_gb=20.0)
    assert memory_loss["verdict"] == "NO-GO"
    assert memory_loss["memory_no_worse"] is False


# The design target is inclusive even though the shipping floor is strict.
def test_o2_accepts_a_lower_endpoint_equal_to_1_30():
    result = time_comparison([10.0] * ROUNDS, [13.0] * ROUNDS)
    assert decide_o2(result)["verdict"] == "AT-TARGET"


# O3 reports the ordinary interval reading, not a post-hoc target threshold.
def test_o3_reports_the_cell_c_interval_verdict():
    result = time_comparison([10.0] * ROUNDS, [10.0] * ROUNDS)
    assert decide_o3(result)["verdict"] == "REFUSED"


def test_one_floor_rule_serves_every_outcome(_amendment_3=None):
    """Amendment 3. These samples used to read GO at O1 and REFUSED at O3,
    which is not two readings of one rule. The floor now lives in the
    comparison, so the endpoints clearing 1.0 is no longer enough on its own
    and every outcome inherits the same answer."""
    noisy = time_comparison([1.0, 100.0], [111.0, 112.0])

    assert noisy["ratio_lo"] > 1.10, "the endpoints alone would have claimed"
    assert abs(noisy["delta_pct"]) < noisy["noise_floor_pct"]
    assert noisy["clears_noise_floor"] is False
    assert noisy["verdict"] == "REFUSED", "a delta inside its floor is noise"

    assert decide_o1(noisy, ours_peak_gb=20.0,
                     stock_peak_gb=20.0)["verdict"] == "NO-GO"
    assert decide_o2(noisy)["verdict"] == "BELOW-TARGET"
    assert decide_o3(noisy)["verdict"] == "REFUSED"


def test_o1_keeps_its_two_registered_conjuncts_on_a_clean_comparison():
    """Narrowing what may be claimed must not make GO unreachable."""
    clean = time_comparison([10.0, 10.1, 10.05], [12.0, 12.1, 12.05])
    assert clean["verdict"] == "WIN" and clean["ratio_lo"] > 1.10
    assert decide_o1(clean, ours_peak_gb=20.0,
                     stock_peak_gb=21.0)["verdict"] == "GO"
    # ... and the peak-footprint conjunct still bites on its own.
    assert decide_o1(clean, ours_peak_gb=22.0,
                     stock_peak_gb=21.0)["verdict"] == "NO-GO"


# Slower forced-stock control is an interface finding only when it clears its floor.
def test_o4_reports_wrapper_cost_at_every_cell_before_o1():
    comparisons = {
        cell: time_comparison([11.0] * ROUNDS, [10.0] * ROUNDS)
        for cell in CELL_BY_NAME
    }
    result = decide_o4(comparisons)
    assert list(result) == ["A", "B", "C", "D"]
    assert all(row["verdict"] == "LOSS" for row in result.values())
    assert all(row["interface_finding"] is True for row in result.values())
    assert result["A"]["wrapper_slowdown_pct"] == pytest.approx(10.0)


# Zero kept kernels is still a complete measurement of the loop's yield.
def test_o5_keeps_zero_and_every_stage_death_in_the_record():
    outcome = validate_loop_outcome({
        "generated": 7,
        "died_by_stage": {"compile": 3, "lint": 2, "tolerance": 2},
        "priced": 0,
        "kept": 0,
        "wall_seconds": 91.5,
    })
    assert outcome == {
        "generated": 7,
        "died_by_stage": {"compile": 3, "lint": 2, "tolerance": 2},
        "priced": 0,
        "kept": 0,
        "wall_seconds": 91.5,
    }


@pytest.mark.parametrize(
    "record",
    [
        {"generated": 1, "died_by_stage": {}, "priced": 1, "kept": 1},
        {"generated": 1, "died_by_stage": {}, "priced": 2, "kept": 1,
         "wall_seconds": 1.0},
        {"generated": 1, "died_by_stage": {}, "priced": 1, "kept": 2,
         "wall_seconds": 1.0},
        {"generated": 1, "died_by_stage": {"compile": 1}, "priced": 1,
         "kept": 1, "wall_seconds": 1.0},
    ],
)
def test_o5_refuses_an_incomplete_or_impossible_census(record):
    with pytest.raises(RunInvalid):
        validate_loop_outcome(record)


# O4 is inserted first, and O3 is never omitted when O1 is quoted.
def test_pre_registered_outcomes_preserve_the_reading_order_and_scope():
    wall = time_comparison([10.0] * ROUNDS, [12.0] * ROUNDS)
    refused = time_comparison([10.0] * ROUNDS, [10.0] * ROUNDS)
    cells = {
        name: {
            "ours_vs_stock": wall if name == "B" else refused,
            "control_vs_stock": refused,
            "peaks_gb": {"ours": 20.0, "stock": 20.0, "control": 20.0},
        }
        for name in CELL_BY_NAME
    }
    loop = validate_loop_outcome({
        "generated": 1,
        "died_by_stage": {},
        "priced": 1,
        "kept": 1,
        "wall_seconds": 4.0,
    })
    result = pre_registered_outcomes(cells, loop)
    assert list(result) == ["O4", "O1", "O2", "O3", "O5", "claim_scope"]
    assert result["O1"]["verdict"] == "GO"
    assert result["O3"]["verdict"] == "REFUSED"
    assert result["claim_scope"] == "prompt-masked instruction tuning"


def test_one_void_cell_does_not_erase_other_registered_outcomes():
    cells = {
        name: build_cell_reading(name, _rounds(cell=name))
        for name in CELL_BY_NAME
    }
    cells["D"] = {
        "cell": "D",
        "binding": False,
        "verdict": "LOSS-DIVERGED",
        "loss_gate": {"reasons": ["diverged"]},
    }
    result = pre_registered_outcomes(cells, _plan()["loop_outcome"])
    assert result["O1"]["verdict"] in {"GO", "NO-GO"}
    assert result["O3"]["verdict"] in {"WIN", "LOSS", "REFUSED"}
    assert result["O4"]["D"]["verdict"] == "REFUSED"
    assert result["O4"]["D"]["interface_finding"] is False


# The primary cell can void too, and section 9 still owes every outcome.
# The void reading is REFUSED rather than LOSS: nothing was measured, so no
# direction may be claimed, and O5 must survive whatever the cells did.
def _void_cell(name, verdict="LOSS-DIVERGED"):
    return {"cell": name, "binding": False, "verdict": verdict,
            "loss_gate": {"reasons": ["diverged"]}}


def test_a_void_primary_cell_is_no_go_rather_than_an_exception():
    cells = {name: build_cell_reading(name, _rounds(cell=name))
             for name in CELL_BY_NAME}
    cells["B"] = _void_cell("B")
    result = pre_registered_outcomes(cells, _plan()["loop_outcome"])

    assert result["O1"]["verdict"] == "NO-GO"
    assert result["O1"]["clears_noise_floor"] is False
    assert result["O2"]["verdict"] == "BELOW-TARGET"
    # O4 is still read first, and O5 is still reported.
    assert list(result)[0] == "O4"
    assert result["O5"]["kept"] == _plan()["loop_outcome"]["kept"]


def test_a_void_secondary_cell_is_refused_and_still_reported():
    cells = {name: build_cell_reading(name, _rounds(cell=name))
             for name in CELL_BY_NAME}
    cells["C"] = _void_cell("C")
    result = pre_registered_outcomes(cells, _plan()["loop_outcome"])

    assert result["O3"]["verdict"] == "REFUSED"
    assert result["O3"]["void"] is True
    assert result["O1"]["verdict"] in {"GO", "NO-GO"}


def test_an_unmeasured_wrapper_cost_is_never_a_finding_against_the_interface():
    """O4 calls a LOSS clear of the floor a finding against the interface. An
    absent measurement is not a loss, and reading one out of it would be
    inventing a verdict."""
    cells = {name: build_cell_reading(name, _rounds(cell=name))
             for name in CELL_BY_NAME}
    cells["A"] = _void_cell("A", verdict="REFUSED")
    result = pre_registered_outcomes(cells, _plan()["loop_outcome"])

    assert result["O4"]["A"]["verdict"] == "REFUSED"
    assert result["O4"]["A"]["interface_finding"] is False
    assert result["O4"]["A"]["wrapper_slowdown_pct"] is None


def test_every_cell_void_still_produces_every_registered_outcome():
    """The worst case the sprint can end in still owes the same record."""
    cells = {name: _void_cell(name) for name in CELL_BY_NAME}
    result = pre_registered_outcomes(cells, _plan()["loop_outcome"])

    assert list(result)[0] == "O4"
    assert set(result) >= {"O1", "O2", "O3", "O4", "O5"}
    assert result["O1"]["verdict"] == "NO-GO"
    assert all(result["O4"][name]["interface_finding"] is False
               for name in CELL_BY_NAME)
    assert result["O5"]["generated"] >= result["O5"]["kept"]


# Section 9 owes O4 at every cell and O3 alongside O1 whatever else happened,
# so a void cell may not delete the readings the other cells did produce.
def test_a_void_cell_does_not_erase_the_other_outcomes_from_the_record():
    """The driver, not the pure function: reporting only O5 because one cell
    voided would throw away three registered readings that were measured."""
    runtime = _Runtime()
    real_run_arm = runtime.run_arm

    def one_bad_cell(plan, provenance, cell, round_number, arm):
        record = real_run_arm(plan, provenance, cell, round_number, arm)
        if cell.name == "D" and arm == "ours" and round_number > 1:
            record["adapter_written"] = False
        return record

    runtime.run_arm = one_bad_cell
    assert run_binding(_plan(), runtime, results_dir="/results") == 1

    [(path, record)] = runtime.writes
    assert record["binding"] is False, "a void cell must not bind"
    assert path.name.endswith(".REFUSED.json")
    # ... and yet every registered outcome is still on the page.
    assert set(record["outcomes"]) >= {"O1", "O2", "O3", "O4", "O5"}
    assert list(record["outcomes"])[0] == "O4"
    assert record["outcomes"]["O4"]["D"]["interface_finding"] is False
    assert record["outcomes"]["O3"]["verdict"] in {"WIN", "LOSS", "REFUSED"}


# O4 registers the distance the interval sits BELOW 1.0, which is not the same
# number as how much longer the wrapper takes, and is always the smaller of the
# two. Using the larger one manufactures findings against the interface.
def test_o4_measures_the_wrapper_against_the_registered_distance():
    """Section 9 registers how far the interval sits BELOW 1.0, which is not
    how much longer the wrapper takes. The second is always the larger, so
    gating on it would report a finding against the interface on a comparison
    the floor rule calls noise."""
    comparison = time_comparison([10.475, 11.0, 11.52], [10.0, 10.0, 10.0])
    below_one = 100 * (1 - comparison["ratio"])
    slowdown = 100 * (1 / comparison["ratio"] - 1)

    assert below_one < comparison["noise_floor_pct"] < slowdown, (
        "the two quantities must straddle the floor for this to be a test")
    assert comparison["verdict"] == "REFUSED"

    result = decide_o4({name: comparison for name in CELL_BY_NAME})["A"]
    assert result["wrapper_below_one_pct"] == pytest.approx(below_one)
    assert result["wrapper_slowdown_pct"] == pytest.approx(slowdown)
    assert result["interface_finding"] is False


def test_o4_still_finds_a_wrapper_cost_that_clears_its_floor():
    """Narrowing what counts must not make the finding unreachable."""
    comparison = time_comparison([12.0, 12.1, 12.05], [10.0, 10.05, 10.02])
    assert comparison["verdict"] == "LOSS"
    result = decide_o4({name: comparison for name in CELL_BY_NAME})["A"]
    assert result["interface_finding"] is True


# One warmed sample is a point, not an interval, and its spread is 0.0.
def test_a_single_warmed_round_refuses_rather_than_reading_a_verdict():
    rounds = _rounds()
    for number in (3, 4, 5):
        rounds[number - 1]["ours"]["adapter_written"] = False
    reading = build_cell_reading("B", rounds)

    assert reading["binding"] is False
    assert reading["eligible_rounds"] == [1, 2]
    assert "fewer than two eligible rounds after the first" in reading["reasons"]


def test_arm_metrics_refuses_a_single_warmed_sample_directly():
    rounds = [_round(1), _round(2)]
    with pytest.raises(RunInvalid, match="after the first"):
        arm_metrics(rounds, "ours")


def test_a_zero_width_warmed_interval_would_have_read_as_a_win():
    """Why the guard above is needed, stated as the number it prevents: one
    sample per arm gives a 0.0% floor and an interval of zero width, so any
    direction at all clears it."""
    comparison = time_comparison([2.0], [2.2])
    assert comparison["noise_floor_pct"] == 0.0
    assert comparison["ratio_lo"] == comparison["ratio_hi"]
    assert comparison["verdict"] == "WIN"
    assert clears_floor(comparison["delta_pct"], comparison["noise_floor_pct"])


# A larger batch is throughput at equal memory and is never relabelled speed.
def test_larger_batch_has_its_own_equal_memory_throughput_row():
    result = equal_memory_row(
        ours={"batch_size": 6, "supervised_tokens": 1200,
              "wall_seconds": 12.0, "peak_footprint_gb": 23.0,
              "memory_ceiling_gb": 24.0},
        stock={"batch_size": 4, "supervised_tokens": 800,
               "wall_seconds": 10.0, "peak_footprint_gb": 22.0,
               "memory_ceiling_gb": 24.0},
    )
    assert result["label"] == "throughput at equal memory"
    assert result["same_batch_speed"] is False
    assert result["ours_tokens_per_s"] == 100.0
    assert result["stock_tokens_per_s"] == 80.0
    assert result["ratio"] == 1.25


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 4),
        ("memory_ceiling_gb", 25.0),
        ("peak_footprint_gb", 25.0),
    ],
)
def test_equal_memory_row_refuses_a_non_comparable_claim(field, value):
    ours = {"batch_size": 6, "supervised_tokens": 1200,
            "wall_seconds": 12.0, "peak_footprint_gb": 23.0,
            "memory_ceiling_gb": 24.0}
    ours[field] = value
    stock = {"batch_size": 4, "supervised_tokens": 800,
             "wall_seconds": 10.0, "peak_footprint_gb": 22.0,
             "memory_ceiling_gb": 24.0}
    with pytest.raises(RunInvalid):
        equal_memory_row(ours=ours, stock=stock)


# Child exits retain shared meanings; unknown failures become child death.
def test_child_refusals_use_the_shared_exit_code_vocabulary():
    assert child_exit_for_parent(0) == 0
    assert child_exit_for_parent(EXIT_BUDGET_REFUSAL) == EXIT_BUDGET_REFUSAL
    assert child_exit_for_parent(EXIT_LOW_MEMORY) == EXIT_LOW_MEMORY
    assert child_exit_for_parent(EXIT_NO_DEVICE) == EXIT_NO_DEVICE
    assert child_exit_for_parent(EXIT_PRECONDITION) == EXIT_PRECONDITION
    assert child_exit_for_parent(EXIT_ORPHANED) == EXIT_ORPHANED
    assert child_exit_for_parent(1) == EXIT_CHILD_DEATH
    assert child_exit_for_parent(99) == EXIT_CHILD_DEATH


# The lock and idle refusals are parent-only and keep their shared numbers.
def test_parent_refusal_numbers_do_not_collide_with_child_failures():
    assert len({
        EXIT_BUDGET_REFUSAL,
        EXIT_LOCK_HELD,
        EXIT_LOW_MEMORY,
        EXIT_CHILD_DEATH,
        EXIT_NO_DEVICE,
        EXIT_PRECONDITION,
        EXIT_NOT_IDLE,
        EXIT_ORPHANED,
    }) == 8


# A dirty closing gate cannot publish under the canonical binding name.
def test_closing_non_idle_record_is_quarantined():
    clean = recording_path("/results", date(2026, 8, 19), closing_idle=True)
    dirty = recording_path("/results", date(2026, 8, 19), closing_idle=False)
    assert clean.name == "train-lora-e2e-2026-08-19.json"
    assert dirty.name == "train-lora-e2e-2026-08-19.REFUSED.json"


def _plan() -> dict:
    return {
        "schema_version": 1,
        "data": "/fixed/data",
        "steps": 3,
        "warmup_steps": 1,
        "optimizer": "adam",
        "optimizer_config": {},
        "learning_rate": 1e-5,
        "schedule": None,
        "seed": 0,
        "memory_ceiling_gb": 24.0,
        "wall_cap_seconds": 900.0,
        "measurement_module": "measured_kernel",
        "kept_candidates": ["candidate-sha"],
        "loop_outcome": {
            "generated": 1,
            "died_by_stage": {},
            "priced": 1,
            "kept": 1,
            "wall_seconds": 4.0,
        },
        "equal_memory": None,
    }


# Runtime choices not fixed in the research note must arrive as one hashed plan.
def test_validate_plan_accepts_one_complete_fixed_experiment():
    assert validate_plan(_plan()) == _plan()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("steps", 1),
        ("warmup_steps", 3),
        ("optimizer", ""),
        ("learning_rate", 0.0),
        ("seed", True),
        ("memory_ceiling_gb", float("inf")),
        ("wall_cap_seconds", -1.0),
        ("measurement_module", ""),
        ("kept_candidates", []),
    ],
)
def test_validate_plan_refuses_an_unfixed_or_impossible_run(field, value):
    plan = _plan()
    plan[field] = value
    with pytest.raises(RunInvalid):
        validate_plan(plan)


def test_validate_plan_accepts_the_registered_zero_yield_outcome():
    plan = _plan()
    plan["kept_candidates"] = []
    plan["loop_outcome"] = {
        "generated": 4,
        "died_by_stage": {"verify": 4},
        "priced": 0,
        "kept": 0,
        "wall_seconds": 12.0,
    }
    assert validate_plan(plan) == plan


def test_validate_plan_refuses_an_unregistered_larger_batch_search():
    plan = _plan()
    plan["equal_memory"] = {"ours_batch_size": 8, "stock_batch_size": 4}
    with pytest.raises(RunInvalid, match="equal-memory"):
        validate_plan(plan)


# The actual truncated tokens, mask positions and per-step count are hashed.
def test_trace_batch_counts_the_loss_mask_mlx_lm_uses():
    trace = trace_batch(
        [[10, 11, 12, 0, 0], [20, 21, 22, 23, 0]],
        [[2, 3], [0, 4]],
        step=1,
        split="train",
    )
    assert trace["step"] == 1
    assert trace["supervised_tokens"] == 6
    assert all(len(trace[key]) == 64 for key in (
        "tokens_sha256", "lengths_sha256", "mask_sha256"
    ))


# Equal token totals do not excuse a different order or different mask.
def test_trace_batch_fingerprints_order_and_masks_independently():
    first = trace_batch([[1, 2, 3], [4, 5, 6]], [[0, 3], [1, 3]],
                        step=1, split="train")
    reordered = trace_batch([[4, 5, 6], [1, 2, 3]], [[1, 3], [0, 3]],
                            step=1, split="train")
    remasked = trace_batch([[1, 2, 3], [4, 5, 6]], [[2, 3], [0, 3]],
                           step=1, split="train")
    assert first["supervised_tokens"] == reordered["supervised_tokens"]
    assert first["tokens_sha256"] != reordered["tokens_sha256"]
    assert first["tokens_sha256"] == remasked["tokens_sha256"]
    assert first["mask_sha256"] != remasked["mask_sha256"]


# A cell binds only after all five aligned rounds and the loss gate pass.
def test_build_cell_reading_reports_primary_secondary_memory_and_loss():
    result = build_cell_reading("B", _rounds())
    assert result["binding"] is True
    assert result["ours_vs_stock"]["ratio"] == pytest.approx(10 / 9)
    assert result["ours_vs_control"]["ratio"] == pytest.approx(10.1 / 9)
    assert result["control_vs_stock"]["ratio"] == pytest.approx(10 / 10.1)
    assert result["arms"]["ours"]["warmed_step_median_s"] == 2.0
    assert result["arms"]["stock"]["peak_footprint_gb"] == 21.0
    assert result["arms"]["ours"]["loss_curve"]["last_median"] == 1.2


def test_build_cell_reading_excludes_one_invalid_round_from_eligible_samples():
    rounds = _rounds()
    rounds[3]["ours"]["fairness"]["adapter_init_sha256"] = "other"
    result = build_cell_reading("B", rounds)
    assert result["binding"] is True
    assert result["invalid_rounds"] == [4]
    assert len(result["ours_vs_stock"]["samples_first_s"]) == 4


def test_build_cell_reading_refuses_fewer_than_two_eligible_rounds():
    rounds = _rounds()
    for index in range(4):
        rounds[index]["ours"]["fairness"]["adapter_init_sha256"] = (
            f"other-{index}"
        )
    result = build_cell_reading("B", rounds)
    assert result["binding"] is False
    assert result["verdict"] == "REFUSED"
    assert result["eligible_rounds"] == [5]


def test_build_cell_reading_voids_speed_when_the_loss_curve_diverges():
    rounds = _rounds()
    rounds[0]["ours"]["loss_curve"][1]["loss"] = 9.0
    result = build_cell_reading("B", rounds)
    assert result["binding"] is False
    assert result["verdict"] == "LOSS-DIVERGED"
    assert result["loss_gate"]["divergences"]
    assert result["arms"]["ours"]["loss_curve"]["last_median"] == 1.2


def test_build_cell_reading_refuses_uniform_drift_from_the_frozen_plan():
    rounds = _rounds()
    for records in rounds:
        for record in records.values():
            record["fairness"]["optimization"]["optimizer"] = "sgd"
            record["fairness"]["optimization"]["seeds"] = {
                "python": 9,
                "numpy": 9,
                "mlx": 9,
            }
    expected = {
        "base_model": _fairness()["base_model"],
        "data_source_sha256": "examples-sha",
        "optimization": _fairness()["optimization"],
        "memory_ceiling_gb": 24.0,
        "stack": _fairness()["stack"],
    }
    result = build_cell_reading(
        "B", rounds, expected=expected, kept_candidates=["candidate-sha"]
    )
    assert result["binding"] is False
    assert any(
        "frozen plan" in reason
        for check in result["round_validity"]
        for reason in check["reasons"]
    )


def test_build_cell_reading_refuses_a_routed_unkept_candidate():
    rounds = _rounds()
    for records in rounds:
        records["ours"]["routing"]["routed_candidates"] = ["unkept"]
    result = build_cell_reading(
        "B", rounds, kept_candidates=["candidate-sha"]
    )
    assert result["binding"] is False
    assert any(
        "unkept" in reason
        for check in result["round_validity"]
        for reason in check["reasons"]
    )


class _Runtime:
    def __init__(self):
        self.events = []
        self.lock_result = (True, "acquired")
        self.idle_results = [
            {"idle": True, "blockers": []},
            {"idle": True, "blockers": []},
        ]
        self.require_error = None
        self.preflight_error = None
        self.arm_error = None
        self.backward_error = None
        self.backward = {"exact": True, "cases": 4,
                         "candidate_sha256": "candidate-sha"}
        self.writes = []

    def acquire_lock(self):
        self.events.append("lock")
        return self.lock_result

    def release_lock(self):
        self.events.append("release")

    def fingerprint(self):
        self.events.append("fingerprint")
        return {"cores": 12, "chip": "Apple M3 Pro"}

    def idle_check(self, cores):
        self.events.append("idle")
        return self.idle_results.pop(0)

    def require_memory(self, budget, cell):
        self.events.append(f"memory:{cell}")
        if self.require_error is not None:
            raise self.require_error

    def preflight(self, plan):
        self.events.append("preflight")
        if self.preflight_error is not None:
            raise self.preflight_error
        return {"model_sha256": "model-sha", "data_sha256": "examples-sha",
                "plan_sha256": "plan-sha"}

    def verify_backward(self, plan, provenance):
        self.events.append("backward")
        if self.backward_error is not None:
            raise self.backward_error
        return self.backward

    def run_arm(self, plan, provenance, cell, round_number, arm):
        self.events.append(f"run:{cell.name}:{round_number}:{arm}")
        if self.arm_error is not None:
            raise self.arm_error
        return _record(arm, round_number, cell=cell.name)

    def write_record(self, path, record):
        self.events.append("write")
        self.writes.append((path, record))

    def today(self):
        return date(2026, 8, 19)


# The lock must refuse before even the machine's idle state is sampled.
def test_binding_driver_refuses_a_held_lock_first(capsys):
    runtime = _Runtime()
    runtime.lock_result = (False, "held by test")
    assert run_binding(_plan(), runtime, results_dir="/results") == EXIT_LOCK_HELD
    assert runtime.events == ["lock"]
    assert '"O5"' in capsys.readouterr().out


# Opening non-idle is retryable and no model preflight may run first.
def test_binding_driver_refuses_opening_non_idle_before_model_work():
    runtime = _Runtime()
    runtime.idle_results[0] = {"idle": False, "blockers": ["busy"]}
    assert run_binding(_plan(), runtime, results_dir="/results") == EXIT_NOT_IDLE
    assert "preflight" not in runtime.events
    assert runtime.events[-1] == "release"


def test_binding_driver_uses_the_shared_low_memory_refusal():
    from memory_guard import LowMemoryRefusal

    runtime = _Runtime()
    runtime.require_error = LowMemoryRefusal("startup", 2.0, 24.0)
    assert run_binding(_plan(), runtime, results_dir="/results") == EXIT_LOW_MEMORY
    assert "preflight" not in runtime.events


def test_binding_driver_uses_the_shared_budget_refusal():
    from memory_guard import BudgetExceeded

    runtime = _Runtime()
    runtime.require_error = BudgetExceeded("startup", 25.0, 24.0)
    assert run_binding(_plan(), runtime, results_dir="/results") == (
        EXIT_BUDGET_REFUSAL
    )
    assert "preflight" not in runtime.events


def test_binding_driver_refuses_a_permanent_precondition_before_training():
    runtime = _Runtime()
    runtime.preflight_error = PreconditionFailed("model hash changed")
    assert run_binding(_plan(), runtime, results_dir="/results") == EXIT_PRECONDITION
    assert not any(event.startswith("run:") for event in runtime.events)


# Backward equality is a gate, not another fairness field to waive later.
def test_binding_driver_stops_before_training_on_a_gradient_mismatch():
    runtime = _Runtime()
    runtime.backward = {"exact": False, "cases": 4,
                        "candidate_sha256": "candidate-sha"}
    assert run_binding(_plan(), runtime, results_dir="/results") == 1
    assert not any(event.startswith("run:") for event in runtime.events)
    assert runtime.writes == []


@pytest.mark.parametrize(
    ("child_code", "parent_code"),
    [
        (EXIT_BUDGET_REFUSAL, EXIT_BUDGET_REFUSAL),
        (EXIT_LOW_MEMORY, EXIT_LOW_MEMORY),
        (EXIT_NO_DEVICE, EXIT_NO_DEVICE),
        (EXIT_PRECONDITION, EXIT_PRECONDITION),
        (EXIT_ORPHANED, EXIT_ORPHANED),
        (99, EXIT_CHILD_DEATH),
    ],
)
def test_backward_child_refusals_use_the_shared_parent_code(
    child_code, parent_code
):
    runtime = _Runtime()
    runtime.backward_error = ChildRefusal(
        "backward exactness", child_code, "stopped"
    )
    assert run_binding(_plan(), runtime, results_dir="/results") == parent_code
    assert not any(event.startswith("run:") for event in runtime.events)
    assert runtime.writes == []


@pytest.mark.parametrize(
    ("child_code", "parent_code"),
    [
        (EXIT_BUDGET_REFUSAL, EXIT_BUDGET_REFUSAL),
        (EXIT_LOW_MEMORY, EXIT_LOW_MEMORY),
        (EXIT_NO_DEVICE, EXIT_NO_DEVICE),
        (EXIT_PRECONDITION, EXIT_PRECONDITION),
        (EXIT_ORPHANED, EXIT_ORPHANED),
        (99, EXIT_CHILD_DEATH),
    ],
)
def test_binding_driver_propagates_each_child_refusal(child_code, parent_code):
    runtime = _Runtime()
    runtime.arm_error = ChildRefusal("B round 1 ours", child_code, "stopped")
    assert run_binding(_plan(), runtime, results_dir="/results") == parent_code
    assert runtime.writes == []


# Closing contention preserves evidence only under the quarantine name.
def test_binding_driver_quarantines_a_run_that_goes_non_idle():
    runtime = _Runtime()
    runtime.idle_results[-1] = {"idle": False, "blockers": ["busy"]}
    assert run_binding(_plan(), runtime, results_dir="/results") == 1
    [(path, record)] = runtime.writes
    assert path.name.endswith(".REFUSED.json")
    assert record["binding"] is False
    assert record["outcomes"]["O5"]["kept"] == 1


# GO and NO-GO are both completed measurements, so both exit zero when binding.
def test_binding_driver_writes_one_binding_record_after_all_sixty_children(
    capsys,
):
    runtime = _Runtime()
    assert run_binding(_plan(), runtime, results_dir="/results") == 0
    assert len([event for event in runtime.events if event.startswith("run:")]) == 60
    [(path, record)] = runtime.writes
    assert path.name == "train-lora-e2e-2026-08-19.json"
    assert record["binding"] is True
    assert list(record["outcomes"])[0] == "O4"
    assert runtime.events[-1] == "release"
    output = capsys.readouterr().out
    assert output.rfind('"O4"') < output.rfind('"O1"')


def test_binding_driver_records_o5_when_the_loop_keeps_nothing():
    runtime = _Runtime()
    plan = _plan()
    plan["kept_candidates"] = []
    plan["loop_outcome"] = {
        "generated": 4,
        "died_by_stage": {"verify": 4},
        "priced": 0,
        "kept": 0,
        "wall_seconds": 12.0,
    }
    assert run_binding(plan, runtime, results_dir="/results") == 0
    assert "preflight" not in runtime.events
    assert "backward" not in runtime.events
    assert not any(event.startswith("run:") for event in runtime.events)
    [(path, record)] = runtime.writes
    assert path.name == "train-lora-e2e-2026-08-19.json"
    assert record["outcomes"] == {"O5": plan["loop_outcome"]}
    assert "no speed claim" in record["reason"]


def test_training_args_freeze_mlx_lm_defaults_and_the_registered_cell(tmp_path):
    provenance = {
        "base_model": {"directory": "/pinned/model"},
        "data": {"directory": "/fixed/data"},
    }
    args = build_training_args(
        _plan(), provenance, CELL_BY_NAME["B"], tmp_path / "adapter"
    )
    assert args.model == "/pinned/model"
    assert args.data == "/fixed/data"
    assert args.train is True and args.test is False
    assert args.fine_tune_type == "lora"
    assert args.batch_size == 4 and args.mask_prompt is True
    assert args.iters == 3 and args.steps_per_report == 1
    assert args.max_seq_length == 2048 and args.num_layers == 16
    assert args.lora_parameters == {"rank": 8, "dropout": 0.0, "scale": 20.0}
    assert args.optimizer_config == {"adam": {}}
    assert args.resume_adapter_file is None
    assert args.save_every == 4


def _write_preflight_fixture(tmp_path):
    root = tmp_path / "repo"
    model = root / "bench/.models/qwen3-4b-4bit-g64"
    data = tmp_path / "data"
    wrapper = root / "metalrunner"
    model.mkdir(parents=True)
    data.mkdir()
    wrapper.mkdir(parents=True)
    model_file = model / "model.safetensors"
    model_file.write_bytes(b"pinned model")
    (model / "config.json").write_text("{}")
    (data / "train.jsonl").write_text('{"prompt":"p","completion":"c"}\n')
    (wrapper / "routing.py").write_text("CERTIFIED = ()\n")
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    pins = root / "bench/.models/PINNED-HASHES.txt"
    pins.write_text(
        f"{digest}  bench/.models/qwen3-4b-4bit-g64/model.safetensors\n"
        f"{hashlib.sha256(b'{}').hexdigest()}  "
        "qwen3-4b-4bit-g64/config.json\n"
    )
    plan = _plan()
    plan["data"] = str(data)
    machine = {
        "chip": "Apple M3 Pro",
        "memory_bytes": 36 * 2**30,
        "cores": 12,
    }
    packages = {
        "mlx": {"version": "0.32.0", "package_sha256": "mlx-sha"},
        "mlx_lm": {
            "version": "0.31.3",
            "package_sha256": "mlx-lm-sha",
        },
    }
    return root, plan, machine, packages, digest


def test_preflight_binds_local_model_data_stack_wrapper_and_plan(tmp_path):
    root, plan, machine, packages, model_digest = _write_preflight_fixture(
        tmp_path
    )
    result = preflight_inputs(
        plan,
        root=root,
        machine=machine,
        package_records=packages,
        find_module=lambda name: object(),
    )
    assert result["base_model"]["sha256"] == model_digest
    assert result["base_model"]["file"].endswith("model.safetensors")
    assert result["data"]["files"] == ["train.jsonl"]
    assert len(result["data"]["sha256"]) == 64
    assert result["stack"] == packages
    assert len(result["wrapper_sha256"]) == 64
    assert len(result["plan_sha256"]) == 64


@pytest.mark.parametrize(
    "fault", ["model", "artifact", "chip", "memory", "module"]
)
def test_preflight_refuses_each_permanent_identity_failure(tmp_path, fault):
    root, plan, machine, packages, _ = _write_preflight_fixture(tmp_path)
    finder = lambda name: object()
    if fault == "model":
        (root / "bench/.models/qwen3-4b-4bit-g64/model.safetensors").write_bytes(
            b"changed"
        )
    elif fault == "artifact":
        (root / "bench/.models/qwen3-4b-4bit-g64/config.json").write_text(
            '{"changed":true}'
        )
    elif fault == "chip":
        machine["chip"] = "Apple M3 Max"
    elif fault == "memory":
        machine["memory_bytes"] = 16 * 2**30
    else:
        finder = lambda name: None
    with pytest.raises(PreconditionFailed):
        preflight_inputs(
            plan,
            root=root,
            machine=machine,
            package_records=packages,
            find_module=finder,
        )


class _FakeProcess:
    pid = 12345

    def __init__(self, argv, *, returncode=0, write_result=True,
                 timeout=False):
        self.argv = argv
        self.returncode = returncode
        self.write_result = write_result
        self.timeout = timeout

    def wait(self, timeout=None):
        if self.timeout:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        if self.returncode == 0 and self.write_result:
            out = self.argv[self.argv.index("--child-out") + 1]
            with open(out, "w") as handle:
                json.dump({
                    "completed": True,
                    "adapter_written_monotonic_ns": 1_500_000_000,
                }, handle)
        return self.returncode

    def poll(self):
        return self.returncode


def test_spawn_child_uses_a_fresh_session_and_parent_clock_boundary(tmp_path):
    calls = []

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return _FakeProcess(argv)

    ticks = iter((1_000_000_000, 2_000_000_000))
    reaped = []
    result = spawn_child(
        {"kind": "train", "cell": "B round 1 ours"},
        tmp_path,
        wall_cap_s=10.0,
        popen_factory=popen,
        clock_ns=lambda: next(ticks),
        reap=lambda proc: reaped.append(proc.pid),
    )
    argv, kwargs = calls[0]
    task_path = argv[argv.index("--child-task") + 1]
    assert kwargs == {"start_new_session": True}
    assert result["full_job_wall_s"] == 0.5
    assert result["process_launch_monotonic_ns"] == 1_000_000_000
    assert not tmp_path.joinpath(task_path).exists()
    assert reaped == [12345]


def test_spawn_child_treats_exit_zero_without_a_result_as_child_death(tmp_path):
    with pytest.raises(ChildRefusal) as caught:
        spawn_child(
            {"kind": "backward", "cell": "backward"},
            tmp_path,
            wall_cap_s=10.0,
            popen_factory=lambda argv, **kwargs: _FakeProcess(
                argv, write_result=False
            ),
            reap=lambda proc: None,
        )
    assert caught.value.returncode == EXIT_CHILD_DEATH


def test_spawn_child_preserves_a_shared_child_refusal(tmp_path):
    with pytest.raises(ChildRefusal) as caught:
        spawn_child(
            {"kind": "backward", "cell": "backward"},
            tmp_path,
            wall_cap_s=10.0,
            popen_factory=lambda argv, **kwargs: _FakeProcess(
                argv, returncode=EXIT_ORPHANED, write_result=False
            ),
            reap=lambda proc: None,
        )
    assert caught.value.returncode == EXIT_ORPHANED


def test_spawn_child_reaps_a_timed_out_process_group(tmp_path):
    reaped = []
    with pytest.raises(ChildRefusal) as caught:
        spawn_child(
            {"kind": "train", "cell": "B round 1 ours"},
            tmp_path,
            wall_cap_s=2.0,
            popen_factory=lambda argv, **kwargs: _FakeProcess(
                argv, timeout=True
            ),
            reap=lambda proc: reaped.append(proc.pid),
        )
    assert caught.value.returncode == EXIT_CHILD_DEATH
    assert reaped == [12345]


def test_cli_refuses_a_missing_plan_without_starting_a_run(tmp_path):
    assert main(["--plan", str(tmp_path / "missing.json")]) == EXIT_PRECONDITION


def _child_task(tmp_path):
    task = tmp_path / "task.json"
    out = tmp_path / "out.json"
    task.write_text(json.dumps({
        "kind": "train",
        "cell": "B round 1 ours",
        "budget_gb": 24.0,
        "parent_pid": 123,
    }))
    return task, out


def test_child_mode_refuses_forced_no_metal_before_reading_a_model(
        tmp_path, monkeypatch):
    """The flag is set here rather than inherited from the shell. A test that
    reads its own precondition out of the environment passes or fails for a
    reason that is not in the test, and this one asserts the ordering of two
    refusals: on a machine that has a device the door cannot fire at all, and
    the orphan check answers first with a different code."""
    monkeypatch.setenv("KV_FORCE_NO_METAL", "1")
    task, out = _child_task(tmp_path)
    assert main(["--child-task", str(task), "--child-out", str(out)]) == (
        EXIT_NO_DEVICE
    )
    assert not out.exists()


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [
        (harness.BudgetExceeded("child", 25.0, 24.0), EXIT_BUDGET_REFUSAL),
        (harness.Orphaned("child", 123, 1), EXIT_ORPHANED),
    ],
)
def test_child_guard_refusals_keep_the_shared_code(
    tmp_path, monkeypatch, error, exit_code
):
    task, out = _child_task(tmp_path)
    monkeypatch.delenv("KV_FORCE_NO_METAL", raising=False)

    class RefusingGuard:
        def __init__(self, *args, **kwargs):
            pass

        def check(self, cell):
            raise error

    monkeypatch.setattr(runner, "BudgetGuard", RefusingGuard)
    assert harness.child_main(task, out) == exit_code
    assert not out.exists()


def test_child_low_memory_refusal_keeps_the_shared_code(tmp_path, monkeypatch):
    task, out = _child_task(tmp_path)
    monkeypatch.delenv("KV_FORCE_NO_METAL", raising=False)

    class Guard:
        def __init__(self, *args, **kwargs):
            pass

        def check(self, cell):
            return 1.0

    def refuse(needed, cell):
        raise harness.LowMemoryRefusal(cell, 2.0, needed)

    monkeypatch.setattr(runner, "BudgetGuard", Guard)
    monkeypatch.setattr(runner, "require_available_memory", refuse)
    assert harness.child_main(task, out) == EXIT_LOW_MEMORY
    assert not out.exists()


def test_child_precondition_refusal_happens_before_training(tmp_path, monkeypatch):
    task, out = _child_task(tmp_path)
    monkeypatch.delenv("KV_FORCE_NO_METAL", raising=False)

    class Guard:
        def __init__(self, *args, **kwargs):
            pass

        def check(self, cell):
            return 1.0

    monkeypatch.setattr(runner, "BudgetGuard", Guard)
    monkeypatch.setattr(runner, "require_available_memory", lambda *args: 1.0)
    monkeypatch.setattr(
        harness,
        "stack_record",
        lambda: (_ for _ in ()).throw(PreconditionFailed("changed stack")),
    )
    assert harness.child_main(task, out) == EXIT_PRECONDITION
    assert not out.exists()


# A misspelled measurement module is a permanent refusal, never a crash: the
# detached runner reads exit 1 with a traceback as something worth retrying.
@pytest.mark.parametrize("name", [
    "definitely_not_a_module",          # absent top level: find_spec returns None
    "definitely_not_a_module.lora",     # absent parent: find_spec RAISES
    "metalrunnr.lora",                  # the transposition that motivated this
])
def test_an_unimportable_measurement_module_refuses_by_number(name):
    plan = _plan()
    plan["measurement_module"] = name
    with pytest.raises(PreconditionFailed, match="measurement module"):
        preflight_inputs(
            plan,
            machine={"chip": "Apple M3 Pro", "memory_bytes": 36 * 2**30},
        )


def test_find_spec_really_does_raise_on_an_absent_parent():
    """The premise of the guard above, asserted against the interpreter rather
    than assumed, so the guard cannot outlive its reason."""
    import importlib.util

    assert importlib.util.find_spec("definitely_not_a_module") is None
    with pytest.raises(ModuleNotFoundError):
        importlib.util.find_spec("definitely_not_a_module.lora")


# ---------------------------------------------------------------------------
# Amendment 3: the reference arm's own spread is what detects a moved clock.
# These arms are separate processes in rotated slots, not interleaved, so
# nothing else in the harness can see an excursion that lands on one arm.
# ---------------------------------------------------------------------------
def test_a_reference_arm_that_moved_rejects_rather_than_claiming():
    """The scenario that motivated the amendment, with its own numbers: the
    floor does not catch it, because a contaminated reference arm widens the
    floor more slowly than it moves the ratio."""
    ours = [100.0, 101.0, 102.0, 101.0, 100.0]
    contaminated = [130.0, 190.0, 200.0, 195.0, 205.0]

    comparison = time_comparison(ours, contaminated)
    assert comparison["reference_spread"] > harness.MAX_REFERENCE_SPREAD
    assert comparison["verdict"] == "REJECTED"

    # Without the rule this reads as a comfortable win that clears its floor.
    assert comparison["ratio_lo"] > 1.10
    assert abs(comparison["delta_pct"]) > comparison["noise_floor_pct"]

    # And the truth, had the machine held still, is nowhere near a GO.
    honest = time_comparison(ours, [104.0, 105.0, 104.0, 106.0, 105.0])
    assert honest["verdict"] == "WIN" and honest["ratio_lo"] < 1.10


def test_a_rejected_comparison_never_reaches_a_go():
    rejected = time_comparison([100.0, 101.0], [130.0, 205.0])
    assert rejected["verdict"] == "REJECTED"
    assert decide_o1(rejected, ours_peak_gb=20.0,
                     stock_peak_gb=21.0)["verdict"] == "NO-GO"
    assert decide_o2(rejected)["verdict"] == "BELOW-TARGET"
    assert decide_o3(rejected)["verdict"] == "REJECTED"


def test_a_rejected_comparison_stops_its_whole_cell_binding():
    """Every comparison in a cell reads the same rounds, so a clock that moved
    under one of them moved under all of them."""
    rounds = _rounds()
    for number, wall in enumerate([130.0, 190.0, 200.0, 195.0, 205.0], 1):
        rounds[number - 1]["stock"]["full_job_wall_s"] = wall

    reading = build_cell_reading("B", rounds)
    assert reading["binding"] is False
    assert reading["verdict"] == "REJECTED"
    assert "reference arm spread" in reading["reasons"][0]


def test_a_steady_reference_arm_still_binds():
    """The rule must not refuse an ordinary run: a limit that rejects
    everything measures nothing."""
    reading = build_cell_reading("B", _rounds())
    assert reading["binding"] is True
    assert reading["ours_vs_stock"]["verdict"] != "REJECTED"


def test_the_reference_arm_is_the_second_of_the_two():
    """Orientation matters: a wild FIRST arm is not what this detects, because
    the interval endpoints already pair its worst against the reference's
    best. Only the reference arm's own movement is invisible to them."""
    wild_first = time_comparison([1.0, 100.0], [10.0, 10.0])
    assert wild_first["verdict"] != "REJECTED"
    wild_second = time_comparison([10.0, 10.0], [1.0, 100.0])
    assert wild_second["verdict"] == "REJECTED"


# ---------------------------------------------------------------------------
# The measurement module's own refusal is permanent; everything else is a bug.
# The detached runner reads exit 1 plus a traceback as a crash and spends a
# retry on it, and no retry will make an unkept candidate kept.
# ---------------------------------------------------------------------------
def _measurement_module(raises=None):
    """A stand-in measurement module that declares its refusal type the way
    the real one does, and raises whatever the test hands it."""
    module = types.ModuleType("fake_measurement_module")

    class MeasurementRefusal(RuntimeError):
        pass

    def verify_backward_exact(**_kwargs):
        raise raises if raises is not None else MeasurementRefusal("refused")

    module.MeasurementRefusal = MeasurementRefusal
    module.verify_backward_exact = verify_backward_exact
    module.install = lambda **_kwargs: None
    return module


def _backward_task(tmp_path):
    plan = _plan()
    return {"kind": "backward", "cell": "backward", "plan": plan,
            "provenance": {"base_model": {"directory": str(tmp_path)}}}


def test_a_typed_measurement_refusal_maps_to_the_precondition_exit(
        tmp_path, monkeypatch):
    module = _measurement_module()
    module.verify_backward_exact = lambda **_k: (_ for _ in ()).throw(
        module.MeasurementRefusal("candidate ab is not kept"))
    monkeypatch.setattr(harness, "_load_measurement_module", lambda _p: module)

    with pytest.raises(PreconditionFailed, match="is not kept"):
        harness._backward_child(_backward_task(tmp_path), guard=None)


def test_an_untyped_measurement_crash_still_escapes(tmp_path, monkeypatch):
    """A real bug keeps its retryable traceback: narrowing the mapping to
    the declared type is the whole point of reading it off the module."""
    module = _measurement_module(ValueError("a genuine bug"))
    monkeypatch.setattr(harness, "_load_measurement_module", lambda _p: module)

    with pytest.raises(ValueError, match="a genuine bug"):
        harness._backward_child(_backward_task(tmp_path), guard=None)


def test_a_module_declaring_no_refusal_type_is_not_special_cased(
        tmp_path, monkeypatch):
    module = _measurement_module(RuntimeError("no declared type here"))
    del module.MeasurementRefusal
    monkeypatch.setattr(harness, "_load_measurement_module", lambda _p: module)

    with pytest.raises(RuntimeError, match="no declared type"):
        harness._backward_child(_backward_task(tmp_path), guard=None)


# The wrapper fingerprint has one implementation, and the harness uses it.
def test_the_wrapper_hash_is_the_wrappers_own_implementation():
    """Three places compare this digest and any drift refuses every run, so
    identity is the guarantee rather than two algorithms kept in step."""
    from metalrunner.measurement import tree_sha256

    wrapper = harness.ROOT / "metalrunner"
    assert harness._wrapper_sha256(wrapper) == tree_sha256(wrapper)


def test_the_wrapper_hash_ignores_interpreter_caches(tmp_path):
    """The child imports metalrunner between the preflight and the evidence,
    which writes .pyc files; if those counted, every run would refuse."""
    (tmp_path / "real.py").write_text("x = 1\n")
    before = harness._wrapper_sha256(tmp_path)

    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "real.cpython-312.pyc").write_bytes(b"\x00\x01")
    (tmp_path / "stray.pyc").write_bytes(b"\x00\x02")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x03")

    assert harness._wrapper_sha256(tmp_path) == before
    (tmp_path / "real.py").write_text("x = 2\n")
    assert harness._wrapper_sha256(tmp_path) != before
