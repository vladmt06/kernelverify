"""The two gates that decide whether a baseline number may be published.

Both exist because this project has already been burned once each: the
first-pass baselines were hand-run on a shared machine, and kv-runner-e9
measured that sub-millisecond GPU timings move by up to 4x with power state.
Neither rule is allowed to be a comment.
"""

import json

import pytest

import machine_state
import measure_baselines
from machine_state import binding_verdict, timing_verdict


def state(load1=0.5, source="AC", lpm="0", thermal=False, blockers=None):
    """An idle_check result, without touching the real machine."""
    if blockers is None:
        blockers = []
        if load1 > 3.0:
            blockers.append(f"load1 {load1:.2f} over threshold 3.00")
        if source != "AC":
            blockers.append(f"power source {source}")
        if lpm not in ("0", "?"):
            blockers.append("low power mode on")
        if thermal:
            blockers.append("thermal warning recorded")
    return {
        "load1": load1, "load5": load1, "load15": load1, "load_threshold": 3.0,
        "power": {"source": source, "low_power_mode": lpm, "thermal_warning": thermal},
        "idle": not blockers, "blockers": blockers,
    }


# --- the idle gate --------------------------------------------------------


def test_idle_threshold_scales_with_core_count(monkeypatch):
    monkeypatch.setattr(machine_state, "load_averages", lambda: (2.0, 2.0, 2.0))
    monkeypatch.setattr(
        machine_state, "power_state",
        lambda: {"source": "AC", "low_power_mode": "0", "thermal_warning": False},
    )
    monkeypatch.setattr(machine_state, "competing_processes", lambda *a, **k: [])
    # 2.0 is idle on 12 cores and busy on 4: a fixed threshold would call one
    # of those wrong, and this machine has both kinds of core.
    assert machine_state.idle_check(cores=12)["idle"] is True
    assert machine_state.idle_check(cores=4)["idle"] is False


@pytest.mark.parametrize(
    "kwargs, expected_blocker",
    [
        ({"load1": 9.0}, "load1"),
        ({"source": "battery"}, "power source"),
        ({"lpm": "1"}, "low power mode"),
        ({"thermal": True}, "thermal"),
    ],
)
def test_every_condition_blocks_binding(kwargs, expected_blocker):
    clean = timing_verdict(50.0)
    verdict = binding_verdict(state(**kwargs), state(), clean)
    assert verdict["binding"] is False
    assert any(expected_blocker in b for b in verdict["binding_blockers"])


def test_a_machine_that_goes_busy_mid_run_also_blocks():
    verdict = binding_verdict(state(), state(load1=9.0), timing_verdict(50.0))
    assert verdict["binding"] is False
    assert any(b.startswith("after:") for b in verdict["binding_blockers"])


def test_clean_machine_and_real_timescale_binds():
    verdict = binding_verdict(state(), state(), timing_verdict(50.0))
    assert verdict == {"binding": True, "binding_blockers": []}


# --- the dispersion gate --------------------------------------------------


def test_disagreeing_repeats_block_binding_on_a_quiet_machine():
    """The case that forced this gate.

    A run reported load average 1.93 from start to finish while one spec's
    third sample came in at 22.07 against 99.38 and 90.06. Load is averaged
    over a minute and cannot see a transient that lands inside one sample, so
    an idle machine is not evidence that a measurement succeeded.
    """
    dispersion = machine_state.dispersion_verdict(85.84)
    verdict = binding_verdict(state(), state(), timing_verdict(50.0), dispersion)
    assert verdict["binding"] is False
    assert any("repeats disagree" in b for b in verdict["binding_blockers"])


def test_tight_repeats_still_bind():
    dispersion = machine_state.dispersion_verdict(1.76)
    assert dispersion["over_spread_limit"] is False
    assert binding_verdict(state(), state(), timing_verdict(50.0), dispersion)["binding"]


def test_a_single_sample_has_no_spread_and_is_not_blocked_by_it():
    # one rep cannot disagree with itself; the other gates still apply
    assert machine_state.dispersion_verdict(None)["over_spread_limit"] is False


# --- the GPU contention gate ----------------------------------------------


def test_an_interactive_app_on_the_gpu_blocks_even_at_low_load(monkeypatch):
    """The case the kernels lane measured.

    An AC-powered machine at load average 1.79 produced 2.2x-9.7x spreads
    because Terminal and VS Code were driving the display stack. Load average
    and power state both passed it, so neither is sufficient on its own.
    """
    monkeypatch.setattr(machine_state, "load_averages", lambda: (1.79, 1.8, 1.8))
    monkeypatch.setattr(
        machine_state, "power_state",
        lambda: {"source": "AC", "low_power_mode": "0", "thermal_warning": False},
    )
    monkeypatch.setattr(
        machine_state, "competing_processes",
        lambda *a, **k: [{"name": "Terminal", "pcpu": 36.0},
                         {"name": "Code Helper (GPU)", "pcpu": 25.0}],
    )
    check = machine_state.idle_check(cores=12)
    assert check["idle"] is False
    assert any("Terminal" in b and "36%" in b for b in check["blockers"])


def test_the_harness_does_not_flag_its_own_subprocesses(monkeypatch):
    monkeypatch.setattr(machine_state, "load_averages", lambda: (1.0, 1.0, 1.0))
    monkeypatch.setattr(
        machine_state, "power_state",
        lambda: {"source": "AC", "low_power_mode": "0", "thermal_warning": False},
    )
    # competing_processes already excludes our own tree; with nothing else busy
    # the machine is idle even though our own benchmark pegs a core
    monkeypatch.setattr(machine_state, "competing_processes", lambda *a, **k: [])
    assert machine_state.idle_check(cores=12)["idle"] is True


def test_own_process_tree_contains_this_process():
    import os

    assert os.getpid() in machine_state._own_process_tree()


# --- the timing floor -----------------------------------------------------


def test_sub_millisecond_samples_are_refused_as_absolute_claims():
    # 200 us is the scale kv-runner-e9 measured moving 4x with power state.
    assert timing_verdict(0.2)["below_timing_floor"] is True
    assert timing_verdict(50.0)["below_timing_floor"] is False


def test_timing_floor_blocks_binding_on_an_otherwise_perfect_machine():
    verdict = binding_verdict(state(), state(), timing_verdict(0.2))
    assert verdict["binding"] is False
    assert any("timing floor" in b for b in verdict["binding_blockers"])


# --- the consumer contract ------------------------------------------------


def valid_row():
    return {
        "schema_version": 3, "run_id": "r", "row_id": "r/x", "measured_at": "t",
        "provenance_tier": "owner-run", "machine": {}, "idle_before": state(),
        "idle_after": state(), "stack": {"name": "llama.cpp"},
        "model": {"name": "Qwen3-4B-Q4_K_M.gguf", "logical_name": "qwen3-4b"},
        "measurement": {"kind": "decode"},
        "result": {"metric": "tokens_per_s", "median": 1.0, "reps": 3,
                   "samples": [1.0], "min_sample_ms": 50.0, "floor_ms": 1.0,
                   "below_timing_floor": False},
        "roofline": {"bandwidth_utilisation_pct": 80.0,
                     "binding_resource": "memory"},
        "sampling": {"interleaved": True, "rotation": "arm-alternation",
                     "rounds": 3, "group": "r/decode-w1", "group_members": ["x"]},
        "binding": True, "binding_blockers": [],
    }


def test_a_complete_row_validates_and_round_trips_as_jsonl():
    row = measure_baselines.validate_row(valid_row())
    assert json.loads(json.dumps(row))["row_id"] == "r/x"


@pytest.mark.parametrize("field", ["run_id", "provenance_tier", "machine", "binding"])
def test_a_missing_field_fails_the_producer_not_the_consumer(field):
    row = valid_row()
    del row[field]
    with pytest.raises(ValueError):
        measure_baselines.validate_row(row)


def test_unknown_provenance_tier_is_rejected():
    row = valid_row() | {"provenance_tier": "vibes"}
    with pytest.raises(ValueError, match="provenance tier"):
        measure_baselines.validate_row(row)


def test_a_row_cannot_claim_to_bind_while_carrying_blockers():
    row = valid_row() | {"binding": True, "binding_blockers": ["load1 9.0"]}
    with pytest.raises(ValueError, match="cannot bind"):
        measure_baselines.validate_row(row)


def test_a_model_row_without_a_roofline_placement_is_rejected():
    row = valid_row()
    del row["roofline"]
    with pytest.raises(ValueError, match="no roofline placement"):
        measure_baselines.validate_row(row)


def test_a_row_that_cannot_say_how_it_was_sampled_is_rejected():
    """Comparability is a separate gate from bindingness.

    The kernels lane ran an A/B whose every dispatch was past 5 ms and still
    got the sign backwards, because the arms ran in separate passes. A row
    that cannot state it was interleaved must not be compared against another.
    """
    row = valid_row()
    del row["sampling"]
    with pytest.raises(ValueError, match="sampling missing"):
        measure_baselines.validate_row(row)


def test_a_binding_row_can_still_be_incomparable():
    # binding is about absolutes, interleaving is about comparisons; a row may
    # legitimately pass one gate and fail the other
    row = valid_row()
    row["sampling"] = {"interleaved": False, "group": None, "rounds": 1}
    assert measure_baselines.validate_row(row)["binding"] is True
    assert row["sampling"]["interleaved"] is False


# --- the v3 contract: constrained vocabulary ------------------------------


def test_v3_constrains_binding_resource_to_the_known_set():
    """The renderer picks the utilisation column by this value, so a producer
    writing 'bandwidth' would render every such row as an unknown column."""
    row = valid_row()
    row["roofline"]["binding_resource"] = "bandwidth"
    with pytest.raises(ValueError, match="binding_resource"):
        measure_baselines.validate_row(row)
    del row["roofline"]["binding_resource"]
    with pytest.raises(ValueError, match="binding_resource"):
        measure_baselines.validate_row(row)


def test_v3_requires_a_canonical_logical_model_name():
    """The logical name is the cross-stack cell identity; the artefact
    spelling would split one cell into two, one per stack."""
    row = valid_row()
    row["model"]["logical_name"] = "Qwen3-4B"
    with pytest.raises(ValueError, match="logical_name"):
        measure_baselines.validate_row(row)
    del row["model"]["logical_name"]
    with pytest.raises(ValueError, match="logical_name"):
        measure_baselines.validate_row(row)


def test_a_v2_row_is_not_held_to_v3_fields():
    """The shipped 2026-08-14 record is v2; the validator must not
    retroactively reject the contract those rows were written under."""
    row = valid_row()
    row["schema_version"] = 2
    del row["model"]
    row["roofline"] = {"bandwidth_utilisation_pct": 80.0}
    assert measure_baselines.validate_row(row)


def test_a_v3_ceiling_row_needs_no_model_or_binding_resource():
    row = valid_row()
    row["measurement"] = {"kind": "ceiling", "name": "bandwidth_read"}
    row["model"] = None
    del row["roofline"]
    del row["sampling"]
    assert measure_baselines.validate_row(row)


def test_the_producer_writes_a_logical_name_the_validator_accepts():
    assert measure_baselines.LOGICAL_MODEL in measure_baselines.LOGICAL_MODEL_NAMES


# --- the restructured loop: a sampling group is one A/B session (D1) ------


def spec(spec_id, kind, batch, stack="llama.cpp", n_prompt=0):
    return {"id": spec_id, "stack": stack, "kind": kind, "batch": batch,
            "n_prompt": n_prompt, "n_gen": 0, "model_path": None}


def cell_specs():
    return [
        spec("llamacpp/decode", "decode", 1),
        spec("llamacpp/prefill", "prefill", 1024, n_prompt=1024),
        spec("llamacpp/width8", "matmul_width", 8),
        spec("mlx/decode", "decode", 1, stack="mlx-lm", n_prompt=1024),
        spec("mlx/width8", "matmul_width", 8, stack="mlx-lm", n_prompt=32),
    ]


def fake_measure(_spec):
    return {"tokens_per_s": 1.0, "prompt_tps": 1.0, "generation_tps": 1.0,
            "sample_ms": 5.0}


def test_consecutive_samples_within_a_cell_alternate_arms():
    """The mandated D1 regression: the old loop rotated ALL specs round-robin,
    so the two arms of one comparison sat a full rotation - minutes - apart
    and a sampling group was a relabel of the run. The arms of a cell must be
    measured shoulder to shoulder: consecutive samples alternate arms.
    """
    specs = cell_specs()
    _, sequence = measure_baselines.measure_cells(specs, rounds=3,
                                                  measure=fake_measure)
    for label, members in measure_baselines.group_cells(specs):
        ids = {m["id"] for m in members}
        block = [i for i, spec_id in enumerate(sequence) if spec_id in ids]
        assert block == list(range(block[0], block[0] + len(block))), \
            f"cell {label} was interleaved with other cells, not run as one session"
        if len(ids) < 2:
            continue  # a lone arm has nothing to alternate with
        arms = [sequence[i] for i in block]
        assert all(first != second for first, second in zip(arms, arms[1:])), \
            f"cell {label} measured the same arm twice in a row"


def test_every_arm_gets_every_round():
    samples, sequence = measure_baselines.measure_cells(cell_specs(), rounds=3,
                                                        measure=fake_measure)
    assert all(len(v) == 3 for v in samples.values())
    assert len(sequence) == 3 * len(cell_specs())


def test_a_sampling_group_is_one_workload_cell_not_the_run():
    """Never a relabel: decode and width8 land in different groups, and each
    group holds exactly the arms of its own A/B."""
    cells = measure_baselines.group_cells(cell_specs())
    assert [(label, [m["id"] for m in members]) for label, members in cells] == [
        ("decode-w1", ["llamacpp/decode", "mlx/decode"]),
        ("prefill-w1024", ["llamacpp/prefill"]),
        ("matmul_width-w8", ["llamacpp/width8", "mlx/width8"]),
    ]
