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
        "schema_version": 1, "run_id": "r", "row_id": "r/x", "measured_at": "t",
        "provenance_tier": "owner-run", "machine": {}, "idle_before": state(),
        "idle_after": state(), "stack": {"name": "llama.cpp"},
        "measurement": {"kind": "decode"},
        "result": {"metric": "tokens_per_s", "median": 1.0, "reps": 3,
                   "samples": [1.0], "min_sample_ms": 50.0, "floor_ms": 1.0,
                   "below_timing_floor": False},
        "roofline": {"bandwidth_utilisation_pct": 80.0},
        "sampling": {"interleaved": True, "rotation": "per-round", "rounds": 3,
                     "group": "r", "group_members": ["x"]},
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
