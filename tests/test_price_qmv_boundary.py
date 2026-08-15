"""The boundary-pricing verdict rules: when a point may speak and when not.

The pricing exists to move (or confirm) the routing boundary, so two things
are worth pinning: the refusal discipline (a point whose own spread could
explain its ratio must refuse to claim a direction, and a round whose
reference arm moved too much must be rejected as the machine, not the
kernels - both from the measured mistakes in AGENTS.md), and the ruled
sweep parameters, which are decisions, not defaults.

Pure verdict logic, no MLX and no timing here.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

import price_qmv_boundary as probe
from price_qmv_boundary import MAX_CANARY_SPREAD, classify


def test_ruled_parameters_are_pinned():
    """The sweep and the bit width ARE the ruled decisions (D1: the block
    serves B = 1..16; D4: 2-bit is cut), so a drive-by edit must fail here."""
    assert probe.BITS == 3
    assert probe.M_SWEEP == tuple(range(1, 17))


def test_rejects_the_round_when_the_reference_arm_moved():
    """A reference-arm spread past the class limit means a clock excursion,
    so the point describes the machine and must be REJECTED outright."""
    quiet = [10.0] * 7
    moved = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    assert max(moved) / min(moved) > MAX_CANARY_SPREAD
    assert classify(quiet, moved).verdict == "REJECTED"


def test_refuses_when_spread_swallows_the_ratio():
    """If the ratio interval [min_mlx/max_ours, max_mlx/min_ours] contains
    1.0, the direction could be noise, so no WIN or LOSS may be claimed."""
    ours = [10.0, 11.0, 10.5]
    mlx = [10.4, 10.6, 10.5]
    point = classify(ours, mlx)
    assert point.verdict == "REFUSED"
    assert point.ratio_lo < 1.0 < point.ratio_hi


def test_win_only_when_the_whole_interval_clears_one():
    """A WIN claim must survive the worst pairing of the samples."""
    point = classify([10.0, 10.2, 10.1], [13.0, 13.4, 13.2])
    assert point.verdict == "WIN"
    assert point.ratio_lo > 1.0
    assert point.ratio == pytest.approx(13.2 / 10.1)


def test_loss_when_the_whole_interval_sits_below_one():
    point = classify([13.0, 13.4, 13.2], [10.0, 10.2, 10.1])
    assert point.verdict == "LOSS"
    assert point.ratio_hi < 1.0


def test_every_point_states_its_spread():
    """The ruling requires the pricing to state its spread per point, so the
    verdict object must carry both arms' spreads whatever the outcome."""
    point = classify([10.0, 12.0], [20.0, 21.0])
    assert point.spread_ours == pytest.approx(1.2)
    assert point.spread_mlx == pytest.approx(1.05)


# ---------------------------------------------------------------------------
# Results-write ordering: the recording is written AFTER the final idle gate.
# The bug being pinned: main() wrote the results JSON before the idle_after
# check, so a run that went busy left a complete-looking recording on disk
# and exited 1 with only a printed warning.
# ---------------------------------------------------------------------------
def _idle(flag, why="test blocker"):
    return {"idle": flag, "blockers": [] if flag else [why]}


@pytest.fixture()
def hardware_free(monkeypatch, tmp_path):
    """main() with every hardware seam faked: no GPU, no sysctl, a private
    measurement lock, results in a tmp dir. The fake verify honours the
    guard-callback seam (it calls the guard once) so the refusal paths run
    the way the real gate drives them. Returns the results dir."""
    import machine_state

    monkeypatch.setattr(probe, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(probe, "build", lambda _mx: None)
    monkeypatch.setattr(probe, "MetalRunner", lambda: None)
    monkeypatch.setattr(
        probe, "MeasurementLock",
        lambda owner: machine_state.MeasurementLock(owner,
                                                    path=tmp_path / "lock"))

    def fake_verify(_runner, e2e_m=None, guard=None, **_kw):
        if guard is not None:
            guard("fake verification cell")
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(probe.gate, "verify", fake_verify)
    monkeypatch.setattr(
        probe.gate, "E2E_SHAPES",
        (SimpleNamespace(name="q_proj", d_out=256, d_in=256),))
    point = classify([10.0, 10.2], [13.0, 13.4])
    monkeypatch.setattr(probe, "price_shape", lambda *a, **k: [point])
    monkeypatch.setattr(probe.machine_state, "fingerprint",
                        lambda: {"cores": 12})
    return tmp_path


def _sequence_idle_checks(monkeypatch, verdicts):
    calls = iter(verdicts)
    monkeypatch.setattr(probe.machine_state, "idle_check",
                        lambda _cores: next(calls))


def test_a_clean_run_records_under_the_canonical_name(hardware_free,
                                                      monkeypatch):
    _sequence_idle_checks(monkeypatch, [_idle(True), _idle(True)])
    assert probe.main([]) == 0
    [written] = list(hardware_free.glob("*.json"))
    assert "REFUSED" not in written.name
    assert written.name.startswith("qmv-boundary-pricing-")


def test_a_run_that_went_busy_quarantines_its_recording(hardware_free,
                                                        monkeypatch):
    """The final gate runs BEFORE any write: a busy machine at idle_after
    must never leave a complete-looking recording under the canonical name.
    The samples land under a quarantine name instead, so the evidence
    survives without being quotable as a binding recording."""
    _sequence_idle_checks(monkeypatch, [_idle(True), _idle(False, "went busy")])
    assert probe.main([]) == 1
    [written] = list(hardware_free.glob("*.json"))
    assert "REFUSED" in written.name
    assert not list(hardware_free.glob("qmv-boundary-pricing-*[0-9].json"))


# ---------------------------------------------------------------------------
# Guard refusals (T1/T3): the machine lock, the footprint budget and the
# low-memory gate each refuse with their own exit code from the shared
# vocabulary, and a refusal writes NOTHING - the probe has no checkpoint, so
# a refused run is discarded deliberately.
# ---------------------------------------------------------------------------
def test_a_held_measurement_lock_refuses_before_the_idle_gate(
        hardware_free, monkeypatch, tmp_path):
    """One heavy measurement at a time: a held lock refuses with
    EXIT_LOCK_HELD, and the idle gate must not even sample - co-armed idle
    gates are exactly how the 03:29 two-harness collapse started."""
    import machine_state

    holder = machine_state.MeasurementLock("test-holder",
                                           path=tmp_path / "lock")
    acquired, _ = holder.acquire()
    assert acquired

    def never(_cores):
        raise AssertionError("idle gate must not co-fire behind a held lock")

    monkeypatch.setattr(probe.machine_state, "idle_check", never)
    try:
        assert probe.main([]) == probe.EXIT_LOCK_HELD
    finally:
        holder.release()
    assert not list(hardware_free.glob("*.json"))


def test_an_injected_tiny_budget_refuses_with_no_results(hardware_free,
                                                         monkeypatch):
    """The acceptance case: under a tiny budget the REAL footprint reader
    convicts this very process, and the refusal leaves no results file."""
    _sequence_idle_checks(monkeypatch, [_idle(True)])
    assert probe.main(["--budget-gb", "0.001"]) == probe.EXIT_BUDGET_REFUSAL
    assert not list(hardware_free.glob("*.json"))


def test_low_machine_memory_refuses_with_no_results(hardware_free,
                                                    monkeypatch):
    _sequence_idle_checks(monkeypatch, [_idle(True)])

    def parched(needed_gb, cell, reader=None):
        raise probe.LowMemoryRefusal(cell, 1.0, needed_gb)

    monkeypatch.setattr(probe, "require_available_memory", parched)
    assert probe.main([]) == probe.EXIT_LOW_MEMORY
    assert not list(hardware_free.glob("*.json"))


def test_the_probe_speaks_the_shared_exit_vocabulary():
    """The numbering must be THE shared one, never a probe-local copy."""
    import memory_guard

    assert probe.EXIT_BUDGET_REFUSAL is memory_guard.EXIT_BUDGET_REFUSAL
    assert probe.EXIT_LOCK_HELD is memory_guard.EXIT_LOCK_HELD
    assert probe.EXIT_LOW_MEMORY is memory_guard.EXIT_LOW_MEMORY
