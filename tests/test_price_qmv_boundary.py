"""The boundary-pricing verdict rules: when a point may speak and when not.

The pricing exists to move (or confirm) the routing boundary, so two things
are worth pinning: the refusal discipline (a point whose own spread could
explain its ratio must refuse to claim a direction, and a round whose
reference arm moved too much must be rejected as the machine, not the
kernels - both from the measured mistakes in AGENTS.md), and the ruled
sweep parameters, which are decisions, not defaults.

Pure verdict logic, no MLX and no timing here.
"""

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
from types import SimpleNamespace  # noqa: E402


def _idle(flag, why="test blocker"):
    return {"idle": flag, "blockers": [] if flag else [why]}


@pytest.fixture()
def hardware_free(monkeypatch, tmp_path):
    """main() with every hardware seam faked: no GPU, no sysctl, results in a
    tmp dir. Returns the results dir."""
    monkeypatch.setattr(probe, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(probe, "build", lambda _mx: None)
    monkeypatch.setattr(probe, "MetalRunner", lambda: None)
    monkeypatch.setattr(probe.gate, "verify",
                        lambda *a, **k: SimpleNamespace(ok=True))
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
