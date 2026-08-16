"""The device calibration harness's decisions, taken on CPU over built records.

`bench/calibrate_quant_device.py` needs a Metal device to MEASURE, but every
decision it takes afterwards - which K its own grid demands, whether the
independent draw misses it, what it prints and what it persists - is arithmetic
over records it already holds.
Those decisions are what an ADR reads, so they are pinned here against records
built by hand, with `measure` and `DeviceMemberSession` monkeypatched out.
No test in this module opens a Metal device, which is why it lives apart from
`test_quant_device_members.py`: that module is skipped wholesale without a GPU,
and the harness's arithmetic must stay checkable on any machine.
"""

import json

import pytest

import calibrate_quant_device as dev
from kernelverify.schemas.native_ops import K_QUANT as SHIPPED_K


def _record(seed, member_errs, heldout_err, base=1e-9):
    """One record in the harness's own shape, with every field its readers touch."""
    names = list(dev.ALL_MEMBERS)
    return {"bits": 3, "shape": "512x512", "draw": "normal-0.02", "seed": seed,
            "heldout_draw": seed in dev.HELDOUT_SEEDS, "mode": "unit",
            "dtype": "float32", "base_tol": base,
            "members": dict(zip(names, member_errs)),
            "heldout": {"block-tiled": heldout_err, "mlx-on-device": heldout_err},
            "faults": {name: 1.0 for name in dev.FAULTS},
            "boundary_fp16_dequant": 0.0}


def _measured(*records):
    """What measure() returns: main() reads bit_exact before it reads records."""
    return {"records": list(records), "bit_exact": True}


def _no_metal(monkeypatch, measured):
    monkeypatch.setattr(dev, "measure", lambda *a, **k: measured)
    monkeypatch.setattr(dev, "DeviceMemberSession", lambda: object())


# ---------------------------------------------------------------------------
# V1: this harness's K is a reading of three synthetic shapes, not what ships
# ---------------------------------------------------------------------------
def test_the_device_json_names_its_own_k_and_the_one_that_ships(monkeypatch, tmp_path):
    """Calling this harness's reading `shipped K` is how a K derived on three
    synthetic shapes nearly became the verifier's K, at a value the serving
    grid's own DEMAND MISS branch would then have refused (ADR 0016).
    Two readings, two keys, and the ambiguous name is gone."""
    monkeypatch.setattr(dev, "OUT_PATH", tmp_path / "out.json")
    _no_metal(monkeypatch, _measured(_record(0, [1.0] * 9, 2.5),
                                     _record(100, [1.0] * 9, 2.0)))
    dev.main(["--bits", "3"])

    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload["k_grid"] == 3.0, "the harness's own reading of its own grid"
    assert payload["k_shipped"] == SHIPPED_K, "what the verifier actually ships"
    assert "k_ship" not in payload, "the name that meant both is gone"


def test_the_device_grid_k_prints_as_a_reading_beside_the_shipped_value(
        monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(dev, "OUT_PATH", tmp_path / "out.json")
    _no_metal(monkeypatch, _measured(_record(0, [1.0] * 9, 2.5),
                                     _record(100, [1.0] * 9, 2.0)))
    dev.main(["--bits", "3"])

    out = capsys.readouterr().out
    assert "device-grid K: 3.0" in out
    assert f"native_ops.K_QUANT = {SHIPPED_K}" in out
    assert "shipped K:" not in out, "the harness never claims to ship a K"


# ---------------------------------------------------------------------------
# V5: K comes from the calibration draw; the independent draw tests it
# ---------------------------------------------------------------------------
def test_k_is_one_value_across_widths_chosen_from_calibration_alone():
    cal3 = [_record(0, [1.0] * 9, heldout_err=2.5)]      # demands 2.5
    cal4 = [_record(0, [1.0] * 9, heldout_err=1.2)]
    indep3 = [_record(100, [1.0] * 9, heldout_err=3.5)]  # 3.5 > 3.0: a miss at bits 3
    indep4 = [_record(100, [1.0] * 9, heldout_err=1.0)]
    k_grid, misses = dev.choose_k({3: (cal3, indep3), 4: (cal4, indep4)})
    assert k_grid == 3.0, (
        "one value across widths: max(K_CPU = 3.0, cover(2.5) = 3.0); this is "
        "the harness's reading, not native_ops.K_QUANT")
    assert set(misses) == {3} and misses[3][0] == 3.5, (
        "the held-out excess is REPORTED per width, never folded into K")


def test_an_independent_miss_stops_the_run(monkeypatch, tmp_path, capsys):
    """The amendment's whole point: the held-out draw can now say no.
    ADR 0012 rolled K up to cover it, which made the next gate true by
    construction again."""
    monkeypatch.setattr(dev, "OUT_PATH", tmp_path / "out.json")
    monkeypatch.setattr(dev, "choose_k", lambda per_bits: (
        3.0, {3: (3.5, "heldout mlx-on-device @ 512x512 normal-0.02 s100 "
                  "constant-rows float32")}))
    _no_metal(monkeypatch, _measured(_record(0, [1.0] * 9, 0.0),
                                     _record(100, [1.0] * 9, 3.5)))
    assert dev.main(["--bits", "3"]) == 1
    assert "INDEPENDENT MISS: bits=3" in capsys.readouterr().out


def test_cover_is_the_one_grid_lookup():
    from calibrate_quant_bits import K_GRID, cover

    assert cover(2.766) == 3.0 and cover(3.120) == 4.0 and cover(0.5) == K_GRID[0]
    with pytest.raises(ValueError):
        cover(9.0)          # above the grid: the KILL branch, never a silent clamp


def test_choose_k_with_no_usable_width_is_a_named_verdict_not_a_crash(
        monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(dev, "OUT_PATH", tmp_path / "out.json")
    assert dev.choose_k({}) == (dev.K_CPU, {})
    _no_metal(monkeypatch, {"records": [], "bit_exact": False})   # G0 fails
    assert dev.main(["--bits", "3"]) == 1
    assert "no usable width" in capsys.readouterr().out


def test_g1_is_scored_on_the_independent_draw_only():
    # a calibration record with heldout at 10x floor exists in the run, but
    # G1 never sees it:
    indep = [_record(100, [1.0] * 9, heldout_err=1.5)]
    report = dev.gates(indep, k=3.0)                    # G1 sees indep only
    assert report["gates"]["G1_no_false_positives"] is True
    assert report["false_positives"] == []


def test_the_old_reading_keys_are_still_persisted(monkeypatch, tmp_path):
    """REGRESSION (iron rule): tests/ and ADR 0016 read k_needed_calibration and
    k_needed_independent from the device JSON; both-readings continuity keeps them."""
    monkeypatch.setattr(dev, "OUT_PATH", tmp_path / "out.json")
    _no_metal(monkeypatch, _measured(_record(0, [1.0] * 9, 2.5),
                                     _record(100, [1.0] * 9, 2.0)))
    dev.main(["--bits", "3"])
    out = json.loads((tmp_path / "out.json").read_text())["reports"]["3"]
    for key in ("k_needed_calibration", "k_needed_independent",
                "k_needed_calibration_distinct", "k_needed_independent_distinct",
                "independent_miss"):
        assert key in out, key


def test_k_demand_collapses_value_duplicates():
    errs = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # members 0 and 1 bit-identical
    recs = [_record(0, errs, heldout_err=0.0)]
    needed_all, _ = dev.k_demand(recs, distinct_only=False)
    needed_distinct, binding = dev.k_demand(recs, distinct_only=True)
    assert needed_all == 1.0, "a duplicate is capped at 1.0 by its twin and looks harmless"
    assert needed_distinct == 2.0 and "member" in binding, (
        "collapsed, the pair binds at its true ratio")
