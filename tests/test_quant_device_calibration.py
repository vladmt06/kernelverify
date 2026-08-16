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
import sys

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
    monkeypatch.setattr(sys, "argv", ["calibrate_quant_device", "--bits", "3"])
    dev.main()

    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload["k_grid"] == 3.0, "the harness's own reading of its own grid"
    assert payload["k_shipped"] == SHIPPED_K, "what the verifier actually ships"
    assert "k_ship" not in payload, "the name that meant both is gone"


def test_the_device_grid_k_prints_as_a_reading_beside_the_shipped_value(
        monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(dev, "OUT_PATH", tmp_path / "out.json")
    _no_metal(monkeypatch, _measured(_record(0, [1.0] * 9, 2.5),
                                     _record(100, [1.0] * 9, 2.0)))
    monkeypatch.setattr(sys, "argv", ["calibrate_quant_device", "--bits", "3"])
    dev.main()

    out = capsys.readouterr().out
    assert "device-grid K: 3.0" in out
    assert f"native_ops.K_QUANT = {SHIPPED_K}" in out
    assert "shipped K:" not in out, "the harness never claims to ship a K"
