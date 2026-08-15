"""The structured input modes must actually produce their structure.

Each mode exists to reach a fault mechanism (ADR 0003). If a mode silently
degenerates to an iid draw, the battery loses that reach while its scores
still look plausible, so these tests pin the construction itself.
"""

import numpy as np
import pytest

from measure_escape import load_meta, make_inputs
from score_oracles import CORPUS_MODE, IID_MODES, INPUT_MODES, make_mode_inputs

ATTENTION = load_meta("attention_triton")
SOFTMAX = load_meta("softmax_triton")
ATT_DIMS = {"M": 4, "N": 8, "D": 16}
SM_DIMS = {"B": 2, "S": 3, "H": 8}


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_opposed_signs_alternates_sign_by_input(dtype):
    inputs = make_mode_inputs(ATTENTION, ATT_DIMS, dtype, 7, "opposed signs")
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    assert np.all(q >= 0) and np.all(v >= 0)
    assert np.all(k <= 0)
    base = make_inputs(ATTENTION, ATT_DIMS, dtype, 7, CORPUS_MODE)
    for name in inputs:
        assert np.array_equal(np.abs(inputs[name]), np.abs(base[name]))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_near_zero_scales_down_by_1e3(dtype):
    inputs = make_mode_inputs(SOFTMAX, SM_DIMS, dtype, 7, "near zero")
    base = make_inputs(SOFTMAX, SM_DIMS, dtype, 7, CORPUS_MODE)
    arr, ref = inputs["input"], base["input"]
    assert float(np.max(np.abs(arr))) <= 0.011
    assert np.array_equal(arr, (ref * np.float32(1e-3)).astype(ref.dtype))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_constant_rows_are_constant(dtype):
    inputs = make_mode_inputs(ATTENTION, ATT_DIMS, dtype, 7, "constant rows")
    for arr in inputs.values():
        assert np.array_equal(arr, np.broadcast_to(arr[..., :1], arr.shape))
        assert arr.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("mode", IID_MODES)
def test_iid_modes_pass_through_unchanged(mode):
    via_mode = make_mode_inputs(SOFTMAX, SM_DIMS, "float32", 7, mode)
    direct = make_inputs(SOFTMAX, SM_DIMS, "float32", 7, mode)
    for name in direct:
        assert np.array_equal(via_mode[name], direct[name])


@pytest.mark.parametrize("mode", INPUT_MODES)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_every_mode_keeps_shape_dtype_and_finiteness(mode, dtype):
    inputs = make_mode_inputs(ATTENTION, ATT_DIMS, dtype, 7, mode)
    for spec in ATTENTION["op_schema"]["inputs"]:
        arr = inputs[spec["name"]]
        assert arr.shape == tuple(ATT_DIMS[d] for d in spec["dims"])
        assert str(arr.dtype) == dtype
        assert np.all(np.isfinite(arr.astype(np.float64)))


def test_modes_are_distinct_data():
    drawn = {
        mode: make_mode_inputs(SOFTMAX, SM_DIMS, "float32", 7, mode)["input"]
        for mode in INPUT_MODES
    }
    modes = list(INPUT_MODES)
    for i, a in enumerate(modes):
        for b in modes[i + 1:]:
            assert not np.array_equal(drawn[a], drawn[b]), f"{a} == {b}"
