"""Device-arithmetic ensemble members: runnable, correct, distinct, and typed
for the freeze.

Correctness here is a sanity envelope, not the calibrated tolerance: whether a
correct device member exceeds the CPU-calibrated tolerance is the pre-registered
re-derivation trigger, a measurement the calibration harness owns, and a suite
must stay green on either outcome of a measurement. What a test may pin is the
difference between a working kernel and a broken one, which is orders of
magnitude, and the structural properties the class gate needs: distinct
outputs, exact carriers, specs that validate before any process starts.

Skipped wholesale without PyObjC Metal or a device, matching the runner tests.
"""

import itertools

import numpy as np
import pytest

pytest.importorskip("Metal")

from kernelverify.runners.device import MetalDevice  # noqa: E402
from kernelverify.runners.result import RunResult  # noqa: E402
from kernelverify.schemas.quant_contract import (  # noqa: E402
    FAULTS,
    QuantContract,
    canonical_quantize,
    r_contract,
)
from kernelverify.schemas.quant_device import (  # noqa: E402
    DEVICE_MEMBERS,
    DeviceMemberSession,
    device_member_case,
    device_member_spec,
    device_member_template,
    result_outputs,
)

try:
    _DEVICE = MetalDevice()
except RuntimeError:
    _DEVICE = None

pytestmark = pytest.mark.skipif(_DEVICE is None, reason="no Metal device on this machine")

# A broken kernel (wrong indexing, wrong domain, half accumulation) misses by
# orders of magnitude; a working one sits near the CPU floor. Relative to the
# output scale, the smoke measurements sit at 5e-7 (fp32) and 3e-4 (fp16).
SANITY_RELATIVE = {"float32": 1e-4, "float16": 2e-2}


@pytest.fixture(scope="module")
def session():
    return DeviceMemberSession(_DEVICE)


@pytest.fixture(scope="module")
def artefact():
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((512, 512)) * 0.02).astype(np.float16)
    return canonical_quantize(w, QuantContract(bits=4, group_size=64))


def _x(dtype, seed=11):
    return np.random.default_rng(seed).standard_normal((2, 512)).astype(dtype)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("member", DEVICE_MEMBERS)
@pytest.mark.parametrize("x_dtype", ["float32", "float16"])
def test_spec_is_final_msl_and_validates(member, x_dtype, artefact):
    template = device_member_template(member, x_dtype)
    assert "${XT}" in template.source, "the template carries the activation hole"
    spec = device_member_spec(member, x_dtype)
    assert "$" not in spec.source, "specialization must produce final source"
    case = device_member_case(_x(np.dtype(x_dtype)), artefact)
    case.validate_against(spec)  # raises SpecError on any mismatch


def test_unknown_member_or_dtype_is_rejected():
    with pytest.raises(KeyError):
        device_member_spec("device-nonexistent", "float32")
    with pytest.raises(KeyError):
        device_member_spec("device-dequant-loop", "float64")


# ---------------------------------------------------------------------------
# Correctness envelope and class structure
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.parametrize("member", DEVICE_MEMBERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_member_is_correct_within_sanity_envelope(member, dtype, session, artefact):
    x = _x(dtype)
    ref = r_contract(x, artefact)
    out = session.run(member, x, artefact)
    assert out.shape == ref.shape
    assert str(out.dtype) == str(x.dtype)
    error = float(np.abs(out.astype(np.float64) - ref).max())
    scale = float(np.abs(ref).max())
    assert error / scale < SANITY_RELATIVE[str(x.dtype)]


@pytest.mark.gpu
def test_members_are_pairwise_distinct(session, artefact):
    """The structural gate counts members on outputs, never on names."""
    x = _x(np.float32)
    outputs = {m: session.run(m, x, artefact) for m in DEVICE_MEMBERS}
    for a, b in itertools.combinations(DEVICE_MEMBERS, 2):
        assert not np.array_equal(outputs[a], outputs[b]), f"{a} == {b}"


@pytest.mark.gpu
@pytest.mark.parametrize("bits", [2, 3, 8])
def test_members_are_bits_generic(bits, session):
    """One compiled spec serves every width: only the artefact values change."""
    rng = np.random.default_rng(23)
    w = (rng.standard_normal((256, 256)) * 0.02).astype(np.float16)
    a = canonical_quantize(w, QuantContract(bits=bits, group_size=64))
    x = rng.standard_normal((2, 256)).astype(np.float32)
    ref = r_contract(x, a)
    scale = float(np.abs(ref).max())
    for member in DEVICE_MEMBERS:
        error = float(np.abs(session.run(member, x, a).astype(np.float64) - ref).max())
        assert error / scale < SANITY_RELATIVE["float32"], (member, bits)


@pytest.mark.gpu
def test_faulted_artefact_is_visible_to_a_device_member(session, artefact):
    """A member that could not see a bias-dropped artefact would be reading
    something other than the artefact it was handed."""
    x = _x(np.float32)
    ref = r_contract(x, artefact)
    clean = float(np.abs(session.run("device-dequant-loop", x, artefact).astype(np.float64) - ref).max())
    faulted = FAULTS["bias-dropped"](artefact)
    broken = float(np.abs(session.run("device-dequant-loop", x, faulted).astype(np.float64) - ref).max())
    assert broken > 100 * max(clean, 1e-12)


def test_code_carrier_is_exact_at_every_width():
    for bits in (2, 3, 4, 8):
        codes = np.arange((1 << bits), dtype=np.int32)
        assert np.array_equal(codes.astype(np.float16).astype(np.int32), codes)


# ---------------------------------------------------------------------------
# The frozen shapes
# ---------------------------------------------------------------------------
def test_case_speaks_the_frozen_vocabulary(artefact):
    x = _x(np.float16)
    case = device_member_case(x, artefact, label="frozen")
    assert tuple(case.output_shapes) == (((2, 512), "float16"),)
    assert case.label == "frozen"


def test_result_reading_returns_a_list():
    payload = np.ones((2, 2), np.float32)
    outputs = result_outputs(RunResult(outputs=[payload]))
    assert isinstance(outputs, list) and len(outputs) == 1
    assert np.array_equal(outputs[0], payload)


@pytest.mark.gpu
def test_session_compiles_each_spec_exactly_once(session):
    first = session.compiled("device-dequant-simd", "float32")
    second = session.compiled("device-dequant-simd", "float32")
    assert first is second
