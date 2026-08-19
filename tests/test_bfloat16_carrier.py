"""bfloat16 enters the contract: its eps, its carrier, and the device fact.

Training is a bf16 problem (ADR 0019), and three things had to be true
before a bf16 kernel could be gated at all: the contract needs an eps for
the dtype, numpy needs a way to hold the values, and the reference needs to
agree with what the device actually computes.

The last one is where the surprise was. Metal flushes bf16 subnormals to
zero and MLX's own CPU stream does not, so a bf16 kernel judged against a
CPU-built reference disagrees with the device in that band for reasons that
have nothing to do with the kernel. The band is declared outside the
contract and pinned here, so an MLX release that changes it turns this red
rather than quietly widening what the gate claims.

The helper tests are pure numpy and need no GPU.
"""

import numpy as np
import pytest

from conftest import requires_metal
from kernelverify.runners.spec import TENSOR_DTYPES
from kernelverify.schemas import bfloat16 as bf
from kernelverify.schemas.native_ops import _EPS

ALL_PATTERNS = np.arange(1 << 16, dtype=np.uint16)


# ---------------------------------------------------------------------------
# The contract entry
# ---------------------------------------------------------------------------
def test_the_eps_table_follows_one_convention_across_all_three_dtypes():
    """The table's own rule is 2**-(stored mantissa bits). numpy can check two
    of the three entries against its own finfo, which is where the pre-existing
    entries came from, rounded to three figures. bfloat16, which numpy cannot
    hold at all, is the third reading of the same rule and is written exactly."""
    assert _EPS["float32"] == pytest.approx(float(np.finfo(np.float32).eps),
                                            rel=5e-3)
    assert _EPS["float16"] == pytest.approx(float(np.finfo(np.float16).eps),
                                            rel=5e-3)
    assert _EPS["bfloat16"] == 2.0 ** -7 == bf.EPS


def test_bfloat16_is_far_coarser_than_float16_so_no_tolerance_carries_over():
    """The reason the entry had to exist rather than borrow float16's: the
    same tolerance formula written for float16 would be 8x too tight."""
    assert _EPS["bfloat16"] / _EPS["float16"] > 7.0


def test_the_carrier_is_a_declared_tensor_dtype_of_the_right_width():
    assert TENSOR_DTYPES["uint16"] is np.uint16
    assert np.dtype(TENSOR_DTYPES["uint16"]).itemsize == 2


# ---------------------------------------------------------------------------
# The two conversions
# ---------------------------------------------------------------------------
def test_decoding_is_exact_for_every_one_of_the_65536_patterns():
    """bf16 is float32's top 16 bits, so decoding invents nothing: re-encoding
    every finite pattern returns the pattern it came from."""
    values = bf.from_bits(ALL_PATTERNS)
    finite = np.isfinite(values)
    np.testing.assert_array_equal(bf.to_bits(values[finite]),
                                  ALL_PATTERNS[finite])


def test_encoding_rounds_to_nearest_with_ties_to_even():
    """The two exact ties either side of 1.0: one has an even neighbour below
    and stays, the other has an odd one and steps up. A round-half-up encoder
    passes the first and fails the second, which is why both are here."""
    ties = np.array([0x3F808000, 0x3F818000], dtype=np.uint32).view(np.float32)
    np.testing.assert_array_equal(bf.to_bits(ties),
                                  np.array([0x3F80, 0x3F82], dtype=np.uint16))


def test_the_specials_survive_encoding_as_themselves():
    """Zeros keep their sign, infinities stay infinite, and a NaN whose whole
    payload sits in the discarded half becomes a quiet NaN rather than the
    infinity a plain shift-and-round would produce."""
    words = np.array([0x00000000, 0x80000000, 0x7F800000, 0xFF800000,
                      0x7F800001, 0xFF800001], dtype=np.uint32)
    got = bf.to_bits(words.view(np.float32))
    np.testing.assert_array_equal(got[:4], np.array(
        [0x0000, 0x8000, 0x7F80, 0xFF80], dtype=np.uint16))
    assert np.all(np.isnan(bf.from_bits(got[4:]))), (
        "a NaN must not round into an infinity")


def test_the_largest_float32_overflows_to_infinity_as_rounding_requires():
    """Not an edge case worth arguing about: float32's largest finite value
    sits above the midpoint between bf16's largest finite value and the next
    representable step, so nearest-rounding sends it to infinity."""
    biggest = np.array([0x7F7FFFFF], dtype=np.uint32).view(np.float32)
    assert bf.to_bits(biggest)[0] == np.uint16(0x7F80)


# ---------------------------------------------------------------------------
# Agreement with MLX, which is the ground-truth honesty rule for this repo,
# and the one measured place the two disagree.
# ---------------------------------------------------------------------------
@requires_metal
def test_the_encoder_agrees_with_mlx_on_the_gpu_outside_the_subnormal_band():
    import mlx.core as mx

    values = bf.from_bits(ALL_PATTERNS)
    eligible = np.isfinite(values) & (np.abs(values) >= 2.0 ** -126)
    with mx.stream(mx.gpu):
        theirs = np.array(mx.array(values).astype(mx.bfloat16).view(mx.uint16))
    np.testing.assert_array_equal(theirs[eligible], ALL_PATTERNS[eligible])


@requires_metal
def test_metal_flushes_bf16_subnormals_and_mlxs_cpu_stream_does_not():
    """The declared exclusion, measured rather than assumed. Both readings are
    pinned: if Metal stops flushing, or if the CPU stream starts, the contract's
    excluded band is wrong and this says so."""
    import mlx.core as mx

    values = bf.from_bits(ALL_PATTERNS)
    subnormal = np.isfinite(values) & (values != 0) & (np.abs(values) < 2.0 ** -126)
    assert int(subnormal.sum()) == 254, "the bf16 subnormal band is 254 patterns"

    with mx.stream(mx.gpu):
        on_gpu = np.array(mx.array(values).astype(mx.bfloat16).view(mx.uint16))
    with mx.stream(mx.cpu):
        on_cpu = np.array(mx.array(values).astype(mx.bfloat16).view(mx.uint16))

    assert np.all(on_gpu[subnormal] & np.uint16(0x7FFF) == 0), (
        "Metal is expected to flush every bf16 subnormal to zero")
    np.testing.assert_array_equal(on_cpu[subnormal], ALL_PATTERNS[subnormal])


@requires_metal
def test_a_bf16_array_cannot_reach_numpy_without_the_carrier():
    """The fact the uint16 carrier exists for: numpy has no bfloat16, so the
    direct conversion raises and the uint16 view is the only route."""
    import mlx.core as mx

    array = mx.array([1.0, 2.0], dtype=mx.bfloat16)
    with pytest.raises(RuntimeError):
        np.array(array)
    np.testing.assert_array_equal(bf.from_bits(np.array(array.view(mx.uint16))),
                                  np.array([1.0, 2.0], dtype=np.float32))
