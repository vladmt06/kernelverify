"""The pack's wide-tile quantized matvec: correct at every tile width it claims.

The kernel's whole reason to exist is holding M input vectors in one weight
pass, so the properties worth pinning are the ones that break when the tile
widens: agreement with the contract across M, the R switch at the register
wall, a d_out that does not divide R, and the artefact layout it assumes.

Skipped wholesale when MLX/Metal is unavailable so the suite stays green on
non-Mac CI, matching tests/test_metal_runner.py.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal unavailable", allow_module_level=True)

from kernelverify.pack.wide_qmv import (
    MIN_PROFITABLE_M,
    SUPPORTED_BITS,
    build,
    launch_config,
    pack_codes,
    rows_per_simdgroup,
    should_dispatch,
)
from kernelverify.schemas.native_ops import K_QUANT
from kernelverify.schemas.quant_contract import (
    ENSEMBLE,
    QuantContract,
    canonical_quantize,
    r_contract,
)

FP16_EPS = 9.77e-4


@pytest.fixture(scope="module")
def kernel():
    return build(mx)


def _artefact(d_out, d_in, seed=7, bits=4):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((d_out, d_in)).astype(np.float32) * 0.02).astype(np.float16)
    return w, canonical_quantize(w, QuantContract(bits=bits, group_size=64))


def _run(kernel, x, art, d_out, bits=4):
    m = x.shape[0]
    grid, threadgroup, r = launch_config(d_out, m)
    out = kernel(
        inputs=[mx.array(x), mx.array(pack_codes(art.q, bits)),
                mx.array(art.scales), mx.array(art.biases)],
        output_shapes=[(m, d_out)], output_dtypes=[mx.float16],
        grid=grid, threadgroup=threadgroup,
        template=[("T", mx.float16), ("M", m), ("R", r), ("BITS", bits)])[0]
    mx.eval(out)
    return np.array(out)


def _tolerance(x, art, ref):
    floor = max(float(np.max(np.abs(fn(x, art).astype(np.float64) - ref)))
                for fn in ENSEMBLE.values())
    base = 4.0 * FP16_EPS * float(np.max(np.abs(ref)))
    return max(base, K_QUANT * floor)


@pytest.mark.parametrize("m", [1, 2, 4, 5, 6, 7, 8, 10, 11])
def test_agrees_with_contract_at_every_tile_width(kernel, m):
    """One weight pass must give the same answer as the contract at any M."""
    d_out, d_in = 256, 512
    _, art = _artefact(d_out, d_in)
    x = np.random.default_rng(m).standard_normal((m, d_in)).astype(np.float16)
    ref = r_contract(x, art)
    err = float(np.max(np.abs(_run(kernel, x, art, d_out).astype(np.float64) - ref)))
    assert err <= _tolerance(x, art, ref)


def test_rows_per_simdgroup_drops_at_the_register_wall():
    """R = 8 and M*R past ~40 spilled when measured; the switch is at M = 10."""
    assert [rows_per_simdgroup(m) for m in (1, 8, 10)] == [4, 4, 4]
    assert rows_per_simdgroup(11) == 2


def test_defers_to_mlx_below_the_profitable_tile_width():
    """At 2 bits this kernel measured 0.81-0.89x at M = 1 and 2, so the pack
    must route those to MLX instead of shipping a regression."""
    assert not any(should_dispatch(m) for m in range(1, MIN_PROFITABLE_M))
    assert all(should_dispatch(m) for m in range(MIN_PROFITABLE_M, 12))


def test_handles_d_out_not_divisible_by_r(kernel):
    """Out-of-range rows read clamped and must never be stored."""
    d_out, d_in = 254, 512  # 254 % 4 == 2
    _, art = _artefact(d_out, d_in)
    x = np.random.default_rng(3).standard_normal((6, d_in)).astype(np.float16)
    ref = r_contract(x, art)
    got = _run(kernel, x, art, d_out)
    assert got.shape == (6, d_out)
    assert float(np.max(np.abs(got.astype(np.float64) - ref))) <= _tolerance(x, art, ref)


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
def test_artefact_layout_matches_mlx(kernel, bits):
    """The kernel reads MLX's own packing, so the two artefacts must be equal."""
    w, art = _artefact(128, 256, bits=bits)
    wq, scales, biases = mx.quantize(mx.array(w), group_size=64, bits=bits)
    mx.eval(wq, scales, biases)
    assert np.array_equal(pack_codes(art.q, bits), np.array(wq))
    assert np.array_equal(art.scales, np.array(scales))
    assert np.array_equal(art.biases, np.array(biases))


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
@pytest.mark.parametrize("m", [1, 6, 8])
def test_agrees_with_contract_at_every_bit_width(kernel, bits, m):
    """Sub-4-bit codes straddle words; the block striding must still be exact."""
    d_out, d_in = 256, 512
    _, art = _artefact(d_out, d_in, seed=bits, bits=bits)
    x = np.random.default_rng(m).standard_normal((m, d_in)).astype(np.float16)
    ref = r_contract(x, art)
    got = _run(kernel, x, art, d_out, bits=bits)
    assert float(np.max(np.abs(got.astype(np.float64) - ref))) <= _tolerance(x, art, ref)


def test_matches_mlx_quantized_matmul_within_contract(kernel):
    """Both are admissible implementations, so they agree within the floor."""
    d_out, d_in, m = 256, 512, 8
    w, art = _artefact(d_out, d_in)
    x = np.random.default_rng(5).standard_normal((m, d_in)).astype(np.float16)
    ref = r_contract(x, art)
    wq, scales, biases = mx.quantize(mx.array(w), group_size=64, bits=4)
    theirs = np.array(mx.quantized_matmul(mx.array(x), wq, scales, biases,
                                          transpose=True, group_size=64, bits=4))
    ours = _run(kernel, x, art, d_out)
    tol = _tolerance(x, art, ref)
    assert float(np.max(np.abs(theirs.astype(np.float64) - ref))) <= tol
    assert float(np.max(np.abs(ours.astype(np.float64) - ref))) <= tol
