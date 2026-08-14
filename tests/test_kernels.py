"""Semantic pins for the reference kernels.

Cross-implementation agreement (flash vs plain attention, padded vs block
softmax) and fp64 naive formulas pin what "correct by default" means, without
duplicating the kernels' own code. Shapes deliberately include non-powers of
two and several flash tiles so the padding and rescale paths actually run.
"""

import numpy as np

from kernelverify.reference.kernels import (
    KERNELS,
    attention,
    flash_attention,
    gelu,
    leaky_relu,
    matmul,
    silu,
    softmax,
    softmax_padded,
)

RNG = np.random.default_rng(42)


def arr(*shape, dtype=np.float32):
    return RNG.standard_normal(shape).astype(dtype)


def test_softmax_rows_sum_to_one_including_padded_rows():
    x = arr(2, 3, 7)
    out = softmax({"input": x}).astype(np.float64)
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-6)


def test_padded_softmax_agrees_with_block_softmax():
    x = arr(2, 3, 7)
    a = softmax({"input": x}).astype(np.float64)
    b = softmax_padded({"input": x}).astype(np.float64)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_matmul_matches_fp64_matmul():
    a, b = arr(8, 32), arr(32, 5)
    out = matmul({"a": a, "b": b}).astype(np.float64)
    np.testing.assert_allclose(out, a.astype(np.float64) @ b.astype(np.float64),
                               atol=1e-5)


def test_attention_matches_naive_fp64_attention():
    q, k, v = arr(7, 96, 16), arr(7, 96, 16), arr(7, 96, 16)
    inputs = {"q": q[0], "k": k[0], "v": v[0]}
    out = attention(inputs).astype(np.float64)
    q64, k64, v64 = (inputs[n].astype(np.float64) for n in ("q", "k", "v"))
    scores = (q64 @ k64.T) * (16 ** -0.5)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(out, p @ v64, atol=1e-5)


def test_flash_attention_agrees_with_plain_attention_across_tiles():
    inputs = {"q": arr(7, 16), "k": arr(96, 16), "v": arr(96, 16)}
    a = attention(inputs).astype(np.float64)
    b = flash_attention(inputs).astype(np.float64)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_elementwise_fixed_points():
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(leaky_relu({"input": x}), [-0.01, 0.0, 1.0], atol=1e-7)
    assert silu({"input": x})[1] == 0.0
    assert gelu({"input": x})[1] == 0.0


def test_outputs_keep_the_input_dtype():
    from kernelverify.reference.native_kernels import NATIVE_KERNELS

    for name, fn in KERNELS.items():
        if name in NATIVE_KERNELS:
            continue  # native ops have their own input schemas and test suite
        if name == "matmul":
            out = fn({"a": arr(4, 8, dtype=np.float16), "b": arr(8, 3, dtype=np.float16)})
        elif name in ("attention", "flash_attention"):
            out = fn({"q": arr(4, 64, 8, dtype=np.float16)[0],
                      "k": arr(4, 64, 8, dtype=np.float16)[1],
                      "v": arr(4, 64, 8, dtype=np.float16)[2]})
        else:
            out = fn({"input": arr(2, 3, 5, dtype=np.float16)})
        assert out.dtype == np.float16, name


def test_pinned_seams_change_the_output():
    x = {"input": arr(2, 3, 8)}
    assert np.max(np.abs(gelu(x) - gelu(x, leading_scale=1.0))) > 1e-2
    assert np.max(np.abs(silu(x) - silu(x, beta=2.0))) > 1e-2
    rms = KERNELS["rmsnorm"]
    assert np.max(np.abs(rms(x) - rms(x, use_sqrt=False))) > 1e-2
    a, b = arr(8, 32), arr(32, 5)
    good = matmul({"a": a, "b": b})
    bad = matmul({"a": a, "b": b}, accumulate=False)
    assert np.max(np.abs(good - bad)) > 1e-2
