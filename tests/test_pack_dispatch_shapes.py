"""The E2E dispatch-shape derivation: the shapes the serving path prices.

The pack's boundary pricing and gate coverage run at the qmv shapes mlx-lm
actually dispatches when decoding Qwen3-4B (ruling D1). Until the runner
lane records the live shapes (risk RR1), they are derived from the model
config and the mlx-lm qwen3 module structure, so the derivation itself is
what these tests pin: every projection present, GQA respected, duplicates
merged, and the kernel's own alignment requirement checked at the source.

Pure-python module, no MLX needed.
"""

import pytest

from kernelverify.pack.dispatch_shapes import (
    PINNED_QWEN3_4B,
    qmv_dispatch_shapes,
)


def test_derives_the_six_distinct_qwen3_4b_decode_shapes():
    """One entry per distinct (d_out, d_in); shared shapes merge their names.

    The values trace to Qwen/Qwen3-4B config.json: hidden 2560, 32 heads x
    head_dim 128 (explicit, NOT hidden/heads), 8 KV heads, intermediate 9728,
    vocab 151936, tied embeddings dispatching lm_head through the quantized
    embedding table.
    """
    shapes = qmv_dispatch_shapes(PINNED_QWEN3_4B)
    assert [(s.name, s.d_out, s.d_in) for s in shapes] == [
        ("q_proj", 4096, 2560),
        ("k_proj/v_proj", 1024, 2560),
        ("o_proj", 2560, 4096),
        ("gate_proj/up_proj", 9728, 2560),
        ("down_proj", 2560, 9728),
        ("lm_head", 151936, 2560),
    ]


def test_gqa_width_follows_kv_heads_not_heads():
    """k/v projections are kv_heads * head_dim wide; a config with full MHA
    (kv_heads == heads) must merge q with k/v instead of duplicating."""
    mha = dict(PINNED_QWEN3_4B, num_key_value_heads=32)
    shapes = {s.name: (s.d_out, s.d_in) for s in qmv_dispatch_shapes(mha)}
    assert "q_proj/k_proj/v_proj" in shapes
    assert shapes["q_proj/k_proj/v_proj"] == (4096, 2560)


def test_head_dim_is_read_explicitly_never_inferred():
    """Qwen3-4B has head_dim 128 while hidden/heads is 80; inferring would
    derive the wrong o_proj and q_proj silently, so a missing key must raise."""
    config = {k: v for k, v in PINNED_QWEN3_4B.items() if k != "head_dim"}
    with pytest.raises(KeyError):
        qmv_dispatch_shapes(config)


def test_every_shape_satisfies_the_kernel_alignment():
    """wide_qmv requires d_in divisible by 64 (one group is a whole number of
    8-code blocks); the derivation must refuse a config that breaks this
    rather than let the gate fail deep inside Metal."""
    for s in qmv_dispatch_shapes(PINNED_QWEN3_4B):
        assert s.d_in % 64 == 0, s
    bad = dict(PINNED_QWEN3_4B, hidden_size=2500)
    with pytest.raises(ValueError):
        qmv_dispatch_shapes(bad)
