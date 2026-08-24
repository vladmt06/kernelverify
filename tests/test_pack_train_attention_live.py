"""The training-attention forward kernel on the device, judged by the battery.

Every verdict here comes from `NATIVE_OPS["train_attention"]` through the shared
`kernelverify.pack.verify` path, so this file states no tolerance of its own.
The one comparison that is NOT a verdict is against
`mx.fast.scaled_dot_product_attention`: it is a second opinion at the two
registered cells, where an fp64 reference would need half a gigabyte of scores,
and it is labelled as the smoke it is.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from conftest import requires_metal
from kernelverify.pack import train_attention as ta
from kernelverify.pack.verify import attn_inputs, group_heads, verify_output
from kernelverify.schemas.native_ops import attn_lse_reference

DEFAULT_KNOBS = {"SGROUPS": 2, "BKEY": 16}

# One row past a 64-wide tile, one row short of it, the exact boundaries, and
# the degenerate widths where a whole tile is masked but one key still is not.
EDGE_WIDTHS = (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
               127, 128, 129)


def _dtype_name(dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _operands(mx, batch, n_q, n_kv, t_len, head_dim, dtype, seed=0):
    """Drawn at fp32 and cast on the device, so the reference can read back
    exactly what the kernel was given: bfloat16 has no numpy dtype."""
    rng = np.random.default_rng(seed)

    def draw(heads, scale):
        return mx.array((rng.standard_normal((batch, heads, t_len, head_dim))
                         * scale).astype(np.float32)).astype(dtype)

    return draw(n_q, 0.4), draw(n_kv, 0.9), draw(n_kv, 0.9)


def _f32(mx, x) -> np.ndarray:
    return np.array(x.astype(mx.float32))


def _verdict(mx, q, k, v, out, dtype):
    inputs = attn_inputs(_f32(mx, q), _f32(mx, k), _f32(mx, v))
    return verify_output("train_attention", inputs,
                         group_heads(_f32(mx, out), k.shape[1]),
                         _dtype_name(dtype))


@functools.lru_cache(maxsize=1)
def _limits():
    """What this chip can launch, as this chip reports it. Read through the
    project's own probe rather than hardcoded, because the whole point of a
    knob space is that its launchable half is a property of the machine."""
    from kernelverify.compiler.search_space import Limits
    from kernelverify.runners.metal import MetalRunner

    return Limits.probe(MetalRunner())


def _skip_if_unlaunchable(knobs, head_dim):
    """A setting this device cannot stage is skipped with the arithmetic that
    says why, rather than failing as though the kernel were wrong."""
    needed = ta.fwd_threadgroup_bytes(knobs, head_dim)
    limit = _limits().max_threadgroup_memory
    if needed > limit:
        pytest.skip(f"{knobs} stages {needed} bytes, over this device's {limit}")
    if ta.fwd_threads(knobs) > _limits().max_threads_per_threadgroup:
        pytest.skip(f"{knobs} wants more threads than this device allows")


def _forward(mx, q, k, v, knobs=None):
    knobs = knobs or DEFAULT_KNOBS
    _skip_if_unlaunchable(knobs, q.shape[-1])
    kern = ta.build_fwd(mx, knobs, q.shape[-1])
    out, lse = ta.run_fwd(mx, kern, knobs, q, k, v)
    mx.eval(out, lse)
    return out, lse


@requires_metal
@pytest.mark.parametrize("t_len", EDGE_WIDTHS)
def test_the_forward_holds_at_every_tile_edge(t_len):
    """A tiled kernel's faults live at the edges: the last key tile is partly
    masked, the last query tile is partly out of range, and at these widths the
    causal bound falls inside a tile rather than on its boundary."""
    mx = pytest.importorskip("mlx.core")
    q, k, v = _operands(mx, 1, 4, 2, t_len, 128, mx.float16)
    out, _lse = _forward(mx, q, k, v)
    verdict = _verdict(mx, q, k, v, out, mx.float16)
    assert verdict.ok, f"width {t_len}: {verdict}"


@requires_metal
@pytest.mark.parametrize("t_len", (65, 128, 1057))
def test_the_forward_holds_at_bfloat16_which_is_what_the_seam_receives(t_len):
    """Measured on the real training path: the seam is handed bfloat16, because
    the shipped 4-bit artifacts store their scales in it and mlx-lm takes the
    model dtype from the scales. A kernel verified only at float16 would be a
    kernel verified at a dtype this product never sees."""
    mx = pytest.importorskip("mlx.core")
    q, k, v = _operands(mx, 1, 4, 2, t_len, 128, mx.bfloat16)
    out, _lse = _forward(mx, q, k, v)
    verdict = _verdict(mx, q, k, v, out, mx.bfloat16)
    assert verdict.ok, f"width {t_len}: {verdict}"


@requires_metal
@pytest.mark.parametrize("n_kv", (1, 2, 4, 8))
def test_every_group_ratio_indexes_its_own_key_value_head(n_kv):
    """Eight query heads over one, two, four or eight key-value heads. A kernel
    that computes `hq / group` wrongly is silently correct at a ratio of one."""
    mx = pytest.importorskip("mlx.core")
    q, k, v = _operands(mx, 2, 8, n_kv, 65, 128, mx.float16)
    out, _lse = _forward(mx, q, k, v)
    verdict = _verdict(mx, q, k, v, out, mx.float16)
    assert verdict.ok, f"{n_kv} key-value heads: {verdict}"


@requires_metal
def test_the_row_logsumexp_is_the_statistic_the_backward_will_read():
    """The second output carries its own correctness claim: an L wrong by a
    constant per row leaves the forward's own output exactly right, because the
    constant cancels in the normalisation, and only the backward would see it."""
    mx = pytest.importorskip("mlx.core")
    q, k, v = _operands(mx, 2, 8, 2, 129, 128, mx.float16)
    _out, lse = _forward(mx, q, k, v)
    reference = attn_lse_reference(attn_inputs(_f32(mx, q), _f32(mx, k),
                                               _f32(mx, v)))
    got = group_heads(np.array(lse)[..., None], k.shape[1])[..., 0]
    assert np.max(np.abs(got - reference)) < 1e-4


@requires_metal
def test_a_key_a_row_may_not_see_cannot_change_that_row():
    """Causality, stated as the experiment rather than as the mask: move the
    last key and value, and every row but the last must be bit-identical. This
    is the fault a training attention kernel must never ship."""
    mx = pytest.importorskip("mlx.core")
    q, k, v = _operands(mx, 1, 4, 2, 65, 128, mx.float16)
    before, _lse = _forward(mx, q, k, v)

    bump = mx.zeros(k.shape, dtype=mx.float32)
    bump[:, :, -1, :] = 3.0
    moved_k = (k.astype(mx.float32) + bump).astype(mx.float16)
    moved_v = (v.astype(mx.float32) + bump).astype(mx.float16)
    after, _lse = _forward(mx, q, moved_k, moved_v)

    assert np.array_equal(np.array(before[:, :, :-1, :]),
                          np.array(after[:, :, :-1, :]))
    assert not np.array_equal(np.array(before[:, :, -1, :]),
                              np.array(after[:, :, -1, :]))


@requires_metal
@pytest.mark.parametrize("knobs", (
    {"SGROUPS": 1, "BKEY": 16},
    {"SGROUPS": 2, "BKEY": 32},
    {"SGROUPS": 4, "BKEY": 16},
    {"SGROUPS": 8, "BKEY": 64},
))
def test_a_knob_setting_changes_the_schedule_and_not_the_answer(knobs):
    """The whole premise of tuning: these settings rescale the running maximum
    at different points and stage different amounts, and every one of them must
    land on the same answer."""
    mx = pytest.importorskip("mlx.core")
    q, k, v = _operands(mx, 1, 4, 2, 129, 128, mx.float16)
    out, _lse = _forward(mx, q, k, v, knobs)
    verdict = _verdict(mx, q, k, v, out, mx.float16)
    assert verdict.ok, f"{knobs}: {verdict}"


@requires_metal
@pytest.mark.parametrize("cell,batch,t_len", (("B4:T128", 4, 65),
                                              ("B2:T2048", 2, 1057)))
def test_the_registered_cells_agree_with_the_fused_reference(cell, batch, t_len):
    """The two cells at the real Qwen3-4B geometry, against MLX's own fused
    attention. A SMOKE and not a verdict: at 32 heads and width 1057 an fp64
    score matrix is half a gigabyte, so the battery judges the same kernel at
    the smaller widths above and this says the big shapes dispatch and agree."""
    mx = pytest.importorskip("mlx.core")
    assert ta.cell_key(batch, t_len) == cell
    q, k, v = _operands(mx, batch, ta.N_Q_HEADS, ta.N_KV_HEADS, t_len,
                        ta.HEAD_DIM, mx.bfloat16)
    out, _lse = _forward(mx, q, k, v)
    fused = mx.fast.scaled_dot_product_attention(q, k, v, scale=ta.SCALE,
                                                 mask="causal")
    mx.eval(fused)
    gap = float(mx.max(mx.abs(out.astype(mx.float32) - fused.astype(mx.float32))))
    assert gap < 2e-2, f"{cell}: {gap}"


@requires_metal
def test_the_simdgroup_fragment_layout_is_the_one_this_kernel_indexes_by():
    """The assumption the register softmax rests on, measured rather than read.

    The kernel applies the causal mask, the row reductions and the output store
    directly to the two elements each lane holds of an 8x8 fragment, which
    needs the mapping from lane to (row, column). MSL does not promise a
    mapping, so this test states the one the kernel indexes by and asks the
    chip. A machine whose layout differs fails here, and would in any case fail
    the on-device verification and route to stock rather than answer wrongly.
    """
    mx = pytest.importorskip("mlx.core")

    probe = mx.fast.metal_kernel(
        name="fragment_layout", input_names=["src"], output_names=["out"],
        source="""
    const uint lane = thread_position_in_threadgroup.x;
    threadgroup float tile[64];
    for (uint e = lane; e < 64; e += 32) { tile[e] = src[e]; }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::simdgroup_float8x8 m;
    metal::simdgroup_load(m, tile, 8);
    out[lane * 2 + 0] = m.thread_elements()[0];
    out[lane * 2 + 1] = m.thread_elements()[1];
""")
    held = probe(inputs=[mx.array(np.arange(64, dtype=np.float32))],
                 output_shapes=[(64,)], output_dtypes=[mx.float32],
                 grid=(32, 1, 1), threadgroup=(32, 1, 1))[0]
    mx.eval(held)
    held = np.array(held).astype(int)

    for lane in range(32):
        row = ((lane % 8) // 2) + (lane // 16) * 4      # the kernel's sg_row
        col = (lane % 2) * 2 + ((lane // 8) % 2) * 4    # the kernel's sg_col
        assert held[2 * lane] == row * 8 + col, lane
        assert held[2 * lane + 1] == row * 8 + col + 1, lane

    # And the four lanes that share a row are at exclusive-or distances 1 and
    # 8, which is what makes a row reduction two shuffles.
    for lane in range(32):
        row = ((lane % 8) // 2) + (lane // 16) * 4
        for partner in (lane ^ 1, lane ^ 8, lane ^ 9):
            assert ((partner % 8) // 2) + (partner // 16) * 4 == row, (lane, partner)
