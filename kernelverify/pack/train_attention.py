"""Causal grouped-query attention at TRAINING widths, and the cells it is tuned at.

`kernelverify/pack/kv_attention.py` is the decode sibling: one query position
against a quantized cache, where the whole cost is reading the cache. Training
is the other regime. Every position in the step attends at once, the mask is
causal, nothing is quantized, and the operator carries a backward.

What mlx-lm actually dispatches, read off the installed stack rather than
assumed: `mlx_lm.models.qwen3.Attention.__call__` calls

    scaled_dot_product_attention(queries, keys, values, cache=cache,
                                 scale=self.scale, mask=mask)

which is a re-export of `mlx_lm.models.base.scaled_dot_product_attention`, and
in a training step `cache` is None and `mask` is the literal string "causal".
That function forwards to `mx.fast.scaled_dot_product_attention`, which fuses
in a plain call and DECOMPOSES under a gradient trace into two matmuls and a
softmax, because MLX implements no fused attention backward at all. Reading the
graph MLX builds says so directly: the fused primitive appears once in a plain
call and zero times under `mx.grad`, where a Softmax appears instead.

So stock's training attention is the composed path in both directions, at every
one of the 36 layers, and the composed path materialises a
(batch, heads, width, width) score matrix per layer: 286 MB at batch 2 and
width 1057 in fp32. Not building it is worth as much as the time is, because it
is what puts a 16 GB machine in reach.

The geometry below is Qwen3-4B's and only Qwen3-4B's. A call site whose heads,
group ratio or head dimension differ routes to stock rather than to a kernel
verified somewhere else, the same rule `train_qmm.shape_name` applies to the
projections.
"""

from __future__ import annotations

import re

# Qwen3-4B's attention geometry, as `mlx_lm.models.qwen3.ModelArgs` carries it.
# The group ratio is what makes this grouped-query rather than multi-head: four
# query heads share each key-value head, so a backward must sum a group's four
# gradients onto the one head they read.
N_Q_HEADS = 32
N_KV_HEADS = 8
GQA_GROUP = N_Q_HEADS // N_KV_HEADS
HEAD_DIM = 128

# `head_dim**-0.5`, the value qwen3's Attention computes and hands to the seam.
# Written out as the constant rather than recomputed, because a kernel that
# scales by a different value is wrong in a way no shape check would catch.
SCALE = HEAD_DIM ** -0.5

# The batch sizes a tuned configuration exists for. A step at any other batch
# routes to stock: the tile geometry that wins at four rows of queries is not
# the one that wins at sixteen, and a configuration measured at one batch says
# nothing about another.
BATCHES = (1, 2, 4)

# Width is data-dependent (mlx-lm pads a batch to one past the next multiple of
# 32 above its longest row), so a tuned configuration is stored against a
# BUCKET. Powers of two, because that is the axis along which a tile
# configuration's behaviour changes; floored at 128 because below that the
# whole step is small enough that the composed path's overhead dominates
# anything a kernel could win; capped because past the cap the key loop is
# long enough that its own steady state, not its edges, sets the winner.
T_BUCKET_FLOOR = 128
T_BUCKET_CAP = 2048

# The two registered bands, from the two pinned corpus slices: the short band
# is width 65 at batch 4, the long band is width 1057 at batch 2. Every gate
# and every timing arm in this pack reports at both, because attention is the
# one region whose share is quadratic in the width, so a reading at one band
# says nothing about the other.
CELLS = ("B4:T128", "B2:T2048")


def t_bucket(t: int) -> int:
    """The bucket a step's width is tuned and looked up under.

    Rounds UP to the next power of two so a bucket's tuned configuration was
    measured on a problem at least as large as the one it is applied to.
    Rounding down would credit a configuration with a speed it was never shown
    to reach at the width it runs at.
    """
    if t < 1:
        raise ValueError(f"a step {t} tokens wide has nothing to attend over")
    bucket = T_BUCKET_FLOOR
    while bucket < t and bucket < T_BUCKET_CAP:
        bucket *= 2
    return bucket


def cell_key(batch: int, t: int) -> str:
    """The knob map's key: one tuned configuration per batch per bucket."""
    if batch not in BATCHES:
        raise ValueError(f"batch {batch} is not one of {list(BATCHES)}")
    return f"B{batch}:T{t_bucket(t)}"


def cell_name(batch: int, t: int) -> str | None:
    """Which cell this call site is, or None if it is not one of ours.

    None is a route to stock, never a guess at the nearest cell: a
    configuration measured at one batch says nothing about another.
    """
    if batch not in BATCHES:
        return None
    return cell_key(batch, t)


def geometry_ok(n_q_heads: int, n_kv_heads: int, head_dim: int) -> bool:
    """Whether a call site is the geometry this pack is verified for.

    Checked at every call rather than assumed from the model path: a config
    file names the model, and the tensors name what the model actually built.
    """
    return (n_q_heads == N_Q_HEADS and n_kv_heads == N_KV_HEADS
            and head_dim == HEAD_DIM)



# ---------------------------------------------------------------------------
# The forward kernel.
# ---------------------------------------------------------------------------
# One simdgroup owns eight query rows of one (batch, query head) and walks the
# keys in BKEY-wide tiles, carrying a running maximum and a running normaliser
# per row. The full score matrix is never built, which is the memory half of
# the claim; the two matmuls run on the simdgroup matrix units, which is the
# time half.
#
#     q rows (8, DHEAD)        k tile (BKEY, DHEAD)      v tile (BKEY, DHEAD)
#          |                          |                         |
#          +--- S = Q K^T, on 8x8 ----+                         |
#          |    matrix fragments                                |
#     softmax IN REGISTERS: mask, running max, exp, running sum |
#          |                                                    |
#          +--------------- O += P V, same units ---------------+
#
# Three decisions carry the speed, and each was measured against the
# alternative rather than assumed.
#
# NOTHING IS STAGED IN THREADGROUP MEMORY. A fragment load wants a float
# source, and the model holds its tensors at two bytes, so an earlier design
# converted each tile into threadgroup memory on the way past. That staging
# cost more than the matmuls it fed: measured at the long band, 15.6 ms
# against 6.8 ms for this arrangement. The door converts the whole operand
# once with MLX's own cast and the kernel reads its fragments straight from
# device memory, where the cache serves the rows every simdgroup shares.
#
# THE SOFTMAX STAYS IN THE FRAGMENTS. The scores land in registers as 8x8
# matrices, and the mask, the running maximum, the exponential and the running
# sum are applied to those registers in place, so the probabilities are already
# the left operand of the second matmul. The earlier design stored the scores
# to threadgroup memory, ran a scalar softmax over them, and loaded them back;
# that round trip is what the register form removes.
#
# A ROW OF A FRAGMENT IS HELD BY FOUR LANES, at exclusive-or distances 1 and 8,
# so a row-wide maximum or sum is two shuffles rather than a threadgroup
# reduction. That mapping is a property of the chip: it is measured, pinned by
# `test_the_simdgroup_fragment_layout_is_the_one_this_kernel_indexes_by`, and a
# machine whose layout differs fails the on-device verification and routes to
# stock rather than returning a wrong answer.
#
# Every accumulator is fp32. The multiplicands are fp32 too, because the door
# converts once; contract clause C1 is satisfied by construction and the source
# lint has no narrow declaration to find.
#
# Two outputs, not one. `l` is the row logsumexp, which this forward computes
# on its way through the key loop and which the backward needs to rebuild the
# probabilities without a second pass over the scores. Handing it back costs
# one float per query row; recomputing it in the backward would cost a whole
# extra forward.
ATTN_FWD_MSL = """
    constexpr uint LANES = 32u;
    constexpr uint ROWS  = SGROUPS * 8u;      // query rows per threadgroup
    constexpr uint DB    = DHEAD / 8u;        // head-dimension blocks
    constexpr uint KB    = BKEY / 8u;         // key blocks per tile

    const uint lane = thread_position_in_threadgroup.x;
    const uint sg   = thread_position_in_threadgroup.y;

    const uint t_len = tlen[0];               // rows that exist
    const uint t_pad = T_PAD;                 // rows the operands are padded to
    const uint n_q   = HEADS_Q;
    const uint n_kv  = HEADS_KV;
    const uint group = n_q / n_kv;

    const uint bh  = threadgroup_position_in_grid.z;
    const uint bat = bh / n_q;
    const uint hq  = bh - bat * n_q;
    const uint hkv = hq / group;

    const uint q0   = threadgroup_position_in_grid.x * ROWS;
    const uint row0 = q0 + sg * 8u;

    const ulong q_base  = ((ulong)(bat * n_q  + hq ) * (ulong)t_pad) * DHEAD;
    const ulong kv_base = ((ulong)(bat * n_kv + hkv) * (ulong)t_pad) * DHEAD;

    // Where this lane's two elements of any 8x8 fragment live.
    const uint sg_row = ((lane % 8u) / 2u) + (lane / 16u) * 4u;
    const uint sg_col = (lane % 2u) * 2u + ((lane / 8u) % 2u) * 4u;
    const uint gi = row0 + sg_row;

    metal::simdgroup_float8x8 qf[DB];
    #pragma clang loop unroll(full)
    for (uint db = 0; db < DB; ++db) {
        metal::simdgroup_load(qf[db], q + q_base + (ulong)row0 * DHEAD + db * 8u,
                              DHEAD);
    }
    metal::simdgroup_float8x8 of[DB];
    #pragma clang loop unroll(full)
    for (uint db = 0; db < DB; ++db) {
        of[db] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m_run = -INFINITY;
    float l_run = 0.0f;

    // The causal bound, applied to the whole threadgroup: a key past its last
    // query row can never contribute to any of its rows, so that tile is never
    // read rather than read and masked.
    const uint k_end = min(q0 + ROWS, t_len);

    for (uint k0 = 0; k0 < k_end; k0 += BKEY) {
        metal::simdgroup_float8x8 sf[KB];
        #pragma clang loop unroll(full)
        for (uint kb = 0; kb < KB; ++kb) {
            sf[kb] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        #pragma clang loop unroll(full)
        for (uint db = 0; db < DB; ++db) {
            #pragma clang loop unroll(full)
            for (uint kb = 0; kb < KB; ++kb) {
                metal::simdgroup_float8x8 kf;
                metal::simdgroup_load(kf,
                    k + kv_base + (ulong)(k0 + kb * 8u) * DHEAD + db * 8u,
                    DHEAD, ulong2(0, 0), true);
                metal::simdgroup_multiply_accumulate(sf[kb], qf[db], kf, sf[kb]);
            }
        }

        float m_tile = -INFINITY;
        #pragma clang loop unroll(full)
        for (uint kb = 0; kb < KB; ++kb) {
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 2u; ++e) {
                uint gj = k0 + kb * 8u + sg_col + e;
                // A padded row is not a row: gj < t_len is what stops a key
                // that only exists to round the operand up from being read.
                float s = (gi < t_len && gj < t_len && gj <= gi)
                        ? sf[kb].thread_elements()[e] * SCALE_LIT : -INFINITY;
                sf[kb].thread_elements()[e] = s;
                m_tile = metal::max(m_tile, s);
            }
        }
        m_tile = metal::max(m_tile, metal::simd_shuffle_xor(m_tile, 1u));
        m_tile = metal::max(m_tile, metal::simd_shuffle_xor(m_tile, 8u));

        float m_new = metal::max(m_run, m_tile);
        // A row that has seen nothing yet carries -INFINITY, and -INF minus
        // -INF is a NaN rather than the zero the algebra wants.
        float safe = metal::isfinite(m_new) ? m_new : 0.0f;
        float corr = metal::isfinite(m_run) ? metal::exp(m_run - safe) : 0.0f;
        float l_tile = 0.0f;
        #pragma clang loop unroll(full)
        for (uint kb = 0; kb < KB; ++kb) {
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 2u; ++e) {
                float p = metal::exp(sf[kb].thread_elements()[e] - safe);
                sf[kb].thread_elements()[e] = p;
                l_tile += p;
            }
        }
        l_tile += metal::simd_shuffle_xor(l_tile, 1u);
        l_tile += metal::simd_shuffle_xor(l_tile, 8u);
        l_run = l_run * corr + l_tile;
        m_run = m_new;

        #pragma clang loop unroll(full)
        for (uint db = 0; db < DB; ++db) {
            of[db].thread_elements()[0] *= corr;
            of[db].thread_elements()[1] *= corr;
        }

        #pragma clang loop unroll(full)
        for (uint kb = 0; kb < KB; ++kb) {
            #pragma clang loop unroll(full)
            for (uint db = 0; db < DB; ++db) {
                metal::simdgroup_float8x8 vf;
                metal::simdgroup_load(vf,
                    v + kv_base + (ulong)(k0 + kb * 8u) * DHEAD + db * 8u, DHEAD);
                metal::simdgroup_multiply_accumulate(of[db], sf[kb], vf, of[db]);
            }
        }
    }

    // Every element of both outputs is written, including the rows that exist
    // only to round the operands up to whole tiles: a row past the true width
    // is masked everywhere, so its normaliser is zero and it stores zeros
    // rather than being left as whatever the allocator last held. A kernel
    // that leaves an output element untouched is what the unwritten-output
    // gate exists to refuse, and a padded row is not an excuse.
    float inv = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    #pragma clang loop unroll(full)
    for (uint db = 0; db < DB; ++db) {
        ulong at = q_base + (ulong)gi * DHEAD + db * 8u + sg_col;
        o[at]     = (T)(of[db].thread_elements()[0] * inv);
        o[at + 1] = (T)(of[db].thread_elements()[1] * inv);
    }
    // One lane per row holds column zero, so this writes once per row.
    if (sg_col == 0u) {
        l[(ulong)(bat * n_q + hq) * (ulong)t_pad + gi] =
            (l_run > 0.0f) ? (m_run + metal::log(l_run)) : -INFINITY;
    }
"""

FWD_KERNEL_NAME = "kv_train_attention_fwd"
FWD_INPUT_NAMES = ["q", "k", "v", "tlen"]
FWD_OUTPUT_NAMES = ["o", "l"]

# The knob axes. Nothing here is dropped for being slow: launchability is the
# only filter, it is read from the device, and a merely inefficient setting
# loses on the clock, which is a measurement rather than an opinion.
#
# SGROUPS is simdgroups per threadgroup, which sets how many query rows share a
# dispatch; BKEY is the key tile width, which sets how often the running
# maximum is rescaled and how many score fragments a lane holds at once. Both
# are multiples of eight because a fragment is 8x8, and the head dimension is a
# multiple of eight for the same reason.
FWD_KNOB_AXES = {
    "SGROUPS": [1, 2, 4, 8],
    "BKEY": [16, 32, 64],
}

# The dtypes this kernel routes for. bfloat16 is what the shipped 4-bit
# artifacts carry and therefore what the seam actually receives, measured on
# the real training path rather than assumed; float16 is admitted because the
# same source serves it and it is the sharper check of the two.
ROUTED_DTYPES = ("bfloat16", "float16")


def fwd_axes() -> dict:
    """The forward knob space, in the shape `search_space.enumerate_knobs` walks."""
    return {name: list(values) for name, values in FWD_KNOB_AXES.items()}


def fwd_threads(knobs: dict) -> int:
    """Threads per threadgroup: one simdgroup of 32 lanes per SGROUPS."""
    return 32 * int(knobs["SGROUPS"])


def fwd_threadgroup_bytes(knobs: dict, head_dim: int = HEAD_DIM) -> int:
    """Zero, and that is the design rather than an oversight: every operand is
    read into fragments straight from device memory, so this kernel declares no
    threadgroup array at all. The knob filter still calls it, because
    launchability is asked the same way of every operation."""
    del knobs, head_dim
    return 0


def fwd_row_multiple(knobs: dict) -> int:
    """The row count the operands must be a whole number of.

    A fragment load reads eight rows whether or not eight rows exist, so the
    door pads the operands rather than letting the last threadgroup read past
    the end of a tensor. Both the query tile and the key tile have to divide
    it, so it is the larger of the two.
    """
    return max(8 * int(knobs["SGROUPS"]), int(knobs["BKEY"]))


def fwd_launch(knobs: dict, batch: int, n_q_heads: int, t_len: int) -> tuple:
    """(grid, threadgroup) in MLX's convention, where the grid counts THREADS."""
    rows = 8 * int(knobs["SGROUPS"])
    tiles = (t_len + rows - 1) // rows
    return ((tiles * 32, int(knobs["SGROUPS"]), batch * n_q_heads),
            (32, int(knobs["SGROUPS"]), 1))


def _fwd_body(knobs: dict, head_dim: int, scale: float, t_pad_expr: str,
              heads_q_expr: str, heads_kv_expr: str) -> str:
    """The shared body with its compile-time constants and shape spellings
    substituted. One body serves every door so they cannot drift apart.

    The knob values, the head dimension and the scale are LITERALS, because
    every one of them bounds an unrolled loop, sizes a fragment array, or is
    the constant a wrong kernel would get wrong. The head counts and the padded
    width stay expressions read from the shapes, and the TRUE width arrives as
    a one-element input rather than a literal, because a compiled kernel per
    step width would recompile on nearly every batch and mlx-lm's widths move
    with the data.
    """
    substitutions = {
        "SGROUPS": str(int(knobs["SGROUPS"])),
        "BKEY": str(int(knobs["BKEY"])),
        "DHEAD": str(int(head_dim)),
        "SCALE_LIT": f"{float(scale)!r}f",
        "T_PAD": t_pad_expr,
        "HEADS_Q": heads_q_expr,
        "HEADS_KV": heads_kv_expr,
    }
    source = ATTN_FWD_MSL
    for name, value in substitutions.items():
        # Word boundaries, not a bare replace: `T_LEN` must never be rewritten
        # by a substitution for `T`, and no knob name may be caught inside a
        # longer identifier.
        source = re.sub(rf"\b{name}\b", value, source)
    return source


def fwd_door_source(knobs: dict, head_dim: int = HEAD_DIM,
                    scale: float | None = None) -> str:
    """The body in the MLX door's spelling, which reads its own shape buffers
    for the padded width and its `tlen` input for the true one."""
    if scale is None:
        scale = head_dim ** -0.5
    return _fwd_body(knobs, head_dim, scale, "q_shape[2]", "q_shape[1]",
                     "k_shape[1]")


def build_fwd(mx, knobs: dict, head_dim: int = HEAD_DIM,
              scale: float | None = None):
    """The MLX callable for one knob setting. Takes mx so this module imports
    with no device and no mlx present."""
    return mx.fast.metal_kernel(name=FWD_KERNEL_NAME,
                                input_names=FWD_INPUT_NAMES,
                                output_names=FWD_OUTPUT_NAMES,
                                source=fwd_door_source(knobs, head_dim, scale))


def pad_rows(mx, x, multiple: int):
    """Round a (batch, heads, width, head_dim) operand up to whole tiles.

    A fragment load reads eight rows whether or not eight rows exist, so
    without this the last threadgroup of the last head reads past the end of
    the tensor. The padding is zeros and every padded position is masked out by
    the true width, so it changes no answer.
    """
    width = x.shape[2]
    over = width % multiple
    if over == 0:
        return x
    return mx.pad(x, [(0, 0), (0, 0), (0, multiple - over), (0, 0)])


def run_fwd(mx, kern, knobs: dict, q, k, v):
    """Dispatch one knob setting, returning (output, row logsumexp).

    The single call path the gate, the tuner and the installed product all use,
    so a configuration cannot be timed one way and routed another. The cast to
    fp32 is part of the operation rather than preparation for it: the fragments
    the matrix units read are four bytes wide, the model's tensors are two, and
    converting once here is what buys the staging-free inner loop. `T` is the
    storage type the answer goes back to and reaches the kernel as a template
    parameter, which is how MLX's door names it.
    """
    batch, n_q_heads, t_len, head_dim = q.shape
    multiple = fwd_row_multiple(knobs)
    qf = pad_rows(mx, q, multiple).astype(mx.float32)
    kf = pad_rows(mx, k, multiple).astype(mx.float32)
    vf = pad_rows(mx, v, multiple).astype(mx.float32)
    t_pad = qf.shape[2]
    grid, threadgroup = fwd_launch(knobs, batch, n_q_heads, t_pad)
    out, lse = kern(inputs=[qf, kf, vf,
                            mx.array([t_len], dtype=mx.uint32)],
                    template=[("T", q.dtype)],
                    output_shapes=[(batch, n_q_heads, t_pad, head_dim),
                                   (batch, n_q_heads, t_pad)],
                    output_dtypes=[q.dtype, mx.float32],
                    grid=grid, threadgroup=threadgroup)
    if t_pad == t_len:
        return out, lse
    return out[:, :, :t_len, :], lse[:, :, :t_len]
