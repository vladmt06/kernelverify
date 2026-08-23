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
# One threadgroup owns ROWS = SGROUPS * RQ query rows of one (batch, query
# head) and walks the keys in BKEY-wide tiles, carrying a running maximum and
# a running normaliser per row. The full score matrix is never built, which is
# the memory half of the claim, and each key tile is read once and reused by
# every row in the threadgroup, which is the time half.
#
#     q rows (ROWS, DHEAD)          k tile (BKEY, DHEAD)      v tile (BKEY, DHEAD)
#     +------DHEAD------+           +------DHEAD-----+        +------DHEAD-----+
#     |      qs[][]     | ROWS      |   ks[][] (T)   | BKEY   |   vs[][] (T)   | BKEY
#     +-----------------+           +----------------+        +----------------+
#              |                            |                         |
#              +----- phase 1: scores ------+                         |
#                          |                                          |
#                     ps[ROWS][BKEY], online max and sum               |
#                          |                                          |
#                          +---------- phase 2: acc += p * v ---------+
#
# The two phases exist because one data layout cannot serve both. In phase 1 a
# lane owns a KEY and loops the head dimension, so the reduction that makes a
# score is inside one lane and needs no cross-lane traffic; the key tile is
# staged TRANSPOSED for it, so the 32 lanes of a simdgroup read 32 consecutive
# addresses instead of 32 addresses 256 bytes apart. In phase 2 a lane owns
# COLUMNS of the output and loops the keys, so the sum that makes an output
# element is again inside one lane; the probabilities cross between the two
# through threadgroup memory rather than through a shuffle.
#
# Every accumulator is fp32 and every staged query row is fp32. The key and
# value tiles are staged at MLX's `T`, which is the storage type the model
# already holds them in, and every read of them is a cast to fp32 before it is
# used: contract clause C1 forbids a narrower intermediate, and the source
# lint refuses the kernel outright if one appears.
#
# Two outputs, not one. `l` is the row logsumexp, which this forward computes
# on its way through the key loop and which the backward needs to rebuild the
# probabilities without a second pass over the scores. Handing it back costs
# one float per query row; recomputing it in the backward would cost a whole
# extra forward.
ATTN_FWD_MSL = """
    constexpr uint LANES = 32u;                  // one simdgroup
    constexpr uint KPL   = BKEY / LANES;         // keys owned per lane
    constexpr uint DPL   = DHEAD / LANES;        // output columns owned per lane
    constexpr uint ROWS  = SGROUPS * RQ;         // query rows per threadgroup

    const uint lane = thread_position_in_threadgroup.x;
    const uint sg   = thread_position_in_threadgroup.y;
    const uint tid  = sg * LANES + lane;
    const uint nthreads = SGROUPS * LANES;

    const uint t_len = T_LEN;
    const uint n_q   = HEADS_Q;
    const uint n_kv  = HEADS_KV;
    const uint group = n_q / n_kv;

    const uint bh  = threadgroup_position_in_grid.z;
    const uint bat = bh / n_q;
    const uint hq  = bh - bat * n_q;
    const uint hkv = hq / group;

    const uint q0   = threadgroup_position_in_grid.x * ROWS;
    const uint row0 = q0 + sg * RQ;

    const ulong q_base  = ((ulong)(bat * n_q  + hq ) * (ulong)t_len) * DHEAD;
    const ulong kv_base = ((ulong)(bat * n_kv + hkv) * (ulong)t_len) * DHEAD;

    threadgroup T ks[DHEAD * BKEY];      // transposed: ks[d * BKEY + j]
    threadgroup T vs[BKEY * DHEAD];      // natural:    vs[j * DHEAD + d]
    threadgroup float qs[ROWS * DHEAD];
    threadgroup float ps[ROWS * BKEY];

    // The query rows are read once for the whole key loop. An out-of-range row
    // stages zero rather than being skipped, so every thread reaches every
    // barrier.
    for (uint e = tid; e < ROWS * DHEAD; e += nthreads) {
        uint r = e / DHEAD;
        uint d = e - r * DHEAD;
        uint gi = q0 + r;
        qs[e] = (gi < t_len) ? (float)q[q_base + (ulong)gi * DHEAD + d] : 0.0f;
    }

    float acc[RQ][DPL];
    float m_run[RQ];
    float l_run[RQ];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < RQ; ++r) {
        m_run[r] = -INFINITY;
        l_run[r] = 0.0f;
        #pragma clang loop unroll(full)
        for (uint c = 0; c < DPL; ++c) { acc[r][c] = 0.0f; }
    }

    // The causal bound, applied to the whole threadgroup: a key past this
    // group's last query row can never contribute to any of its rows, so the
    // tile is never read rather than read and masked. That is where the factor
    // of two against a dense pass comes from.
    const uint k_end = min(q0 + ROWS, t_len);

    for (uint k0 = 0; k0 < k_end; k0 += BKEY) {
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        for (uint e = tid; e < BKEY * DHEAD; e += nthreads) {
            uint j = e / DHEAD;
            uint d = e - j * DHEAD;
            uint gj = k0 + j;
            bool live = gj < t_len;
            ks[d * BKEY + j] = live ? k[kv_base + (ulong)gj * DHEAD + d] : (T)0.0f;
            vs[e]            = live ? v[kv_base + (ulong)gj * DHEAD + d] : (T)0.0f;
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        // Phase 1: this lane's keys against every one of its rows.
        float s[RQ][KPL];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < RQ; ++r) {
            #pragma clang loop unroll(full)
            for (uint c = 0; c < KPL; ++c) { s[r][c] = 0.0f; }
        }
        for (uint d = 0; d < DHEAD; ++d) {
            float kv[KPL];
            #pragma clang loop unroll(full)
            for (uint c = 0; c < KPL; ++c) {
                kv[c] = (float)ks[d * BKEY + c * LANES + lane];
            }
            #pragma clang loop unroll(full)
            for (uint r = 0; r < RQ; ++r) {
                float qv = qs[(sg * RQ + r) * DHEAD + d];
                #pragma clang loop unroll(full)
                for (uint c = 0; c < KPL; ++c) {
                    s[r][c] = metal::fma(qv, kv[c], s[r][c]);
                }
            }
        }

        // Mask, scale, and fold this tile into the running softmax.
        #pragma clang loop unroll(full)
        for (uint r = 0; r < RQ; ++r) {
            uint gi = row0 + r;
            float local_max = -INFINITY;
            #pragma clang loop unroll(full)
            for (uint c = 0; c < KPL; ++c) {
                uint gj = k0 + c * LANES + lane;
                // gj <= gi already implies gj < t_len for a live row, so the
                // causal test is the only test the mask needs.
                s[r][c] = (gi < t_len && gj <= gi) ? s[r][c] * SCALE_LIT : -INFINITY;
                local_max = metal::max(local_max, s[r][c]);
            }
            float m_tile = metal::simd_max(local_max);
            float m_new = metal::max(m_run[r], m_tile);
            // A row that has seen nothing yet carries -INFINITY, and -INF minus
            // -INF is a NaN rather than the zero the algebra wants.
            float safe = metal::isfinite(m_new) ? m_new : 0.0f;
            float corr = metal::isfinite(m_run[r]) ? metal::exp(m_run[r] - safe) : 0.0f;
            float local_sum = 0.0f;
            #pragma clang loop unroll(full)
            for (uint c = 0; c < KPL; ++c) {
                float p = metal::exp(s[r][c] - safe);
                ps[(sg * RQ + r) * BKEY + c * LANES + lane] = p;
                local_sum += p;
            }
            l_run[r] = l_run[r] * corr + metal::simd_sum(local_sum);
            #pragma clang loop unroll(full)
            for (uint c = 0; c < DPL; ++c) { acc[r][c] *= corr; }
            m_run[r] = m_new;
        }

        // The probabilities were written by this simdgroup and are read by this
        // simdgroup, so the handoff needs a simdgroup barrier and not a whole
        // threadgroup one.
        metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);

        // Phase 2: this lane's output columns over every key in the tile.
        for (uint j = 0; j < BKEY; ++j) {
            float pv[RQ];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < RQ; ++r) {
                pv[r] = ps[(sg * RQ + r) * BKEY + j];
            }
            #pragma clang loop unroll(full)
            for (uint c = 0; c < DPL; ++c) {
                float vv = (float)vs[j * DHEAD + c * LANES + lane];
                #pragma clang loop unroll(full)
                for (uint r = 0; r < RQ; ++r) {
                    acc[r][c] = metal::fma(pv[r], vv, acc[r][c]);
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < RQ; ++r) {
        uint gi = row0 + r;
        if (gi >= t_len) { continue; }
        float denom = l_run[r];
        float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        #pragma clang loop unroll(full)
        for (uint c = 0; c < DPL; ++c) {
            o[q_base + (ulong)gi * DHEAD + c * LANES + lane] = (T)(acc[r][c] * inv);
        }
        if (lane == 0u) {
            l[(ulong)(bat * n_q + hq) * (ulong)t_len + gi] =
                (denom > 0.0f) ? (m_run[r] + metal::log(denom)) : -INFINITY;
        }
    }
"""

FWD_KERNEL_NAME = "kv_train_attention_fwd"
FWD_INPUT_NAMES = ["q", "k", "v"]
FWD_OUTPUT_NAMES = ["o", "l"]

# The knob axes. Nothing here is dropped for being slow: launchability is the
# only filter, it is read from the device, and a merely inefficient setting
# loses on the clock, which is a measurement rather than an opinion.
#
# RQ is query rows per simdgroup, which sets how much key traffic each staged
# tile pays for; SGROUPS is simdgroups per threadgroup, which sets how many
# rows share one staged tile; BKEY is the key tile width, which sets how often
# the running maximum is rescaled. BKEY is a multiple of the 32-lane simdgroup
# because a lane owns whole keys, and the head dimension is a multiple of 32
# for the same reason in the other phase.
FWD_KNOB_AXES = {
    "RQ": [2, 4, 8, 16],
    "SGROUPS": [1, 2, 4],
    "BKEY": [32, 64],
}

# The storage width of one staged key or value element. The routed dtypes are
# bfloat16 and float16, both two bytes, and the door declines anything else, so
# the threadgroup-memory arithmetic below is exact rather than an estimate.
STORAGE_BYTES = 2

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
    """Staged bytes per threadgroup: the key and value tiles at storage width,
    the query rows and the probabilities at fp32."""
    rows = int(knobs["SGROUPS"]) * int(knobs["RQ"])
    bkey = int(knobs["BKEY"])
    tiles = 2 * head_dim * bkey * STORAGE_BYTES
    return tiles + rows * head_dim * 4 + rows * bkey * 4


def fwd_launch(knobs: dict, batch: int, n_q_heads: int, t_len: int) -> tuple:
    """(grid, threadgroup) in MLX's convention, where the grid counts THREADS.

    A partial tile at the query edge is covered by a whole threadgroup whose
    out-of-range rows stage zero and store nothing, so no width needs to divide
    a tile and no edge case needs a second kernel.
    """
    rows = int(knobs["SGROUPS"]) * int(knobs["RQ"])
    tiles = (t_len + rows - 1) // rows
    return ((tiles * 32, int(knobs["SGROUPS"]), batch * n_q_heads),
            (32, int(knobs["SGROUPS"]), 1))


def _fwd_body(knobs: dict, head_dim: int, scale: float, t_len_expr: str,
              heads_q_expr: str, heads_kv_expr: str) -> str:
    """The shared body with its compile-time constants and shape spellings
    substituted. One body serves every door so they cannot drift apart.

    The knob values, the head dimension and the scale are LITERALS, because
    every one of them sizes an array, bounds an unrolled loop, or is the
    constant a wrong kernel would get wrong. The batch, the head counts and the
    width stay expressions read from the shapes, because a compiled kernel per
    step width would recompile on nearly every batch.
    """
    substitutions = {
        "RQ": str(int(knobs["RQ"])),
        "SGROUPS": str(int(knobs["SGROUPS"])),
        "BKEY": str(int(knobs["BKEY"])),
        "DHEAD": str(int(head_dim)),
        "SCALE_LIT": f"{float(scale)!r}f",
        "T_LEN": t_len_expr,
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
    """The body in the MLX door's spelling, which reads its own shape buffers."""
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


def run_fwd(mx, kern, knobs: dict, q, k, v):
    """Dispatch one knob setting, returning (output, row logsumexp).

    The single call path the gate, the tuner and the installed product all use,
    so a configuration cannot be timed one way and routed another. `T` is the
    storage type and reaches the kernel as a template parameter, which is how
    MLX's door names it; the output follows the queries' dtype, the way stock's
    own operator does, and the row statistic is fp32 whatever the storage is.
    """
    batch, n_q_heads, t_len, head_dim = q.shape
    grid, threadgroup = fwd_launch(knobs, batch, n_q_heads, t_len)
    return kern(inputs=[q, k, v],
                template=[("T", q.dtype)],
                output_shapes=[(batch, n_q_heads, t_len, head_dim),
                               (batch, n_q_heads, t_len)],
                output_dtypes=[q.dtype, mx.float32],
                grid=grid, threadgroup=threadgroup)
