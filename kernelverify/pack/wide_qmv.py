"""Quantized matvec with a wide input-vector tile: one weight pass for any M.

The defect this fixes is algorithmic, not a bandwidth shortfall. MLX routes
batch 2 to 11 of a transposed quantized matmul to `qmv_wide`, which sets

    n_tiles = ceil(M / 5)

and gives each threadgroup a tile of at most five input vectors. Every tile
re-reads the entire weight matrix, so batch 6 to 10 read the weights twice and
batch 11 reads them three times. Counted against the traffic it actually
generates MLX is at 75-98% of the achievable bandwidth on this machine, so it
is not wasting bandwidth; it is generating extra bandwidth to stay under a
register wall (docs/research/2026-08-14-mlx-pack-feasibility.md, section 4b).

This kernel keeps all M vectors in one pass and buys back the register budget
elsewhere, by giving each simdgroup R output rows so a loaded input element
feeds R dot products instead of one.

Two register facts decided the shape, both measured (`R` and `M` sweeps at
2560x2560, 4-bit group 64):

- hoisting all M input vectors into registers costs 8*M floats and spills at
  M = 11, so exactly one vector is live at a time and the R rows' unpacked
  codes are the hoisted quantity instead;
- accumulators go as M*R, and R = 8 spills at every M (60 to 267 us against
  MLX's 35 to 86), so R tops out at 4 and drops to 2 at M = 11.

Layout matches MLX's own affine artefact so both read identical bytes: `w_q`
is uint32 with eight 4-bit codes packed low to high, and a 64-wide group spans
exactly eight words, so a word never straddles a group boundary. Per word the
fused form `s * dot(x, q) + b * sum(x)` accumulates in fp32, which keeps every
intermediate at or above fp32 as the contract requires, then the 32 lanes
sharing a row simd-reduce.

Requires d_in divisible by 64. d_out need not divide R; out-of-range rows read
clamped and are not stored.
"""

from __future__ import annotations

import numpy as np

SIMD_WIDTH = 32
SIMDGROUPS_PER_THREADGROUP = 8

WIDE_QMV_MSL = """
    uint sg_row = thread_position_in_grid.y;   // this simdgroup's row block
    uint lane   = thread_position_in_grid.x;
    uint d_in   = x_shape[1];
    uint d_out  = scales_shape[0];
    uint words  = d_in / 8;
    uint n_groups = d_in / 64;
    const device half4* x4 = (const device half4*)x;

    float acc[M][R];
    #pragma clang loop unroll(full)
    for (int m = 0; m < M; ++m) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < R; ++r) { acc[m][r] = 0.0f; }
    }

    for (uint wi = lane; wi < words; wi += 32) {
        uint g = wi / 8;

        // The R rows' weights are loaded and unpacked once per word, then
        // reused by every one of the M vectors. This is the single pass.
        float4 qlo[R], qhi[R];
        float s[R], b[R];
        #pragma clang loop unroll(full)
        for (int r = 0; r < R; ++r) {
            uint row = metal::min(sg_row * R + r, d_out - 1);
            uint word = w_q[row * words + wi];
            s[r] = (float)scales[row * n_groups + g];
            b[r] = (float)biases[row * n_groups + g];
            qlo[r] = float4(float(word & 0xF), float((word >> 4) & 0xF),
                            float((word >> 8) & 0xF), float((word >> 12) & 0xF));
            qhi[r] = float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                            float((word >> 24) & 0xF), float((word >> 28) & 0xF));
        }

        // One input vector is live at a time: hoisting all M spills at M = 11.
        #pragma clang loop unroll(full)
        for (int m = 0; m < M; ++m) {
            float4 lo = float4(x4[m * (d_in / 4) + wi * 2]);
            float4 hi = float4(x4[m * (d_in / 4) + wi * 2 + 1]);
            float xs = metal::dot(lo, float4(1.0f)) + metal::dot(hi, float4(1.0f));
            #pragma clang loop unroll(full)
            for (int r = 0; r < R; ++r) {
                float xq = metal::dot(lo, qlo[r]) + metal::dot(hi, qhi[r]);
                acc[m][r] = metal::fma(s[r], xq, metal::fma(b[r], xs, acc[m][r]));
            }
        }
    }

    #pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) {
        uint row = sg_row * R + r;
        #pragma clang loop unroll(full)
        for (int m = 0; m < M; ++m) {
            float v = metal::simd_sum(acc[m][r]);
            if (lane == 0 && row < d_out) { out[m * d_out + row] = (T)v; }
        }
    }
"""

INPUT_NAMES = ["x", "w_q", "scales", "biases"]
OUTPUT_NAMES = ["out"]
KERNEL_NAME = "kv_wide_qmv"


def rows_per_simdgroup(m: int) -> int:
    """R, from the measured register wall: 4 up to M = 10, then 2."""
    return 4 if m <= 10 else 2


def pack_nibbles(q: np.ndarray) -> np.ndarray:
    """Eight 4-bit codes per uint32, low to high: the MLX affine packing."""
    rows, cols = q.shape
    q8 = q.reshape(rows, cols // 8, 8).astype(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * 4)[None, None, :]
    return np.bitwise_or.reduce(q8 << shifts, axis=2).astype(np.uint32)


def launch_config(d_out: int, m: int) -> tuple:
    """(grid, threadgroup, R) for this output width and vector tile."""
    r = rows_per_simdgroup(m)
    row_blocks = (d_out + r - 1) // r
    return ((SIMD_WIDTH, row_blocks, 1),
            (SIMD_WIDTH, SIMDGROUPS_PER_THREADGROUP, 1),
            r)


def kernel_spec():
    """The runner's view of this kernel, for verification."""
    from kernelverify.runners.runner import KernelSpec

    return KernelSpec(name=KERNEL_NAME, source=WIDE_QMV_MSL,
                      input_names=INPUT_NAMES, output_names=OUTPUT_NAMES)


def build(mx):
    """The MLX callable. Takes mx so this module imports without it."""
    return mx.fast.metal_kernel(name=KERNEL_NAME, input_names=INPUT_NAMES,
                                output_names=OUTPUT_NAMES, source=WIDE_QMV_MSL)
