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

Bit widths 2, 3 and 4 share one kernel. MLX's `ceil(M / 5)` tiling does not
depend on the width, so the same extra pass is paid at every width and the
same fix applies. The only new problem below 4 bits is that codes stop
aligning to 32-bit words, which is why lanes stride fixed 8-CODE blocks rather
than words: the activation access pattern is then identical at every width and
only the weight-side bit arithmetic changes. An 8-code block spans at most 24
bits, so it touches at most two words, read as a pair and shifted once.

Striding whole words instead was measured and is much worse below 4 bits -
0.52x at 2 bits and 0.26x at 3 bits against MLX at M=8 - because the
per-lane activation stride then grows with the width and the loads stop
coalescing.

Layout matches MLX's own affine artefact so both read identical bytes: a
contiguous little-endian code stream, lowest element in the lowest bits, and a
64-wide group is always a whole number of blocks so scale and bias are
constant within one. Per block the fused form `s * dot(x, q) + b * sum(x)`
accumulates in fp32, which keeps every intermediate at or above fp32 as the
contract requires, then the 32 lanes sharing a row simd-reduce.

Requires d_in divisible by 64. d_out need not divide R; out-of-range rows read
clamped and are not stored.
"""

from __future__ import annotations

import numpy as np

from kernelverify.pack.routed_windows import window_for

SIMD_WIDTH = 32
SIMDGROUPS_PER_THREADGROUP = 8

# The body is door-neutral: `D_IN` and `D_OUT` are spelled differently per
# door. The MLX door substitutes MLX's generated shape buffers, the runner
# door substitutes scalar bindings, and everything else is shared, so the two
# doors cannot drift (the spike's W4 pattern, bench/spike_dequant_gemv.py).
WIDE_QMV_MSL = """
    uint sg_row = thread_position_in_grid.y;   // this simdgroup's row block
    uint lane   = thread_position_in_grid.x;
    uint d_in   = D_IN;
    uint d_out  = D_OUT;
    uint row_words = d_in * BITS / 32;
    uint n_groups  = d_in / 64;
    uint blocks    = d_in / 8;                 // 8 codes per block, any width
    const device half4* x4 = (const device half4*)x;
    const uint mask = (1u << BITS) - 1u;

    float acc[M][R];
    #pragma clang loop unroll(full)
    for (int m = 0; m < M; ++m) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < R; ++r) { acc[m][r] = 0.0f; }
    }

    for (uint bi = lane; bi < blocks; bi += 32) {
        uint bit0 = bi * 8 * BITS;
        uint w0   = bit0 >> 5;
        uint sh0  = bit0 & 31;
        uint g    = bi / 8;

        // The R rows' codes are unpacked once per block, then reused by every
        // one of the M vectors. This is the single pass.
        float4 qlo[R], qhi[R];
        float s[R], b[R];
        #pragma clang loop unroll(full)
        for (int r = 0; r < R; ++r) {
            uint row = metal::min(sg_row * R + r, d_out - 1);
            s[r] = (float)scales[row * n_groups + g];
            b[r] = (float)biases[row * n_groups + g];
            uint base = row * row_words + w0;
            uint lo_w = w_q[base];
            uint hi_w = (sh0 + 8 * BITS > 32) ? w_q[base + 1] : 0u;
            ulong v = (((ulong)lo_w) | (((ulong)hi_w) << 32)) >> sh0;
            float e[8];
            #pragma clang loop unroll(full)
            for (int j = 0; j < 8; ++j) {
                e[j] = (float)((uint)(v >> (j * BITS)) & mask);
            }
            qlo[r] = float4(e[0], e[1], e[2], e[3]);
            qhi[r] = float4(e[4], e[5], e[6], e[7]);
        }

        // One input vector is live at a time: hoisting all M spills at M = 11.
        #pragma clang loop unroll(full)
        for (int m = 0; m < M; ++m) {
            uint xi = m * (d_in / 4) + bi * 2;
            float4 lo = float4(x4[xi]);
            float4 hi = float4(x4[xi + 1]);
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


def should_dispatch(m: int, bits: int, d_out: int, d_in: int) -> bool:
    """Whether the pack should use this kernel for THIS cell, or defer to MLX.

    Every argument is required: routing is a property of the whole
    (tile width, bit width, shape) cell, and a caller that supplies less than
    that is asking to be routed on evidence nobody collected.

    Shipped scope, deliberately narrow: the six distinct qmv shapes one model
    (Qwen3-4B) dispatches at decode, at 3 bits, from one recording
    (kernelverify/pack/routed_windows.py). Five of them route M = 5..9;
    lm_head routes M = 5..10, because at 151936 rows MLX's second weight pass
    still costs more than this kernel's single pass one width further. 2-bit
    and 4-bit route NOWHERE, not because they lose but because they have no
    recording: the 4-bit pricing run is queued (ruling D1, TODOS.md), and
    until it lands MLX serves that width. Any shape outside the six is
    likewise unrouted.

    Each window is conditional on the launch config it was priced at - the R
    that rows_per_simdgroup returns - and on the kernel body the recording
    timed, both pinned by sha in routed_windows. Change either and the windows
    describe a kernel that no longer exists.

    What is BELOW a window is not uniformly a measured loss, which the old
    single-boundary docstring got wrong. At four of the six shapes M = 4 came
    back REFUSED: its ratio interval straddles 1.0, so the point cannot claim
    a direction either way - undecided, not bad. M = 4 is a measured LOSS at
    q_proj only, and a measured WIN at lm_head that ruling D2 does not route
    (routed_windows.EXCLUDED_CELLS has the reason). Two more REFUSED cells sit
    lower still, at M = 2 (gate/up_proj) and M = 3 (down_proj). Only M <= 3 is
    a measured loss everywhere it was priced. An undecided cell routes to MLX
    for the same reason a losing one does - no evidence supports taking it -
    but it is one pricing run away from moving, not a closed question.
    """
    return m in window_for(bits, d_out, d_in)


def rows_per_simdgroup(m: int) -> int:
    """R, from the measured register wall: 4 up to M = 10, then 2."""
    return 4 if m <= 10 else 2


SUPPORTED_BITS = (2, 3, 4)


def pack_codes(q: np.ndarray, bits: int) -> np.ndarray:
    """Contiguous little-endian code stream: the MLX affine packing, any width.

    Verified bit-identical to mx.quantize at 2, 3, 4, 5, 6 and 8 bits by
    bench/mlx_probes/probe_affine_contract.py.
    """
    rows, cols = q.shape
    total_bits = cols * bits
    stream = np.zeros((rows, total_bits // 64 + 2), dtype=np.uint64)
    for i in range(cols):
        word, offset = divmod(i * bits, 64)
        val = q[:, i].astype(np.uint64)
        stream[:, word] |= val << np.uint64(offset)
        if offset + bits > 64:
            stream[:, word + 1] |= val >> np.uint64(64 - offset)
    byts = stream.view(np.uint8)[:, : total_bits // 8]
    return np.ascontiguousarray(byts).view(np.uint32)


def pack_nibbles(q: np.ndarray) -> np.ndarray:
    """The 4-bit case, kept as a name because callers use it."""
    return pack_codes(q, 4)


def launch_config(d_out: int, m: int) -> tuple:
    """(grid, threadgroup, R) for this output width and vector tile."""
    r = rows_per_simdgroup(m)
    row_blocks = (d_out + r - 1) // r
    return ((SIMD_WIDTH, row_blocks, 1),
            (SIMD_WIDTH, SIMDGROUPS_PER_THREADGROUP, 1),
            r)


def kernel_spec():
    """The runner-door template, for verification through `MetalRunner`.

    A raw-MSL wrapper around the shared body: `specialize()` fills the
    per-combination compile-time constants (`T`, `BITS`, `M`, `R`), the
    run-time dimensions arrive as scalar bindings, and the grid's row-block
    count is the `row_blocks` extent each case supplies from `launch_config`.
    """
    from kernelverify.runners import Binding, BindingKind, KernelSpec, LaunchSpec

    source = f"""
#include <metal_stdlib>
using namespace metal;
using T = $T;
constant constexpr uint BITS = $BITS;
constant constexpr int  M    = $M;
constant constexpr int  R    = $R;
kernel void {KERNEL_NAME}(
    device const half* x      [[buffer(0)]],
    device const uint* w_q    [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const half* biases [[buffer(3)]],
    device T* out             [[buffer(4)]],
    constant uint& d_in_arg   [[buffer(5)]],
    constant uint& d_out_arg  [[buffer(6)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]) {{
{WIDE_QMV_MSL.replace("D_IN", "d_in_arg").replace("D_OUT", "d_out_arg")}
}}
"""
    return KernelSpec(
        source=source,
        entry_point=KERNEL_NAME,
        name=f"{KERNEL_NAME}-$BITS bit-M$M-R$R-$T",
        bindings=(Binding(BindingKind.INPUT, "x"),
                  Binding(BindingKind.INPUT, "w_q"),
                  Binding(BindingKind.INPUT, "scales"),
                  Binding(BindingKind.INPUT, "biases"),
                  Binding(BindingKind.OUTPUT),
                  Binding(BindingKind.SCALAR, "d_in_arg", "uint32"),
                  Binding(BindingKind.SCALAR, "d_out_arg", "uint32")),
        launch=LaunchSpec(grid=(SIMD_WIDTH, "row_blocks", 1),
                          threadgroup=(SIMD_WIDTH, SIMDGROUPS_PER_THREADGROUP, 1)),
    )


def mlx_door_source() -> str:
    """The body in the MLX door's spelling, the single source `build()` uses."""
    return WIDE_QMV_MSL.replace("D_IN", "x_shape[1]").replace("D_OUT", "scales_shape[0]")


def build(mx):
    """The MLX callable. Takes mx so this module imports without it."""
    return mx.fast.metal_kernel(name=KERNEL_NAME, input_names=INPUT_NAMES,
                                output_names=OUTPUT_NAMES, source=mlx_door_source())
