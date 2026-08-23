"""The quantized projection matmul at TRAINING widths, and the knobs that tune it.

`kernelverify/pack/wide_qmv.py` is the decode-shaped sibling: one input vector at
a time, or at most eleven, where the whole cost is reading the weights. Training
is the other regime. A step carries hundreds or thousands of tokens at once, so
the same operator becomes a real matmul whose arithmetic dominates, and a kernel
tuned for the matvec regime is the wrong kernel here.

What mlx-lm actually dispatches, read off the installed stack rather than
assumed: every LoRA projection reaches `mx.quantized_matmul(x, w_q, scales,
biases, transpose=True)` through `nn.QuantizedLinear.__call__`, and the tied
output head reaches the same operator through `nn.QuantizedEmbedding.as_linear`.
Both compute

    out[m, n] = sum_k x[m, k] * dequant(w)[n, k]

with x of shape (M, d_in), the packed weight of logical shape (d_out, d_in), and
the output (M, d_out). M is batch times sequence width, so it moves with the
data; d_in and d_out are fixed by the model.

ORIENTATION is "tt" and only "tt" here, and that is a scoping decision rather
than an oversight. The backward pass needs the other orientation, dX = dY @
dequant(w), but the pre-registration requires a kept kernel's gradient to equal
stock's gradient by exact array equality, and a differently-ordered accumulation
cannot be bit-equal. So the shipped vjp calls stock's own transpose=False
operator, no kernel of ours runs in that direction, and the nt orientation gets
its own certified entry later under its own amendment.

The knobs exist because a kernel that is fastest on one Apple GPU is not
fastest on the next one. The tile widths, the per-thread output block and the
reduction depth trade threadgroup memory against register pressure against
occupancy, and where the optimum sits is a property of the chip. Nothing here
picks a winner: `axes()` describes the space, `kernelverify.compiler.search_space`
drops what this device cannot launch, and the winner is measured on whatever
machine the user actually has.
"""

from __future__ import annotations

# The six distinct projection shapes Qwen3-4B dispatches in a training step, as
# (d_out, d_in). Held here as product data rather than imported from the bench
# tree, because the on-device tuner ships to users and bench/ does not.
#
# S6 is k_proj and v_proj: 8 key-value heads at head dimension 128 gives 1024
# output rows, which is the shape a five-entry table missed for a while.
# S5 is the tied output head, the largest by two orders of magnitude.
TRAIN_SHAPES = {
    "S1": (4096, 2560),      # q_proj
    "S2": (2560, 4096),      # o_proj
    "S3": (9728, 2560),      # gate_proj and up_proj
    "S4": (2560, 9728),      # down_proj
    "S5": (151936, 2560),    # the tied output head
    "S6": (1024, 2560),      # k_proj and v_proj
}

ORIENTATION = "tt"

# The quantization this pack is verified for. Anything else declines rather
# than routing: a kernel checked at one format must never run at another.
BITS = 4
GROUP_SIZE = 64

# M is data-dependent, so a tuned configuration is stored against a BUCKET
# rather than an exact token count. Powers of two, because that is the axis
# along which a tile configuration's behaviour actually changes, floored at 64
# because below that the matvec regime and wide_qmv own the problem, and capped
# because past the cap the arithmetic is saturated and the same tile wins.
M_BUCKET_FLOOR = 64
M_BUCKET_CAP = 2048


def m_bucket(m: int) -> int:
    """The bucket a token count is tuned and looked up under.

    Rounds UP to the next power of two so a bucket's tuned configuration was
    measured on a problem at least as large as the one it is being applied to.
    Rounding down would credit a configuration with a speed it was never shown
    to reach at the width it is used at.
    """
    if m < 1:
        raise ValueError(f"a step with {m} tokens has nothing to multiply")
    bucket = M_BUCKET_FLOOR
    while bucket < m and bucket < M_BUCKET_CAP:
        bucket *= 2
    return bucket


def cell_key(shape_name: str, m: int) -> str:
    """The knob map's key: one tuned configuration per shape per bucket."""
    if shape_name not in TRAIN_SHAPES:
        raise KeyError(f"{shape_name!r} is not one of {sorted(TRAIN_SHAPES)}")
    return f"{shape_name}:M{m_bucket(m)}"


def shape_name(d_out: int, d_in: int) -> str | None:
    """Which registered shape this call site is, or None if it is not one.

    None is a route to stock, never a guess at the nearest shape: a
    configuration measured at one shape says nothing about another.
    """
    for name, dims in TRAIN_SHAPES.items():
        if dims == (d_out, d_in):
            return name
    return None


# ---------------------------------------------------------------------------
# The kernel.
# ---------------------------------------------------------------------------
# One threadgroup owns a BM by BN tile of the output and walks the reduction in
# BK-deep steps. Within a step the tile of x and the DEQUANTIZED tile of w are
# staged in threadgroup memory once and then read many times, which is the whole
# reason a training-width kernel differs from the decode one: at M in the
# hundreds each weight element is reused across BM rows instead of one.
#
#     x (M, d_in)                 w codes (d_out, d_in/8 words)
#     +----BK----+                +----BK----+
#     |          | BM             |          | BN        out (M, d_out)
#     |  xs[][]  |                |  ws[][]  |           +--BN--+
#     +----------+                +----------+           |      | BM
#          |                           |                 +------+
#          +------ accumulate fp32 ----+
#
# Every accumulator is fp32 and every staged weight is fp32. The contract's
# working-precision clause forbids half accumulators outside load and store
# idioms, and the source lint rejects the kernel outright if one appears.
#
# The dequantization is MLX's own affine layout, the same bytes wide_qmv reads:
# a contiguous little-endian code stream, lowest element in the lowest bits,
# with a 64-wide group always a whole number of 8-code blocks so scale and bias
# are constant across a block.
TRAIN_QMM_TT_MSL = """
    // Tile geometry. TM by TN outputs per thread, TX by TY threads per group.
    constexpr uint BM = TY * TM;
    constexpr uint BN = TX * TN;

    const uint d_in  = D_IN;
    const uint d_out = D_OUT;
    const uint m_rows = M_ROWS;

    const uint words_per_row  = d_in / 8u;     // 8 four-bit codes per uint32
    const uint groups_per_row = d_in / 64u;

    const uint tx = thread_position_in_threadgroup.x;
    const uint ty = thread_position_in_threadgroup.y;
    const uint tid = ty * TX + tx;
    const uint nthreads = TX * TY;

    const uint n0 = threadgroup_position_in_grid.x * BN;   // first output col
    const uint m0 = threadgroup_position_in_grid.y * BM;   // first output row

    threadgroup float xs[BM * BK];
    threadgroup float ws[BN * BK];

    float acc[TM][TN];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < TM; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < TN; ++j) { acc[i][j] = 0.0f; }
    }

    for (uint k0 = 0; k0 < d_in; k0 += BK) {
        // Stage the activation tile. Out-of-range rows read as zero rather
        // than being skipped, so every thread reaches the same barriers.
        for (uint e = tid; e < BM * BK; e += nthreads) {
            uint r = e / BK;
            uint c = e % BK;
            uint gm = m0 + r;
            uint gk = k0 + c;
            xs[e] = (gm < m_rows && gk < d_in)
                  ? (float)x[gm * d_in + gk] : 0.0f;
        }

        // Stage the dequantized weight tile. Each element is one code, its
        // group's scale and its group's bias: s * q + b, evaluated in fp32.
        for (uint e = tid; e < BN * BK; e += nthreads) {
            uint r = e / BK;
            uint c = e % BK;
            uint gn = n0 + r;
            uint gk = k0 + c;
            float v = 0.0f;
            if (gn < d_out && gk < d_in) {
                uint word = w_q[gn * words_per_row + (gk >> 3)];
                uint code = (word >> ((gk & 7u) * 4u)) & 0xFu;
                uint g = gk / 64u;
                float s = (float)scales[gn * groups_per_row + g];
                float b = (float)biases[gn * groups_per_row + g];
                v = metal::fma(s, (float)code, b);
            }
            ws[e] = v;
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        // The multiply. Each thread's TM by TN block reuses one loaded x value
        // across TN weights and one loaded weight across TM activations.
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; ++kk) {
            float xv[TM];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < TM; ++i) {
                xv[i] = xs[(ty * TM + i) * BK + kk];
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < TN; ++j) {
                float wv = ws[(tx * TN + j) * BK + kk];
                #pragma clang loop unroll(full)
                for (uint i = 0; i < TM; ++i) {
                    acc[i][j] = metal::fma(xv[i], wv, acc[i][j]);
                }
            }
        }

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (uint i = 0; i < TM; ++i) {
        uint gm = m0 + ty * TM + i;
        if (gm >= m_rows) { continue; }
        #pragma clang loop unroll(full)
        for (uint j = 0; j < TN; ++j) {
            uint gn = n0 + tx * TN + j;
            if (gn < d_out) { out[gm * d_out + gn] = (T)acc[i][j]; }
        }
    }
"""

KERNEL_NAME = "kv_train_qmm_tt"
INPUT_NAMES = ["x", "w_q", "scales", "biases"]
OUTPUT_NAMES = ["out"]

# The knob axes. Nothing here is dropped for being slow: launchability is the
# only filter, it is read from the device, and a merely inefficient setting
# loses on the clock, which is a measurement rather than an opinion.
KNOB_AXES = {
    "TM": [2, 4, 8],
    "TN": [2, 4, 8],
    "TX": [8, 16, 32],
    "TY": [4, 8, 16],
    "BK": [32, 64, 128],
}


def axes() -> dict:
    """The knob space, in the shape `search_space.enumerate_knobs` walks."""
    return {name: list(values) for name, values in KNOB_AXES.items()}


def threads(knobs: dict) -> int:
    """Threads per threadgroup, the first device limit a setting must clear."""
    return int(knobs["TX"]) * int(knobs["TY"])


def threadgroup_bytes(knobs: dict) -> int:
    """Staged bytes per threadgroup: the activation tile plus the dequantized
    weight tile, both fp32 because the contract forbids half accumulation."""
    bm = int(knobs["TY"]) * int(knobs["TM"])
    bn = int(knobs["TX"]) * int(knobs["TN"])
    bk = int(knobs["BK"])
    return (bm * bk + bn * bk) * 4


def tile(knobs: dict) -> tuple[int, int, int]:
    """(BM, BN, BK) for a setting: what one threadgroup covers per step."""
    return (int(knobs["TY"]) * int(knobs["TM"]),
            int(knobs["TX"]) * int(knobs["TN"]),
            int(knobs["BK"]))


def launch(knobs: dict, m_rows: int, d_out: int) -> tuple:
    """(grid, threadgroup) in MLX's convention, where the grid counts THREADS.

    A partial tile at either edge is covered by a whole threadgroup whose
    out-of-range lanes read zero and store nothing, so no shape needs to divide
    a tile and no edge case needs a second kernel.
    """
    bm, bn, _bk = tile(knobs)
    tx, ty = int(knobs["TX"]), int(knobs["TY"])
    groups_n = (d_out + bn - 1) // bn
    groups_m = (m_rows + bm - 1) // bm
    return ((groups_n * tx, groups_m * ty, 1), (tx, ty, 1))


def _body(knobs: dict, d_in_expr: str, d_out_expr: str,
          m_rows_expr: str) -> str:
    """The shared body with its compile-time constants and shape spellings
    substituted. One body serves both doors so they cannot drift apart."""
    source = TRAIN_QMM_TT_MSL
    for name in ("TM", "TN", "TX", "TY", "BK"):
        source = source.replace(name, str(int(knobs[name])))
    return (source.replace("D_IN", d_in_expr)
                  .replace("D_OUT", d_out_expr)
                  .replace("M_ROWS", m_rows_expr))


def mlx_door_source(knobs: dict) -> str:
    """The body in the MLX door's spelling, which reads its own shape buffers."""
    return _body(knobs, "x_shape[1]", "scales_shape[0]", "x_shape[0]")


def build(mx, knobs: dict):
    """The MLX callable for one knob setting. Takes mx so this module imports
    with no device and no mlx present."""
    return mx.fast.metal_kernel(name=KERNEL_NAME, input_names=INPUT_NAMES,
                                output_names=OUTPUT_NAMES,
                                source=mlx_door_source(knobs))


def run(mx, kern, knobs: dict, x, w_q, scales, biases):
    """Dispatch one knob setting. The single call path both the tuner and the
    installed product use, so a configuration cannot be timed one way and then
    routed another.

    `T` is the output element type and reaches the kernel as a template
    parameter, which is how MLX's door names it; the output dtype follows the
    activation's, the way stock's own operator does.
    """
    m_rows, _d_in = x.shape
    d_out = scales.shape[0]
    grid, threadgroup = launch(knobs, m_rows, d_out)
    return kern(inputs=[x, w_q, scales, biases],
                template=[("T", x.dtype)],
                output_shapes=[(m_rows, d_out)],
                output_dtypes=[x.dtype],
                grid=grid, threadgroup=threadgroup)[0]
