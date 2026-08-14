"""Fused top-2 MoE decode over quantized experts, Qwen3-class routing.

Two kernels, because routing and dispatch have different shapes and fusing
them would be slower, not faster. Routing needs every expert's logit for a
token, so it is one simdgroup per token; dispatch needs the selected experts'
weight rows, so it is one threadgroup per (token, row block). Fusing them into
the dispatch kernel would recompute the routing once per row block, which at
d_out = 768 and R = 4 is 192 redundant copies per token.

The contract implemented is the one pinned in
`kernelverify/reference/native_kernels.moe_dispatch`: logits in fp32, softmax
over ALL experts, top-2 by probability with ties to the LOWER index,
renormalize the top-2 weights, then sum weight_e * (x @ expert_e.T).

Routing compares probabilities rather than logits even though softmax is
monotonic, because in floating point exp() can map two distinct logits onto
the same probability; comparing logits would then pick a winner where the
contract sees a tie and takes the lower index.

Experts are quantized under the MLX-affine contract, so `dispatch` folds the
dequantization into the accumulation with the same fused per-group form the
wide-tile matvec uses: s * dot(x, q) + b * sum(x), fp32 throughout.
"""

from __future__ import annotations

TOP_K = 2
SIMD_WIDTH = 32
SIMDGROUPS_PER_THREADGROUP = 8
ROWS_PER_SIMDGROUP = 4

# --------------------------------------------------------------------------
# routing: one simdgroup per token
# --------------------------------------------------------------------------
ROUTING_MSL = """
    threadgroup float lg[E];
    uint token = threadgroup_position_in_grid.x;
    uint lane  = thread_index_in_simdgroup;
    uint d_in  = x_shape[1];
    const device half4* x4 = (const device half4*)(x + token * d_in);

    // Every lane takes a slice of the experts and writes that logit.
    for (uint e = lane; e < E; e += 32) {
        const device half4* r4 = (const device half4*)(router + e * d_in);
        float dot_acc = 0.0f;
        for (uint j = 0; j < d_in / 4; ++j) {
            dot_acc += metal::dot(float4(x4[j]), float4(r4[j]));
        }
        lg[e] = dot_acc;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // The selection is serial in one lane: E is small next to the expert
    // matmuls, and a serial scan is the only cheap way to make the tie rule
    // exactly "first index wins".
    if (lane == 0) {
        float peak = -INFINITY;
        for (uint e = 0; e < E; ++e) { peak = metal::max(peak, lg[e]); }
        float denom = 0.0f;
        for (uint e = 0; e < E; ++e) { denom += metal::exp(lg[e] - peak); }

        float p0 = -1.0f, p1 = -1.0f;
        uint i0 = 0, i1 = 0;
        for (uint e = 0; e < E; ++e) {
            float p = metal::exp(lg[e] - peak) / denom;
            if (p > p0) { p1 = p0; i1 = i0; p0 = p; i0 = e; }
            else if (p > p1) { p1 = p; i1 = e; }
        }
        idx[token * 2 + 0] = i0;
        idx[token * 2 + 1] = i1;
        float total = p0 + p1;
        gate[token * 2 + 0] = p0 / total;
        gate[token * 2 + 1] = p1 / total;
    }
"""

# --------------------------------------------------------------------------
# dispatch: one threadgroup per (token, row block), both slots at once
# --------------------------------------------------------------------------
DISPATCH_MSL = """
    uint sg_row = thread_position_in_grid.y;
    uint lane   = thread_position_in_grid.x;
    uint token  = thread_position_in_grid.z;
    uint d_in   = x_shape[1];
    uint d_out  = scales_shape[1];
    uint words  = d_in / 8;
    uint n_groups = d_in / 64;
    const device half4* x4 = (const device half4*)(x + token * d_in);

    uint  e0 = idx[token * 2 + 0], e1 = idx[token * 2 + 1];
    float g0 = gate[token * 2 + 0], g1 = gate[token * 2 + 1];

    float acc[R];
    #pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) { acc[r] = 0.0f; }

    // Both experts are combined in one pass, so the token's activations are
    // read once for 2*R dot products.
    for (uint wi = lane; wi < words; wi += 32) {
        uint g = wi / 8;
        float4 lo = float4(x4[wi * 2]);
        float4 hi = float4(x4[wi * 2 + 1]);
        float xs = metal::dot(lo, float4(1.0f)) + metal::dot(hi, float4(1.0f));

        #pragma clang loop unroll(full)
        for (int r = 0; r < R; ++r) {
            uint row = metal::min(sg_row * R + r, d_out - 1);
            float slot_sum = 0.0f;
            #pragma clang loop unroll(full)
            for (int k = 0; k < 2; ++k) {
                uint e = (k == 0) ? e0 : e1;
                float gk = (k == 0) ? g0 : g1;
                uint flat = e * d_out + row;
                uint word = w_q[flat * words + wi];
                float s = (float)scales[flat * n_groups + g];
                float b = (float)biases[flat * n_groups + g];
                float4 qlo = float4(float(word & 0xF), float((word >> 4) & 0xF),
                                    float((word >> 8) & 0xF), float((word >> 12) & 0xF));
                float4 qhi = float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                                    float((word >> 24) & 0xF), float((word >> 28) & 0xF));
                float xq = metal::dot(lo, qlo) + metal::dot(hi, qhi);
                slot_sum = metal::fma(gk, metal::fma(s, xq, b * xs), slot_sum);
            }
            acc[r] += slot_sum;
        }
    }

    #pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) {
        uint row = sg_row * R + r;
        float v = metal::simd_sum(acc[r]);
        if (lane == 0 && row < d_out) { out[token * d_out + row] = (T)v; }
    }
"""

ROUTING_INPUTS = ["x", "router"]
ROUTING_OUTPUTS = ["idx", "gate"]
DISPATCH_INPUTS = ["x", "idx", "gate", "w_q", "scales", "biases"]
DISPATCH_OUTPUTS = ["out"]
ROUTING_NAME = "kv_moe_routing"
DISPATCH_NAME = "kv_moe_dispatch"


def routing_launch(n_tokens: int) -> tuple:
    """(grid, threadgroup): one simdgroup per token."""
    return (SIMD_WIDTH * n_tokens, 1, 1), (SIMD_WIDTH, 1, 1)


def dispatch_launch(d_out: int, n_tokens: int) -> tuple:
    """(grid, threadgroup, R): row blocks over d_out, one z-slice per token."""
    row_blocks = (d_out + ROWS_PER_SIMDGROUP - 1) // ROWS_PER_SIMDGROUP
    return ((SIMD_WIDTH, row_blocks, n_tokens),
            (SIMD_WIDTH, SIMDGROUPS_PER_THREADGROUP, 1),
            ROWS_PER_SIMDGROUP)


def routing_spec():
    from kernelverify.runners.runner import KernelSpec

    return KernelSpec(name=ROUTING_NAME, source=ROUTING_MSL,
                      input_names=ROUTING_INPUTS, output_names=ROUTING_OUTPUTS)


def dispatch_spec():
    from kernelverify.runners.runner import KernelSpec

    return KernelSpec(name=DISPATCH_NAME, source=DISPATCH_MSL,
                      input_names=DISPATCH_INPUTS, output_names=DISPATCH_OUTPUTS)


def build_routing(mx):
    return mx.fast.metal_kernel(name=ROUTING_NAME, input_names=ROUTING_INPUTS,
                                output_names=ROUTING_OUTPUTS, source=ROUTING_MSL)


def build_dispatch(mx):
    return mx.fast.metal_kernel(name=DISPATCH_NAME, input_names=DISPATCH_INPUTS,
                                output_names=DISPATCH_OUTPUTS, source=DISPATCH_MSL)
