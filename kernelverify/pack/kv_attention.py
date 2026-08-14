"""Quantized-KV attention decode: one launch for the whole step.

The incumbent is mlx-lm's own quantized KV path, which composes one decode
step from separate ops: two `mx.quantized_matmul` calls (scores and combine),
a precise softmax, and the concat/split arithmetic for the entry the step
itself appends. That is at least six dispatches per attention call, paid every
layer of every token, and the MoE finding (pack findings, section 4) already
measured what that overhead is worth at decode: most of the win there came
from collapsing launches, and this operator has more of them.

This kernel runs the whole step in one launch: one threadgroup per
(batch row, head), eight simdgroups each.

- Scores: one thread per cached position; each thread dequantizes its K row
  element-by-element and dots it against the query held in threadgroup
  memory. The thread whose position index equals T takes the step's own new
  entry, reading `new_k` at full precision instead of the cache, so the T+1
  softmax is one code path rather than a concat.
- Softmax over T+1 at fp32, max-subtracted, `precise::exp`.
- Combine: each simdgroup takes cached positions strided by 8, each lane owns
  DH/32 contiguous output dims and accumulates prob * dequant(V) at fp32;
  simdgroup partials tree-reduce through threadgroup memory. Simdgroup 0 adds
  the new entry's `prob_new * new_v` before the reduction.

Contract clauses (frozen kv_attention contract, 8ec7eca), and where each is
met: every intermediate is fp32 (scores, probs, both accumulators - C1, the
accumulator-dtype clause the certificate attests structurally); scores are
computed and stored at fp32, never through half (the fp16-score canary);
the new entry enters both the scores and the combine (the t == T arm and the
simdgroup-0 add); 1/sqrt(DH) uses `precise::sqrt` at compile-time-constant
DH; the cache is dequantized per element as s * code + b from the stored
scale and bias, against the same MLX-affine artefact the oracle derives.

Dequantization is per element rather than the fused per-group form the
wide-tile matvec uses. The fused form saves nothing here (no output-row
reuse to amortise the activation sum), and per-element `fma(s, c, b)` is the
arithmetic shape of the contract ensemble's own members.

Codes are read as 8-code blocks through a two-word window, the sub-4-bit
striding rule from the wide-tile work; at 4 and 8 bits the window degenerates
to whole aligned words, so one reader covers every supported width.

Bounds: T <= TCAP (the softmax buffer is threadgroup memory, sized at
compile time), DH in {64, 128} (one or two 64-wide quantization groups, and
32 lanes must cover DH with a whole number of dims each).
t > TCAP is rejected loudly at both doors rather than left to the caller,
because overrunning a compile-time-sized threadgroup buffer is silent
memory corruption: the kernel body itself poisons the output with NaN and
returns (a kernel cannot raise, and a poisoned output fails every oracle
and every generation instantly), and the MLX door additionally raises
before dispatch, where the cache shape is visible host-side.
`should_dispatch` remains the routing predicate; the doors' rejection is
the backstop for callers that skip it.
"""

from __future__ import annotations

import numpy as np

from kernelverify.pack.wide_qmv import pack_codes
from kernelverify.schemas.quant_contract import QuantContract, canonical_quantize

SIMD_WIDTH = 32
SIMDGROUPS_PER_THREADGROUP = 8
TCAP = 1024                      # max cached positions one threadgroup holds
SUPPORTED_BITS = (2, 3, 4, 8)
SUPPORTED_DH = (64, 128)

# The body is door-neutral: `NUM_HEADS` and `T_CACHED` are spelled differently
# per door (MLX's generated shape buffers, or the runner door's scalar
# bindings), and everything else is shared so the doors cannot drift (the
# spike's W4 pattern, bench/spike_dequant_gemv.py).
KV_ATTENTION_MSL = """
    constexpr uint NSG   = 8;                 // simdgroups per threadgroup
    constexpr uint NT    = 32 * NSG;          // threads per threadgroup
    constexpr uint TCAP  = 1024;
    constexpr uint WORDS = DH * BITS / 32;    // packed words per cache row
    constexpr uint G     = DH / 64;           // quantization groups per row
    constexpr uint L     = DH / 32;           // output dims per lane
    const uint mask = (1u << BITS) - 1u;
    const float scale = 1.0f / metal::precise::sqrt((float)DH);

    uint lane = thread_position_in_threadgroup.x;
    uint sg   = thread_position_in_threadgroup.y;
    uint lin  = sg * 32 + lane;
    uint z    = thread_position_in_grid.z;    // b * H + h
    uint H    = NUM_HEADS;
    uint h    = z % H;
    uint Tc   = T_CACHED;                // cached positions

    // Capacity bound, in the shared body so both doors inherit it: Tc > TCAP
    // would overrun sc_arr, whose size is fixed at compile time. A kernel
    // cannot raise, so the loudest defined behaviour is to poison the output
    // and return; the condition is uniform across the grid, so every thread
    // leaves together and no barrier below is stranded.
    if (Tc > TCAP) {
        for (uint d = lin; d < DH; d += NT) { out[z * DH + d] = (T)NAN; }
        return;
    }

    threadgroup float qsh[DH];
    threadgroup float sc_arr[TCAP + 1];
    threadgroup float red[NSG];
    threadgroup float outbuf[NSG * DH];

    // The query this threadgroup answers, once, at fp32.
    for (uint d = lin; d < DH; d += NT) { qsh[d] = (float)q[z * DH + d]; }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Scores over T cached positions plus the step's own new entry: position
    // Tc reads new_k at full precision, so the T+1 softmax needs no concat.
    for (uint t = lin; t <= Tc; t += NT) {
        float acc = 0.0f;
        if (t == Tc) {
            for (uint d = 0; d < DH; ++d) {
                acc = metal::fma(qsh[d], (float)new_k[z * DH + d], acc);
            }
        } else {
            uint row = h * Tc + t;
            for (uint g = 0; g < G; ++g) {
                float s = (float)k_scales[row * G + g];
                float b = (float)k_biases[row * G + g];
                for (uint blk = 0; blk < 8; ++blk) {   // 8 blocks of 8 codes
                    uint bi = g * 8 + blk;
                    uint bit0 = bi * 8 * BITS;
                    uint w0 = bit0 >> 5;
                    uint sh0 = bit0 & 31;
                    uint base = row * WORDS + w0;
                    uint lo_w = k_wq[base];
                    uint hi_w = (sh0 + 8 * BITS > 32) ? k_wq[base + 1] : 0u;
                    ulong v = (((ulong)lo_w) | (((ulong)hi_w) << 32)) >> sh0;
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 8; ++j) {
                        float c = (float)((uint)(v >> (j * BITS)) & mask);
                        acc = metal::fma(qsh[bi * 8 + j], metal::fma(s, c, b), acc);
                    }
                }
            }
        }
        sc_arr[t] = acc * scale;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Softmax over T+1, max-subtracted, all fp32.
    float m_local = -INFINITY;
    for (uint t = lin; t <= Tc; t += NT) { m_local = metal::max(m_local, sc_arr[t]); }
    m_local = metal::simd_max(m_local);
    if (lane == 0) { red[sg] = m_local; }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (lin == 0) {
        float m = red[0];
        for (uint i = 1; i < NSG; ++i) { m = metal::max(m, red[i]); }
        red[0] = m;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float m = red[0];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    float s_local = 0.0f;
    for (uint t = lin; t <= Tc; t += NT) {
        float e = metal::precise::exp(sc_arr[t] - m);
        sc_arr[t] = e;
        s_local += e;
    }
    s_local = metal::simd_sum(s_local);
    if (lane == 0) { red[sg] = s_local; }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (lin == 0) {
        float d = 0.0f;
        for (uint i = 0; i < NSG; ++i) { d += red[i]; }
        red[0] = d;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float denom = red[0];

    // Combine: simdgroup sg takes cached positions sg, sg+8, ...; lane owns
    // dims [lane*L, lane*L + L). An L-dim block never crosses a group.
    uint d0 = lane * L;
    uint g_out = d0 >> 6;
    float acc[L];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < L; ++j) { acc[j] = 0.0f; }

    for (uint t = sg; t < Tc; t += NSG) {
        float p = sc_arr[t] / denom;
        uint row = h * Tc + t;
        uint bit0 = d0 * BITS;
        uint w0 = bit0 >> 5;
        uint sh0 = bit0 & 31;
        uint base = row * WORDS + w0;
        uint lo_w = v_wq[base];
        uint hi_w = (sh0 + L * BITS > 32) ? v_wq[base + 1] : 0u;
        ulong vv = (((ulong)lo_w) | (((ulong)hi_w) << 32)) >> sh0;
        float s = (float)v_scales[row * G + g_out];
        float b = (float)v_biases[row * G + g_out];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < L; ++j) {
            float c = (float)((uint)(vv >> (j * BITS)) & mask);
            acc[j] = metal::fma(p, metal::fma(s, c, b), acc[j]);
        }
    }
    if (sg == 0) {
        float p_new = sc_arr[Tc] / denom;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < L; ++j) {
            acc[j] = metal::fma(p_new, (float)new_v[z * DH + d0 + j], acc[j]);
        }
    }

    #pragma clang loop unroll(full)
    for (uint j = 0; j < L; ++j) { outbuf[sg * DH + d0 + j] = acc[j]; }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (uint d = lin; d < DH; d += NT) {
        float v = 0.0f;
        for (uint i = 0; i < NSG; ++i) { v += outbuf[i * DH + d]; }
        out[z * DH + d] = (T)v;
    }
"""

INPUT_NAMES = ["q", "k_wq", "k_scales", "k_biases",
               "v_wq", "v_scales", "v_biases", "new_k", "new_v"]
OUTPUT_NAMES = ["out"]
KERNEL_NAME = "kv_attn_decode"


def should_dispatch(t: int) -> bool:
    """Whether the pack should use this kernel at all, or defer to MLX.

    T <= TCAP is a hard capacity bound: the softmax buffer lives in
    threadgroup memory. Within it no profitability boundary has been
    measured yet; the interval claim waits on the binding run, per the
    scoped-claims rule (pack findings, section 5).
    """
    return t <= TCAP


def quantize_cache(cache: np.ndarray, bits: int):
    """Packed artefact triplet for one (H, T, DH) cache.

    Quantizes the same 2D (H*T, DH) reshape the shipped reference quantizes,
    so kernel and oracle read one artefact; packing is the verified MLX
    affine code stream from the wide-tile matvec.
    """
    h, t, dh = cache.shape
    art = canonical_quantize(cache.reshape(h * t, dh),
                             QuantContract(bits=bits, group_size=64))
    packed = pack_codes(art.q, bits).reshape(h, t, dh * bits // 32)
    groups = dh // 64
    return (packed, art.scales.reshape(h, t, groups),
            art.biases.reshape(h, t, groups))


def launch_config(b: int, h: int) -> tuple:
    """(grid, threadgroup): one threadgroup per (batch row, head)."""
    return ((SIMD_WIDTH, SIMDGROUPS_PER_THREADGROUP, b * h),
            (SIMD_WIDTH, SIMDGROUPS_PER_THREADGROUP, 1))


def kernel_spec():
    """The runner-door template, for verification through `MetalRunner`.

    `specialize()` fills the compile-time constants (`T`, `BITS`, `DH`); the
    head count and cached length arrive as scalar bindings, and the grid's z
    extent is the `b_rows` x `n_heads` product each case supplies.
    """
    from kernelverify.runners import Binding, BindingKind, KernelSpec, LaunchSpec

    source = f"""
#include <metal_stdlib>
using namespace metal;
using T = $T;
constant constexpr uint BITS = $BITS;
constant constexpr uint DH   = $DH;
kernel void {KERNEL_NAME}(
    device const half* q        [[buffer(0)]],
    device const uint* k_wq     [[buffer(1)]],
    device const half* k_scales [[buffer(2)]],
    device const half* k_biases [[buffer(3)]],
    device const uint* v_wq     [[buffer(4)]],
    device const half* v_scales [[buffer(5)]],
    device const half* v_biases [[buffer(6)]],
    device const half* new_k    [[buffer(7)]],
    device const half* new_v    [[buffer(8)]],
    device T* out               [[buffer(9)]],
    constant uint& n_heads      [[buffer(10)]],
    constant uint& t_cached     [[buffer(11)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 thread_position_in_grid        [[thread_position_in_grid]]) {{
{KV_ATTENTION_MSL.replace("NUM_HEADS", "n_heads").replace("T_CACHED", "t_cached")}
}}
"""
    return KernelSpec(
        source=source,
        entry_point=KERNEL_NAME,
        name=f"{KERNEL_NAME}-$BITS bit-DH$DH-$T",
        bindings=(Binding(BindingKind.INPUT, "q"),
                  Binding(BindingKind.INPUT, "k_wq"),
                  Binding(BindingKind.INPUT, "k_scales"),
                  Binding(BindingKind.INPUT, "k_biases"),
                  Binding(BindingKind.INPUT, "v_wq"),
                  Binding(BindingKind.INPUT, "v_scales"),
                  Binding(BindingKind.INPUT, "v_biases"),
                  Binding(BindingKind.INPUT, "new_k"),
                  Binding(BindingKind.INPUT, "new_v"),
                  Binding(BindingKind.OUTPUT),
                  Binding(BindingKind.SCALAR, "n_heads", "uint32"),
                  Binding(BindingKind.SCALAR, "t_cached", "uint32")),
        launch=LaunchSpec(grid=(SIMD_WIDTH, SIMDGROUPS_PER_THREADGROUP,
                                ["b_rows", "n_heads"]),
                          threadgroup=(SIMD_WIDTH, SIMDGROUPS_PER_THREADGROUP, 1)),
    )


def build(mx):
    """The MLX callable. Takes mx so this module imports without it.

    The callable rejects t > TCAP before dispatch: the cache shape is visible
    host-side here, so the capacity bound can raise instead of relying on the
    in-body NaN guard alone.
    """
    source = (KV_ATTENTION_MSL.replace("NUM_HEADS", "q_shape[1]")
              .replace("T_CACHED", "k_wq_shape[1]"))
    kernel = mx.fast.metal_kernel(name=KERNEL_NAME, input_names=INPUT_NAMES,
                                  output_names=OUTPUT_NAMES, source=source)
    k_wq_pos = INPUT_NAMES.index("k_wq")

    def guarded(*, inputs, **kwargs):
        t = inputs[k_wq_pos].shape[1]
        if t > TCAP:
            raise ValueError(
                f"kv_attention: t_cached={t} exceeds TCAP={TCAP}; the softmax "
                f"buffer is threadgroup memory sized at compile time, so this "
                f"launch would corrupt it - route long caches to MLX "
                f"(should_dispatch)")
        return kernel(inputs=inputs, **kwargs)

    return guarded
