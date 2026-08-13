"""LLM-style buggy flash-attention: forgot to rescale the accumulator on max update.

The online-softmax invariant requires that when the running max updates from
m to m_new, both the normalizer l AND the accumulator acc must be multiplied
by alpha = exp(m - m_new). Forgetting the acc * alpha line is a classic LLM
transcription bug.

Shape-dependent illusion:
  - For N <= BLOCK_N (single tile, no max update mid-stream): correct.
  - For N >  BLOCK_N (multiple tiles, max updates as more keys are seen):
    accumulator is left at the OLD scale while l_i is correctly rescaled,
    so out = acc / l_i is wrong by a multiplicative factor that depends on
    how many tiles had a smaller max.
"""

import math
import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
    _TRITON_OK = torch.cuda.is_available()
except Exception:
    torch = triton = tl = None
    _TRITON_OK = False


_TORCH_DT = {"float16": "float16", "float32": "float32"}


if _TRITON_OK:

    @triton.jit
    def _flash_attn_buggy_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr,
        M, N, D, SCALE,
        stride_qm, stride_qd,
        stride_kn, stride_kd,
        stride_vn, stride_vd,
        stride_om, stride_od,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        m_mask = offs_m < M
        d_mask = offs_d < D

        q = tl.load(
            q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=m_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N
            k = tl.load(
                k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=n_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=n_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)

            s = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * SCALE
            s = tl.where(n_mask[None, :], s, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            # BUG: missing  acc = acc * alpha[:, None]  +  ...
            # Accumulator is left at the old scale; output is wrong on N > BLOCK_N.
            acc = acc + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
            m_i = m_new

        out = acc / l_i[:, None]
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
            out.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & d_mask[None, :],
        )


def run(inputs):
    if not _TRITON_OK:
        raise RuntimeError("flash_attention_triton_buggy requires torch + triton + CUDA")
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    dt = q.dtype
    M, D = q.shape
    N, _ = k.shape
    torch_dt = getattr(torch, _TORCH_DT[str(dt)])
    q_t = torch.from_numpy(np.ascontiguousarray(q)).to(device="cuda", dtype=torch_dt)
    k_t = torch.from_numpy(np.ascontiguousarray(k)).to(device="cuda", dtype=torch_dt)
    v_t = torch.from_numpy(np.ascontiguousarray(v)).to(device="cuda", dtype=torch_dt)
    o_t = torch.empty((M, D), device="cuda", dtype=torch_dt)
    BLOCK_M = max(4, triton.next_power_of_2(min(M, 8)))
    BLOCK_N = max(8, triton.next_power_of_2(min(N, 32)))
    BLOCK_D = triton.next_power_of_2(D)
    grid = ((M + BLOCK_M - 1) // BLOCK_M,)
    _flash_attn_buggy_kernel[grid](
        q_t, k_t, v_t, o_t,
        M, N, D, 1.0 / math.sqrt(D),
        q_t.stride(0), q_t.stride(1),
        k_t.stride(0), k_t.stride(1),
        v_t.stride(0), v_t.stride(1),
        o_t.stride(0), o_t.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o_t.detach().cpu().numpy().astype(dt)
