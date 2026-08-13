"""Correct Triton scaled-dot-product attention (single batch, single head).

  scores = Q @ K^T * (1/sqrt(D))      # shape [M, N]
  probs  = softmax(scores, axis=-1)    # shape [M, N]
  out    = probs @ V                   # shape [M, D]

One program per query row. Loads full K and V via BLOCK_N x BLOCK_D tiles
(works for the small test shapes; not flash-attention efficient, but a real
Triton kernel for the operator).
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
    def _attn_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr,
        M, N, D, SCALE,
        stride_qm, stride_qd,
        stride_kn, stride_kd,
        stride_vn, stride_vd,
        stride_om, stride_od,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)
        d_mask = offs_d < D
        n_mask = offs_n < N

        # Q[pid, :D]
        q = tl.load(q_ptr + pid * stride_qm + offs_d * stride_qd,
                    mask=d_mask, other=0.0).to(tl.float32)

        # K[:N, :D]: shape (BLOCK_N, BLOCK_D)
        k = tl.load(
            k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        # Scores: sum over D of q * k[n] -> (BLOCK_N,)
        scores = tl.sum(q[None, :] * k, axis=1) * SCALE
        scores = tl.where(n_mask, scores, -float("inf"))

        # Stable softmax
        scores = scores - tl.max(scores, axis=0)
        e = tl.exp(scores)
        p = e / tl.sum(e, axis=0)

        v = tl.load(
            v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        out = tl.sum(p[:, None] * v, axis=0)  # (BLOCK_D,)
        tl.store(out_ptr + pid * stride_om + offs_d * stride_od,
                 out.to(out_ptr.dtype.element_ty), mask=d_mask)


def run(inputs):
    if not _TRITON_OK:
        raise RuntimeError("attention_triton requires torch + triton + CUDA")
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    dt = q.dtype
    M, D = q.shape
    N, D2 = k.shape
    assert D == D2 and v.shape == (N, D), (q.shape, k.shape, v.shape)
    torch_dt = getattr(torch, _TORCH_DT[str(dt)])
    q_t = torch.from_numpy(np.ascontiguousarray(q)).to(device="cuda", dtype=torch_dt)
    k_t = torch.from_numpy(np.ascontiguousarray(k)).to(device="cuda", dtype=torch_dt)
    v_t = torch.from_numpy(np.ascontiguousarray(v)).to(device="cuda", dtype=torch_dt)
    o_t = torch.empty((M, D), device="cuda", dtype=torch_dt)
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_D = triton.next_power_of_2(D)
    _attn_kernel[(M,)](
        q_t, k_t, v_t, o_t, M, N, D, 1.0 / math.sqrt(D),
        q_t.stride(0), q_t.stride(1),
        k_t.stride(0), k_t.stride(1),
        v_t.stride(0), v_t.stride(1),
        o_t.stride(0), o_t.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o_t.detach().cpu().numpy().astype(dt)
