"""LLM-style buggy matmul: assignment instead of accumulation inside the K loop.

  acc = ... (BUG)   instead of   acc += ...

Result: only the LAST K-step contributes. Correct iff K == 1; for K > 1 the
output is drastically wrong. Shape-dependent illusion that boundary fuzzing
on K (the schema includes K=1) catches conditionally.
"""

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
    def _matmul_buggy_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(K):
            a = tl.load(a_ptr + offs_m * stride_am + k * stride_ak, mask=m_mask, other=0.0)
            b = tl.load(b_ptr + k * stride_bk + offs_n * stride_bn, mask=n_mask, other=0.0)
            # BUG: assignment instead of accumulation.
            acc = a[:, None].to(tl.float32) * b[None, :].to(tl.float32)
        c = acc.to(c_ptr.dtype.element_ty)
        c_mask = m_mask[:, None] & n_mask[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=c_mask)


def run(inputs):
    if not _TRITON_OK:
        raise RuntimeError("matmul_triton_buggy requires torch + triton + CUDA")
    a, b = inputs["a"], inputs["b"]
    dt = a.dtype
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    torch_dt = getattr(torch, _TORCH_DT[str(dt)])
    a_t = torch.from_numpy(np.ascontiguousarray(a)).to(device="cuda", dtype=torch_dt)
    b_t = torch.from_numpy(np.ascontiguousarray(b)).to(device="cuda", dtype=torch_dt)
    c_t = torch.empty((M, N), device="cuda", dtype=torch_dt)
    BLOCK_M = 32; BLOCK_N = 32
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    _matmul_buggy_kernel[grid](
        a_t, b_t, c_t,
        M, N, K,
        a_t.stride(0), a_t.stride(1),
        b_t.stride(0), b_t.stride(1),
        c_t.stride(0), c_t.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return c_t.detach().cpu().numpy().astype(dt)
