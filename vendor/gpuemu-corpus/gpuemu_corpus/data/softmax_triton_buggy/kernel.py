"""LLM-style buggy Triton softmax — the *real GPU* analog of softmax_llm_buggy.

The bug is one character different from `softmax_triton/kernel.py`: the OOB
load uses `other=0.0` instead of `other=-float("inf")`. exp(0)=1 then leaks
into the denominator for every padding lane, so the kernel is:

  - CORRECT  when n_cols is a power of two (BLOCK == n_cols, zero padding)
  - WRONG    when n_cols is anything else (e.g. 3, 1025) — and very wrong for
             small reductions where the padding dwarfs the real elements.

This is the "correctness illusion" demonstrated on real hardware: the kernel
passes a benchmark oracle's single H=256 case but fuzzing exposes it.
"""

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
    _TRITON_OK = torch.cuda.is_available()
except Exception:
    torch = None
    triton = None
    tl = None
    _TRITON_OK = False


if _TRITON_OK:

    @triton.jit
    def _softmax_kernel_buggy(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols,
                              BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        mask = col < n_cols
        # BUG: padding lanes get 0.0 instead of -inf → exp(0)=1 contaminates sum.
        x = tl.load(in_ptr + row * in_row_stride + col, mask=mask, other=0.0)
        x = x - tl.max(x, axis=0)
        e = tl.exp(x)
        y = e / tl.sum(e, axis=0)
        tl.store(out_ptr + row * out_row_stride + col, y, mask=mask)


_TORCH_DT = {"float16": "float16", "float32": "float32"}


def run(inputs):
    if not _TRITON_OK:
        raise RuntimeError("softmax_triton_buggy requires torch + triton + CUDA")
    x = inputs["input"]
    dt = x.dtype
    orig_shape = x.shape
    flat = np.ascontiguousarray(x).reshape(-1, orig_shape[-1])
    n_rows, n_cols = flat.shape

    torch_dt = getattr(torch, _TORCH_DT[str(dt)])
    x_t = torch.from_numpy(flat).to(device="cuda", dtype=torch_dt)
    y_t = torch.empty_like(x_t)
    block = triton.next_power_of_2(n_cols)
    _softmax_kernel_buggy[(n_rows,)](y_t, x_t, x_t.stride(0), y_t.stride(0), n_cols, BLOCK=block)
    return y_t.detach().cpu().numpy().reshape(orig_shape).astype(dt)
