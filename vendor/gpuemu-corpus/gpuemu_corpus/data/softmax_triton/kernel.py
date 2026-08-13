"""Correct Triton softmax kernel (real GPU compute).

Reduces over the last dim. The driver wraps `run()` in try/except, so on a
host without CUDA/Triton, import failures + run() failures are recorded as
per-iteration "exception" results rather than killing the run.

Triton's `@triton.jit` reads the kernel's annotations (`BLOCK: tl.constexpr`,
...) against the kernel's *module* globals, so `triton` and `tl` MUST be
imported at module level — not inside a helper.
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
    def _softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols,
                        BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        mask = col < n_cols
        # Correct pattern: mask OOB to -inf so exp(-inf)=0 contributes nothing.
        x = tl.load(in_ptr + row * in_row_stride + col, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        e = tl.exp(x)
        y = e / tl.sum(e, axis=0)
        tl.store(out_ptr + row * out_row_stride + col, y, mask=mask)


_TORCH_DT = {"float16": "float16", "float32": "float32"}


def run(inputs):
    if not _TRITON_OK:
        raise RuntimeError("softmax_triton requires torch + triton + CUDA")
    x = inputs["input"]
    dt = x.dtype
    orig_shape = x.shape
    flat = np.ascontiguousarray(x).reshape(-1, orig_shape[-1])
    n_rows, n_cols = flat.shape

    torch_dt = getattr(torch, _TORCH_DT[str(dt)])
    x_t = torch.from_numpy(flat).to(device="cuda", dtype=torch_dt)
    y_t = torch.empty_like(x_t)
    block = triton.next_power_of_2(n_cols)
    _softmax_kernel[(n_rows,)](y_t, x_t, x_t.stride(0), y_t.stride(0), n_cols, BLOCK=block)
    return y_t.detach().cpu().numpy().reshape(orig_shape).astype(dt)
