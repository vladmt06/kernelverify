"""LLM-style buggy Triton RMSNorm: forgot the sqrt.

  y_buggy = x / (mean(x^2) + eps)             # missing the sqrt!
  y_true  = x / sqrt(mean(x^2) + eps)

Classic copy-from-formula bug. For typical input magnitudes mean(x^2) ≫ 1, so
y_buggy is much smaller than y_true. Detectable on essentially every shape.
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


EPS = 1e-5


if _TRITON_OK:

    @triton.jit
    def _rmsnorm_buggy_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols,
                              BLOCK: tl.constexpr, EPS: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        mask = col < n_cols
        x = tl.load(in_ptr + row * in_row_stride + col, mask=mask, other=0.0)
        sq = x * x
        mean_sq = tl.sum(sq, axis=0) / n_cols
        # BUG: forgot the sqrt.
        inv_rms = 1.0 / (mean_sq + EPS)
        y = x * inv_rms
        tl.store(out_ptr + row * out_row_stride + col, y, mask=mask)


_TORCH_DT = {"float16": "float16", "float32": "float32"}


def run(inputs):
    if not _TRITON_OK:
        raise RuntimeError("rmsnorm_triton_buggy requires torch + triton + CUDA")
    x = inputs["input"]
    dt = x.dtype
    orig_shape = x.shape
    flat = np.ascontiguousarray(x).reshape(-1, orig_shape[-1])
    n_rows, n_cols = flat.shape
    torch_dt = getattr(torch, _TORCH_DT[str(dt)])
    x_t = torch.from_numpy(flat).to(device="cuda", dtype=torch_dt)
    y_t = torch.empty_like(x_t)
    BLOCK = triton.next_power_of_2(n_cols)
    _rmsnorm_buggy_kernel[(n_rows,)](y_t, x_t, x_t.stride(0), y_t.stride(0), n_cols,
                                     BLOCK=BLOCK, EPS=EPS)
    return y_t.detach().cpu().numpy().reshape(orig_shape).astype(dt)
