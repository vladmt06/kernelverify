"""LLM-style buggy L2-norm: forgot the sqrt. y = x / (sum(x^2) + eps)."""

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


EPS = 1e-12


if _TRITON_OK:

    @triton.jit
    def _l2norm_buggy_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols,
                             BLOCK: tl.constexpr, EPS: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        mask = col < n_cols
        x = tl.load(in_ptr + row * in_row_stride + col, mask=mask, other=0.0)
        x_f = x.to(tl.float32)
        sq = x_f * x_f
        sum_sq = tl.sum(sq, axis=0)
        # BUG: forgot sqrt.
        inv = 1.0 / (sum_sq + EPS)
        y = (x_f * inv).to(x.dtype)
        tl.store(out_ptr + row * out_row_stride + col, y, mask=mask)


def run(inputs):
    if not _TRITON_OK:
        raise RuntimeError("l2norm_triton_buggy requires torch + triton + CUDA")
    x = inputs["input"]; dt = x.dtype; orig = x.shape
    flat = np.ascontiguousarray(x).reshape(-1, orig[-1])
    n_rows, n_cols = flat.shape
    torch_dt = getattr(torch, _TORCH_DT[str(dt)])
    x_t = torch.from_numpy(flat).to(device="cuda", dtype=torch_dt)
    y_t = torch.empty_like(x_t)
    BLOCK = triton.next_power_of_2(n_cols)
    _l2norm_buggy_kernel[(n_rows,)](y_t, x_t, x_t.stride(0), y_t.stride(0), n_cols,
                                    BLOCK=BLOCK, EPS=EPS)
    return y_t.detach().cpu().numpy().reshape(orig).astype(dt)
