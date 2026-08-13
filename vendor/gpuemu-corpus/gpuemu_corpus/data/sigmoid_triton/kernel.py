"""Correct Triton sigmoid: y = 1/(1+exp(-x)). fp32-internal pattern (silu lesson)."""

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
    def _sigmoid_kernel(out_ptr, in_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        x_f = x.to(tl.float32)
        y = (1.0 / (1.0 + tl.exp(-x_f))).to(x.dtype)
        tl.store(out_ptr + offsets, y, mask=mask)


def run(inputs):
    if not _TRITON_OK:
        raise RuntimeError("sigmoid_triton requires torch + triton + CUDA")
    x = inputs["input"]; dt = x.dtype; shape = x.shape
    flat = np.ascontiguousarray(x).reshape(-1); n = flat.size
    torch_dt = getattr(torch, _TORCH_DT[str(dt)])
    x_t = torch.from_numpy(flat).to(device="cuda", dtype=torch_dt)
    y_t = torch.empty_like(x_t)
    BLOCK = 1024; grid = ((n + BLOCK - 1) // BLOCK,)
    _sigmoid_kernel[grid](y_t, x_t, n, BLOCK=BLOCK)
    return y_t.detach().cpu().numpy().reshape(shape).astype(dt)
