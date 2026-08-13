"""LLM-style buggy SiLU: confuses SiLU with Swish-β (uses sigmoid(2x) instead of sigmoid(x)).

A common transcription bug — the writer remembered "Swish family" and grabbed
a different β. Outputs are uniformly wrong: y_buggy = x * sigmoid(2x), so for
positive x it's roughly 2x the correct value, for negative x it's near zero.
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


if _TRITON_OK:

    @triton.jit
    def _silu_buggy_kernel(out_ptr, in_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        # Same fp32-internal pattern as the correct kernel (see silu_triton).
        x_f = x.to(tl.float32)
        # BUG: sigmoid(2*x) instead of sigmoid(x).
        sig = 1.0 / (1.0 + tl.exp(-2.0 * x_f))
        y = (x_f * sig).to(x.dtype)
        tl.store(out_ptr + offsets, y, mask=mask)


_TORCH_DT = {"float16": "float16", "float32": "float32"}


def run(inputs):
    if not _TRITON_OK:
        raise RuntimeError("silu_triton_buggy requires torch + triton + CUDA")
    x = inputs["input"]
    dt = x.dtype
    shape = x.shape
    flat = np.ascontiguousarray(x).reshape(-1)
    n = flat.size
    torch_dt = getattr(torch, _TORCH_DT[str(dt)])
    x_t = torch.from_numpy(flat).to(device="cuda", dtype=torch_dt)
    y_t = torch.empty_like(x_t)
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    _silu_buggy_kernel[grid](y_t, x_t, n, BLOCK=BLOCK)
    return y_t.detach().cpu().numpy().reshape(shape).astype(dt)
