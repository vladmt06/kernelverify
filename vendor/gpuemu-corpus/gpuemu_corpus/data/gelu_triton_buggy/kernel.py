"""LLM-style buggy Triton GELU: forgot the 0.5 factor.

A classic transcription error: the writer remembered the tanh-approx GELU
formula but dropped the leading 0.5, so outputs are 2x too large. Unlike
the tail-mask bug in softmax_triton_buggy, this bug is *not* shape-dependent
— so gpuemu should catch it on essentially every test case.
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


_SQRT_2_OVER_PI = 0.7978845608028654


if _TRITON_OK:

    @triton.jit
    def _gelu_buggy_kernel(out_ptr, in_ptr, n_elements,
                           BLOCK: tl.constexpr, KAPPA: tl.constexpr, COEFF: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        inner = KAPPA * (x + COEFF * x * x * x)
        # Stable manual tanh (see gelu_triton/kernel.py for rationale).
        abs_inner = tl.where(inner >= 0.0, inner, -inner)
        sign = tl.where(inner >= 0.0, 1.0, -1.0)
        th = sign * (1.0 - 2.0 / (tl.exp(2.0 * abs_inner) + 1.0))
        # BUG: missing 0.5 factor.
        y = x * (1.0 + th)
        tl.store(out_ptr + offsets, y, mask=mask)


_TORCH_DT = {"float16": "float16", "float32": "float32"}


def run(inputs):
    if not _TRITON_OK:
        raise RuntimeError("gelu_triton_buggy requires torch + triton + CUDA")
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
    _gelu_buggy_kernel[grid](y_t, x_t, n, BLOCK=BLOCK, KAPPA=_SQRT_2_OVER_PI, COEFF=0.044715)
    return y_t.detach().cpu().numpy().reshape(shape).astype(dt)
