"""Correct Triton GELU kernel (tanh approximation).

  y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Elementwise; flattens to 1D for the launch grid.
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
    def _gelu_kernel(out_ptr, in_ptr, n_elements,
                     BLOCK: tl.constexpr, KAPPA: tl.constexpr, COEFF: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        inner = KAPPA * (x + COEFF * x * x * x)
        # Stable manual tanh (Triton's tl.math.tanh / libdevice.tanh paths are
        # version-specific): tanh(z) = sign(z) * (1 - 2 / (exp(2|z|) + 1)).
        # The fp16 overflow case is well-behaved: exp -> +inf -> 2/inf -> 0 -> 1.
        abs_inner = tl.where(inner >= 0.0, inner, -inner)
        sign = tl.where(inner >= 0.0, 1.0, -1.0)
        th = sign * (1.0 - 2.0 / (tl.exp(2.0 * abs_inner) + 1.0))
        y = 0.5 * x * (1.0 + th)
        tl.store(out_ptr + offsets, y, mask=mask)


_TORCH_DT = {"float16": "float16", "float32": "float32"}


def run(inputs):
    if not _TRITON_OK:
        raise RuntimeError("gelu_triton requires torch + triton + CUDA")
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
    _gelu_kernel[grid](y_t, x_t, n, BLOCK=BLOCK, KAPPA=_SQRT_2_OVER_PI, COEFF=0.044715)
    return y_t.detach().cpu().numpy().reshape(shape).astype(dt)
