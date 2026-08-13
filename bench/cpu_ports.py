"""The gpuemu corpus's 10 seeded faults, expressed as parameterisations.

The corpus kernels under `vendor/gpuemu-corpus` are Triton and refuse to run
without CUDA. Rather than reimplement each one separately, every entry here is
the shared parameterised kernel from `kernelverify/reference/kernels.py` with
one keyword pinned to the wrong value. The correct variant is the same function
with no keyword at all.

That indirection is the point. It means the hand-written faults published in
the paper and the faults our mutation engine synthesises are drawn from exactly
the same space, so a mutation score measured against synthetic faults is
measuring something that provably contains the real ones.

Fidelity limits of the CPU ports, stated rather than hidden:

- Reduction order inside `tl.sum` is not reproduced; numpy reduces in a
  different order than a GPU tree reduction. This shifts results by a few ULP
  and cannot change any verdict here, where the smallest real signal is 0.048
  against a 5e-2 threshold.
- The flash-attention port does not tile over BLOCK_M. Query rows are
  independent in that kernel, so BLOCK_M cannot affect the output values.
  BLOCK_N tiling, which the fault does depend on, is reproduced exactly.

The ports are checked two ways by `bench/measure_escape.py`: every correct
variant must pass its own schema sweep against the corpus fp64 references, and
`softmax_llm_buggy` must match the corpus original bit for bit, since that one
kernel is pure numpy and runs unmodified without a GPU.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernelverify.reference.kernels import (  # noqa: E402
    attention,
    flash_attention,
    gelu,
    l2norm,
    leaky_relu,
    matmul,
    rmsnorm,
    silu,
    softmax,
    softmax_padded,
)

# Corpus kernel name -> the parameterisation that reproduces it.
# A correct entry passes no keyword; a buggy entry pins exactly one.
PORTS = {
    "gelu_triton": gelu,
    "gelu_triton_buggy": functools.partial(gelu, leading_scale=1.0),
    "silu_triton": silu,
    "silu_triton_buggy": functools.partial(silu, beta=2.0),
    "leaky_relu_triton": leaky_relu,
    "leaky_relu_triton_buggy": functools.partial(leaky_relu, alpha=0.1),
    "rmsnorm_triton": rmsnorm,
    "rmsnorm_triton_buggy": functools.partial(rmsnorm, use_sqrt=False),
    "l2norm_triton": l2norm,
    "l2norm_triton_buggy": functools.partial(l2norm, use_sqrt=False),
    "softmax_triton": softmax,
    "softmax_triton_buggy": functools.partial(softmax, pad_fill=0.0),
    "softmax_llm_buggy": functools.partial(softmax_padded, mask_tail=False),
    "matmul_triton": matmul,
    "matmul_triton_buggy": functools.partial(matmul, accumulate=False),
    "attention_triton": attention,
    "attention_triton_buggy": functools.partial(attention, scale_power=0.0),
    "flash_attention_triton": flash_attention,
    "flash_attention_triton_buggy": functools.partial(flash_attention, rescale_acc=False),
}

# Each buggy kernel and the correct kernel that acts as its port control.
BUGGY_TO_CONTROL = {
    "gelu_triton_buggy": "gelu_triton",
    "silu_triton_buggy": "silu_triton",
    "leaky_relu_triton_buggy": "leaky_relu_triton",
    "rmsnorm_triton_buggy": "rmsnorm_triton",
    "l2norm_triton_buggy": "l2norm_triton",
    "softmax_triton_buggy": "softmax_triton",
    "softmax_llm_buggy": "softmax_triton",
    "matmul_triton_buggy": "matmul_triton",
    "attention_triton_buggy": "attention_triton",
    "flash_attention_triton_buggy": "flash_attention_triton",
}
