"""LLM-style softmax with a classic tail-masking bug.

A frequent failure mode in generated GPU kernels: the reduction is sized to a
fixed BLOCK and the tail is not masked, so when the reduced dimension is not a
multiple of BLOCK, padding elements leak into the denominator.

This kernel is CORRECT when H is a multiple of BLOCK (e.g. the benchmark's
H=256) and WRONG otherwise (e.g. H=3, H=1025) — the "correctness illusion".
The reference is the correct fp64 softmax, so gpuemu flags it under fuzzing.
"""

import numpy as np

BLOCK = 128


def run(inputs):
    x = inputs["input"]
    dt = x.dtype
    x = x.astype(np.float32)
    H = x.shape[-1]
    padded = ((H + BLOCK - 1) // BLOCK) * BLOCK
    if padded != H:
        pad = np.zeros(x.shape[:-1] + (padded - H,), np.float32)
        xp = np.concatenate([x, pad], axis=-1)
    else:
        xp = x
    m = xp.max(axis=-1, keepdims=True)
    e = np.exp(xp - m)
    s = e.sum(axis=-1, keepdims=True)  # BUG: denominator includes padded zeros
    out = (e / s)[..., :H]
    return out.astype(dt)
