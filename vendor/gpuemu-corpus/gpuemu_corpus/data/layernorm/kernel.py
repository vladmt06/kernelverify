"""LayerNorm kernel under test (no affine), computed in float32."""

import numpy as np

EPS = 1e-5


def run(inputs):
    x = inputs["input"]
    dt = x.dtype
    x = x.astype(np.float32)
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    out = (x - mu) / np.sqrt(var + EPS)
    return out.astype(dt)
