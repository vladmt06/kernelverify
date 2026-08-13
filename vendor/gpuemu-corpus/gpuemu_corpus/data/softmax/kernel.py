"""Softmax kernel under test (CPU/numpy stand-in; computes in float32).

The driver imports `run`. On a GPU host this would call a Triton/CUDA kernel;
the numpy version lets the pipeline run locally. `run` returns the input dtype.
"""

import numpy as np


def run(inputs):
    x = inputs["input"]
    dt = x.dtype
    x = x.astype(np.float32)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    out = e / e.sum(axis=-1, keepdims=True)
    return out.astype(dt)
