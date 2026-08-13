"""Matmul kernel under test: C[M,N] = A[M,K] @ B[K,N], accumulated in float32."""

import numpy as np


def run(inputs):
    a = inputs["a"]
    b = inputs["b"]
    dt = a.dtype
    out = a.astype(np.float32) @ b.astype(np.float32)
    return out.astype(dt)
