#!/usr/bin/env python3
"""fp64 reference for matmul: C[M,N] = A[M,K] @ B[K,N]."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from _refkit import emit, read_inputs  # noqa: E402

def main():
    inputs, _ = read_inputs()
    a, b = inputs["a"], inputs["b"]
    dt = a.dtype
    out = (a.astype(np.float64) @ b.astype(np.float64)).astype(dt)
    emit(out)

if __name__ == "__main__":
    main()
