#!/usr/bin/env python3
"""fp64 reference for matmul C[M,N] = A[M,K] @ B[K,N]."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from _refkit import emit, read_inputs  # noqa: E402


def main():
    inputs, _ = read_inputs()
    a = inputs["a"]
    b = inputs["b"]
    dt = a.dtype
    out = a.astype(np.float64) @ b.astype(np.float64)
    emit(out.astype(dt))


if __name__ == "__main__":
    main()
