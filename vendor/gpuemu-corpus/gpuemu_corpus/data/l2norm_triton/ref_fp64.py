#!/usr/bin/env python3
"""fp64 reference for L2-norm over last axis."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from _refkit import emit, read_inputs  # noqa: E402

EPS = 1e-12

def main():
    inputs, _ = read_inputs()
    x = inputs["input"]; dt = x.dtype
    xd = x.astype(np.float64)
    sum_sq = (xd * xd).sum(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(sum_sq + EPS)
    emit((xd * inv).astype(dt))

if __name__ == "__main__":
    main()
