#!/usr/bin/env python3
"""fp64 reference for ReLU."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from _refkit import emit, read_inputs  # noqa: E402

def main():
    inputs, _ = read_inputs()
    x = inputs["input"]; dt = x.dtype
    xd = x.astype(np.float64)
    emit(np.maximum(xd, 0.0).astype(dt))

if __name__ == "__main__":
    main()
