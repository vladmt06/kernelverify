#!/usr/bin/env python3
"""fp64 reference for scaled-dot-product attention."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from _refkit import emit, read_inputs  # noqa: E402

def main():
    inputs, _ = read_inputs()
    q = inputs["q"].astype(np.float64)
    k = inputs["k"].astype(np.float64)
    v = inputs["v"].astype(np.float64)
    dt = inputs["q"].dtype
    D = q.shape[-1]
    scores = (q @ k.T) / np.sqrt(D)
    scores -= scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    p = e / e.sum(axis=-1, keepdims=True)
    out = p @ v
    emit(out.astype(dt))

if __name__ == "__main__":
    main()
