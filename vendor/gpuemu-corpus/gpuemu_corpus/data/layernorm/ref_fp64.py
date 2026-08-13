#!/usr/bin/env python3
"""fp64 reference for LayerNorm (no affine) over the last axis."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from _refkit import emit, read_inputs  # noqa: E402

EPS = 1e-5


def main():
    inputs, _ = read_inputs()
    x = inputs["input"]
    dt = x.dtype
    xd = x.astype(np.float64)
    mu = xd.mean(axis=-1, keepdims=True)
    var = xd.var(axis=-1, keepdims=True)
    out = (xd - mu) / np.sqrt(var + EPS)
    emit(out.astype(dt))


if __name__ == "__main__":
    main()
