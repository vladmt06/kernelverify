#!/usr/bin/env python3
"""fp64 reference for RMSNorm: y = x * rsqrt(mean(x^2, last) + eps)."""

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
    mean_sq = (xd * xd).mean(axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(mean_sq + EPS)
    emit((xd * inv_rms).astype(dt))


if __name__ == "__main__":
    main()
