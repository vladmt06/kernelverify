#!/usr/bin/env python3
"""fp64 reference for tanh-approx GELU: y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from _refkit import emit, read_inputs  # noqa: E402

_KAPPA = np.sqrt(2.0 / np.pi)


def main():
    inputs, _ = read_inputs()
    x = inputs["input"]
    dt = x.dtype
    xd = x.astype(np.float64)
    inner = _KAPPA * (xd + 0.044715 * xd ** 3)
    out = 0.5 * xd * (1.0 + np.tanh(inner))
    emit(out.astype(dt))


if __name__ == "__main__":
    main()
