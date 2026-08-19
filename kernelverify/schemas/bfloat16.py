"""bfloat16 as a host-side carrier, and its two exact conversions.

Training is where bf16 arrives. The shipped 4-bit artifacts store their
scales and biases as bfloat16, and mlx-lm takes the model's dtype straight
from the scales (`mlx_lm/utils.py`), so a training kernel this repository
gates is a bf16 kernel whether or not the contract says so.

numpy has no bfloat16, and it will not view one: `np.array(bf16_array)`
raises outright. So bf16 travels the host side as its raw 16 bits in a
uint16, exactly the way packed quantized codes travel as uint32, and the
shader declares `bfloat` on its own. `to_bits` and `from_bits` are the only
two places that know the layout.

`from_bits` is exact in the strong sense: every bf16 value is a float32
value, because bf16 is float32's top 16 bits. `to_bits` is the rounding
direction and rounds to nearest with ties to even, which is what the
hardware store does; it is the operation that loses information, and the
contract's eps for bfloat16 is what bounds that loss.
"""

from __future__ import annotations

import numpy as np

# One ulp at 1.0, by this repository's convention of 2**-(stored mantissa
# bits): bf16 keeps 7, against float16's 10 and float32's 23. The contract's
# _EPS table carries the same number.
EPS = 2.0 ** -7

_QUIET_NAN = np.uint16(0x7FC0)
_SIGN = np.uint16(0x8000)


def to_bits(values: np.ndarray) -> np.ndarray:
    """float32 -> the uint16 carrier, rounding to nearest with ties to even."""
    source = np.ascontiguousarray(values, dtype=np.float32)
    word = source.view(np.uint32)
    # Half an ulp of the discarded low half, plus one more when the bit being
    # kept is already odd, which is what sends an exact tie to the even
    # neighbour rather than always upward.
    bias = ((word >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    with np.errstate(over="ignore"):  # a NaN's high payload wraps; handled below
        bits = ((word + bias) >> np.uint32(16)).astype(np.uint16)
    # A NaN whose payload lives entirely in the discarded half would round to
    # an infinity, turning "not a number" into a finite-looking claim, so NaN
    # is written as a quiet NaN explicitly.
    nan_bits = _QUIET_NAN | ((word >> np.uint32(16)).astype(np.uint16) & _SIGN)
    return np.where(np.isnan(source), nan_bits, bits).astype(np.uint16)


def from_bits(bits: np.ndarray) -> np.ndarray:
    """The uint16 carrier -> float32, exactly: bf16 is float32's top half."""
    word = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32)
    return (word << np.uint32(16)).view(np.float32)
