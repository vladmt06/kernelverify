"""The quantized projection matmul at TRAINING widths, and the knobs that tune it.

`kernelverify/pack/wide_qmv.py` is the decode-shaped sibling: one input vector at
a time, or at most eleven, where the whole cost is reading the weights. Training
is the other regime. A step carries hundreds or thousands of tokens at once, so
the same operator becomes a real matmul whose arithmetic dominates, and a kernel
tuned for the matvec regime is the wrong kernel here.

What mlx-lm actually dispatches, read off the installed stack rather than
assumed: every LoRA projection reaches `mx.quantized_matmul(x, w_q, scales,
biases, transpose=True)` through `nn.QuantizedLinear.__call__`, and the tied
output head reaches the same operator through `nn.QuantizedEmbedding.as_linear`.
Both compute

    out[m, n] = sum_k x[m, k] * dequant(w)[n, k]

with x of shape (M, d_in), the packed weight of logical shape (d_out, d_in), and
the output (M, d_out). M is batch times sequence width, so it moves with the
data; d_in and d_out are fixed by the model.

ORIENTATION is "tt" and only "tt" here, and that is a scoping decision rather
than an oversight. The backward pass needs the other orientation, dX = dY @
dequant(w), but the pre-registration requires a kept kernel's gradient to equal
stock's gradient by exact array equality, and a differently-ordered accumulation
cannot be bit-equal. So the shipped vjp calls stock's own transpose=False
operator, no kernel of ours runs in that direction, and the nt orientation gets
its own certified entry later under its own amendment.

The knobs exist because a kernel that is fastest on one Apple GPU is not
fastest on the next one. The tile widths, the per-thread output block and the
reduction depth trade threadgroup memory against register pressure against
occupancy, and where the optimum sits is a property of the chip. Nothing here
picks a winner: `axes()` describes the space, `kernelverify.compiler.search_space`
drops what this device cannot launch, and the winner is measured on whatever
machine the user actually has.
"""

from __future__ import annotations

# The six distinct projection shapes Qwen3-4B dispatches in a training step, as
# (d_out, d_in). Held here as product data rather than imported from the bench
# tree, because the on-device tuner ships to users and bench/ does not.
#
# S6 is k_proj and v_proj: 8 key-value heads at head dimension 128 gives 1024
# output rows, which is the shape a five-entry table missed for a while.
# S5 is the tied output head, the largest by two orders of magnitude.
TRAIN_SHAPES = {
    "S1": (4096, 2560),      # q_proj
    "S2": (2560, 4096),      # o_proj
    "S3": (9728, 2560),      # gate_proj and up_proj
    "S4": (2560, 9728),      # down_proj
    "S5": (151936, 2560),    # the tied output head
    "S6": (1024, 2560),      # k_proj and v_proj
}

ORIENTATION = "tt"

# The quantization this pack is verified for. Anything else declines rather
# than routing: a kernel checked at one format must never run at another.
BITS = 4
GROUP_SIZE = 64

# M is data-dependent, so a tuned configuration is stored against a BUCKET
# rather than an exact token count. Powers of two, because that is the axis
# along which a tile configuration's behaviour actually changes, floored at 64
# because below that the matvec regime and wide_qmv own the problem, and capped
# because past the cap the arithmetic is saturated and the same tile wins.
M_BUCKET_FLOOR = 64
M_BUCKET_CAP = 2048


def m_bucket(m: int) -> int:
    """The bucket a token count is tuned and looked up under.

    Rounds UP to the next power of two so a bucket's tuned configuration was
    measured on a problem at least as large as the one it is being applied to.
    Rounding down would credit a configuration with a speed it was never shown
    to reach at the width it is used at.
    """
    if m < 1:
        raise ValueError(f"a step with {m} tokens has nothing to multiply")
    bucket = M_BUCKET_FLOOR
    while bucket < m and bucket < M_BUCKET_CAP:
        bucket *= 2
    return bucket


def cell_key(shape_name: str, m: int) -> str:
    """The knob map's key: one tuned configuration per shape per bucket."""
    if shape_name not in TRAIN_SHAPES:
        raise KeyError(f"{shape_name!r} is not one of {sorted(TRAIN_SHAPES)}")
    return f"{shape_name}:M{m_bucket(m)}"


def shape_name(d_out: int, d_in: int) -> str | None:
    """Which registered shape this call site is, or None if it is not one.

    None is a route to stock, never a guess at the nearest shape: a
    configuration measured at one shape says nothing about another.
    """
    for name, dims in TRAIN_SHAPES.items():
        if dims == (d_out, d_in):
            return name
    return None
