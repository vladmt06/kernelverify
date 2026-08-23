"""Causal grouped-query attention at TRAINING widths, and the cells it is tuned at.

`kernelverify/pack/kv_attention.py` is the decode sibling: one query position
against a quantized cache, where the whole cost is reading the cache. Training
is the other regime. Every position in the step attends at once, the mask is
causal, nothing is quantized, and the operator carries a backward.

What mlx-lm actually dispatches, read off the installed stack rather than
assumed: `mlx_lm.models.qwen3.Attention.__call__` calls

    scaled_dot_product_attention(queries, keys, values, cache=cache,
                                 scale=self.scale, mask=mask)

which is a re-export of `mlx_lm.models.base.scaled_dot_product_attention`, and
in a training step `cache` is None and `mask` is the literal string "causal".
That function forwards to `mx.fast.scaled_dot_product_attention`, which fuses
in a plain call and DECOMPOSES under a gradient trace into two matmuls and a
softmax, because MLX implements no fused attention backward at all. Reading the
graph MLX builds says so directly: the fused primitive appears once in a plain
call and zero times under `mx.grad`, where a Softmax appears instead.

So stock's training attention is the composed path in both directions, at every
one of the 36 layers, and the composed path materialises a
(batch, heads, width, width) score matrix per layer: 286 MB at batch 2 and
width 1057 in fp32. Not building it is worth as much as the time is, because it
is what puts a 16 GB machine in reach.

The geometry below is Qwen3-4B's and only Qwen3-4B's. A call site whose heads,
group ratio or head dimension differ routes to stock rather than to a kernel
verified somewhere else, the same rule `train_qmm.shape_name` applies to the
projections.
"""

from __future__ import annotations

# Qwen3-4B's attention geometry, as `mlx_lm.models.qwen3.ModelArgs` carries it.
# The group ratio is what makes this grouped-query rather than multi-head: four
# query heads share each key-value head, so a backward must sum a group's four
# gradients onto the one head they read.
N_Q_HEADS = 32
N_KV_HEADS = 8
GQA_GROUP = N_Q_HEADS // N_KV_HEADS
HEAD_DIM = 128

# `head_dim**-0.5`, the value qwen3's Attention computes and hands to the seam.
# Written out as the constant rather than recomputed, because a kernel that
# scales by a different value is wrong in a way no shape check would catch.
SCALE = HEAD_DIM ** -0.5

# The batch sizes a tuned configuration exists for. A step at any other batch
# routes to stock: the tile geometry that wins at four rows of queries is not
# the one that wins at sixteen, and a configuration measured at one batch says
# nothing about another.
BATCHES = (1, 2, 4)

# Width is data-dependent (mlx-lm pads a batch to one past the next multiple of
# 32 above its longest row), so a tuned configuration is stored against a
# BUCKET. Powers of two, because that is the axis along which a tile
# configuration's behaviour changes; floored at 128 because below that the
# whole step is small enough that the composed path's overhead dominates
# anything a kernel could win; capped because past the cap the key loop is
# long enough that its own steady state, not its edges, sets the winner.
T_BUCKET_FLOOR = 128
T_BUCKET_CAP = 2048

# The two registered bands, from the two pinned corpus slices: the short band
# is width 65 at batch 4, the long band is width 1057 at batch 2. Every gate
# and every timing arm in this pack reports at both, because attention is the
# one region whose share is quadratic in the width, so a reading at one band
# says nothing about the other.
CELLS = ("B4:T128", "B2:T2048")


def t_bucket(t: int) -> int:
    """The bucket a step's width is tuned and looked up under.

    Rounds UP to the next power of two so a bucket's tuned configuration was
    measured on a problem at least as large as the one it is applied to.
    Rounding down would credit a configuration with a speed it was never shown
    to reach at the width it runs at.
    """
    if t < 1:
        raise ValueError(f"a step {t} tokens wide has nothing to attend over")
    bucket = T_BUCKET_FLOOR
    while bucket < t and bucket < T_BUCKET_CAP:
        bucket *= 2
    return bucket


def cell_key(batch: int, t: int) -> str:
    """The knob map's key: one tuned configuration per batch per bucket."""
    if batch not in BATCHES:
        raise ValueError(f"batch {batch} is not one of {list(BATCHES)}")
    return f"B{batch}:T{t_bucket(t)}"


def cell_name(batch: int, t: int) -> str | None:
    """Which cell this call site is, or None if it is not one of ours.

    None is a route to stock, never a guess at the nearest cell: a
    configuration measured at one batch says nothing about another.
    """
    if batch not in BATCHES:
        return None
    return cell_key(batch, t)


def geometry_ok(n_q_heads: int, n_kv_heads: int, head_dim: int) -> bool:
    """Whether a call site is the geometry this pack is verified for.

    Checked at every call rather than assumed from the model path: a config
    file names the model, and the tensors name what the model actually built.
    """
    return (n_q_heads == N_Q_HEADS and n_kv_heads == N_KV_HEADS
            and head_dim == HEAD_DIM)
