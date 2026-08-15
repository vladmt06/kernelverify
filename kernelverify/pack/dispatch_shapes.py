"""The qmv shapes mlx-lm dispatches when decoding a Qwen3-class model.

The serving block (ruling D1) prices and gates the pack's wide_qmv at the
shapes the E2E path actually runs, not at microbenchmark shapes. At decode
every quantized linear in mlx-lm's qwen3 module sees x of M rows (M = batch,
L = 1) against a (d_out, d_in) weight, and with tied embeddings the logits
step dispatches one more through the quantized embedding table
(`embed_tokens.as_linear`, mlx_lm/models/qwen3.py).

DERIVED, NOT RECORDED: these shapes come from the model config plus the
mlx-lm module structure (separate q/k/v, gate/up/down MLP, no fusion). The
runner lane records the live dispatch shapes (risk RR1 in the pivot design
doc); if the recording disagrees, it wins and this derivation gets fixed.

The pinned fallback is Qwen/Qwen3-4B's config.json (transformers 4.51.0).
The block's own 3-bit artifact carries the same architecture fields in its
config.json; callers pass that when it is present and fall back to the pin
while the coordinator's conversion is still landing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# The alignment wide_qmv requires: a 64-wide quant group is a whole number
# of 8-code blocks, so d_in must divide by 64 (kernelverify/pack/wide_qmv.py).
D_IN_ALIGNMENT = 64

# Qwen/Qwen3-4B config.json, the architecture fields the derivation reads.
# head_dim is explicit because Qwen3 decouples it from hidden_size:
# hidden / heads is 80, the real head width is 128.
PINNED_QWEN3_4B = {
    "hidden_size": 2560,
    "head_dim": 128,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 9728,
    "vocab_size": 151936,
}


@dataclass(frozen=True)
class DispatchShape:
    """One distinct (d_out, d_in) the decode path dispatches, with the
    projection names that share it."""

    name: str
    d_out: int
    d_in: int


def qmv_dispatch_shapes(config: Mapping) -> tuple[DispatchShape, ...]:
    """Every distinct qmv (d_out, d_in) of one decode step, in module order.

    Projections landing on the same (d_out, d_in) merge into one entry with
    their names joined, so a gate or probe prices each distinct shape once.
    Raises KeyError on a missing architecture field (head_dim is never
    inferred) and ValueError when a shape breaks the kernel's alignment.
    """
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    intermediate = config["intermediate_size"]
    vocab = config["vocab_size"]

    projections = [
        ("q_proj", heads * head_dim, hidden),
        ("k_proj", kv_heads * head_dim, hidden),
        ("v_proj", kv_heads * head_dim, hidden),
        ("o_proj", hidden, heads * head_dim),
        ("gate_proj", intermediate, hidden),
        ("up_proj", intermediate, hidden),
        ("down_proj", hidden, intermediate),
        ("lm_head", vocab, hidden),
    ]

    merged: dict[tuple[int, int], list[str]] = {}
    for name, d_out, d_in in projections:
        merged.setdefault((d_out, d_in), []).append(name)

    shapes = tuple(DispatchShape("/".join(names), d_out, d_in)
                   for (d_out, d_in), names in merged.items())
    for s in shapes:
        if s.d_in % D_IN_ALIGNMENT:
            raise ValueError(
                f"{s.name}: d_in {s.d_in} is not divisible by "
                f"{D_IN_ALIGNMENT}, which wide_qmv requires")
    return shapes
