#!/usr/bin/env python3
"""Minimal GGUF reader, for the flop and byte models the roofline needs.

The first version of the baseline scored prompt processing as
`2 * model_n_params * tokens`, and produced 110% of the machine's measured flop
ceiling for Qwen2.5-0.5B, which is impossible and therefore falsified the
model. The cause is that `model_n_params` counts the token embedding table,
which is a gather at input, not a matmul, and on a 0.5B model with a 151k
vocabulary that table is 22% of all parameters.

So the flop count is built from the tensors themselves:

  body      every 2-D weight that is a matmul operand, run once per token
  head      the output projection, run once per decode call, because llama.cpp
            only materialises logits for the last position of a prompt
  attention the two score matmuls, which scale with context length, not with
            parameter count

Reading a header costs milliseconds and no dependencies: the GGUF layout is a
fixed prelude, a key-value section, and a tensor table.

    .venv/bin/python bench/gguf_info.py <model.gguf>
"""

from __future__ import annotations

import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

# value type tags from the GGUF spec
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

_SCALAR = {
    U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
    U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
}

# ggml_type -> (block size in elements, bytes per block), from ggml-common.h.
GGML_TYPES = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24), 8: ("q8_0", 32, 34), 9: ("q8_1", 32, 40),
    10: ("q2_K", 256, 84), 11: ("q3_K", 256, 110), 12: ("q4_K", 256, 144),
    13: ("q5_K", 256, 176), 14: ("q6_K", 256, 210), 15: ("q8_K", 256, 292),
    30: ("bf16", 1, 2),
}


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    ggml_type: str
    n_elements: int
    n_bytes: int


@dataclass(frozen=True)
class Model:
    path: Path
    kv: dict
    tensors: tuple[Tensor, ...]

    @property
    def arch(self) -> str:
        return self.kv.get("general.architecture", "?")

    def a(self, suffix: str, default=None):
        """Architecture-scoped metadata, e.g. block_count."""
        return self.kv.get(f"{self.arch}.{suffix}", default)


class _Reader:
    def __init__(self, buf: bytes):
        self.b, self.i = buf, 0

    def take(self, n: int) -> bytes:
        out = self.b[self.i : self.i + n]
        if len(out) != n:
            raise ValueError("truncated GGUF header")
        self.i += n
        return out

    def scalar(self, tag: int):
        fmt, size = _SCALAR[tag]
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        return self.take(self.scalar(U64)).decode("utf-8", "replace")

    def value(self, tag: int):
        if tag == STRING:
            return self.string()
        if tag == ARRAY:
            elem = self.scalar(U32)
            count = self.scalar(U64)
            # Tokenizer arrays run to hundreds of thousands of entries and
            # nothing here needs them, so they are consumed and dropped.
            if elem == STRING:
                for _ in range(count):
                    self.string()
                return f"<{count} strings>"
            fmt, size = _SCALAR[elem]
            self.take(size * count)
            return f"<{count} values>"
        return self.scalar(tag)


def read(path: Path, header_bytes: int = 64 << 20) -> Model:
    with path.open("rb") as fh:
        buf = fh.read(header_bytes)
    r = _Reader(buf)
    if r.take(4) != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    version = r.scalar(U32)
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    n_tensors = r.scalar(U64)
    n_kv = r.scalar(U64)

    kv = {}
    for _ in range(n_kv):
        key = r.string()
        kv[key] = r.value(r.scalar(U32))

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        dims = tuple(r.scalar(U64) for _ in range(r.scalar(U32)))
        tag = r.scalar(U32)
        r.scalar(U64)  # data offset, not needed
        if tag not in GGML_TYPES:
            raise ValueError(f"unknown ggml type {tag} on {name}")
        tname, blk, blk_bytes = GGML_TYPES[tag]
        n = 1
        for d in dims:
            n *= d
        tensors.append(Tensor(name, dims, tname, n, n // blk * blk_bytes))

    return Model(path, kv, tuple(tensors))


def _is_embedding(name: str) -> bool:
    return name.startswith("token_embd") or name.endswith("token_embd.weight")


def _is_head(name: str) -> bool:
    return name.startswith("output.") or name == "output.weight"


def cost_model(m: Model) -> dict:
    """Per-token flops and per-token bytes, split by what runs when."""
    body_params = head_params = 0
    body_bytes = head_bytes = embd_bytes = 0

    for t in m.tensors:
        is_weight = len(t.shape) == 2 and t.name.endswith(".weight")
        if _is_embedding(t.name):
            embd_bytes += t.n_bytes
        elif _is_head(t.name) and is_weight:
            head_params += t.n_elements
            head_bytes += t.n_bytes
        elif is_weight:
            body_params += t.n_elements
            body_bytes += t.n_bytes
        else:
            # norms and biases: negligible flops, but their bytes are read.
            body_bytes += t.n_bytes

    # A model with tied embeddings has no separate output tensor; the
    # embedding table is the head, and is fully read whenever logits are.
    tied = head_params == 0
    if tied:
        embd = next((t for t in m.tensors if _is_embedding(t.name)), None)
        if embd is not None:
            head_params = embd.n_elements
            head_bytes = embd.n_bytes
            embd_bytes = 0

    n_layer = m.a("block_count") or 0
    n_head = m.a("attention.head_count") or 0
    n_head_kv = m.a("attention.head_count_kv") or n_head
    n_embd = m.a("embedding_length") or 0

    # Head dimension is not always embedding_length / head_count. Qwen3-4B
    # declares key_length 128 against an implied 80, so deriving it understates
    # attention flops and cache traffic by 1.6x on exactly the models this
    # baseline reports. Take the declared value whenever the file states one.
    implied = (n_embd // n_head) if n_head else 0
    d_head_k = m.a("attention.key_length") or implied
    d_head_v = m.a("attention.value_length") or implied

    return {
        "arch": m.arch,
        "tied_embeddings": tied,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "d_head_k": d_head_k,
        "d_head_v": d_head_v,
        "d_head_implied": implied,
        "d_head_declared": bool(m.a("attention.key_length")),
        "body_params": body_params,
        "head_params": head_params,
        "body_flops_per_token": 2 * body_params,
        "head_flops_per_call": 2 * head_params,
        "body_bytes": body_bytes,
        "head_bytes": head_bytes,
        "embedding_bytes_resident": embd_bytes,
    }


def _attn_flops_per_query_token(cost: dict) -> float:
    """QK-transpose plus the value matmul, per query token per cache position."""
    return 2.0 * cost["n_layer"] * cost["n_head"] * (cost["d_head_k"] + cost["d_head_v"])


def prompt_flops(cost: dict, n_tokens: int) -> float:
    """Flops for one decode call over n_tokens of prompt.

    The attention score matmuls are causal, so over a prompt of T tokens they
    cost the per-token rate times T(T+1)/2 rather than T squared.
    """
    body = cost["body_flops_per_token"] * n_tokens
    head = cost["head_flops_per_call"]
    attn = _attn_flops_per_query_token(cost) * n_tokens * (n_tokens + 1) / 2.0
    return body + head + attn


def gen_flops(cost: dict, context: int) -> float:
    """Flops to generate one token with `context` tokens already in the cache."""
    attn = _attn_flops_per_query_token(cost) * context
    return cost["body_flops_per_token"] + cost["head_flops_per_call"] + attn


def gen_bytes(cost: dict) -> int:
    """Weight bytes read to generate one token: the whole body plus the head."""
    return cost["body_bytes"] + cost["head_bytes"]


def kv_bytes(cost: dict, context: int, bytes_per_elem: int = 2) -> int:
    """Cache bytes touched at a given context length, K and V summed."""
    per_position = cost["n_head_kv"] * (cost["d_head_k"] + cost["d_head_v"])
    return int(cost["n_layer"] * per_position * context * bytes_per_elem)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 1
    m = read(Path(sys.argv[1]))
    c = cost_model(m)
    c["file_bytes"] = Path(sys.argv[1]).stat().st_size
    c["n_tensors"] = len(m.tensors)
    print(json.dumps(c, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
