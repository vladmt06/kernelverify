"""Streamed-bytes model for an MLX safetensors checkpoint.

The MLX rows in the matrix need the same treatment the GGUF rows get in
bench/gguf_info.py: a utilisation column divided by file size is wrong,
because the embedding table is a gather rather than a matmul unless the model
ties it to the output head.

Safetensors makes this cheap. The file opens with an 8-byte little-endian
header length, then a JSON header giving every tensor's dtype, shape and byte
range, so the byte accounting needs no tensor data and no MLX import.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

# The byte models are shared with the GGUF side deliberately: both cost
# models emit the same keys (body_bytes, head_bytes, n_head_kv, d_head_*,
# n_layer), so bytes-per-token is one formula, not two copies. The readers
# stay separate because the file formats have nothing in common.
from gguf_info import gen_bytes, kv_bytes  # noqa: F401 - re-exported

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "U32": 4, "U16": 2, "BOOL": 1,
}


def read_header(path: Path) -> dict:
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n).decode("utf-8"))


def _nbytes(info: dict) -> int:
    if "data_offsets" in info:
        lo, hi = info["data_offsets"]
        return hi - lo
    n = 1
    for d in info.get("shape", []):
        n *= d
    return n * DTYPE_BYTES.get(info.get("dtype", "F16"), 2)


def cost_model(model_dir: Path) -> dict:
    """Bytes read per generated token, split the same way as the GGUF model."""
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {model_dir}")

    config = json.loads((model_dir / "config.json").read_text())
    tied = bool(config.get("tie_word_embeddings", False))

    body_bytes = head_bytes = embd_bytes = 0
    tensors = 0
    for shard in shards:
        for name, info in read_header(shard).items():
            if name == "__metadata__":
                continue
            tensors += 1
            n = _nbytes(info)
            if "embed_tokens" in name:
                embd_bytes += n
            elif "lm_head" in name:
                head_bytes += n
            else:
                body_bytes += n

    if tied and head_bytes == 0:
        # The embedding table is the output head, so it is fully read whenever
        # logits are produced rather than indexed once.
        head_bytes, embd_bytes = embd_bytes, 0

    n_head = config.get("num_attention_heads", 0)
    n_head_kv = config.get("num_key_value_heads", n_head)
    hidden = config.get("hidden_size", 0)
    d_head = config.get("head_dim") or (hidden // n_head if n_head else 0)

    return {
        "arch": config.get("model_type", "?"),
        "tied_embeddings": tied,
        "n_layer": config.get("num_hidden_layers", 0),
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "d_head_k": d_head,
        "d_head_v": d_head,
        "body_bytes": body_bytes,
        "head_bytes": head_bytes,
        "embedding_bytes_resident": embd_bytes,
        "n_tensors": tensors,
        "quantization": config.get("quantization"),
    }


def resolve_hf_model(repo: str, cache: Path | None = None) -> Path:
    """Locate a cached HF snapshot directory without touching the network."""
    cache = cache or Path.home() / ".cache" / "huggingface" / "hub"
    root = cache / ("models--" + repo.replace("/", "--")) / "snapshots"
    snaps = sorted(root.glob("*")) if root.exists() else []
    if not snaps:
        raise FileNotFoundError(f"{repo} is not in the local cache at {root}")
    return snaps[-1]
