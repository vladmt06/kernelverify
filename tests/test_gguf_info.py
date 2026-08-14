"""The GGUF cost model behind the roofline placement (ADR 0005).

The baseline's first flop model charged `2 * model_n_params` per token and
scored prompt processing at 110% of the machine's measured flop ceiling, which
is impossible. These tests pin the three corrections that fixed it: the
embedding table is a gather and not a matmul, the output head runs once per
decode call rather than once per prompt token, and the attention score matmuls
grow with context instead of with parameter count.
"""

import struct

import pytest

import gguf_info
from gguf_info import cost_model, gen_bytes, gen_flops, kv_bytes, prompt_flops

U32, U64, STRING = 4, 10, 8


def _string(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, tag: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", tag) + payload


def build_gguf(tensors, tied: bool, extra_kv: bytes = b"") -> bytes:
    """A GGUF header with no tensor data behind it; only the table is parsed."""
    kv = [
        _kv("general.architecture", STRING, _string("testarch")),
        _kv("testarch.block_count", U32, struct.pack("<I", 2)),
        _kv("testarch.attention.head_count", U32, struct.pack("<I", 4)),
        _kv("testarch.attention.head_count_kv", U32, struct.pack("<I", 2)),
        _kv("testarch.embedding_length", U32, struct.pack("<I", 64)),
    ]
    n_kv = len(kv) + (2 if extra_kv else 0)
    body = b"".join(kv) + extra_kv
    for name, shape, ggml_type in tensors:
        body += _string(name) + struct.pack("<I", len(shape))
        body += b"".join(struct.pack("<Q", d) for d in shape)
        body += struct.pack("<I", ggml_type) + struct.pack("<Q", 0)
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), n_kv)
    assert tied == all(not n.startswith("output.") for n, _, _ in tensors)
    return head + body


F32, F16 = 0, 1

UNTIED = [
    ("token_embd.weight", (64, 100), F16),
    ("blk.0.attn_q.weight", (64, 64), F16),
    ("blk.0.ffn_up.weight", (64, 128), F16),
    ("blk.0.attn_norm.weight", (64,), F32),
    ("output.weight", (64, 100), F16),
]
TIED = [t for t in UNTIED if not t[0].startswith("output.")]


def cost_of(tensors, tied, tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(build_gguf(tensors, tied))
    return cost_model(gguf_info.read(path))


def test_embedding_is_not_charged_as_a_matmul(tmp_path):
    c = cost_of(UNTIED, tied=False, tmp_path=tmp_path)
    # body is attn_q + ffn_up only: the embedding table and the head are not in it
    assert c["body_params"] == 64 * 64 + 64 * 128
    assert c["head_params"] == 64 * 100
    assert c["body_flops_per_token"] == 2 * c["body_params"]


def test_tied_embeddings_become_the_head(tmp_path):
    c = cost_of(TIED, tied=True, tmp_path=tmp_path)
    assert c["tied_embeddings"] is True
    assert c["head_params"] == 64 * 100
    # a tied table is fully read whenever logits are produced, so it is head
    # bytes rather than resident-but-untouched bytes
    assert c["embedding_bytes_resident"] == 0
    assert c["head_bytes"] == 64 * 100 * 2


def test_untied_embedding_bytes_are_resident_not_streamed(tmp_path):
    c = cost_of(UNTIED, tied=False, tmp_path=tmp_path)
    assert c["embedding_bytes_resident"] == 64 * 100 * 2
    assert gen_bytes(c) == c["body_bytes"] + c["head_bytes"]


def test_head_is_charged_once_per_decode_call_not_per_prompt_token(tmp_path):
    c = cost_of(UNTIED, tied=False, tmp_path=tmp_path)
    only_head = prompt_flops(c, 1) - prompt_flops_body_and_attn(c, 1)
    assert only_head == pytest.approx(c["head_flops_per_call"])
    # doubling the prompt doubles the body term but not the head term
    grew_by = prompt_flops(c, 2) - prompt_flops(c, 1)
    assert grew_by < 2 * c["body_flops_per_token"] + c["head_flops_per_call"]


def prompt_flops_body_and_attn(c, n):
    per_token = 2.0 * c["n_layer"] * c["n_head"] * (c["d_head_k"] + c["d_head_v"])
    return c["body_flops_per_token"] * n + per_token * n * (n + 1) / 2.0


def test_attention_flops_grow_quadratically_with_prompt(tmp_path):
    c = cost_of(UNTIED, tied=False, tmp_path=tmp_path)
    body = c["body_flops_per_token"]
    head = c["head_flops_per_call"]
    attn = lambda n: prompt_flops(c, n) - body * n - head  # noqa: E731
    assert attn(200) / attn(100) == pytest.approx(200 * 201 / (100 * 101), rel=1e-9)


def test_generation_charges_one_token_of_body_plus_the_cache(tmp_path):
    c = cost_of(UNTIED, tied=False, tmp_path=tmp_path)
    at_zero = gen_flops(c, 0)
    assert at_zero == c["body_flops_per_token"] + c["head_flops_per_call"]
    # two layers, two kv heads, 16 for K plus 16 for V, ten positions, fp16
    assert kv_bytes(c, 10) == 2 * 2 * (16 + 16) * 10 * 2


def test_declared_head_dim_beats_the_implied_one(tmp_path):
    """Qwen3-4B declares key_length 128 against an implied 80.

    Deriving the head dimension from embedding_length / head_count understates
    both attention flops and cache bytes by the ratio between them, on exactly
    the model class this baseline reports.
    """
    declared = _kv("testarch.attention.key_length", U32, struct.pack("<I", 48))
    declared += _kv("testarch.attention.value_length", U32, struct.pack("<I", 48))
    path = tmp_path / "declared.gguf"
    path.write_bytes(build_gguf(UNTIED, tied=False, extra_kv=declared))
    c = cost_model(gguf_info.read(path))

    assert c["d_head_implied"] == 16
    assert c["d_head_k"] == 48 and c["d_head_declared"] is True
    # three times the implied head dim means three times the cache traffic
    assert kv_bytes(c, 10) == 2 * 2 * (48 + 48) * 10 * 2

    plain = cost_of(UNTIED, tied=False, tmp_path=tmp_path)
    attn = lambda cost, n: (  # noqa: E731
        prompt_flops(cost, n) - cost["body_flops_per_token"] * n - cost["head_flops_per_call"]
    )
    assert attn(c, 100) == pytest.approx(3 * attn(plain, 100))


def test_rejects_a_file_that_is_not_gguf(tmp_path):
    path = tmp_path / "not.gguf"
    path.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ValueError, match="not a GGUF file"):
        gguf_info.read(path)
