"""The serving harness's interception layer, judged before any measurement.

The properties under test are the pre-registered ones from
docs/research/2026-08-15-sub4bit-serve-findings.md section 4:

- interception is scoped to the installed model tree only, never global:
  oracle calls (direct mx.quantized_matmul) and a second model instance must
  never route through the kernel under test;
- routing is delegated entirely to kernelverify.pack.wide_qmv.should_dispatch,
  so the harness inherits the pack's boundary rather than encoding one, and
  since ADR 0015 that boundary is per shape, so the harness has to ask with
  the whole cell and count dispatches per site rather than per model;
- the registered win zone is a claim ABOUT the pack, so it is checked against
  the pack's own table here rather than derived from it;
- the forced-stock mode (arm 4) evaluates eligibility identically and then
  takes the stock path, bit-identically to an unpatched model;
- pin verification refuses a corrupted artifact;
- the batched decode driver's step accounting and the perplexity proxy's
  windowing are exact.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from kernelverify.pack import routed_windows  # noqa: E402
from kernelverify.pack.wide_qmv import should_dispatch  # noqa: E402

import serve_sub4bit  # noqa: E402

# k_proj/v_proj's real decode shape, the smallest of the five the harness
# intercepts. It has to be a real one: routing is keyed by (bits, d_out,
# d_in) since ADR 0015, so a made-up shape routes nowhere and every routing
# assertion below would pass vacuously.
D_IN, D_OUT = 2560, 1024
# A shape the pricing run never priced, which is therefore routed nowhere at
# any tile width; the per-site accounting is what it exists to expose.
UNPRICED_D_IN, UNPRICED_D_OUT = 512, 128
BITS = 3
B_GRID = [1, 4, 5, 6, 8, 11, 12, 16]
PINNED_MODEL = "qwen3-4b-3bit-g64"


class Holder(nn.Module):
    """The smallest tree with a QuantizedLinear leaf one level down."""

    def __init__(self, bits: int = BITS, group_size: int = 64,
                 d_in: int = D_IN, d_out: int = D_OUT):
        super().__init__()
        self.proj = nn.QuantizedLinear(d_in, d_out, bits=bits,
                                       group_size=group_size, bias=False)

    def __call__(self, x):
        return self.proj(x)


def make_holder(bits: int = BITS, group_size: int = 64,
                d_in: int = D_IN, d_out: int = D_OUT) -> Holder:
    h = Holder(bits=bits, group_size=group_size, d_in=d_in, d_out=d_out)
    h.set_dtype(mx.float16)
    mx.eval(h.parameters())
    return h


def x_rows(m: int, seed: int = 3, d_in: int = D_IN) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((m, d_in)).astype(np.float16))


def fp64_reference(holder: Holder, x: mx.array) -> np.ndarray:
    p = holder.proj
    w = mx.dequantize(p.weight, p.scales, p.biases,
                      group_size=p.group_size, bits=p.bits)
    w64 = np.array(w.astype(mx.float32), dtype=np.float64)
    x64 = np.array(x.astype(mx.float32), dtype=np.float64)
    return x64.reshape(-1, D_IN) @ w64.T


def dispatched_m() -> int:
    """An M the pack currently routes at the holder's shape; the tests
    derive it from should_dispatch so they keep passing when the boundary
    moves."""
    return next(m for m in B_GRID if should_dispatch(m, BITS, D_OUT, D_IN))


@contextmanager
def patched(model):
    """Install for the body, uninstall however the body ends: a failing
    assertion must not leave a wrapped tree behind for the next test."""
    patch = serve_sub4bit.install_patch(model)
    try:
        yield patch
    finally:
        patch.uninstall()


# ---------------------------------------------------------------------------
# scoping: the brief's non-negotiable
# ---------------------------------------------------------------------------
def test_oracle_and_second_model_never_routed():
    a, b = make_holder(), make_holder()
    with patched(a) as patch:
        x = x_rows(dispatched_m())

        mx.eval(a(x))
        assert patch.calls == 1

        # A second, unpatched model must not route.
        mx.eval(b(x))
        assert patch.calls == 1

        # The oracle path: a direct mx.quantized_matmul over the same
        # arrays must not route either.
        p = a.proj.inner
        oracle = mx.quantized_matmul(x, p.weight, p.scales, p.biases,
                                     transpose=True, group_size=p.group_size,
                                     bits=p.bits)
        mx.eval(oracle)
        assert patch.calls == 1


def test_uninstall_restores_the_original_module():
    a = make_holder()
    original = a.proj
    patch = serve_sub4bit.install_patch(a)
    assert a.proj is not original
    patch.uninstall()
    assert a.proj is original


def test_install_counts_wrapped_leaves():
    a = make_holder()
    with patched(a) as patch:
        assert patch.n_wrapped == 1


# ---------------------------------------------------------------------------
# routing: delegated to should_dispatch, decode-scoped
# ---------------------------------------------------------------------------
def test_routing_follows_should_dispatch_across_the_grid():
    a = make_holder()
    with patched(a) as patch:
        for m in B_GRID:
            patch.reset()
            mx.eval(a(x_rows(m)))
            if should_dispatch(m, BITS, D_OUT, D_IN):
                assert patch.calls == 1, f"M={m} should have dispatched"
            else:
                assert patch.calls == 0, f"M={m} should have fallen back"
                assert any(k.startswith("m-") for k in patch.fallbacks)


def test_an_unpriced_shape_routes_nowhere_on_the_whole_grid():
    """Routing is keyed by shape, so a shape the pricing run never priced
    falls back at every tile width, including the widths that win at the
    real projections."""
    a = make_holder(d_in=UNPRICED_D_IN, d_out=UNPRICED_D_OUT)
    with patched(a) as patch:
        for m in B_GRID:
            patch.reset()
            mx.eval(a(x_rows(m, d_in=UNPRICED_D_IN)))
            assert patch.calls == 0, f"M={m} routed an unpriced shape"
            assert any(k.startswith("m-") for k in patch.fallbacks)


class TwoShapes(nn.Module):
    """One routed projection and one unpriced one in the same tree, which
    is the case a whole-model dispatch count cannot describe."""

    def __init__(self):
        super().__init__()
        self.routed = nn.QuantizedLinear(D_IN, D_OUT, bits=BITS,
                                         group_size=64, bias=False)
        self.unpriced = nn.QuantizedLinear(UNPRICED_D_IN, UNPRICED_D_OUT,
                                           bits=BITS, group_size=64,
                                           bias=False)

    def __call__(self, x, y):
        return self.routed(x), self.unpriced(y)


def test_expected_calls_counts_routed_sites_not_wrapped_sites():
    model = TwoShapes()
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    m, steps = dispatched_m(), 7
    with patched(model) as patch:
        assert patch.n_wrapped == 2
        # One of the two sites routes at this width, so the expectation is
        # steps, not 2 x steps; n_wrapped x steps would be the old answer.
        assert serve_sub4bit.expected_calls(patch, m, steps) == steps
        assert serve_sub4bit.routed_shapes(patch, m) == [f"{D_OUT}x{D_IN}"]

        # And the count the harness asserts is the count the wrapper makes.
        patch.reset()
        out = model(x_rows(m), x_rows(m, d_in=UNPRICED_D_IN))
        mx.eval(out)
        assert patch.calls == serve_sub4bit.expected_calls(patch, m, 1)


def test_site_cell_reads_d_out_and_d_in_in_should_dispatch_order():
    """scales is (d_out, d_in // group_size); reading the pair the other way
    round routes nowhere and looks like a boundary rather than a bug."""
    a = make_holder()
    assert serve_sub4bit.site_cell(a.proj) == (BITS, D_OUT, D_IN)


# ---------------------------------------------------------------------------
# the registered zone: a claim about the pack, checked against the pack
# ---------------------------------------------------------------------------
def test_registered_zone_is_what_the_recording_routes_on_the_grid():
    for (d_out, d_in), pinned in serve_sub4bit.PINNED_ZONE.items():
        window = routed_windows.window_for(BITS, d_out, d_in)
        assert pinned == window & frozenset(B_GRID), f"{d_out}x{d_in}"
    serve_sub4bit.require_pinned_zone()          # must not raise today


def test_registered_zone_excludes_the_never_intercepted_lm_head():
    """lm_head is priced and wins at M = 5..10, but it is tied embeddings
    and not an nn.QuantizedLinear leaf, so the patch never wraps it and a
    zone entry for it would register a claim nothing here enforces."""
    lm_head = (151936, 2560)
    assert routed_windows.window_for(BITS, *lm_head)
    assert lm_head not in serve_sub4bit.PINNED_ZONE
    assert len(serve_sub4bit.PINNED_ZONE) == 5


def test_registered_zone_refuses_a_boundary_that_moved_at_one_shape(
        monkeypatch, capsys):
    moved = dict(serve_sub4bit.PINNED_ZONE)
    shape = (2560, 4096)
    moved[shape] = moved[shape] | frozenset({11})
    monkeypatch.setattr(serve_sub4bit, "PINNED_ZONE", moved)
    with pytest.raises(SystemExit) as exc:
        serve_sub4bit.require_pinned_zone()
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "1 of 5" in out and "2560x4096" in out


def test_prefill_shaped_input_falls_back_and_matches_stock():
    a = make_holder()
    x3 = mx.array(np.random.default_rng(5)
                  .standard_normal((2, 7, D_IN)).astype(np.float16))
    stock = a(x3)
    mx.eval(stock)
    with patched(a) as patch:
        out = a(x3)
        mx.eval(out)
        assert patch.calls == 0
        assert any(k.startswith("prefill-") for k in patch.fallbacks)
        assert mx.array_equal(out, stock).item()


def test_decode_shaped_3d_input_dispatches():
    a = make_holder()
    with patched(a) as patch:
        m = dispatched_m()
        x3 = mx.array(np.random.default_rng(7)
                      .standard_normal((m, 1, D_IN)).astype(np.float16))
        out = a(x3)
        mx.eval(out)
        assert patch.calls == 1
        assert out.shape == (m, 1, D_OUT)


def test_ineligible_group_size_falls_back_with_reason():
    a = make_holder(group_size=32)
    with patched(a) as patch:
        mx.eval(a(x_rows(dispatched_m())))
        assert patch.calls == 0
        assert any(k.startswith("quant-") for k in patch.fallbacks)


# ---------------------------------------------------------------------------
# correctness of the routed output
# ---------------------------------------------------------------------------
def test_routed_output_sits_at_the_fp64_reference():
    a = make_holder()
    m = dispatched_m()
    x = x_rows(m)
    ref = fp64_reference(a, x)          # taken before the patch exists
    with patched(a) as patch:
        out = a(x)
        mx.eval(out)
        assert patch.calls == 1
        gap = float(np.max(np.abs(
            np.array(out.astype(mx.float32), dtype=np.float64) - ref)))
        # 4 fp16 ulps of the output magnitude, the spike's wiring gate.
        bound = 4e-3 * max(1.0, float(np.max(np.abs(ref))))
        assert gap < bound, f"gap {gap} exceeds {bound}"


# ---------------------------------------------------------------------------
# arm 4: forced stock
# ---------------------------------------------------------------------------
def test_forced_stock_never_dispatches_and_is_bit_identical():
    a = make_holder()
    m = dispatched_m()
    x = x_rows(m)
    stock = a(x)
    mx.eval(stock)
    with patched(a) as patch:
        patch.mode = "stock"
        out = a(x)
        mx.eval(out)
        assert patch.calls == 0
        # Eligibility was still evaluated: the forced call is counted.
        assert patch.fallbacks.get("forced-stock") == 1
        assert mx.array_equal(out, stock).item()


# ---------------------------------------------------------------------------
# pin verification
# ---------------------------------------------------------------------------
def write_pin_manifest(root: Path, spelled_as: str, path: Path) -> None:
    """A one-entry PINNED-HASHES.txt over `path`, listing it under the
    spelling given; the two spellings the coordinator uses are what these
    tests are about."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "PINNED-HASHES.txt").write_text(f"{digest}  {spelled_as}\n")


def test_pin_verification_accepts_then_refuses_corruption(tmp_path):
    weights = tmp_path / PINNED_MODEL / "weights.bin"
    weights.parent.mkdir()
    weights.write_bytes(b"\x01\x02\x03")
    write_pin_manifest(tmp_path, f"{PINNED_MODEL}/weights.bin", weights)
    serve_sub4bit.verify_pins(tmp_path, [PINNED_MODEL])

    weights.write_bytes(b"\x01\x02\x04")
    with pytest.raises(RuntimeError):
        serve_sub4bit.verify_pins(tmp_path, [PINNED_MODEL])


def test_pin_verification_accepts_repo_root_relative_paths(tmp_path):
    """The coordinator's manifest spells paths from the repo root
    (bench/.models/<model>/<file>); both spellings must verify."""
    artifact = tmp_path / PINNED_MODEL / "model.safetensors"
    artifact.parent.mkdir()
    artifact.write_bytes(b"\x07\x08")
    write_pin_manifest(
        tmp_path, f"bench/.models/{PINNED_MODEL}/model.safetensors", artifact)
    serve_sub4bit.verify_pins(tmp_path, [PINNED_MODEL])


def test_pin_verification_refuses_an_unpinned_extra_file(tmp_path):
    """Every non-hidden file in the model directory must be pinned: an
    unlisted tokenizer or config would load unverified while the recorded
    numbers claim to bind to the manifest."""
    artifact = tmp_path / PINNED_MODEL / "model.safetensors"
    artifact.parent.mkdir()
    artifact.write_bytes(b"\x01\x02\x03")
    (artifact.parent / "tokenizer.json").write_bytes(b"{}")
    write_pin_manifest(tmp_path, f"{PINNED_MODEL}/model.safetensors", artifact)
    with pytest.raises(RuntimeError, match="tokenizer.json"):
        serve_sub4bit.verify_pins(tmp_path, [PINNED_MODEL])


def test_pin_verification_ignores_hidden_files(tmp_path):
    artifact = tmp_path / PINNED_MODEL / "model.safetensors"
    artifact.parent.mkdir()
    artifact.write_bytes(b"\x01\x02\x03")
    (artifact.parent / ".DS_Store").write_bytes(b"noise")
    write_pin_manifest(tmp_path, f"{PINNED_MODEL}/model.safetensors", artifact)
    serve_sub4bit.verify_pins(tmp_path, [PINNED_MODEL])


def test_pin_verification_refuses_unlisted_model(tmp_path):
    (tmp_path / "PINNED-HASHES.txt").write_text("")
    with pytest.raises(RuntimeError):
        serve_sub4bit.verify_pins(tmp_path, [PINNED_MODEL])


# ---------------------------------------------------------------------------
# the batched decode driver's accounting
# ---------------------------------------------------------------------------
class ChainModel(nn.Module):
    """Deterministic fake: argmax of the logits at the last position is
    always (last input token + 1) mod VOCAB, so the driver's feedback loop
    is checkable token by token."""

    VOCAB = 32

    def __init__(self):
        super().__init__()
        self.layers = []      # make_prompt_cache walks these; none are needed
        self.seen_shapes = []
        self.seen_last = []

    def __call__(self, x, cache=None):
        self.seen_shapes.append(tuple(x.shape))
        self.seen_last.append(np.array(x[:, -1]).tolist())
        nxt = (x[:, -1] + 1) % self.VOCAB
        zeros = mx.zeros((x.shape[0], x.shape[1], self.VOCAB),
                         dtype=mx.float16)
        idx = nxt[:, None].astype(mx.int32)
        hit = mx.arange(self.VOCAB)[None, None, :] == idx[:, :, None]
        return mx.where(hit, mx.array(10.0, dtype=mx.float16), zeros)


def test_decode_window_step_accounting_and_feedback():
    model = ChainModel()
    b, t, g = 3, 5, 6
    prompts = mx.tile(mx.arange(t, dtype=mx.int32)[None, :], (b, 1))
    dt, steps = serve_sub4bit.decode_window(model, prompts, gen_tokens=g)
    assert steps == g - 1
    assert dt > 0
    # One prefill call of (B, T), then G-1 decode calls of (B, 1).
    assert model.seen_shapes[0] == (b, t)
    assert model.seen_shapes[1:] == [(b, 1)] * (g - 1)
    # The feedback loop itself: after the prompt's last token t-1, each
    # decode call must receive the previous call's argmax, so every stream
    # sees t, t+1, ... in order; a driver feeding a stale token would
    # still pass the shape checks above.
    assert model.seen_last[1:] == [[t - 1 + k] * b for k in range(1, g)]


# ---------------------------------------------------------------------------
# the perplexity proxy's windowing
# ---------------------------------------------------------------------------
class UniformModel(nn.Module):
    """Zero logits everywhere: every token costs exactly log(VOCAB), so the
    perplexity of any stream is VOCAB and only the windowing can break it."""

    VOCAB = 24

    def __init__(self):
        super().__init__()
        self.seen_shapes = []

    def __call__(self, x, cache=None):
        self.seen_shapes.append(tuple(x.shape))
        return mx.zeros((x.shape[0], x.shape[1], self.VOCAB),
                        dtype=mx.float16)


def test_perplexity_windowing_is_exact_on_the_uniform_model():
    model = UniformModel()
    window, n_windows = 8, 3
    ids = list(range(window * n_windows + 5))     # tail must be dropped
    ppl = serve_sub4bit.perplexity(model, ids, window=window,
                                   n_windows=n_windows)
    assert ppl == pytest.approx(float(model.VOCAB), rel=1e-4)
    assert model.seen_shapes == [(1, window)] * n_windows


def test_perplexity_refuses_a_short_stream():
    model = UniformModel()
    with pytest.raises(RuntimeError):
        serve_sub4bit.perplexity(model, list(range(10)), window=8,
                                 n_windows=3)
