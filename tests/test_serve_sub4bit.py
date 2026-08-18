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
# The device check comes BEFORE the mlx.nn import, and that order is the whole
# point. importorskip cannot help here: mlx.nn evaluates on the device while it
# imports, so where there is no usable device the interpreter ABORTS rather than
# raising ImportError. A fatal abort during collection takes down the entire
# session, so this one line used to cost every other test file in the suite,
# not just this module's own.
#
# The probe is conftest's, and it has to be: `mx.metal.is_available()` answers
# whether the Metal framework loaded, NOT whether a device can be created, and
# it returns True inside a sandbox that then aborts on first use. METAL_DEVICE
# asks the only question that matters by trying it in a worker process, where
# an abort costs that process instead of this one.
from conftest import METAL_DEVICE, requires_metal  # noqa: E402

if METAL_DEVICE is None:
    pytest.skip("the interception tests build a real quantized model on Metal",
                allow_module_level=True)

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
@pytest.mark.gpu
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


@pytest.mark.gpu
def test_uninstall_restores_the_original_module():
    a = make_holder()
    original = a.proj
    patch = serve_sub4bit.install_patch(a)
    assert a.proj is not original
    patch.uninstall()
    assert a.proj is original


@pytest.mark.gpu
def test_install_counts_wrapped_leaves():
    a = make_holder()
    with patched(a) as patch:
        assert patch.n_wrapped == 1


# ---------------------------------------------------------------------------
# routing: delegated to should_dispatch, decode-scoped
# ---------------------------------------------------------------------------
@pytest.mark.gpu
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


@pytest.mark.gpu
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


@pytest.mark.gpu
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


@pytest.mark.gpu
def test_site_cell_reads_d_out_and_d_in_in_should_dispatch_order():
    """scales is (d_out, d_in // group_size); reading the pair the other way
    round routes nowhere and looks like a boundary rather than a bug."""
    a = make_holder()
    assert serve_sub4bit.site_cell(a.proj) == (BITS, D_OUT, D_IN)


# ---------------------------------------------------------------------------
# the registered zone: a claim about the pack, checked against the pack
# ---------------------------------------------------------------------------
def test_registered_zone_is_the_whole_window_the_recording_routes():
    """The registration is the pack's full routed window, not its overlap
    with the timing grid.

    It used to be `window & B_GRID`, which was blind at M = 7 and M = 9
    because B_GRID holds neither. That was harmless while the sequence check
    made those widths unreachable and stopped being harmless on 2026-08-18,
    when a verification step became exactly an M = 7 call. Registered and
    reasoned in the findings doc, section 4 amendment.
    """
    scan = frozenset(serve_sub4bit.ZONE_SCAN)
    for (d_out, d_in), pinned in serve_sub4bit.PINNED_ZONE.items():
        window = routed_windows.window_for(BITS, d_out, d_in)
        assert pinned == window, f"{d_out}x{d_in}"
        assert window <= scan, (
            f"{d_out}x{d_in}: the recording routes {sorted(window - scan)} "
            "outside ZONE_SCAN, so the guard would miss part of the window "
            "without ever saying so")
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
    # A typed refusal, not SystemExit(2): a moved boundary is permanent, so
    # the detached runner must give up rather than wait for a quiet machine,
    # and 2 belongs to the interpreter (memory_guard's exit table).
    with pytest.raises(serve_sub4bit.PreconditionFailed):
        serve_sub4bit.require_pinned_zone()
    out = capsys.readouterr().out
    assert "1 of 5" in out and "2560x4096" in out


@pytest.mark.gpu
def test_sequence_and_batch_spellings_are_bit_identical_when_routed():
    """One M = 7 certificate covers both rank spellings of the same rows.

    The routed-vs-routed comparison alone would be near vacuous: `_fused`
    flattens with `x.reshape(-1, d_in)`, so both spellings become the same
    kernel call on the same rows and comparing them tests determinism rather
    than routing. The load-bearing assertions are that BOTH spellings routed,
    and that each matches the STOCK output the unpatched module produces.
    """
    a = make_holder()
    rows = x_rows(7)
    sequence = rows.reshape(1, 7, D_IN)
    batch = rows.reshape(7, 1, D_IN)
    stock = a(sequence)                  # unpatched: the reference
    mx.eval(stock)
    with patched(a) as patch:
        sequence_out = a(sequence)
        batch_out = a(batch)
        mx.eval(sequence_out, batch_out)
        assert patch.calls == 2, patch.fallbacks
        assert mx.array_equal(
            sequence_out.reshape(7, D_OUT),
            batch_out.reshape(7, D_OUT),
        ).item()
    assert mx.allclose(sequence_out.reshape(7, D_OUT),
                       stock.reshape(7, D_OUT), atol=2e-2, rtol=2e-2).item()


@pytest.mark.gpu
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
        assert patch.fallbacks == {
            f"m-{2 * 7}-outside-dispatch-{D_OUT}x{D_IN}": 1,
        }
        assert mx.array_equal(out, stock).item()


@pytest.mark.gpu
def test_real_prefill_is_refused_by_flattened_width():
    a = make_holder()
    prefill = x_rows(64).reshape(1, 64, D_IN)
    with patched(a) as patch:
        mx.eval(a(prefill))
        assert patch.calls == 0
        assert patch.fallbacks == {
            f"m-64-outside-dispatch-{D_OUT}x{D_IN}": 1,
        }


@pytest.mark.gpu
def test_no_prefill_fallback_reason_remains():
    import inspect

    source = inspect.getsource(serve_sub4bit._RoutedLinear._ineligible)
    assert "prefill-" not in source
    assert serve_sub4bit._WHITELIST_PREFIXES == ("m-", "forced-stock")


@pytest.mark.gpu
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


@pytest.mark.gpu
def test_ineligible_group_size_falls_back_with_reason():
    a = make_holder(group_size=32)
    with patched(a) as patch:
        mx.eval(a(x_rows(dispatched_m())))
        assert patch.calls == 0
        assert any(k.startswith("quant-") for k in patch.fallbacks)


# ---------------------------------------------------------------------------
# correctness of the routed output
# ---------------------------------------------------------------------------
@pytest.mark.gpu
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
@pytest.mark.gpu
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


@pytest.mark.gpu
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


@requires_metal
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


# ---------------------------------------------------------------------------
# the refusal machine: one lock, one budget, one memory gate, typed exits
#
# Every test here drives main() rather than the mode functions, because the
# ORDER is the property: a gate that fires after the first model is already
# resident protects nothing. `_never_load` is the proof of order in each one.
# ---------------------------------------------------------------------------
TIMED = pytest.mark.parametrize("mode", ["--ab", "--mde"])


def _never_load(*a, **k):
    raise AssertionError("a model was loaded before the gate refused")


def _raise(exc):
    def _f(*a, **k):
        raise exc
    return _f


def _pins_ok(monkeypatch):
    monkeypatch.setattr(serve_sub4bit, "verify_pins",
                        lambda *a, **k: {"pins": "ok"})


def _lock_granted(monkeypatch):
    monkeypatch.setattr(serve_sub4bit.MeasurementLock, "acquire",
                        lambda self: (True, "acquired"))
    monkeypatch.setattr(serve_sub4bit.MeasurementLock, "release",
                        lambda self: None)


@TIMED
def test_timed_modes_refuse_when_the_machine_lock_is_held(monkeypatch, mode):
    _pins_ok(monkeypatch)
    monkeypatch.setattr(serve_sub4bit.MeasurementLock, "acquire",
                        lambda self: (False, "held by pid 1 (test)"))
    monkeypatch.setattr(serve_sub4bit, "load_model", _never_load)
    assert serve_sub4bit.main([mode]) == serve_sub4bit.EXIT_LOCK_HELD


def test_smoke_never_takes_the_lock(monkeypatch):
    """main() reaches smoke BEFORE it builds the lock - and only that.

    smoke() is replaced here because the real one loads two models and
    dispatches, so this test can say nothing about what smoke itself does: it
    would pass just as happily if smoke took the lock on its first line, or had
    been reduced to a stub. The property it does pin is main()'s ordering,
    which is where the lock decision actually lives.

    The other half - that nothing smoke calls can take the lock - is
    tests/test_serving_survival.py::test_nothing_reachable_from_smoke_can_take_the_machine_lock,
    which walks the call graph because a lock inside a helper is exactly what
    this test cannot see.
    """
    _pins_ok(monkeypatch)

    def _boom(self):
        raise AssertionError("smoke must not take the machine lock")

    monkeypatch.setattr(serve_sub4bit.MeasurementLock, "acquire", _boom)
    monkeypatch.setattr(serve_sub4bit, "smoke", lambda: 0)
    assert serve_sub4bit.main(["--smoke"]) == 0


@TIMED
def test_permanent_preconditions_exit_with_their_own_code(monkeypatch, mode):
    monkeypatch.setattr(serve_sub4bit, "verify_pins",
                        _raise(RuntimeError("hash mismatch")))
    assert serve_sub4bit.main([mode]) == serve_sub4bit.EXIT_PRECONDITION
    _pins_ok(monkeypatch)
    _lock_granted(monkeypatch)
    monkeypatch.setattr(serve_sub4bit, "require_idle",
                        lambda label: {"idle": True})
    monkeypatch.setattr(serve_sub4bit, "require_pinned_zone",
                        _raise(serve_sub4bit.PreconditionFailed(
                            "zone disagrees")))
    monkeypatch.setattr(serve_sub4bit, "load_model", _never_load)
    assert serve_sub4bit.main([mode]) == serve_sub4bit.EXIT_PRECONDITION


@TIMED
def test_a_busy_machine_is_transient(monkeypatch, mode):
    _pins_ok(monkeypatch)
    _lock_granted(monkeypatch)
    monkeypatch.setattr(serve_sub4bit, "require_idle",
                        _raise(serve_sub4bit.NotIdle("WindowServer at 30%")))
    monkeypatch.setattr(serve_sub4bit, "load_model", _never_load)
    assert serve_sub4bit.main([mode]) == serve_sub4bit.EXIT_NOT_IDLE


@TIMED
def test_timed_modes_refuse_when_the_budget_is_crossed(monkeypatch, mode):
    _pins_ok(monkeypatch)
    _lock_granted(monkeypatch)
    monkeypatch.setattr(serve_sub4bit, "require_idle",
                        lambda label: {"idle": True})
    monkeypatch.setattr(serve_sub4bit, "require_pinned_zone", lambda: None)
    monkeypatch.setattr(serve_sub4bit.BudgetGuard, "check",
                        lambda self, cell: (_ for _ in ()).throw(
                            serve_sub4bit.BudgetExceeded(cell, 30.0, 24.0)))
    monkeypatch.setattr(serve_sub4bit, "load_model", _never_load)
    assert serve_sub4bit.main([mode, "--budget-gb", "24"]) == \
        serve_sub4bit.EXIT_BUDGET_REFUSAL


@TIMED
def test_timed_modes_refuse_a_machine_with_no_room(monkeypatch, mode):
    """The budget bounds THIS process; the memory gate bounds the machine.
    Both must refuse before a model lands, and with different codes."""
    _pins_ok(monkeypatch)
    _lock_granted(monkeypatch)
    monkeypatch.setattr(serve_sub4bit, "require_idle",
                        lambda label: {"idle": True})
    monkeypatch.setattr(serve_sub4bit, "require_pinned_zone", lambda: None)
    monkeypatch.setattr(serve_sub4bit.BudgetGuard, "check",
                        lambda self, cell: 1.0)
    monkeypatch.setattr(serve_sub4bit, "require_available_memory",
                        _raise(serve_sub4bit.LowMemoryRefusal("cell", 1.0,
                                                              23.0)))
    monkeypatch.setattr(serve_sub4bit, "load_model", _never_load)
    assert serve_sub4bit.main([mode]) == serve_sub4bit.EXIT_LOW_MEMORY


def test_the_exit_vocabulary_is_the_shared_one(monkeypatch):
    """serve_sub4bit numbers nothing itself: every refusal code is
    memory_guard's, which is what stops two harnesses meaning different
    things by the same number (the 03:29 collapse's root)."""
    import memory_guard

    assert serve_sub4bit.EXIT_PRECONDITION is memory_guard.EXIT_PRECONDITION
    assert serve_sub4bit.EXIT_NOT_IDLE is memory_guard.EXIT_NOT_IDLE
    assert serve_sub4bit.EXIT_LOCK_HELD is memory_guard.EXIT_LOCK_HELD
    assert {memory_guard.EXIT_PRECONDITION,
            memory_guard.EXIT_NOT_IDLE} == {8, 9}
    assert 2 not in {memory_guard.EXIT_BUDGET_REFUSAL,
                     memory_guard.EXIT_LOCK_HELD,
                     memory_guard.EXIT_LOW_MEMORY,
                     memory_guard.EXIT_CHILD_DEATH,
                     memory_guard.EXIT_NO_DEVICE,
                     memory_guard.EXIT_PRECONDITION,
                     memory_guard.EXIT_NOT_IDLE}


def test_require_idle_raises_the_transient_refusal(monkeypatch, capsys):
    monkeypatch.setattr(serve_sub4bit.machine_state, "idle_check",
                        lambda *a, **k: {"idle": False,
                                         "blockers": ["WindowServer 30%"]})
    with pytest.raises(serve_sub4bit.NotIdle):
        serve_sub4bit.require_idle("before")
    assert "NOT QUIET" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the decision surface
#
# primary_verdict and composed_attribution ARE the experiment: every number
# the A/B prints is read through them, and they had no test at all. The losing
# branches come first on purpose - wide_qmv is a measured loss at batch 1-3
# (ADR 0015), so those are the branches M2 will actually take, and a decision
# function whose losing branches were never executed is not a decision
# function. This is the check this repo learned to write after G1.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prim, noise, expected", [
    (-3.0, 1.0, "regression"),   # the branch wide_qmv is expected to hit at B = 1..3
    (-0.5, 1.0, "null"),         # a loss inside the noise floor is NOT a regression
    (0.5, 1.0, "null"),          # nor is a gain inside it a win
    (3.0, 1.0, "win"),
    (1.0, 1.0, "null"),          # the boundary belongs to null: a gap the noise could produce proves nothing
    (-1.0, 1.0, "null"),
])
def test_primary_verdict_reads_section_6(prim, noise, expected):
    assert serve_sub4bit.primary_verdict(prim, noise) == expected


@pytest.mark.parametrize("comp, base, noise, expected", [
    (-3.0, 0.0, 1.0, "negative"),        # composed loses outright
    (-0.5, 0.0, 1.0, "inconclusive"),    # inside the floor
    (3.0, 2.5, 1.0, "artifact-alone"),   # arm 2 already beat arm 3 beyond noise
    (3.0, 0.5, 1.0, "joint"),            # arm 2 did not; the kernel unlocked it
    (0.0, 5.0, 1.0, "inconclusive"),     # comp inside the floor: base is never consulted
])
def test_composed_attribution_reads_section_6(comp, base, noise, expected):
    assert serve_sub4bit.composed_attribution(comp, base, noise) == expected


def test_the_verdict_can_say_loss():
    """A decision function that can only confirm is not a decision. Both
    losing outcomes must be reachable."""
    assert serve_sub4bit.primary_verdict(-10.0, 0.1) == "regression"
    assert serve_sub4bit.composed_attribution(-10.0, 0.0, 0.1) == "negative"


# ---------------------------------------------------------------------------
# the pre-registered rules the code derived and then dropped
# ---------------------------------------------------------------------------
def _mde_deciders(tmp_path, cells, manifest=None):
    path = tmp_path / "serve_mde.json"
    serve_sub4bit.write_mde(manifest or {"pins": "ok"}, cells, path)
    return path


def test_the_ab_refuses_when_no_mde_record_exists(tmp_path):
    """Section 5's non-decider rule decides which cells the A/B may be read
    as evidence for, so an A/B with no MDE behind it has no such reading.
    Defaulting every cell to "decider" is the failure this refuses."""
    with pytest.raises(serve_sub4bit.PreconditionFailed):
        serve_sub4bit.load_mde({"pins": "ok"}, tmp_path / "absent.json")


def test_the_ab_refuses_an_mde_record_from_different_artifacts(tmp_path):
    path = _mde_deciders(tmp_path, {6: True}, manifest={"pins": "old"})
    with pytest.raises(serve_sub4bit.PreconditionFailed):
        serve_sub4bit.load_mde({"pins": "new"}, path)


def test_the_mde_record_round_trips_its_cells(tmp_path):
    path = _mde_deciders(tmp_path, {5: True, 6: False, 8: True})
    assert serve_sub4bit.load_mde({"pins": "ok"}, path) == {5: True, 6: False,
                                                            8: True}


def _arms(one=100.0, two=100.0, three=100.0, four=100.0):
    return {"1": [one] * 5, "2": [two] * 5, "3": [three] * 5, "4": [four] * 5}


def test_a_non_decider_cell_is_labelled_in_its_ab_row(tmp_path):
    path = _mde_deciders(tmp_path, {5: True, 6: False})
    deciders = serve_sub4bit.load_mde({"pins": "ok"}, path)
    row, code = serve_sub4bit.ab_row(6, _arms(one=110.0), ["1024x2560"],
                                     10, 10, {}, deciders[6])
    assert row["decider"] is False
    assert code == 0                      # not an error: an unreadable win
    assert row["primary_verdict"] == "win"
    decided, _ = serve_sub4bit.ab_row(5, _arms(one=110.0), ["1024x2560"],
                                      10, 10, {}, deciders[5])
    assert decided["decider"] is True


def test_every_ab_row_carries_the_decider_label_even_when_invalid():
    """A cell that never produced a verdict still says whether it could
    have: a reader must not have to infer it from the absence of one."""
    invalid, code = serve_sub4bit.ab_row(6, _arms(), ["1024x2560"], 9, 10,
                                         {"dtype-x": 3}, False)
    assert invalid["decider"] is False and code == 1
    assert invalid["verdict"].startswith("INVALID")
    withheld, code = serve_sub4bit.ab_row(
        6, {"1": [100.0] * 5, "2": [100.0, 200.0, 100.0, 100.0, 100.0],
            "3": [100.0] * 5, "4": [100.0] * 5}, ["1024x2560"], 10, 10, {},
        True)
    assert code == 1 and withheld["verdict"].startswith("WITHHELD")
    assert withheld["decider"] is True


def test_the_summary_line_names_the_non_decider_cells():
    line = serve_sub4bit.non_decider_line({5: True, 6: False, 8: False})
    assert "B=6" in line and "B=8" in line and "B=5" not in line
    assert "NON-DECIDER" in line
    assert "every" in serve_sub4bit.non_decider_line({5: True})


# --- the MDE's own row: the comparison it printed and never made -----------
def _probe_row(shape="2560x4096", routed=True, spread=1.0):
    return {"shape": shape, "count": 1, "routed": routed,
            "stock_us": 100.0, "fused_us": 90.0, "stock_spread_pct": spread}


def test_a_cell_whose_gain_is_under_its_noise_floor_is_a_non_decider():
    """Section 5, verbatim: "a cell whose expected gain is below its noise
    floor is a pre-declared non-decider". The two numbers were printed
    adjacently and never compared."""
    under = serve_sub4bit.mde_row(6, [_probe_row()], i_ms=0.0, saving_ms=0.1,
                                  t_step_ms=10.0, noise_floor_pct=2.0)
    assert under["expected_gain_pct"] == 1.0 and under["decider"] is False
    over = serve_sub4bit.mde_row(6, [_probe_row()], i_ms=0.0, saving_ms=0.5,
                                 t_step_ms=10.0, noise_floor_pct=2.0)
    assert over["expected_gain_pct"] == 5.0 and over["decider"] is True


def test_an_expected_loss_is_a_non_decider_not_a_verdict():
    row = serve_sub4bit.mde_row(1, [_probe_row()], i_ms=1.0, saving_ms=0.0,
                                t_step_ms=10.0, noise_floor_pct=2.0)
    assert row["expected_gain_pct"] == -10.0 and row["decider"] is False


def test_the_mde_withholds_a_cell_whose_reference_arm_disagreed():
    """AGENTS.md: reject any round whose reference samples exceed the class
    spread limit. _op_probe measured that spread, reported it, and nothing
    read it - so a saving built on an unstable stock arm fed the gain."""
    row = serve_sub4bit.mde_row(
        6, [_probe_row(spread=serve_sub4bit.MAX_SPREAD_PCT + 0.1)],
        i_ms=0.0, saving_ms=5.0, t_step_ms=10.0, noise_floor_pct=2.0)
    assert row["verdict"].startswith("WITHHELD")
    assert "2560x4096" in row["verdict"]
    assert row["decider"] is False
    assert "expected_gain_pct" not in row, \
        "a gain derived from a rejected round is not a number"


def test_an_unrouted_shapes_spread_cannot_withhold_a_cell():
    """A shape the table does not route at this B contributes no saving, so
    its stock arm's spread says nothing about this cell."""
    row = serve_sub4bit.mde_row(
        6, [{"shape": "2560x4096", "count": 1, "routed": False}],
        i_ms=0.0, saving_ms=5.0, t_step_ms=10.0, noise_floor_pct=2.0)
    assert "verdict" not in row and row["decider"] is True


# --- the corpus pin --------------------------------------------------------
def test_the_ppl_mode_refuses_a_corpus_that_is_not_the_registered_one(
        tmp_path, monkeypatch):
    """Section 7 registers the corpus by sha256. ppl() computed the digest,
    printed it and never compared it, so any text at that path would have
    been scored and recorded under the registered hash's authority."""
    _pins_ok(monkeypatch)
    _lock_granted(monkeypatch)
    monkeypatch.setattr(serve_sub4bit, "load_model", _never_load)
    wrong = tmp_path / "ppl.txt"
    wrong.write_text("not the wikitext-2 test split")
    assert serve_sub4bit.main(["--ppl", "--corpus", str(wrong)]) == \
        serve_sub4bit.EXIT_PRECONDITION


def test_the_registered_corpus_digest_is_the_one_in_the_findings_doc():
    doc = (Path(serve_sub4bit.__file__).resolve().parents[1] / "docs"
           / "research" / "2026-08-15-sub4bit-serve-findings.md").read_text()
    assert serve_sub4bit.PPL_CORPUS_SHA256 in doc


def test_the_round_count_is_the_registered_one():
    """Section 3 registers 5 rounds per cell; _op_probe took 7."""
    doc = (Path(serve_sub4bit.__file__).resolve().parents[1] / "docs"
           / "research" / "2026-08-15-sub4bit-serve-findings.md").read_text()
    assert f"Rounds: {serve_sub4bit.ROUNDS} per B cell" in doc
    import inspect
    src = inspect.getsource(serve_sub4bit._op_probe)
    assert "range(ROUNDS)" in src and "range(7)" not in src


# ---------------------------------------------------------------------------
# the registration and the code say the same thing
#
# Each of these is a degree of freedom the harness always had and section 4
# never named. A measurement harness's freedoms are registered in writing or
# removed from the code; an unregistered one is a knob nobody agreed to.
# ---------------------------------------------------------------------------
def _findings_doc() -> str:
    return (Path(serve_sub4bit.__file__).resolve().parents[1] / "docs"
            / "research" / "2026-08-15-sub4bit-serve-findings.md").read_text()


def test_the_whitelist_matches_the_registration():
    """One spelling of the rule, in the doc, checked against the code. Two
    spellings is how the doc came to say `prefill-*` and `m-*-outside-
    dispatch` while the code whitelisted any `m-` reason plus forced-stock."""
    import ast
    import re

    found = re.search(r"WHITELIST_PREFIXES = (\([^)]*\))", _findings_doc())
    assert found, "the findings doc must register the whitelist verbatim"
    assert ast.literal_eval(found.group(1)) == \
        serve_sub4bit._WHITELIST_PREFIXES


@pytest.mark.gpu
def test_the_only_m_reason_is_the_routing_table_declining_the_cell():
    """The registration whitelists the `m-` prefix, which is wider than the
    one reason that exists. This is what keeps the widening honest: no other
    `m-` reason may appear and be absorbed without anyone deciding it."""
    import re

    a = make_holder()
    with patched(a) as patch:
        for m in B_GRID:
            mx.eval(a(x_rows(m)))
        m_reasons = [r for r in patch.fallbacks if r.startswith("m-")]
    assert m_reasons, "the grid must exercise the outside-dispatch branch"
    for reason in m_reasons:
        assert re.fullmatch(r"m-\d+-outside-dispatch-\d+x\d+", reason), reason


@pytest.mark.gpu
def test_a_bias_term_falls_back_and_invalidates_the_round():
    """The kernel has no bias path, so a biased layer must run stock - and
    because `bias-term` is deliberately NOT whitelisted, a round in which it
    happened is invalid rather than quietly part-stock."""
    holder = Holder()
    holder.proj = nn.QuantizedLinear(D_IN, D_OUT, bits=BITS, group_size=64,
                                     bias=True)
    holder.set_dtype(mx.float16)
    mx.eval(holder.parameters())
    with patched(holder) as patch:
        mx.eval(holder(x_rows(dispatched_m())))
        assert patch.calls == 0
        assert patch.fallbacks.get("bias-term") == 1
        assert patch.hard_fallbacks() == {"bias-term": 1}
    assert "bias-term" in _findings_doc()


def test_the_fp16_cast_of_both_checkpoints_is_registered():
    """load_model casts every parameter of both artifacts to fp16, which is
    what makes arm 1 eligible at all; it is applied to every arm equally and
    the doc has to say the numbers describe the cast checkpoints."""
    import inspect

    assert "set_dtype(mx.float16)" in inspect.getsource(
        serve_sub4bit.load_model)
    assert "set_dtype" in _findings_doc()


def test_the_corpus_location_is_recorded():
    """bench/.corpus is gitignored, so the file exists per worktree and the
    doc is the only place that can say where it comes from."""
    assert "`bench/.corpus/` is gitignored" in _findings_doc()


# ---------------------------------------------------------------------------
# one copy of the timing discipline
#
# AGENTS.md: "One copy of the discipline, so a timing rule amended in one gate
# cannot silently stay old in another." _op_probe had its own MIN_SAMPLE_MS,
# its own spread_pct and its own calibrate-and-batch loop, all of which
# bench/interleave.py and bench/machine_state.py already own.
# ---------------------------------------------------------------------------
def test_the_harness_owns_no_second_copy_of_the_timing_rules():
    import inspect

    src = inspect.getsource(serve_sub4bit)
    assert "def spread_pct" not in src
    assert "MIN_SAMPLE_MS =" not in src
    assert serve_sub4bit.spread_pct is serve_sub4bit.machine_state.spread_pct


@pytest.mark.gpu
def test_the_op_probe_times_through_the_shared_engine(monkeypatch):
    """Counted, not read off the source: the probe must calibrate once and
    then sample both arms ROUNDS times through interleave.dispatch."""
    calls = []

    def _fake_dispatch(build_one, copies):
        calls.append(copies)
        return 0.010                      # already past MIN_SAMPLE_MS

    monkeypatch.setattr(serve_sub4bit.interleave, "dispatch", _fake_dispatch)
    probe = serve_sub4bit._op_probe(BITS, d_in=128, d_out=64, m=1)
    assert len(calls) == 1 + 2 * serve_sub4bit.ROUNDS
    assert set(calls) == {8}, "one calibration, then every sample at that size"
    assert probe["stock_spread_pct"] == 0.0


def test_the_expected_gain_is_labelled_as_the_cross_pass_composition():
    """_op_probe's saving and arm 2's per-step time come from separate
    passes minutes apart, which is the composition interleaving exists to
    forbid; the row says so rather than reading as a single measurement."""
    row = serve_sub4bit.mde_row(6, [_probe_row()], i_ms=0.0, saving_ms=0.5,
                                t_step_ms=10.0, noise_floor_pct=2.0)
    assert row["composition"] == "cross-pass"
    assert "cross-pass" in _findings_doc()
