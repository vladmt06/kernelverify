"""Host-side contracts for the speculative-decode harness."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from conftest import (  # noqa: E402
    METAL_DEVICE,
    _lock_granted,
    _never_load,
    _pins_ok,
    _raise,
)

if METAL_DEVICE is None:
    pytest.skip(
        "importing the speculative-decode harness reaches mlx.nn",
        allow_module_level=True,
    )

nn = pytest.importorskip("mlx.nn")

# These tests were written before the harness, which is the order this repo
# wants. The guard stays now that the harness exists, for the reason it was
# written: a bare import of a missing module raises during COLLECTION, which
# pytest reports as an error rather than a failure and which aborts the whole
# session, so a deleted or renamed harness would take every other test file
# down with it - the failure mode `tests/test_serve_sub4bit.py` documents for
# the mlx.nn import. A skip here is loud in the summary; an abort is not.
h = pytest.importorskip(
    "serve_spec_decode",
    reason="bench/serve_spec_decode.py is missing; these are its host-side "
           "contracts and cannot run without it",
)


# These tests stop before or wrap the harness boundary, so no honest change
# confined to spec_decode_rules.py can make them red. Each mutation comment
# therefore names the concrete serve_spec_decode.py change it detects.


# Constructing MeasurementLock before verify_pins makes this red.
def test_pins_are_checked_before_the_lock(monkeypatch):
    monkeypatch.setattr(h, "verify_pins", _raise(RuntimeError("hash mismatch")))
    monkeypatch.setattr(
        h,
        "MeasurementLock",
        _raise(AssertionError("lock was built before pins were checked")),
    )
    monkeypatch.setattr(h, "load_model", _never_load)
    assert h.main([]) == h.EXIT_PRECONDITION


# Loading a model before MeasurementLock.acquire makes this red.
def test_refuses_when_the_machine_lock_is_held(monkeypatch):
    _pins_ok(monkeypatch, h)
    monkeypatch.setattr(
        h.MeasurementLock,
        "acquire",
        lambda self: (False, "held by pid 1 (test)"),
    )
    monkeypatch.setattr(h, "load_model", _never_load)
    assert h.main([]) == h.EXIT_LOCK_HELD


# Loading before require_idle or mapping NotIdle to another exit makes this red.
def test_a_busy_machine_is_transient(monkeypatch):
    _pins_ok(monkeypatch, h)
    _lock_granted(monkeypatch, h)
    monkeypatch.setattr(
        h, "require_idle", _raise(h.NotIdle("WindowServer at 30%"))
    )
    monkeypatch.setattr(h, "load_model", _never_load)
    assert h.main([]) == h.EXIT_NOT_IDLE


# Passing a refusal through check_idle_after makes this red. The three tests
# above only catch it on a machine that happens to be non-idle, because there
# check_idle_after rewrites the code to 1; this one catches it anywhere. What it
# costs is not cosmetic: detached_run.judge_exit_code reads EXIT_NOT_IDLE as
# "waiting" and spends no attempt, while it reads 1 as "stopped" and spends one,
# so a busy machine would burn all three attempts without measuring anything.
def test_refusals_do_not_pass_through_the_closing_idle_check(monkeypatch):
    _pins_ok(monkeypatch, h)
    _lock_granted(monkeypatch, h)
    monkeypatch.setattr(
        h, "require_idle", _raise(h.NotIdle("WindowServer at 30%"))
    )
    monkeypatch.setattr(h, "load_model", _never_load)
    monkeypatch.setattr(
        h,
        "check_idle_after",
        _raise(AssertionError("a refusal was passed through the idle check")),
    )
    assert h.main([]) == h.EXIT_NOT_IDLE


def _models_loaded(monkeypatch):
    monkeypatch.setattr(h, "require_idle", lambda label: {"idle": True})
    monkeypatch.setattr(h, "provenance", lambda manifest: {"pins": manifest})
    monkeypatch.setattr(h, "load_model", lambda name: (object(), object()))
    monkeypatch.setattr(h, "make_prompts", lambda *args, **kwargs: [1, 2, 3])
    monkeypatch.setattr(h, "make_sampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(h.mx, "clear_cache", lambda: None)


# Mapping either typed memory refusal to the other's shared exit makes a case red.
@pytest.mark.parametrize(
    "exc,want",
    [
        (lambda: h.BudgetExceeded("draft K=2", 30.0, 24.0),
         lambda: h.EXIT_BUDGET_REFUSAL),
        (lambda: h.LowMemoryRefusal("draft K=2", 1.0, 23.0),
         lambda: h.EXIT_LOW_MEMORY),
    ],
)
def test_memory_refusals_use_the_shared_codes(monkeypatch, exc, want):
    _pins_ok(monkeypatch, h)
    _lock_granted(monkeypatch, h)
    _models_loaded(monkeypatch)
    monkeypatch.setattr(h.ServeGuard, "__call__", _raise(exc()))
    assert h.main([]) == want()


class _Input:
    def __init__(self, shape):
        self.shape = shape


@contextmanager
def _patched_eval(record: list):
    """Record every mx.eval the counter makes, without evaluating anything."""
    original = h.mx.eval
    h.mx.eval = lambda *args: record.append(args)
    try:
        yield
    finally:
        h.mx.eval = original


class _Model(nn.Module):
    def __call__(self, inputs, cache=None):
        return inputs


# These shapes are mlx_lm's, not ours: it verifies with model(y[None], cache=cache)
# on the K+1 candidate TOKEN IDS, so the counted seam sees rank 2 and the
# embedding happens below it (doc, section 4 amendment of 2026-08-18). The first
# version of this file wrote (1, 7, 2560) here, which no test could have caught
# because every other test used the same fiction.
def test_the_counted_seam_is_still_rank_two_token_ids():
    import inspect

    generate = pytest.importorskip("mlx_lm.generate")
    source = inspect.getsource(generate.speculative_generate_step)
    assert "logits = model(y[None], cache=cache)" in source, (
        "mlx_lm no longer verifies through the seam this harness counts; the "
        "width rule in spec_decode_rules.expected_routed_calls reads it"
    )


# Assigning __call__ on the target instance or counting every instance makes this red.
def test_counter_counts_only_the_target_and_restores_the_class():
    target = _Model()
    other = _Model()
    original = type(target).__call__
    with h.counting_calls(target) as counter:
        target(_Input((1, 7)))
        other(_Input((1, 7)))
    assert counter.shapes == [(1, 7)]
    assert type(target).__call__ is original


# MLX is lazy, so a timer around the seam would measure graph construction.
# Forcing evaluation to get a real time perturbs generation_tps, so it is
# confined to the probe: making the timed path sync makes this red.
def test_the_timed_path_never_forces_evaluation():
    evaluated = []
    target = _Model()
    with _patched_eval(evaluated):
        with h.counting_calls(target) as counter:
            target(_Input((1, 7)))
    assert evaluated == []
    assert counter.seconds == []


# Making the probe lazy too makes this red, and the probe would then time nothing.
def test_the_probe_forces_evaluation_and_times_each_pass():
    evaluated = []
    target = _Model()
    with _patched_eval(evaluated):
        with h.counting_calls(target, sync=True) as counter:
            target(_Input((1, 7)))
            target(_Input((1, 7)))
    assert len(evaluated) == 2
    assert len(counter.seconds) == 2
    assert all(s >= 0.0 for s in counter.seconds)


# Omitting the counting context manager's finally restoration makes this red.
def test_counter_restores_the_class_when_an_arm_raises():
    target = _Model()
    original = type(target).__call__
    with pytest.raises(RuntimeError, match="arm failed"):
        with h.counting_calls(target):
            raise RuntimeError("arm failed")
    assert type(target).__call__ is original


class _Patch:
    def __init__(self):
        self.mode = "fused"
        self.calls = 0
        self.site_cells = []

    def reset(self):
        self.calls = 0

    def hard_fallbacks(self):
        return {}

    def uninstall(self):
        return None


def _response(token: int):
    # No generation_time: mlx_lm 0.31.3's GenerationResponse has no such field,
    # and the harness derives it as generation_tokens / generation_tps (doc,
    # section 5 amendment of 2026-08-18). finish_reason is here because the
    # real one carries it and the row records it.
    return SimpleNamespace(
        token=token,
        generation_tokens=h.GEN_TOKENS,
        generation_tps=100.0,
        finish_reason="length",
    )


def _stub_the_machine(monkeypatch):
    """Everything below the harness boundary, faked, so main() walks its
    wiring end to end on a machine that does nothing."""
    _pins_ok(monkeypatch, h)
    _lock_granted(monkeypatch, h)
    monkeypatch.setattr(h, "require_idle", lambda label: {"idle": True})
    monkeypatch.setattr(h, "check_idle_after", lambda exit_code: exit_code)
    monkeypatch.setattr(h, "provenance", lambda manifest: {"pins": manifest})
    monkeypatch.setattr(h, "load_model", lambda name: (_Model(), object()))
    monkeypatch.setattr(h, "make_prompts", lambda *args, **kwargs: [1, 2, 3])
    monkeypatch.setattr(h, "make_sampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(h, "install_patch", lambda model: _Patch())
    monkeypatch.setattr(h, "should_dispatch", lambda *args: False)
    monkeypatch.setattr(h.mx, "clear_cache", lambda: None)
    # The per-cell probe runs the counter with sync=True, and the fake model
    # above returns the fake input rather than an array, so the machine's
    # evaluator is stubbed here for the same reason clear_cache is: this test
    # walks the wiring, and mx.eval is machine.
    monkeypatch.setattr(h.mx, "eval", lambda *args: None)
    monkeypatch.setattr(h, "ROUNDS", 1)

    guarded = []
    monkeypatch.setattr(h, "ServeGuard", lambda budget_gb: guarded.append)

    def stream_generate(
        model,
        tokenizer,
        prompt,
        *,
        max_tokens,
        draft_model=None,
        num_draft_tokens=None,
        sampler,
    ):
        width = 1 if draft_model is None else num_draft_tokens + 1
        model(_Input((1, width)))
        for token in range(max_tokens):
            yield _response(token)

    monkeypatch.setattr(h, "stream_generate", stream_generate)
    return guarded


# Omitting the per-cell guard or changing stream_generate's speculative keywords makes this red.
def test_main_reaches_the_grid_with_the_machine_stubbed(monkeypatch):
    guarded = _stub_the_machine(monkeypatch)
    assert h.main([]) == 0
    assert guarded, "main never reached the first per-cell memory guard"


# The K=4 follow-up's one change (docs/research/2026-08-18-spec-decode-k4-followup.md,
# section 3): the primary cell is a registered input. Dropping the flag, or
# parsing it and not passing it through to the verdicts, makes this red; so
# does letting the default drift off the parent's K=6.
def test_the_primary_cell_flag_reaches_both_verdicts(monkeypatch, capsys):
    _stub_the_machine(monkeypatch)
    assert h.main(["--primary-k", "4"]) == 0
    results = [
        json.loads(line[len("RESULT: "):])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("RESULT: ")
    ]
    primary = {(r["draft"], r["outcome"]): r["primary_k"]
               for r in results if r["outcome"] in ("O3", "O4")}
    assert primary and set(primary.values()) == {4}, primary


def test_the_primary_cell_defaults_to_the_parents_k6(monkeypatch, capsys):
    _stub_the_machine(monkeypatch)
    assert h.main([]) == 0
    primary = {
        json.loads(line[len("RESULT: "):])["primary_k"]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("RESULT: ")
        and json.loads(line[len("RESULT: "):])["outcome"] in ("O3", "O4")
    }
    assert primary == {6}
