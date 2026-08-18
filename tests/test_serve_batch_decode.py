"""Host-side contracts for the batched-decode harness."""

from __future__ import annotations

import inspect
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
        "importing the batched-decode harness reaches mlx.nn",
        allow_module_level=True,
    )

h = pytest.importorskip(
    "serve_batch_decode",
    reason="bench/serve_batch_decode.py is missing; these are its host-side "
           "contracts and cannot run without it",
)


def test_pins_are_checked_before_the_lock(monkeypatch):
    monkeypatch.setattr(h, "verify_pins", _raise(RuntimeError("hash mismatch")))
    monkeypatch.setattr(
        h,
        "MeasurementLock",
        _raise(AssertionError("lock was built before pins were checked")),
    )
    monkeypatch.setattr(h, "load_model", _never_load)
    assert h.main([]) == h.EXIT_PRECONDITION


def test_refuses_when_the_machine_lock_is_held(monkeypatch):
    _pins_ok(monkeypatch, h)
    monkeypatch.setattr(
        h.MeasurementLock,
        "acquire",
        lambda self: (False, "held by pid 1 (test)"),
    )
    monkeypatch.setattr(h, "load_model", _never_load)
    assert h.main([]) == h.EXIT_LOCK_HELD


def test_the_lock_is_held_while_models_are_loaded(monkeypatch):
    _pins_ok(monkeypatch, h)
    held = {"value": False}

    class Lock:
        def __init__(self, name):
            assert name == h.LOCK_NAME

        def acquire(self):
            held["value"] = True
            return True, "acquired"

        def release(self):
            held["value"] = False

    monkeypatch.setattr(h, "MeasurementLock", Lock)
    monkeypatch.setattr(h, "require_idle", lambda label: {"idle": True})
    monkeypatch.setattr(h, "require_pinned_zone", lambda: None)
    monkeypatch.setattr(h, "provenance", lambda manifest: {})

    def load_model(name):
        assert held["value"], "model loaded outside the lock"
        return object(), object()

    monkeypatch.setattr(h, "load_model", load_model)
    monkeypatch.setattr(h, "B_GRID", ())
    monkeypatch.setattr(h, "decide_zone", lambda *args, **kwargs: {})
    monkeypatch.setattr(h, "check_idle_after", lambda code: code)
    assert h.main([]) == 0
    assert held["value"] is False


def test_a_busy_machine_is_transient(monkeypatch):
    _pins_ok(monkeypatch, h)
    _lock_granted(monkeypatch, h)
    monkeypatch.setattr(
        h, "require_idle", _raise(h.NotIdle("WindowServer at 30%"))
    )
    monkeypatch.setattr(h, "load_model", _never_load)
    assert h.main([]) == h.EXIT_NOT_IDLE


class _PromptArray:
    def __init__(self, rows):
        self._rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def tolist(self):
        return [list(row) for row in self._rows]


class _Input:
    def __init__(self, shape):
        self.shape = shape


class _Model:
    def __call__(self, inputs, cache=None):
        return inputs


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


class _FakeBatchGenerator:
    events = []
    instances = []
    short_stream = None
    bad_finish = None

    @classmethod
    def reset(cls):
        cls.events = []
        cls.instances = []
        cls.short_stream = None
        cls.bad_finish = None

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.instance = len(type(self).instances)
        self.uids = []
        self.max_tokens = []
        self._returned = False
        self.insert_calls = 0
        type(self).instances.append(self)
        type(self).events.append(("init", self.instance))

    def insert(self, prompts, max_tokens):
        self.insert_calls += 1
        self.prompts = prompts
        self.max_tokens = max_tokens
        self.uids = list(range(len(prompts)))
        type(self).events.append(("insert", self.instance))
        return self.uids

    @contextmanager
    def stats(self):
        generation_tokens = sum(self.max_tokens)
        stats = SimpleNamespace(
            prompt_tokens=sum(len(prompt) for prompt in self.prompts),
            prompt_tps=1000.0,
            prompt_time=0.1,
            generation_tokens=generation_tokens,
            generation_tps=100.0,
            generation_time=generation_tokens / 100.0,
            peak_memory=1.0,
        )
        type(self).events.append(("stats-enter", self.instance))
        try:
            yield stats
        finally:
            type(self).events.append(("stats-exit", self.instance))

    def next_generated(self):
        if self._returned:
            return []
        self._returned = True
        b = len(self.uids)
        self.model(_Input((b, h.PROMPT_T - 1)))
        self.model(_Input((b, 1)))
        responses = []
        for position in range(h.GEN_TOKENS):
            for uid in self.uids:
                if uid == type(self).short_stream and position == h.GEN_TOKENS - 1:
                    continue
                reason = "length" if position == h.GEN_TOKENS - 1 else None
                if uid == type(self).bad_finish and reason is not None:
                    reason = "stop"
                responses.append(
                    SimpleNamespace(uid=uid, token=position, finish_reason=reason)
                )
        return responses

    def close(self):
        type(self).events.append(("close", self.instance))


def _stub_the_machine(monkeypatch):
    _FakeBatchGenerator.reset()
    _pins_ok(monkeypatch, h)
    _lock_granted(monkeypatch, h)
    monkeypatch.setattr(h, "require_idle", lambda label: {"idle": True})
    monkeypatch.setattr(h, "require_pinned_zone", lambda: None)
    monkeypatch.setattr(h, "check_idle_after", lambda exit_code: exit_code)
    monkeypatch.setattr(h, "provenance", lambda manifest: {"pins": manifest})
    monkeypatch.setattr(h, "load_model", lambda name: (_Model(), object()))
    monkeypatch.setattr(
        h,
        "_stream_prompts",
        lambda tokenizer, t, b, *, stride: (
            _PromptArray([[stream] * t for stream in range(b)]),
            [[stream] * t for stream in range(b)],
        ),
    )
    monkeypatch.setattr(h, "install_patch", lambda model: _Patch())
    monkeypatch.setattr(h, "BatchGenerator", _FakeBatchGenerator)
    monkeypatch.setattr(h.mx, "clear_cache", lambda: None)
    guarded = []
    monkeypatch.setattr(h, "ServeGuard", lambda budget_gb: guarded.append)
    return guarded


@pytest.mark.parametrize(
    "exc,want",
    [
        (
            lambda: h.BudgetExceeded("B=1", 30.0, 24.0),
            lambda: h.EXIT_BUDGET_REFUSAL,
        ),
        (
            lambda: h.LowMemoryRefusal("B=1", 1.0, 23.0),
            lambda: h.EXIT_LOW_MEMORY,
        ),
    ],
)
def test_memory_refusals_use_the_shared_codes(monkeypatch, exc, want):
    _stub_the_machine(monkeypatch)
    monkeypatch.setattr(h, "ServeGuard", lambda budget: _raise(exc()))
    assert h.main([]) == want()


@pytest.mark.parametrize(
    "kind,want",
    [
        ("not-idle", lambda: h.EXIT_NOT_IDLE),
        ("precondition", lambda: h.EXIT_PRECONDITION),
        ("budget", lambda: h.EXIT_BUDGET_REFUSAL),
        ("low-memory", lambda: h.EXIT_LOW_MEMORY),
    ],
)
def test_refusals_do_not_pass_through_the_closing_idle_check(
    monkeypatch, kind, want
):
    _stub_the_machine(monkeypatch)
    if kind == "not-idle":
        monkeypatch.setattr(h, "require_idle", _raise(h.NotIdle("busy")))
    elif kind == "precondition":
        monkeypatch.setattr(
            h,
            "require_pinned_zone",
            _raise(h.PreconditionFailed("zone moved")),
        )
    elif kind == "budget":
        monkeypatch.setattr(
            h,
            "ServeGuard",
            lambda budget: _raise(h.BudgetExceeded("B=1", 30.0, 24.0)),
        )
    else:
        monkeypatch.setattr(
            h,
            "ServeGuard",
            lambda budget: _raise(h.LowMemoryRefusal("B=1", 1.0, 23.0)),
        )
    monkeypatch.setattr(
        h,
        "check_idle_after",
        _raise(AssertionError("a refusal reached the closing idle check")),
    )
    assert h.main([]) == want()


@pytest.mark.parametrize(
    "kind",
    ["success", "not-idle", "precondition", "budget", "low-memory", "invalid"],
)
def test_the_lock_is_released_on_every_acquired_exit(monkeypatch, kind):
    _pins_ok(monkeypatch, h)
    released = []

    class Lock:
        def __init__(self, name):
            pass

        def acquire(self):
            return True, "acquired"

        def release(self):
            released.append(True)

    monkeypatch.setattr(h, "MeasurementLock", Lock)
    monkeypatch.setattr(h, "require_idle", lambda label: {"idle": True})
    monkeypatch.setattr(h, "require_pinned_zone", lambda: None)
    monkeypatch.setattr(h, "provenance", lambda manifest: {})
    monkeypatch.setattr(h, "load_model", lambda name: (_Model(), object()))
    monkeypatch.setattr(h, "check_idle_after", lambda code: code)
    monkeypatch.setattr(h, "B_GRID", ())
    monkeypatch.setattr(h, "decide_zone", lambda *args, **kwargs: {})
    if kind == "not-idle":
        monkeypatch.setattr(h, "require_idle", _raise(h.NotIdle("busy")))
    elif kind == "precondition":
        monkeypatch.setattr(
            h,
            "require_pinned_zone",
            _raise(h.PreconditionFailed("zone moved")),
        )
    elif kind == "budget":
        monkeypatch.setattr(
            h, "ServeGuard", _raise(h.BudgetExceeded("startup", 30.0, 24.0))
        )
    elif kind == "low-memory":
        monkeypatch.setattr(
            h, "ServeGuard", _raise(h.LowMemoryRefusal("startup", 1.0, 23.0))
        )
    elif kind == "invalid":
        monkeypatch.setattr(h, "decide_zone", _raise(h.RunInvalid("bad run")))
    h.main([])
    assert released == [True]


def test_main_reaches_the_grid_with_the_batch_engine_stubbed(monkeypatch):
    guarded = _stub_the_machine(monkeypatch)
    monkeypatch.setattr(h, "ROUNDS", 1)
    assert h.main([]) == 0
    assert guarded, "main never reached the first per-cell memory guard"


def test_every_generator_receives_the_registered_engine_arguments(monkeypatch):
    _stub_the_machine(monkeypatch)
    monkeypatch.setattr(h, "ROUNDS", 1)
    assert h.main([]) == 0
    want = {
        "max_tokens": h.GEN_TOKENS,
        "stop_tokens": None,
        **h.ENGINE,
    }
    assert _FakeBatchGenerator.instances
    assert all(gen.kwargs == want for gen in _FakeBatchGenerator.instances)
    assert all(gen.insert_calls == 1 for gen in _FakeBatchGenerator.instances)


def test_each_generator_closes_after_stats_and_before_the_next_arm(monkeypatch):
    _stub_the_machine(monkeypatch)
    monkeypatch.setattr(h, "ROUNDS", 1)
    assert h.main([]) == 0
    events = _FakeBatchGenerator.events
    for instance in range(len(_FakeBatchGenerator.instances)):
        stats_exit = events.index(("stats-exit", instance))
        close = events.index(("close", instance))
        assert stats_exit < close
        if instance + 1 < len(_FakeBatchGenerator.instances):
            assert close < events.index(("init", instance + 1))


def test_each_round_rotates_its_start_and_runs_every_arm_once(monkeypatch):
    _stub_the_machine(monkeypatch)
    calls = []
    original = h._run_arm

    def recorded(arm, *args, **kwargs):
        calls.append((arm, kwargs["cell"]))
        return original(arm, *args, **kwargs)

    monkeypatch.setattr(h, "_run_arm", recorded)
    assert h.main([]) == 0
    for b in h.B_GRID:
        for round_number in range(1, h.ROUNDS + 1):
            prefix = f"B={b} round {round_number}"
            got = [arm for arm, cell in calls if cell.startswith(prefix)]
            offset = (round_number - 1) % len(h.ARMS)
            want = list(h.ARMS[offset:] + h.ARMS[:offset])
            assert got == want


@pytest.mark.parametrize("failure", ["short", "finish"])
def test_an_incomplete_stream_invalidates_the_run(monkeypatch, failure):
    _stub_the_machine(monkeypatch)
    monkeypatch.setattr(h, "B_GRID", (4,))
    if failure == "short":
        _FakeBatchGenerator.short_stream = 2
    else:
        _FakeBatchGenerator.bad_finish = 2
    assert h.main([]) == 1


def _result_records(capsys):
    return [
        json.loads(line[len("RESULT: "):])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("RESULT: ")
    ]


def _plain_json_numbers(value):
    if isinstance(value, dict):
        return all(_plain_json_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_plain_json_numbers(item) for item in value)
    if isinstance(value, bool):
        # bool is a subclass of int, and a boolean is plain JSON; without this
        # branch every `decider` and `clears_floor_upward` field would read as
        # a non-plain number and the check would fail on correct output.
        return True
    if isinstance(value, (int, float)):
        return type(value) in (int, float)
    return True


def test_result_lines_cover_the_registered_outcomes(monkeypatch, capsys):
    _stub_the_machine(monkeypatch)
    monkeypatch.setattr(h, "ROUNDS", 1)
    assert h.main([]) == 0
    records = _result_records(capsys)
    by_outcome = {
        outcome: [record for record in records if record["outcome"] == outcome]
        for outcome in ("OB1", "OB2", "OB3", "ZONE")
    }
    assert [record["b"] for record in by_outcome["OB1"]] == list(h.B_GRID)
    assert {record["b"] for record in by_outcome["OB2"]} == set(h.IN_ZONE)
    assert {record["b"] for record in by_outcome["OB3"]} == set(h.IN_ZONE)
    assert len(by_outcome["ZONE"]) == 1
    assert all(_plain_json_numbers(record) for record in records)


class _Tokenizer:
    def __init__(self, length=3000):
        self.ids = list(range(length))

    def encode(self, text):
        return list(self.ids)


def test_stream_prompts_use_registered_offsets_and_lengths():
    prompts, token_lists = h._stream_prompts(_Tokenizer(), 8, 4, stride=3)
    assert prompts.shape == (4, 8)
    assert token_lists == [
        list(range(0, 8)),
        list(range(3, 11)),
        list(range(6, 14)),
        list(range(9, 17)),
    ]
    assert len({tuple(tokens) for tokens in token_lists}) == 4


def test_stream_zero_matches_the_existing_prompt_builder():
    tokenizer = _Tokenizer()
    prompts, token_lists = h._stream_prompts(tokenizer, 8, 4, stride=3)
    existing = h.make_prompts(tokenizer, 8, 1)
    assert prompts[0].tolist() == existing[0].tolist()
    assert token_lists[0] == existing[0].tolist()


def test_stream_prompts_refuse_a_short_seed_with_all_numbers_named():
    with pytest.raises(RuntimeError) as raised:
        h._stream_prompts(_Tokenizer(length=12), 8, 4, stride=3)
    message = str(raised.value)
    assert all(str(number) in message for number in (12, 17, 8, 4, 3))


def test_prompt_digest_uses_the_registered_compact_serialisation():
    token_lists = [[1, 2], [3, 4]]
    assert h._prompt_digest(token_lists) == (
        "03a4cc702aa6e4f169ff34dd055714141eae3d2d7c50f6f053150cc72ef12638"
    )


@contextmanager
def _record_evaluations(record):
    original = h.mx.eval
    h.mx.eval = lambda *args: record.append(args)
    try:
        yield
    finally:
        h.mx.eval = original


def test_the_timed_call_counter_never_forces_evaluation():
    evaluated = []
    target = _Model()
    with _record_evaluations(evaluated):
        with h.counting_calls(target) as counter:
            target(_Input((5, 1)))
    assert evaluated == []
    assert counter.seconds == []


def test_the_engine_decode_call_still_crosses_the_counted_seam():
    from mlx_lm.generate import GenerationBatch

    source = inspect.getsource(GenerationBatch._step)
    assert "logits = self.model(inputs[:, None], cache=self.prompt_cache)" in source


def test_generation_batch_construction_still_takes_its_first_step():
    from mlx_lm.generate import GenerationBatch

    source = inspect.getsource(GenerationBatch.__init__)
    assert "self._step()" in source
