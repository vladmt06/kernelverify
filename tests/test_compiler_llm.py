"""The LLM generator: pinned, verbatim, counted, and testable with no model.

Nothing here talks to a network or needs a CLI installed. The model seam is
any callable with a `model_id` and an `unwrap`, so the fakes below script
exact responses; the one test that runs a real subprocess drives CliModel
against `sys.executable -c` one-liners, which is the taxonomy mapping
measured rather than assumed.

The envelope fixtures are trimmed captures from live calls made on
2026-08-19 (claude 2.1.234, codex 0.147.0), not hand-written guesses, so an
unwrap that only works on invented shapes cannot pass.
"""

import json
import sys

import pytest

from kernelverify.compiler.brief import Operation, build
from kernelverify.compiler.funnel import Funnel, Stage, StageOutcome
from kernelverify.compiler.llm import (
    KERNEL_SCHEMA,
    CliModel,
    ExtractionFailed,
    ModelCallFailed,
    claude_model,
    codex_model,
    extract_kernel,
    generate_session,
    unwrap_claude,
    unwrap_codex,
)
from kernelverify.compiler.stages import lint_stage
from kernelverify.compiler.store import CandidateStore

OP = Operation(name="demo", goal="Scale a vector.", signature="out = 2 * x")

GOOD_KERNEL = "kernel void k(device float* out) { float acc = 1.0f; out[0] = acc; }"
HALF_KERNEL = "kernel void k(device float* out) { half acc = 1.0h; out[0] = acc; }"


def reply(source: str) -> str:
    return json.dumps({"kernel_source": source})


class FakeModel:
    """Scripted responses; an entry that is an exception is raised."""

    model_id = "fake-1"
    unwrap = staticmethod(lambda raw: raw)

    def __init__(self, replies):
        self.replies = list(replies)
        self.briefs = []

    def __call__(self, brief):
        self.briefs.append(brief)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture()
def store(tmp_path):
    return CandidateStore(tmp_path)


@pytest.fixture()
def funnel():
    return Funnel([lint_stage()])


# ---------------------------------------------------------------------------
# R14: verbatim, and pinned twice
# ---------------------------------------------------------------------------
def test_prompt_and_response_land_verbatim_in_the_store(store, funnel):
    model = FakeModel([reply(GOOD_KERNEL)])
    [round1] = generate_session(OP, model, funnel, store, context={},
                                rounds=1, per_round=1)

    [candidate] = store.candidates()
    assert store.prompt(candidate) == model.briefs[0], "the brief, byte for byte"
    assert store.response(candidate) == reply(GOOD_KERNEL), "the raw reply"
    assert round1.session.proposed == 1


def test_the_model_id_appears_in_the_journal_twice(store, funnel, tmp_path):
    """Per candidate in the origin, per session in the generator event."""
    generate_session(OP, FakeModel([reply(GOOD_KERNEL)]), funnel, store,
                     context={}, rounds=1, per_round=1)

    [candidate] = store.candidates()
    assert store.get(candidate).origin == "llm fake-1 r1"
    events = [json.loads(line) for line in
              (tmp_path / "journal.jsonl").read_text().splitlines()]
    [header] = [e for e in events if e["event"] == "generator"]
    assert header["model"] == "fake-1"
    assert header["schema"] == KERNEL_SCHEMA


# ---------------------------------------------------------------------------
# Misfires are counted, never guessed around
# ---------------------------------------------------------------------------
def test_garbage_json_is_a_counted_schema_refusal_and_no_candidate(store, funnel):
    model = FakeModel(["Sure! Here's a kernel:\n```metal\nkernel...\n```"])
    [round1] = generate_session(OP, model, funnel, store, context={},
                                rounds=1, per_round=1)

    assert round1.refusals == {"schema": 1}
    assert store.candidates() == []
    assert store.census() == {"no-candidate:schema": 1}


def test_an_empty_kernel_source_is_a_schema_refusal(store, funnel):
    [round1] = generate_session(OP, FakeModel([reply("  ")]), funnel, store,
                                context={}, rounds=1, per_round=1)
    assert round1.refusals == {"schema": 1}


def test_a_model_call_failure_is_counted_with_its_partial_output(store, funnel):
    model = FakeModel([ModelCallFailed("timeout", "no response within 300s",
                                       response="partial text"),
                       reply(GOOD_KERNEL)])
    [round1] = generate_session(OP, model, funnel, store, context={},
                                rounds=1, per_round=2)

    assert round1.asked == 2
    assert round1.refusals == {"timeout": 1}
    assert round1.session.proposed == 1
    assert store.census() == {"no-candidate:timeout": 1, "lint:passed": 1}


def test_a_duplicate_source_is_a_reproposal_not_a_new_candidate(store, funnel):
    model = FakeModel([reply(GOOD_KERNEL), reply(GOOD_KERNEL)])
    [round1] = generate_session(OP, model, funnel, store, context={},
                                rounds=1, per_round=2)

    assert round1.session.proposed == 1
    assert round1.session.reproposed == 1


# ---------------------------------------------------------------------------
# The feedback round-trip, through typed records only
# ---------------------------------------------------------------------------
def test_a_lint_death_in_round_one_reaches_round_twos_brief(store, funnel):
    """The whole point of rounds: the model is told what died and why, from
    the store's typed records, never from anything a stage printed."""
    model = FakeModel([reply(HALF_KERNEL), reply(GOOD_KERNEL)])
    rounds = generate_session(OP, model, funnel, store, context={},
                              rounds=2, per_round=1)

    assert rounds[0].session.census["lint:failed"] == 1
    second_brief = model.briefs[1]
    assert "lint:failed" in second_brief
    assert "half" in second_brief, "the lint's reason is the useful feedback"
    assert rounds[1].session.proposed == 1


def test_a_misfire_shows_up_in_the_next_brief_as_a_count_only(store, funnel):
    """Rule V2: the census line is allowed through; the misfire's detail,
    which may carry stderr, is not."""
    model = FakeModel([ModelCallFailed("exit", "exit 1: AUTH_SECRET_abc123"),
                       reply(GOOD_KERNEL)])
    generate_session(OP, model, funnel, store, context={}, rounds=2,
                     per_round=1)

    second_brief = model.briefs[1]
    assert "no-candidate:exit: 1" in second_brief
    assert "AUTH_SECRET_abc123" not in second_brief


# ---------------------------------------------------------------------------
# Extraction and the two real envelopes
# ---------------------------------------------------------------------------
def test_extraction_refuses_missing_key_empty_source_and_non_object():
    for bad in ("not json", "[1, 2]", "{}", '{"kernel_source": 3}',
                '{"kernel_source": ""}'):
        with pytest.raises(ExtractionFailed) as failure:
            extract_kernel(bad)
        assert failure.value.kind == "schema"
    assert extract_kernel(reply(GOOD_KERNEL)) == GOOD_KERNEL


def test_the_claude_envelope_unwraps_by_its_result_field():
    """Trimmed from a live capture, 2026-08-19."""
    envelope = json.dumps({
        "is_error": False, "subtype": "success",
        "result": "{\"kernel_source\":\"hello\"}",
        "structured_output": {"kernel_source": "hello"},
        "type": "result",
    })
    assert unwrap_claude(envelope) == '{"kernel_source":"hello"}'


def test_a_claude_error_envelope_is_an_envelope_refusal():
    with pytest.raises(ExtractionFailed) as failure:
        unwrap_claude(json.dumps({"is_error": True, "result": "overloaded"}))
    assert failure.value.kind == "envelope"
    with pytest.raises(ExtractionFailed):
        unwrap_claude("this is not json at all")


def test_the_codex_stream_unwraps_by_its_last_agent_message():
    """Trimmed from a live capture, 2026-08-19: a non-fatal error item
    precedes the agent message and must not derail the unwrap."""
    stream = "\n".join([
        json.dumps({"type": "thread.started", "thread_id": "t"}),
        json.dumps({"type": "item.completed",
                    "item": {"id": "item_0", "type": "error",
                             "message": "skill descriptions shortened"}}),
        json.dumps({"type": "item.completed",
                    "item": {"id": "item_1", "type": "agent_message",
                             "text": "{\"kernel_source\":\"hello\"}"}}),
        json.dumps({"type": "turn.completed", "usage": {}}),
    ])
    assert unwrap_codex(stream) == '{"kernel_source":"hello"}'


def test_a_codex_stream_with_no_agent_message_is_an_envelope_refusal():
    with pytest.raises(ExtractionFailed) as failure:
        unwrap_codex(json.dumps({"type": "turn.completed", "usage": {}}))
    assert failure.value.kind == "envelope"


# ---------------------------------------------------------------------------
# CliModel against real subprocesses: the taxonomy, measured
# ---------------------------------------------------------------------------
def _cli(*code_argv, timeout=10.0):
    return CliModel(argv=tuple(code_argv), model_id="probe", timeout=timeout,
                    unwrap=lambda raw: raw)


def test_a_nonzero_exit_is_an_exit_failure_with_the_stderr_tail():
    model = _cli(sys.executable, "-c",
                 "import sys; sys.stderr.write('key expired'); sys.exit(3)")
    with pytest.raises(ModelCallFailed) as failure:
        model("brief")
    assert failure.value.kind == "exit"
    assert "exit 3" in failure.value.detail and "key expired" in failure.value.detail


def test_a_hung_cli_is_killed_and_reported_as_a_timeout():
    model = _cli(sys.executable, "-c", "import time; time.sleep(30)",
                 timeout=0.3)
    with pytest.raises(ModelCallFailed) as failure:
        model("brief")
    assert failure.value.kind == "timeout"


def test_stdout_comes_back_verbatim_and_the_brief_goes_in_on_stdin():
    model = _cli(sys.executable, "-c",
                 "import sys; sys.stdout.write(sys.stdin.read().upper())")
    assert model("the brief") == "THE BRIEF"


# ---------------------------------------------------------------------------
# The pinned argvs
# ---------------------------------------------------------------------------
def test_the_claude_argv_pins_the_exact_model_and_inlines_the_schema():
    model = claude_model("claude-fable-5")
    assert model.argv[:4] == ("claude", "-p", "--model", "claude-fable-5")
    assert json.loads(model.argv[model.argv.index("--json-schema") + 1]) == \
        KERNEL_SCHEMA
    assert model.unwrap is unwrap_claude


def test_the_codex_argv_pins_the_model_the_sandbox_and_stdin():
    model = codex_model("gpt-5.6-sol", "/tmp/schema.json")
    assert model.argv[:4] == ("codex", "exec", "-m", "gpt-5.6-sol")
    assert ("-s", "read-only") == tuple(
        model.argv[model.argv.index("-s"):model.argv.index("-s") + 2])
    assert model.argv[-1] == "-", "stdin is the prompt, spelled explicitly"
    assert model.unwrap is unwrap_codex
