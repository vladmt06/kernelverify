"""The loop's memory: what it keeps, what it refuses, and what it counts.

Three properties earn their tests here. Failures are kept, because a session
whose refusals vanish cannot answer the only question worth asking after it
("where did candidates die?"). A source already seen is refused before it
costs a compile, which is what stops a generator circling. And the journal
carries no clock, so the same session written twice is the same bytes and
the evidence trail can be diffed.

Pure filesystem and JSON, no MLX and no GPU.
"""

import json

import pytest

from kernelverify.compiler.store import (
    Candidate,
    CandidateStore,
    CorruptJournal,
    Reproposed,
    digest,
)

SOURCE = "kernel void a() { }"
OTHER = "kernel void b() { }"


@pytest.fixture()
def store(tmp_path):
    return CandidateStore(tmp_path)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def test_a_candidates_identity_is_its_source(store):
    assert store.propose(SOURCE, origin="seed") == digest(SOURCE)


def test_the_same_source_twice_is_refused_by_hash_before_anything_compiles(store):
    first = store.propose(SOURCE, origin="seed")
    with pytest.raises(Reproposed) as refusal:
        store.propose(SOURCE, origin="generator")
    assert refusal.value.candidate == first, (
        "the refusal must name the candidate that already holds this source, "
        "because that is the feedback the generator needs")
    assert store.candidates() == [first], "a refusal must not start a lineage"


def test_two_different_sources_are_two_candidates(store):
    assert store.propose(SOURCE, origin="seed") != store.propose(OTHER, origin="seed")
    assert len(store.candidates()) == 2


# ---------------------------------------------------------------------------
# What is remembered
# ---------------------------------------------------------------------------
def test_a_candidate_carries_its_source_origin_and_parent(store):
    parent = store.propose(SOURCE, origin="seed")
    child = store.propose(OTHER, origin="generator", parent=parent)
    got = store.get(child)
    assert isinstance(got, Candidate)
    assert got.source == OTHER and got.origin == "generator" and got.parent == parent


def test_the_generating_prompt_is_kept_verbatim(store):
    """The prompt is evidence: a candidate nobody can reproduce the request
    for is not a data point about the generator."""
    candidate = store.propose(SOURCE, origin="generator", prompt="write me a gemv")
    assert store.prompt(candidate) == "write me a gemv"


def test_a_seed_candidate_has_no_prompt_and_says_so(store):
    assert store.prompt(store.propose(SOURCE, origin="seed")) is None


def test_a_failure_is_kept_with_its_reason(store):
    candidate = store.propose(SOURCE, origin="generator")
    store.record(candidate, stage="lint", passed=False,
                 detail="half accumulator outside a store idiom")
    got = store.get(candidate)
    assert got.outcome == "lint:failed" and not got.alive
    assert "half accumulator" in got.events[-1]["detail"]


def test_a_candidate_that_is_still_passing_reads_as_alive(store):
    candidate = store.propose(SOURCE, origin="generator")
    store.record(candidate, stage="compile", passed=True)
    store.record(candidate, stage="lint", passed=True)
    assert store.get(candidate).alive
    assert store.get(candidate).outcome == "lint:passed"


def test_recording_against_an_unknown_candidate_refuses(store):
    with pytest.raises(KeyError):
        store.record("f" * 64, stage="lint", passed=True)


def test_a_stage_that_errored_is_a_third_state_not_a_refusal(store):
    """A stage that raised did not judge the candidate. Counting it as a
    refusal would let a bug in our own stage code read as a bad kernel."""
    candidate = store.propose(SOURCE, origin="generator")
    store.record(candidate, stage="compile", passed=False,
                 detail="RuntimeError: segfault", errored=True)
    assert store.get(candidate).outcome == "compile:errored"


def test_a_stage_cannot_have_both_errored_and_passed(store):
    candidate = store.propose(SOURCE, origin="generator")
    with pytest.raises(ValueError, match="cannot also have passed"):
        store.record(candidate, stage="compile", passed=True, errored=True)


# ---------------------------------------------------------------------------
# The model's side of the record: responses, misfires, and the session header
# ---------------------------------------------------------------------------
def test_a_response_blob_is_stored_verbatim_and_content_addressed(store, tmp_path):
    """R14: prompt and response verbatim. The response is a blob like the
    prompt, so identical text is stored once and the journal names it."""
    reply = '{"kernel_source": "kernel void k() { }"}\n'
    candidate = store.propose(SOURCE, origin="llm test-model r1",
                              prompt="write me a kernel", response=reply)
    assert store.response(candidate) == reply
    from kernelverify.compiler.store import digest
    assert (tmp_path / "blobs" / f"{digest(reply)}.response").exists()


def test_a_candidate_without_a_response_says_none(store):
    assert store.response(store.propose(SOURCE, origin="seed")) is None


def test_an_old_journal_without_the_new_events_still_loads(tmp_path):
    """The change is additive: a journal written before responses and
    no-candidate events existed must read back identically."""
    first = CandidateStore(tmp_path)
    candidate = first.propose(SOURCE, origin="seed")
    first.record(candidate, stage="lint", passed=True)

    reopened = CandidateStore(tmp_path)
    assert reopened.census() == {"lint:passed": 1}
    assert reopened.get(candidate).outcome == "lint:passed"
    assert reopened.response(candidate) is None


def test_a_no_candidate_event_is_counted_and_its_response_readable(store, tmp_path):
    """A model call that yields nothing to hash is still a request the
    session paid for: it gets its own census line, and the text that came
    back is stored verbatim so the refusal can be re-read."""
    store.no_candidate("schema", prompt="the brief",
                       response="I cannot write that kernel.",
                       detail="kernel_source missing")
    store.propose(SOURCE, origin="llm test-model r1")

    assert store.census() == {"no-candidate:schema": 1, "proposed": 1}
    assert store.candidates() == [store.candidates()[0]], (
        "a no-candidate event must never appear as a candidate")
    from kernelverify.compiler.store import digest
    blob = tmp_path / "blobs" / f"{digest('I cannot write that kernel.')}.response"
    assert blob.read_text() == "I cannot write that kernel."


def test_a_no_candidate_event_may_have_no_response_at_all(store):
    """A timeout can kill the process before any text arrives."""
    store.no_candidate("timeout", prompt="the brief")
    assert store.census() == {"no-candidate:timeout": 1}


def test_the_generator_event_carries_model_and_argv_and_breaks_no_reader(store,
                                                                         tmp_path):
    """The R14 audit record: the journal proves which binary, flags, model id
    and schema ran. It names no candidate, so every reader must skip it."""
    import json

    store.note_generator(model="claude-fable-5",
                         argv=["claude", "-p", "--model", "claude-fable-5"],
                         schema={"type": "object"}, timeout=300.0)
    candidate = store.propose(SOURCE, origin="llm claude-fable-5 r1")
    store.record(candidate, stage="lint", passed=True)

    [line] = [json.loads(l) for l in
              (tmp_path / "journal.jsonl").read_text().splitlines()
              if json.loads(l)["event"] == "generator"]
    assert line["model"] == "claude-fable-5"
    assert line["argv"] == ["claude", "-p", "--model", "claude-fable-5"]

    assert store.census() == {"lint:passed": 1}
    assert store.get(candidate).outcome == "lint:passed"
    assert store.candidates() == [candidate]


# ---------------------------------------------------------------------------
# The census, which is reported whether or not anything is kept
# ---------------------------------------------------------------------------
def test_the_census_counts_where_candidates_died(store):
    dead_at_lint = store.propose(SOURCE, origin="generator")
    store.record(dead_at_lint, stage="compile", passed=True)
    store.record(dead_at_lint, stage="lint", passed=False, detail="bf16 tile")

    dead_at_compile = store.propose(OTHER, origin="generator")
    store.record(dead_at_compile, stage="compile", passed=False, detail="no such type")

    survivor = store.propose("kernel void c() { }", origin="seed")
    store.record(survivor, stage="compile", passed=True)
    store.record(survivor, stage="lint", passed=True)

    store.propose("kernel void d() { }", origin="generator")  # never judged

    assert store.census() == {"lint:failed": 1, "compile:failed": 1,
                              "lint:passed": 1, "proposed": 1}


# ---------------------------------------------------------------------------
# The journal on disk
# ---------------------------------------------------------------------------
def test_the_journal_is_append_only_and_reopens_to_the_same_state(store, tmp_path):
    candidate = store.propose(SOURCE, origin="seed")
    store.record(candidate, stage="compile", passed=True)
    before = (tmp_path / "journal.jsonl").read_text()

    reopened = CandidateStore(tmp_path)
    reopened.record(candidate, stage="lint", passed=False, detail="nope")
    after = (tmp_path / "journal.jsonl").read_text()

    assert after.startswith(before), "an existing line must never be rewritten"
    assert reopened.get(candidate).outcome == "lint:failed"
    assert [json.loads(line)["seq"] for line in after.splitlines()] == [0, 1, 2]


def test_the_journal_carries_no_clock_so_two_runs_are_the_same_bytes(tmp_path):
    def session(root):
        shop = CandidateStore(root)
        candidate = shop.propose(SOURCE, origin="generator", prompt="p")
        shop.record(candidate, stage="compile", passed=True)
        shop.record(candidate, stage="lint", passed=False, detail="d")
        return (root / "journal.jsonl").read_bytes()

    assert session(tmp_path / "one") == session(tmp_path / "two")


def test_identical_content_is_stored_once(store, tmp_path):
    """Content addressing, seen from the disk: two candidates generated from
    the same prompt share the one prompt blob."""
    store.propose(SOURCE, origin="generator", prompt="same request")
    store.propose(OTHER, origin="generator", prompt="same request")
    prompts = list((tmp_path / "blobs").glob("*.prompt"))
    assert len(prompts) == 1


def test_a_half_written_journal_line_refuses_and_names_the_line(store, tmp_path):
    """A truncated line means the loop died mid-append. Skipping it would
    under-report the census silently, which is the failure this file exists
    to prevent, so the read refuses and says where."""
    store.propose(SOURCE, origin="seed")
    journal = tmp_path / "journal.jsonl"
    journal.write_text(journal.read_text() + '{"event": "sta\n')
    with pytest.raises(CorruptJournal, match="line 2"):
        CandidateStore(tmp_path)
