"""The language-model generator: pinned, verbatim, and honest about misfires.

Ruling R14 requires three things of any model that writes kernels here: the
model id is pinned, and the prompt and the response are stored verbatim.
Everything in this module exists to make those three properties hold without
anyone having to remember them.

The model seam is structural: anything callable as `brief -> raw response
text`, carrying a `model_id` and an `unwrap` that maps its own envelope to
the assistant's text. Tests inject fakes; production uses `CliModel`, which
runs a pinned argv with the brief on stdin. No API keys and no HTTP client:
both routes this machine has (the claude and codex CLIs) are already
authenticated, and a subprocess is the whole integration.

What is stored is the CLI's raw stdout, before any unwrapping, because that
is the response as received; the envelope is parsed at extraction time and
never before storage, so an envelope change degrades to counted refusals
rather than lost evidence.

Structured output is forced through each CLI's schema flag, so extraction is
json.loads plus three assertions: no fence parsing, no regex, no repair. A
model that answers in prose has refused, and a refusal is recorded, never
patched over.

Every model call that yields no candidate lands in exactly one of four
counted kinds, written to the store so no census can undercount:

    exit      the CLI exited nonzero (auth expiry, bad model id, network)
    timeout   the pinned timeout elapsed and the process was killed
    envelope  stdout is not the CLI's own envelope
    schema    the assistant text is not the pinned shape; an explicit
              textual refusal by the model lands here

The envelope shapes were captured from live calls on 2026-08-19 (claude
2.1.234, codex 0.147.0) rather than guessed, and the fixtures in the tests
are those captures. The codex CLI additionally refuses to run outside a
trusted directory, so a generation session runs from the repository root;
if it does not, the misfire surfaces as a counted `exit`, loudly.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Callable

from kernelverify.compiler import brief as brief_module
from kernelverify.compiler import loop
from kernelverify.compiler.funnel import Funnel
from kernelverify.compiler.loop import Proposal
from kernelverify.compiler.store import CandidateStore

# The one shape a generator's answer may take. Both CLIs enforce it on their
# side (claude inline, codex from a file the session writes), and extraction
# asserts it again here, because the CLI's enforcement is not ours to trust.
KERNEL_SCHEMA = {
    "type": "object",
    "properties": {"kernel_source": {"type": "string"}},
    "required": ["kernel_source"],
    "additionalProperties": False,
}


class ModelCallFailed(Exception):
    """The CLI never delivered a response: nonzero exit or timeout."""

    def __init__(self, kind: str, detail: str, response: str | None = None):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind          # "exit" | "timeout"
        self.detail = detail
        self.response = response  # partial stdout, when any arrived


class ExtractionFailed(Exception):
    """A response arrived and no kernel could be read out of it."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind          # "envelope" | "schema"
        self.detail = detail


@dataclass(frozen=True)
class CliModel:
    """One pinned command that turns a brief into a response.

    The argv carries the model id already spelled into it; `model_id` repeats
    it so the journal can carry it without parsing argv. A hung CLI is killed
    at `timeout` and becomes a counted refusal rather than a hung session.
    """

    argv: tuple[str, ...]
    model_id: str
    timeout: float
    unwrap: Callable[[str], str]

    def __call__(self, brief: str) -> str:
        try:
            done = subprocess.run(list(self.argv), input=brief,
                                  capture_output=True, text=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            raise ModelCallFailed(
                "timeout", f"no response within {self.timeout:.0f}s",
                response=stdout or None) from error
        if done.returncode != 0:
            raise ModelCallFailed(
                "exit", f"exit {done.returncode}: {_tail(done.stderr)}",
                response=done.stdout or None)
        return done.stdout


def _tail(text: str, keep: int = 400) -> str:
    text = (text or "").strip()
    return text[-keep:]


# ---------------------------------------------------------------------------
# The two pinned backends. Envelope shapes captured live on 2026-08-19.
# ---------------------------------------------------------------------------
def unwrap_claude(stdout: str) -> str:
    """claude -p --output-format json: one JSON object; the assistant's text
    is its `result` field and `is_error` says whether to believe it."""
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ExtractionFailed(
            "envelope", f"stdout is not the claude JSON envelope: {error}"
        ) from error
    if not isinstance(envelope, dict) or "result" not in envelope:
        raise ExtractionFailed("envelope",
                               "claude envelope has no `result` field")
    if envelope.get("is_error"):
        raise ExtractionFailed(
            "envelope", f"claude reported an error: {_tail(str(envelope.get('result')))}")
    result = envelope["result"]
    if not isinstance(result, str):
        raise ExtractionFailed("envelope", "claude `result` is not text")
    return result


def unwrap_codex(stdout: str) -> str:
    """codex exec --json: JSONL events; the assistant's text is the last
    `item.completed` event whose item is an `agent_message`."""
    message = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # codex may interleave non-JSON noise; the events matter
        if event.get("type") != "item.completed":
            continue
        item = event.get("item") or {}
        if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
            message = item["text"]
    if message is None:
        raise ExtractionFailed(
            "envelope", "no agent_message event in the codex JSONL stream")
    return message


def claude_model(model_id: str, *, timeout: float = 300.0) -> CliModel:
    """The claude CLI, schema inline (verified: --json-schema takes JSON,
    not a path), exact model id pinned."""
    return CliModel(
        argv=("claude", "-p", "--model", model_id, "--output-format", "json",
              "--json-schema", json.dumps(KERNEL_SCHEMA)),
        model_id=model_id, timeout=timeout, unwrap=unwrap_claude)


def codex_model(model_id: str, schema_path, *, timeout: float = 300.0) -> CliModel:
    """The codex CLI, schema from a file (verified: --output-schema takes a
    path), read-only sandbox per the house rule, prompt from stdin via `-`."""
    return CliModel(
        argv=("codex", "exec", "-m", model_id, "--json",
              "--output-schema", str(schema_path), "-s", "read-only", "-"),
        model_id=model_id, timeout=timeout, unwrap=unwrap_codex)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_kernel(text: str) -> str:
    """The assistant's text -> Metal source, by the pinned schema only."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ExtractionFailed(
            "schema", f"assistant text is not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ExtractionFailed("schema", "assistant JSON is not an object")
    source = payload.get("kernel_source")
    if not isinstance(source, str) or not source.strip():
        raise ExtractionFailed(
            "schema", "no non-empty kernel_source in the assistant JSON")
    return source


# ---------------------------------------------------------------------------
# The session: rounds of brief -> model -> funnel, all of it recorded
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Round:
    asked: int
    refusals: dict = field(default_factory=dict)
    session: loop.Session | None = None

    def summary(self) -> str:
        lines = [f"asked {self.asked}, refusals "
                 f"{sum(self.refusals.values())}"]
        for kind, count in sorted(self.refusals.items()):
            lines.append(f"  no-candidate:{kind}: {count}")
        if self.session is not None:
            lines.append(self.session.summary())
        return "\n".join(lines)


def _proposals_from(model, brief_text: str, store: CandidateStore, *,
                    k: int, origin: str, tally: dict) -> Iterator[Proposal]:
    """Ask the model up to k times, lazily, as the loop pulls.

    Misfires are written to the store under their taxonomy kind and never
    yielded, so the loop's own counts stay counts of real candidates. The raw
    stdout travels on the Proposal and is stored verbatim at propose time.
    """
    for _ in range(k):
        tally["asked"] += 1
        try:
            raw = model(brief_text)
        except ModelCallFailed as error:
            store.no_candidate(error.kind, prompt=brief_text,
                               response=error.response, detail=error.detail)
            tally["refusals"][error.kind] = \
                tally["refusals"].get(error.kind, 0) + 1
            continue
        try:
            source = extract_kernel(model.unwrap(raw))
        except ExtractionFailed as error:
            store.no_candidate(error.kind, prompt=brief_text, response=raw,
                               detail=error.detail)
            tally["refusals"][error.kind] = \
                tally["refusals"].get(error.kind, 0) + 1
            continue
        yield Proposal(source=source, origin=origin, prompt=brief_text,
                       response=raw)


def generate_session(operation, model, funnel: Funnel, store: CandidateStore,
                     *, context: dict, rounds: int, per_round: int,
                     limit: int | None = None) -> list[Round]:
    """Rounds of: build the brief from the store, ask, screen, repeat.

    The brief is rebuilt from the store each round, so feedback reaches the
    model through typed records only (rule V2), and the store is append-only,
    so re-running a session simply continues it: duplicate sources are
    refused by hash into the reproposed count. `limit` caps NEW candidates
    per round, the same meaning it has in loop.run.
    """
    store.note_generator(model=model.model_id,
                         argv=list(getattr(model, "argv", ())),
                         schema=KERNEL_SCHEMA,
                         timeout=float(getattr(model, "timeout", 0.0)))
    results = []
    for number in range(1, rounds + 1):
        brief_text = brief_module.build(operation, store)
        tally = {"asked": 0, "refusals": {}}
        proposals = _proposals_from(model, brief_text, store, k=per_round,
                                    origin=f"llm {model.model_id} r{number}",
                                    tally=tally)
        session = loop.run(proposals, funnel, store, context=context,
                           limit=limit)
        results.append(Round(asked=tally["asked"], refusals=tally["refusals"],
                             session=session))
    return results
