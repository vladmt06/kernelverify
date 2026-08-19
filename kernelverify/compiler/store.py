"""Every candidate the loop ever saw, including the ones that died.

A generate-gate-price-keep loop is only worth running if its failures are
as legible as its successes: the interesting question after a session is not
which kernel was kept but how many were proposed, where each died, and
whether the generator kept walking into the same wall. So the store is
append-only. Nothing is edited and nothing is deleted; a refusal is a record
in its own right, written in the same journal as a pass.

Two pieces, kept apart on purpose:

    blobs/   content, addressed by the sha256 of the bytes, so identical
             text is stored once and a candidate's identity IS its source
    journal.jsonl
             events, one JSON object per line, appended in order, each one
             naming the blob it talks about rather than repeating it

The identity rule does the loop's most useful piece of work for free: a
generator that proposes a source it has already proposed is refused by hash
before anything compiles it, and it is told so, which is the feedback that
stops a lineage circling.

The journal carries no wall-clock time, only a sequence number, so the same
loop run twice writes the same journal byte for byte and the evidence trail
can be diffed. Timing the loop is the loop's job, not the store's.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

JOURNAL_NAME = "journal.jsonl"
BLOB_DIR_NAME = "blobs"


class Reproposed(Exception):
    """This exact source is already in the store, under `candidate`."""

    def __init__(self, candidate: str):
        super().__init__(f"candidate {candidate[:12]} was already proposed")
        self.candidate = candidate


class CorruptJournal(Exception):
    """A journal line does not parse: the writing process died mid-line."""


@dataclass(frozen=True)
class Candidate:
    """One proposal, with its source and everything that happened to it."""

    id: str
    source: str
    origin: str
    parent: str | None
    events: tuple[dict, ...]

    @property
    def outcome(self) -> str:
        """The last stage this candidate reached, and how it went."""
        stages = [e for e in self.events if e["event"] == "stage"]
        if not stages:
            return "proposed"
        last = stages[-1]
        return f"{last['stage']}:{'passed' if last['passed'] else 'failed'}"

    @property
    def alive(self) -> bool:
        """No stage has failed it yet."""
        return all(e["passed"] for e in self.events if e["event"] == "stage")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class CandidateStore:
    """Append-only record of a generation session."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.blobs = self.root / BLOB_DIR_NAME
        self.journal = self.root / JOURNAL_NAME
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.journal.touch()
        # The journal is read once here and kept, because every write appends
        # to both. One loop owns one store, the same way one measurement owns
        # the machine, so a second writer is a bug rather than a case to
        # handle: reopen the store to see another process's writes.
        self._entries = self._read_journal()

    # -- writing ----------------------------------------------------------
    def propose(self, source: str, *, origin: str, parent: str | None = None,
                prompt: str | None = None) -> str:
        """Record a new candidate and return its id, which is its source hash.

        Raises `Reproposed` if this exact source is already here, which is
        the loop's cheapest rejection and the only one that costs no compile.
        """
        candidate = digest(source)
        if self._blob_path(candidate, "metal").exists():
            raise Reproposed(candidate)

        self._write_blob(candidate, "metal", source)
        entry = {"event": "propose", "candidate": candidate, "origin": origin,
                 "parent": parent}
        if prompt is not None:
            entry["prompt"] = self._write_blob(digest(prompt), "prompt", prompt)
        self._append(entry)
        return candidate

    def record(self, candidate: str, *, stage: str, passed: bool,
               detail: str = "") -> None:
        """Note what one stage of the funnel did to one candidate."""
        self._require_known(candidate)
        self._append({"event": "stage", "candidate": candidate, "stage": stage,
                      "passed": passed, "detail": detail})

    # -- reading ----------------------------------------------------------
    def source(self, candidate: str) -> str:
        self._require_known(candidate)
        return self._blob_path(candidate, "metal").read_text()

    def prompt(self, candidate: str) -> str | None:
        """The prompt this candidate was generated from, if it had one."""
        for entry in self._entries:
            if entry["event"] == "propose" and entry["candidate"] == candidate:
                stored = entry.get("prompt")
                return None if stored is None else \
                    self._blob_path(stored, "prompt").read_text()
        return None

    def get(self, candidate: str) -> Candidate:
        self._require_known(candidate)
        events = [e for e in self._entries if e["candidate"] == candidate]
        proposal = events[0]
        return Candidate(id=candidate, source=self.source(candidate),
                         origin=proposal["origin"], parent=proposal["parent"],
                         events=tuple(events[1:]))

    def candidates(self) -> list[str]:
        """Every candidate id, in the order it was proposed."""
        return [e["candidate"] for e in self._entries
                if e["event"] == "propose"]

    def census(self) -> dict[str, int]:
        """How many candidates ended where: the loop's own yield, which is
        reported whether or not anything was kept."""
        last: dict[str, str] = {}
        for entry in self._entries:
            if entry["event"] == "propose":
                last[entry["candidate"]] = "proposed"
            else:
                verdict = "passed" if entry["passed"] else "failed"
                last[entry["candidate"]] = f"{entry['stage']}:{verdict}"
        counts: dict[str, int] = {}
        for outcome in last.values():
            counts[outcome] = counts.get(outcome, 0) + 1
        return counts

    # -- internals --------------------------------------------------------
    def _blob_path(self, blob: str, kind: str) -> Path:
        return self.blobs / f"{blob}.{kind}"

    def _write_blob(self, blob: str, kind: str, text: str) -> str:
        path = self._blob_path(blob, kind)
        if not path.exists():  # content-addressed, so identical text is a no-op
            path.write_text(text)
        return blob

    def _append(self, entry: dict) -> None:
        entry = {"seq": len(self._entries), **entry}
        with self.journal.open("a") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        self._entries.append(entry)

    def _read_journal(self) -> list[dict]:
        entries = []
        for number, line in enumerate(self.journal.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as error:
                # A half-written line means the loop died mid-append. Say which
                # line rather than skipping it: a silently shortened journal is
                # a census that under-reports, which is the one thing this file
                # exists to prevent.
                raise CorruptJournal(
                    f"{self.journal} line {number} does not parse: {error}"
                ) from error
        return entries

    def _require_known(self, candidate: str) -> None:
        if not self._blob_path(candidate, "metal").exists():
            raise KeyError(f"no candidate {candidate[:12]} in {self.root}")
