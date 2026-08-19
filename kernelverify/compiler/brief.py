"""What the generator is told, and the two things it is never told.

The loop improves because each round sees what the last one did. That
feedback is the most dangerous surface in the whole design, because two
kinds of leak turn a verified loop into a loop that only looks verified.

Rule V2 of the pre-registration: the generator sees evidence objects, never
raw process output. A brief assembled from whatever a stage happened to
print would carry compiler noise, timing fragments and, sooner or later, a
number nobody gated. This function has no parameter through which raw output
could arrive; every line it writes comes from the store's typed records or
from the operation's own description.

The sealed draw: a held-out failure reaches the generator as the word FAIL
and nothing else. The stage that runs it writes an empty detail for exactly
this reason, so there is nothing here to redact, and the test that matters
builds a brief over a store full of held-out failures and reads it for
leaks.

One smaller thing this file does on purpose: the writing rules it hands the
generator are derived from the lint's own token list rather than typed out
again. A brief that told the generator a different rule than the lint
enforces would spend the session killing candidates for a reason it never
mentioned, and the two would drift apart the first time either changed.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernelverify.compiler.lint import NARROW_TYPES
from kernelverify.compiler.store import CandidateStore


@dataclass(frozen=True)
class Operation:
    """What the generator is being asked to write, in its own words."""

    name: str
    goal: str
    signature: str
    constraints: tuple[str, ...] = ()


def writing_rules() -> str:
    """The style the lint will hold the generator to, derived from the lint.

    Named types come straight from `lint.NARROW_TYPES`, so the rule the
    generator is given and the rule it is judged by cannot disagree.
    """
    forbidden = ", ".join(sorted(set(NARROW_TYPES)))
    return "\n".join([
        "WRITING RULES (a candidate breaking any of these is refused before "
        "it is ever timed)",
        "",
        "1. Every intermediate is computed in float or wider. These types may "
        "name storage and may appear in a cast, and may never name a value "
        f"you compute with: {forbidden}.",
        "2. Load narrow data through a cast, not into a narrow variable: "
        "`float4 v = float4(*(device const half4*)p);` is accepted, "
        "`half4 v = *(device const half4*)p;` is refused.",
        "3. Round once, at the store: `out[i] = (half)acc;` where acc is float.",
        "4. Threadgroup arrays may be narrow, because they hold storage that "
        "is already at that width.",
    ])


def attempts(store: CandidateStore, *, limit: int = 20) -> str:
    """What has already been tried, newest last, from the typed records only."""
    candidates = store.candidates()
    census = store.census()
    shown = candidates[-limit:] if limit else candidates
    if not shown and not census:
        return "WHAT HAS BEEN TRIED\n\nNothing yet: this is the first round."
    if not shown:
        # Every call so far misfired before producing a candidate. The counts
        # are still feedback: a model told nothing repeats the same failure.
        lines = ["WHAT HAS BEEN TRIED", "",
                 "No candidate has been produced yet; the census below counts "
                 "the calls that misfired.", "", "CENSUS", ""]
        for outcome, count in sorted(census.items()):
            lines.append(f"- {outcome}: {count}")
        return "\n".join(lines)

    lines = ["WHAT HAS BEEN TRIED", ""]
    if len(shown) < len(candidates):
        lines.append(f"({len(candidates) - len(shown)} earlier attempts not "
                     f"shown; the census below counts all of them.)")
        lines.append("")
    for candidate in shown:
        record = store.get(candidate)
        detail = record.events[-1]["detail"] if record.events else ""
        line = f"- {candidate[:12]} ({record.origin}): {record.outcome}"
        lines.append(f"{line} - {detail}" if detail else line)

    lines += ["", "CENSUS", ""]
    for outcome, count in sorted(census.items()):
        lines.append(f"- {outcome}: {count}")
    return "\n".join(lines)


def build(operation: Operation, store: CandidateStore, *,
          limit: int = 20) -> str:
    """The whole prompt: the ask, the rules, and what the loop already knows."""
    sections = [
        f"TASK: write a Metal kernel for {operation.name}.",
        "",
        f"GOAL\n\n{operation.goal}",
        "",
        f"SIGNATURE\n\n{operation.signature}",
    ]
    if operation.constraints:
        sections += ["", "CONSTRAINTS", ""]
        sections += [f"- {c}" for c in operation.constraints]
    sections += ["", writing_rules(), "", attempts(store, limit=limit)]
    return "\n".join(sections)
