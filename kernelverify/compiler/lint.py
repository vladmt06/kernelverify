"""Read a candidate's Metal source and say whether it can be in contract.

Clause C1 of the tolerance contract is the one that keeps a precision fault
a fault: every intermediate is computed in binary32 or wider, inputs are
upcast from storage once, and the result is rounded to storage once at the
end. A kernel that accumulates in `half` is out of contract by construction,
and no measured tolerance can legitimise it, so it must never reach a gate
that would hand it a tolerance to hide behind.

Nothing else in the funnel can see this. The tolerance gate compares numbers
and a half accumulator on a friendly case produces numbers that pass; the
tolerance-free gates ask about NaNs and determinism and a half accumulator
is perfectly deterministic. The only place the violation is visible is the
source, which is why this is an attestation of the text rather than a test
of behaviour.

The rule, in four mechanical parts. A narrow float type is admissible where
it names storage that already exists at that width, and nowhere else:

    device const half* x        pointer to storage           allowed
    threadgroup half tile[64]   an array, storage again      allowed
    half(v) and (half)v         a cast, which rounds once    allowed
    half acc = 0.0h             a value being computed with  REFUSED
    void f(half x)              a narrow value, in a signature REFUSED
    simdgroup_matrix<half,8,8>  an accumulator, narrow       REFUSED

Both cast spellings are allowed because a cast converts one value once,
which is what C1's final rounding is. Declarations are where accumulators
live, and declarations are what this catches. The gap that leaves is a
deliberate round trip in the middle of a loop, `float x = (float)(half)v`,
which narrows without ever declaring anything narrow; no source lint sees
that, and it is left to the tolerance gate, where the fp64 reference
disagrees with it. This file claims to attest declarations, not dataflow,
and the census records it under that name.

The `half4 v = load(...)` idiom is refused too, deliberately, even though a
value upcast on the next line is harmless. Proving "upcast immediately"
needs dataflow this does not have, and the cost of the strict reading is one
sentence in the generator's brief: load through a cast, `float4(*(device
const half4*)p)`, which is allowed above. A lint that lets a real half
accumulator through is worth far more than the idiom it costs.

Template types are resolved before the scan, because `T acc` where T is
bound to half is the same violation written differently, and the source that
gets compiled is the substituted one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The narrow float types Metal offers, with their vector and packed spellings.
_BASES = ("half", "bfloat", "bfloat16_t")
NARROW_TYPES = tuple(
    f"{prefix}{base}{suffix}"
    for base in _BASES
    for suffix in ("", "2", "3", "4")
    for prefix in ("", "packed_")
)

_NARROW = re.compile(r"\b(" + "|".join(sorted(NARROW_TYPES, key=len,
                                              reverse=True)) + r")\b")
_POINTER_OR_FUNCTIONAL_CAST = re.compile(r"\s*[*(]")
_ARRAY_DECLARATION = re.compile(r"\s+[A-Za-z_]\w*\s*\[")
_CLOSES_A_CAST = re.compile(r"\s*\)")
_OPENS_A_CAST = re.compile(r"\(\s*$")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")


@dataclass(frozen=True)
class Violation:
    line: int
    token: str
    text: str

    def __str__(self) -> str:
        return (f"line {self.line}: {self.token} names a value, not storage, "
                f"so an intermediate is narrower than binary32 (contract C1): "
                f"{self.text.strip()}")


@dataclass(frozen=True)
class LintReport:
    violations: tuple[Violation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def reason(self) -> str:
        return "; ".join(str(v) for v in self.violations)


def _blank_comments(source: str) -> str:
    """Comments out, newlines kept, so reported line numbers stay true."""
    def keep_newlines(match: re.Match) -> str:
        return "\n" * match.group().count("\n")

    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub(keep_newlines, source))


def _resolve(source: str, types: dict[str, str]) -> str:
    """Bind template type parameters, since the compiled source is the bound
    one and `T acc` with T = half is the same violation as `half acc`."""
    for name, bound in types.items():
        source = re.sub(rf"\b{re.escape(name)}\b", bound, source)
    return source


def lint(source: str, *, types: dict[str, str] | None = None) -> LintReport:
    """Attest clause C1 from the text of one candidate kernel."""
    scanned = _blank_comments(_resolve(source, types or {}))
    lines = scanned.splitlines()

    violations = []
    for match in _NARROW.finditer(scanned):
        tail = scanned[match.end():]
        if _POINTER_OR_FUNCTIONAL_CAST.match(tail) or _ARRAY_DECLARATION.match(tail):
            continue
        if _OPENS_A_CAST.search(scanned[:match.start()]) and _CLOSES_A_CAST.match(tail):
            continue  # the C-style spelling, (half)v, of the same one rounding
        number = scanned.count("\n", 0, match.start()) + 1
        violations.append(Violation(line=number, token=match.group(),
                                    text=lines[number - 1]))
    return LintReport(violations=tuple(violations))
