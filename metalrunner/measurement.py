"""The one installer that routes kept kernels, and the evidence it leaves.

Two callers share this module and that is its reason to exist. The
end-to-end measurement harness (bench/train_lora_e2e.py) imports it by the
dotted name in its run plan and demands a protocol: install the kept
kernels, uninstall them afterwards, and answer for what actually happened.
The user entry point (metalrunner.lora) routes through the same installer,
so the harness provably measures the code a user runs rather than a
measurement-only twin of it.

The module splits into three kinds of thing, kept deliberately apart:

  data      routing.CERTIFIED entries, written by the keep stage, six flat
            JSON-safe keys, validated here by `entry_problems`
  registry  OPERATIONS, one row of executable pieces per shipped operation,
            keyed by the closed set of KNOWN_OPERATIONS names, EMPTY today
  machinery install / Installed / verify_backward_exact, complete now and
            never rewritten when a kernel ships

Shipping a kernel is therefore one CERTIFIED entry plus one OPERATIONS row.
Callables never enter the data table: a dotted path in data is
stringly-typed code whose typo surfaces mid-run on the GPU machine, while a
registry keyed by three known names is checkable at import.

This module never imports mlx. The registry rows import it inside their own
callables when they exist, which is what lets every refusal path here run
on a machine with no device and no model.

A refusal is `MeasurementRefusal` and it means nothing was changed: a
partial install rolls itself back before raising. The one verdict that is
NOT a refusal is a backward comparison coming back unequal - that is the
measurement, reported as `exact: False` for the harness to rule on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from metalrunner import routing, seams

ENTRY_KEYS = (
    "candidate_sha256",
    "operation",
    "chip",
    "bits",
    "group_size",
    "pricing_recording_sha256",
)

OPERATION_NAMES = frozenset(name for name, _ in routing.KNOWN_OPERATIONS)


class MeasurementRefusal(RuntimeError):
    """A permanent reason this process must not be measured.

    Nothing was changed. The harness maps this one type to its permanent
    precondition exit, so a bad plan is refused rather than retried.
    """


@dataclass(frozen=True)
class Operation:
    """The executable half of a certified kernel, keyed by operation name.

    `replace(entry, original)` builds the routed callable and must verify
    that the kernel source it loaded hashes to entry["candidate_sha256"]
    before returning: refusal over a plausible wrong kernel.
    `backward_cases(entry, model_path)` returns data-only case dicts
    (shapes, dtypes, seeds), so the committed cases_sha256 binds them.
    `backward_check(entry, case, model_path)` compares the candidate's
    backward to stock's gradient by exact array equality, per section 8.4
    of the pre-registration, and returns a bool.
    """

    seams: tuple[seams.Seam, ...]
    replace: Callable
    backward_cases: Callable
    backward_check: Callable


# One row per shipped operation. Empty until the keep stage keeps a kernel;
# the registry test pins that its keys stay inside KNOWN_OPERATIONS.
OPERATIONS: dict[str, Operation] = {}


def entry_problems(entry: Mapping) -> list[str]:
    """Every way a CERTIFIED entry fails its schema, in plain words.

    This is the keep stage's write contract: its own tests call this on
    what it emits, so the writer and the reader cannot drift apart.
    """
    problems = []
    keys = set(entry)
    if keys != set(ENTRY_KEYS):
        problems.append(
            f"entry keys differ: missing={sorted(set(ENTRY_KEYS) - keys)}, "
            f"extra={sorted(keys - set(ENTRY_KEYS))}")
        return problems
    for field in ("candidate_sha256", "pricing_recording_sha256"):
        value = entry[field]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            problems.append(f"{field} is not a lowercase sha256 hex digest")
    if entry["operation"] not in OPERATION_NAMES:
        problems.append(
            f"operation {entry['operation']!r} is not one of the known "
            "training operations")
    if not isinstance(entry["chip"], str) or not entry["chip"]:
        problems.append("chip is not a non-empty string")
    for field in ("bits", "group_size"):
        value = entry[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            problems.append(f"{field} is not a positive integer")
    return problems


def tree_sha256(root: Path) -> str:
    """The wrapper fingerprint, one algorithm for every reader.

    The harness preflight, the harness child and this module's evidence all
    hash the metalrunner tree, and they must agree or every run refuses at
    the evidence gate. The harness's call sites import THIS function, so
    agreement is by identity rather than by keeping two copies in step.
    Interpreter caches and dotfiles are excluded because the child's own
    imports write .pyc files between the preflight and the evidence.
    """
    selected = sorted(
        path for path in Path(root).rglob("*")
        if (path.is_file() and "__pycache__" not in path.parts
            and path.suffix != ".pyc" and not path.name.startswith("."))
    )
    if not selected:
        raise MeasurementRefusal(f"nothing hashable under {root}")
    digest = hashlib.sha256()
    for path in selected:
        relative = str(path.relative_to(root)).encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _wrapper_sha256() -> str:
    return tree_sha256(Path(__file__).resolve().parent)


def _resolve(candidate_sha256s: Sequence[str], certified, operations,
             on_chip: str | None) -> list[tuple[Mapping, Operation]]:
    """Each candidate's entry and registry row, or the refusal that stops
    both arms the same way.

    Resolution is identical for ours and control on purpose: a control that
    skipped it could complete a round whose ours arm refuses, wasting the
    round and blurring what the control attests.
    """
    candidates = list(candidate_sha256s)
    if len(set(candidates)) != len(candidates):
        raise MeasurementRefusal("duplicate candidate in the kept set")
    entries = {}
    for entry in certified:
        if isinstance(entry, Mapping) and "candidate_sha256" in entry:
            entries[entry["candidate_sha256"]] = entry

    chip = None
    resolved: list[tuple[Mapping, Operation]] = []
    operations_seen: set[str] = set()
    for candidate in candidates:
        entry = entries.get(candidate)
        if entry is None:
            raise MeasurementRefusal(
                f"candidate {str(candidate)[:12]} has no certified entry: "
                "nothing may be routed that the keep stage did not keep")
        problems = entry_problems(entry)
        if problems:
            raise MeasurementRefusal(
                f"certified entry for {candidate[:12]} fails its schema: "
                + "; ".join(problems))
        if chip is None:
            # Read only now, so a run with nothing to route never needs a
            # device and an early refusal never touches mlx.
            chip = routing.chip() if on_chip is None else on_chip
        if entry["chip"] != chip:
            raise MeasurementRefusal(
                f"candidate {candidate[:12]} was certified on "
                f"{entry['chip']!r} and this is {chip!r}: a kernel is not "
                "routed onto a chip it was never priced on")
        operation = entry["operation"]
        if operation in operations_seen:
            raise MeasurementRefusal(
                f"two kept candidates both claim {operation!r}: one "
                "operation routes one kernel")
        operations_seen.add(operation)
        row = operations.get(operation)
        if row is None:
            raise MeasurementRefusal(
                f"no installer is registered for {operation!r}: the entry "
                "is data, and the code half of it has not shipped")
        resolved.append((entry, row))
    return resolved


class Installed:
    """The active installation: what is routed, counted, and answered for."""

    def __init__(self, *, candidates: Sequence[str], forced_stock: bool,
                 installation: seams.Installation, wrapper_sha256: str):
        self._candidates = sorted(candidates)
        self._forced_stock = forced_stock
        self._installation = installation
        self._wrapper_sha256 = wrapper_sha256
        self._installed = True
        self._decisions = 0
        self._routed_calls = 0
        self._routed: set[str] = set()

    def uninstall(self) -> None:
        self._installation.remove()
        self._installed = False

    def evidence(self) -> dict:
        """A pure read that never raises.

        This is called after training inside a measurement child; an
        exception here would surface as a crash and burn a detached-runner
        retry on a run that actually completed.
        """
        return {
            "wrapper_installed": self._installed,
            "forced_stock": self._forced_stock,
            "wrapper_sha256": self._wrapper_sha256,
            # A real list: the harness's fairness gate isinstance-checks it.
            "routed_candidates": sorted(self._routed),
            "routed_calls": self._routed_calls,
            "routing_decisions": self._decisions,
            "candidate_sha256s": list(self._candidates),
            "seam_calls": dict(self._installation.counts),
            "foreign_on_removal": list(self._installation.foreign_on_removal),
        }


def install(*, candidate_sha256s: Sequence[str],
            force_stock: bool | None = None,
            guard=None,
            certified: Sequence[Mapping] | None = None,
            operations: Mapping[str, Operation] | None = None,
            on_chip: str | None = None) -> Installed:
    """Route the kept kernels, or change nothing and say why not.

    The harness passes only the three demanded keywords and gets the real
    tables; the keyword-only injection parameters are the test seam, in the
    same style the harness itself uses. `force_stock=None` reads the
    control environment variable, so the user path's control mode and the
    harness's explicit control arm are the same switch.

    Under force-stock the replacement is NEVER built: the control arm pays
    the same interception, counting and guard arithmetic per call and none
    of the kernel's construction cost, which is exactly the wrapper-own-cost
    that outcome O4 measures.
    """
    forced = routing.forced_to_stock() if force_stock is None else force_stock
    certified = routing.CERTIFIED if certified is None else certified
    operations = OPERATIONS if operations is None else operations

    resolved = _resolve(candidate_sha256s, certified, operations, on_chip)
    installation = seams.Installation()
    installed = Installed(candidates=candidate_sha256s, forced_stock=forced,
                          installation=installation,
                          wrapper_sha256=_wrapper_sha256())

    def wrapping(entry: Mapping, row: Operation):
        def wrap(original):
            replacement = (None if forced
                           else row.replace(entry, original))
            candidate = entry["candidate_sha256"]

            def routed(*args, **kwargs):
                # Every call through a seam is a routing decision, in both
                # arms; only the ours arm follows it with a routed call.
                installed._decisions += 1
                count = installed._decisions
                if guard is not None and count & (count - 1) == 0:
                    # Exponentially spaced: a footprint syscall per call
                    # would sit inside the timed region of the primary
                    # metric, and a leak grows monotonically, so powers of
                    # two bound detection to 2x the calls that started it.
                    guard.check(f"measurement decision {count}")
                if replacement is None:
                    return original(*args, **kwargs)
                installed._routed_calls += 1
                installed._routed.add(candidate)
                return replacement(*args, **kwargs)

            return routed
        return wrap

    try:
        for entry, row in resolved:
            for seam in row.seams:
                installation.install(seam, wrapping(entry, row))
    except seams.SeamRefusal as refusal:
        # A seam that is not what it should be refuses the whole install,
        # and everything already installed comes back out first.
        installation.remove()
        raise MeasurementRefusal(
            f"a seam refused installation: {refusal}") from refusal

    if guard is not None:
        # A kernel whose construction alone blows the footprint refuses
        # here, before any training starts.
        guard.check("measurement install")
    return installed


def verify_backward_exact(*, model_path: str,
                          candidate_sha256s: Sequence[str],
                          guard=None,
                          certified: Sequence[Mapping] | None = None,
                          operations: Mapping[str, Operation] | None = None,
                          on_chip: str | None = None) -> dict:
    """Every kept kernel's backward against stock's gradient, exactly.

    Section 8.4: exact array equality on fixed cases, before anything is
    timed. The cases come from the registry row, so a kernel ships with its
    fixed cases in the same commit or refuses; cases_sha256 binds the full
    spec, so a changed shape, seed or dtype changes the committed record.

    An unequal comparison is NOT a refusal. `exact: False` is the
    measurement, and the harness is the judge that stops the run on it.
    """
    certified = routing.CERTIFIED if certified is None else certified
    operations = OPERATIONS if operations is None else operations
    resolved = _resolve(candidate_sha256s, certified, operations, on_chip)

    spec: dict[str, dict] = {}
    per_candidate: dict[str, dict] = {}
    exact = True
    total_cases = 0
    for entry, row in resolved:
        candidate = entry["candidate_sha256"]
        cases = tuple(row.backward_cases(entry, model_path))
        if not cases:
            raise MeasurementRefusal(
                f"the backward case builder for {entry['operation']!r} "
                "returned no cases: a caseless verifier verified nothing")
        candidate_exact = True
        for number, case in enumerate(cases, 1):
            if guard is not None:
                guard.check(f"backward case {number} of {candidate[:12]}")
            verdict = row.backward_check(entry, case, model_path)
            if not isinstance(verdict, bool):
                raise MeasurementRefusal(
                    f"the backward check for {entry['operation']!r} "
                    f"returned {type(verdict).__name__}, not a bool: a "
                    "plausible truthy is not a verdict")
            candidate_exact = candidate_exact and verdict
        spec[candidate] = {"operation": entry["operation"],
                           "cases": [dict(case) for case in cases]}
        per_candidate[candidate] = {"operation": entry["operation"],
                                    "cases": len(cases),
                                    "exact": candidate_exact}
        exact = exact and candidate_exact
        total_cases += len(cases)

    cases_sha256 = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "candidate_sha256s": sorted(spec),
        "cases": total_cases,
        "cases_sha256": cases_sha256,
        "exact": exact,
        "per_candidate": per_candidate,
    }
