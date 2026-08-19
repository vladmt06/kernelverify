"""The bridge between the wrapper and the measurement, and what it refuses.

One installer serves two callers: the end-to-end harness, which imports it
by the dotted name in its run plan, and the user entry point. The tests
that matter most are the ones proving the two halves actually meet, and
they run with no kernel, no model and no Metal device, because the bridge
itself never imports mlx: only a shipped operation's own callables do.

The failure this exists to make impossible is an "ours" arm that installed
something and routed nothing, which would read as a kernel result while
being stock. So the counting is pinned from both directions: the bridge
reports it, and the harness's own fairness gate is driven here with the
bridge's real evidence to prove it would catch a hollow one.

Nothing in this file patches mlx-lm. The seams point at a module built for
the purpose, so a failing test cannot leave the trainer modified.
"""

import sys
import types

import pytest

from metalrunner import measurement, routing, seams
from metalrunner.measurement import (
    MeasurementRefusal,
    Operation,
    entry_problems,
    install,
    verify_backward_exact,
)

_REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

SHA_A = "a" * 64
SHA_B = "b" * 64
RECORDING = "c" * 64
CHIP = "Apple M3 Pro"
OPERATION_A, OPERATION_B = (name for name, _ in routing.KNOWN_OPERATIONS[:2])


def _entry(candidate=SHA_A, *, operation=OPERATION_A, chip=CHIP):
    return {
        "candidate_sha256": candidate,
        "operation": operation,
        "chip": chip,
        "bits": 4,
        "group_size": 64,
        "pricing_recording_sha256": RECORDING,
    }


@pytest.fixture()
def target(monkeypatch):
    """A throwaway module standing in for the trainer's own seam."""
    module = types.ModuleType("metalrunner_measurement_target")

    def step(value):
        return f"stock {value}"

    step.__module__ = module.__name__
    module.step = step
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.fixture()
def row(target):
    """An operation row whose replacement is observable and cheap."""
    def replace(entry, original):
        return lambda value: f"ours {value}"

    return Operation(
        seams=(seams.Seam(target.__name__, "step"),),
        replace=replace,
        backward_cases=lambda entry, model_path: (
            {"shape": [4, 8], "dtype": "float32", "seed": 1},
            {"shape": [8, 8], "dtype": "float32", "seed": 2},
        ),
        backward_check=lambda entry, case, model_path: True,
    )


class RecordingGuard:
    def __init__(self):
        self.labels = []

    def check(self, label):
        self.labels.append(label)


def _tables(row, entry=None):
    entry = _entry() if entry is None else entry
    return {"certified": [entry],
            "operations": {entry["operation"]: row},
            "on_chip": CHIP}


# ---------------------------------------------------------------------------
# Routing, and the control that decides without routing
# ---------------------------------------------------------------------------
def test_ours_calls_reach_the_replacement_and_are_counted(target, row):
    installed = install(candidate_sha256s=[SHA_A], force_stock=False,
                        **_tables(row))
    assert target.step("x") == "ours x"
    assert target.step("y") == "ours y"

    evidence = installed.evidence()
    assert evidence["routed_calls"] == evidence["routing_decisions"] == 2
    assert evidence["routed_candidates"] == [SHA_A]
    assert evidence["forced_stock"] is False
    assert evidence["wrapper_installed"] is True


def test_control_decides_without_routing(target, row):
    """The control arm must observe decisions while routing nothing, or the
    harness cannot tell it apart from an arm that was never installed."""
    installed = install(candidate_sha256s=[SHA_A], force_stock=True,
                        **_tables(row))
    assert target.step("x") == "stock x", "control must fall through to stock"
    assert target.step("y") == "stock y"

    evidence = installed.evidence()
    assert evidence["routing_decisions"] == 2
    assert evidence["routed_calls"] == 0
    assert evidence["routed_candidates"] == []
    assert evidence["forced_stock"] is True


def test_the_replacement_is_never_built_under_force_stock(target):
    """O4 reads the control as the wrapper's own host cost, so the control
    must not pay for constructing a kernel it will never call."""
    def explode(entry, original):
        raise AssertionError("the kernel was built for the control arm")

    row = Operation(seams=(seams.Seam(target.__name__, "step"),),
                    replace=explode,
                    backward_cases=lambda e, m: ({"seed": 1},),
                    backward_check=lambda e, c, m: True)
    installed = install(candidate_sha256s=[SHA_A], force_stock=True,
                        **_tables(row))
    assert target.step("x") == "stock x"
    assert installed.evidence()["routing_decisions"] == 1


def test_force_stock_defaults_to_the_environment(target, row, monkeypatch):
    """The user path's control switch and the harness's control arm are the
    same switch, so a user can measure the wrapper's own cost too."""
    monkeypatch.setenv(routing.FORCE_STOCK_ENV, "1")
    installed = install(candidate_sha256s=[SHA_A], **_tables(row))
    assert target.step("x") == "stock x"
    assert installed.evidence()["forced_stock"] is True


def test_uninstall_restores_and_the_counts_survive(target, row):
    """The receipt and the record are written after the seams come out."""
    original = target.step
    installed = install(candidate_sha256s=[SHA_A], force_stock=False,
                        **_tables(row))
    target.step("x")
    installed.uninstall()

    assert target.step is original
    evidence = installed.evidence()
    assert evidence["wrapper_installed"] is False
    assert evidence["routed_calls"] == 1
    assert evidence["foreign_on_removal"] == []


def test_zero_candidates_install_nothing_and_say_so(row):
    """The user path's everyday state today: nothing is certified, so
    nothing routes, and the evidence says that rather than being absent."""
    installed = install(candidate_sha256s=[], force_stock=False,
                        certified=[], operations={},
                        on_chip=None)  # never read: no candidate to place
    evidence = installed.evidence()
    assert evidence["wrapper_installed"] is True
    assert evidence["routed_candidates"] == []
    assert evidence["routed_calls"] == evidence["routing_decisions"] == 0


# ---------------------------------------------------------------------------
# Refusals: a permanent reason, and nothing changed
# ---------------------------------------------------------------------------
def test_an_unkept_candidate_refuses_and_changes_nothing(target, row):
    original = target.step
    with pytest.raises(MeasurementRefusal, match="no certified entry"):
        install(candidate_sha256s=[SHA_B], force_stock=False, **_tables(row))
    assert target.step is original


def test_a_wrong_chip_entry_refuses_and_changes_nothing(target, row):
    original = target.step
    tables = _tables(row)
    tables["on_chip"] = "Apple M1"
    with pytest.raises(MeasurementRefusal, match="never priced on"):
        install(candidate_sha256s=[SHA_A], force_stock=False, **tables)
    assert target.step is original


def test_an_operation_without_an_installer_row_refuses(target, row):
    """An entry is data; without its code half there is nothing to install,
    and pretending otherwise would route nothing while reporting success."""
    with pytest.raises(MeasurementRefusal, match="no installer is registered"):
        install(candidate_sha256s=[SHA_A], force_stock=False,
                certified=[_entry()], operations={}, on_chip=CHIP)


def test_two_candidates_on_one_operation_refuse(target, row):
    entries = [_entry(SHA_A), _entry(SHA_B)]
    with pytest.raises(MeasurementRefusal, match="one operation routes one"):
        install(candidate_sha256s=[SHA_A, SHA_B], force_stock=False,
                certified=entries, operations={OPERATION_A: row},
                on_chip=CHIP)


def test_a_schema_invalid_entry_refuses(target, row):
    broken = _entry()
    broken["bits"] = "four"
    with pytest.raises(MeasurementRefusal, match="fails its schema"):
        install(candidate_sha256s=[SHA_A], force_stock=False,
                certified=[broken], operations={OPERATION_A: row},
                on_chip=CHIP)


def test_a_foreign_object_at_the_seam_refuses_as_a_measurement_refusal(
        target, row):
    """The seam installer's refusal is re-raised as this module's type, so
    the harness maps it to a permanent precondition rather than a crash."""
    def interloper(value):
        return "theirs"

    interloper.__module__ = "some_other_package"
    target.step = interloper

    with pytest.raises(MeasurementRefusal, match="a seam refused") as caught:
        install(candidate_sha256s=[SHA_A], force_stock=False, **_tables(row))
    assert isinstance(caught.value.__cause__, seams.SeamRefusal)
    assert target.step is interloper


def test_a_partial_install_rolls_itself_back(monkeypatch):
    """Two operations, the second unusable: the first must not be left
    installed. A half-installed process is one nobody wrote down."""
    module = types.ModuleType("metalrunner_measurement_partial")

    def first(value):
        return "stock first"

    def second(value):
        return "stock second"

    first.__module__ = second.__module__ = module.__name__
    module.first, module.second = first, second
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def build(attribute):
        return Operation(
            seams=(seams.Seam(module.__name__, attribute),),
            replace=lambda entry, original: (lambda value: "ours"),
            backward_cases=lambda e, m: ({"seed": 1},),
            backward_check=lambda e, c, m: True)

    # The second operation's seam already holds someone else's object.
    def interloper(value):
        return "theirs"

    interloper.__module__ = "some_other_package"
    module.second = interloper

    entries = [_entry(SHA_A, operation=OPERATION_A),
               _entry(SHA_B, operation=OPERATION_B)]
    with pytest.raises(MeasurementRefusal):
        install(candidate_sha256s=[SHA_A, SHA_B], force_stock=False,
                certified=entries,
                operations={OPERATION_A: build("first"),
                            OPERATION_B: build("second")},
                on_chip=CHIP)

    assert module.first is first, "the first seam must have come back out"
    assert module.second is interloper


@pytest.mark.parametrize(("mutate", "expected"), [
    (lambda e: e.pop("bits"), "entry keys differ"),
    (lambda e: e.update(extra=1), "entry keys differ"),
    (lambda e: e.update(candidate_sha256="short"), "sha256 hex digest"),
    (lambda e: e.update(candidate_sha256="A" * 64), "sha256 hex digest"),
    (lambda e: e.update(operation="something else"), "not one of the known"),
    (lambda e: e.update(chip=""), "non-empty string"),
    (lambda e: e.update(bits=0), "positive integer"),
    (lambda e: e.update(group_size=True), "positive integer"),
])
def test_entry_problems_names_every_schema_fault(mutate, expected):
    """The keep stage writes these entries and calls this to check itself,
    so the writer and the reader cannot drift apart."""
    entry = _entry()
    mutate(entry)
    problems = entry_problems(entry)
    assert any(expected in problem for problem in problems), problems


def test_a_good_entry_has_no_problems():
    assert entry_problems(_entry()) == []


# ---------------------------------------------------------------------------
# The guard: often enough to catch a leak, rare enough to stay out of the clock
# ---------------------------------------------------------------------------
def test_the_guard_is_checked_at_install_and_at_exponential_counts(target, row):
    guard = RecordingGuard()
    install(candidate_sha256s=[SHA_A], force_stock=False, guard=guard,
            **_tables(row))
    assert guard.labels == ["measurement install"]

    for _ in range(10):
        target.step("x")
    decisions = [label for label in guard.labels if "decision" in label]
    assert decisions == [f"measurement decision {n}" for n in (1, 2, 4, 8)]


def test_a_missing_guard_is_not_an_error(target, row):
    """The user path passes no guard: it is measuring nothing."""
    install(candidate_sha256s=[SHA_A], force_stock=False, guard=None,
            **_tables(row))
    assert target.step("x") == "ours x"


# ---------------------------------------------------------------------------
# The backward gate: reported, never ruled on here
# ---------------------------------------------------------------------------
def test_verify_backward_binds_its_cases_by_hash_and_counts_them(row):
    import hashlib
    import json

    result = verify_backward_exact(model_path="/pinned/model",
                                   candidate_sha256s=[SHA_A], **_tables(row))
    assert result["candidate_sha256s"] == [SHA_A]
    assert result["cases"] == 2
    assert result["exact"] is True

    spec = {SHA_A: {"operation": OPERATION_A,
                    "cases": [dict(case) for case in
                              row.backward_cases(_entry(), "/pinned/model")]}}
    expected = hashlib.sha256(json.dumps(
        spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert result["cases_sha256"] == expected


def test_one_false_case_makes_exact_false_rather_than_refusing(target):
    """The bridge reports; the harness rules. A false here IS the
    measurement section 8.4 asked for, not a failure to measure."""
    verdicts = iter([True, False])
    row = Operation(seams=(seams.Seam(target.__name__, "step"),),
                    replace=lambda e, o: o,
                    backward_cases=lambda e, m: ({"seed": 1}, {"seed": 2}),
                    backward_check=lambda e, c, m: next(verdicts))
    result = verify_backward_exact(model_path="/pinned/model",
                                   candidate_sha256s=[SHA_A], **_tables(row))
    assert result["exact"] is False
    assert result["cases"] == 2
    assert result["per_candidate"][SHA_A]["exact"] is False


def test_importing_the_bridge_pulls_in_no_mlx():
    """The property every refusal test here rests on, asserted against a
    fresh interpreter rather than assumed. Only a shipped operation's own
    callables may import mlx; if the bridge did, none of these tests could
    run on a machine with no device, and neither could a user's refusal."""
    import subprocess

    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys, metalrunner.measurement as m; "
         "print(sorted(n for n in sys.modules if n.split('.')[0] == 'mlx'))"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "[]", probe.stdout


def test_verify_backward_refuses_an_unknown_candidate_before_any_row_runs(row):
    """Resolution happens before any row callable runs, so this refusal is
    reachable on a machine with no device: the point of the split."""
    with pytest.raises(MeasurementRefusal, match="no certified entry"):
        verify_backward_exact(model_path="/pinned/model",
                              candidate_sha256s=[SHA_B], **_tables(row))


def test_a_row_with_no_cases_refuses(target):
    row = Operation(seams=(seams.Seam(target.__name__, "step"),),
                    replace=lambda e, o: o,
                    backward_cases=lambda e, m: (),
                    backward_check=lambda e, c, m: True)
    with pytest.raises(MeasurementRefusal, match="verified nothing"):
        verify_backward_exact(model_path="/pinned/model",
                              candidate_sha256s=[SHA_A], **_tables(row))


def test_a_truthy_non_bool_verdict_refuses(target):
    """A plausible truthy is not a verdict about a gradient."""
    row = Operation(seams=(seams.Seam(target.__name__, "step"),),
                    replace=lambda e, o: o,
                    backward_cases=lambda e, m: ({"seed": 1},),
                    backward_check=lambda e, c, m: 1)
    with pytest.raises(MeasurementRefusal, match="not a bool"):
        verify_backward_exact(model_path="/pinned/model",
                              candidate_sha256s=[SHA_A], **_tables(row))


@pytest.mark.parametrize("changed", [
    {"shape": [4, 9], "dtype": "float32", "seed": 1},
    {"shape": [4, 8], "dtype": "float16", "seed": 1},
    {"shape": [4, 8], "dtype": "float32", "seed": 99},
])
def test_cases_sha256_moves_when_a_shape_dtype_or_seed_moves(target, changed):
    def build(cases):
        return Operation(seams=(seams.Seam(target.__name__, "step"),),
                         replace=lambda e, o: o,
                         backward_cases=lambda e, m: cases,
                         backward_check=lambda e, c, m: True)

    base = ({"shape": [4, 8], "dtype": "float32", "seed": 1},)
    first = verify_backward_exact(model_path="/m", candidate_sha256s=[SHA_A],
                                  **_tables(build(base)))
    second = verify_backward_exact(model_path="/m", candidate_sha256s=[SHA_A],
                                   **_tables(build((changed,))))
    assert first["cases_sha256"] != second["cases_sha256"]


# ---------------------------------------------------------------------------
# The registry is empty, and its keys are the closed set
# ---------------------------------------------------------------------------
def test_no_operation_has_shipped_yet():
    """Honest state: nothing is kept, so nothing routes. When this changes
    it changes deliberately, together with a CERTIFIED entry."""
    assert measurement.OPERATIONS == {}
    assert routing.CERTIFIED == ()


def test_every_registered_operation_is_a_known_one():
    assert set(measurement.OPERATIONS) <= measurement.OPERATION_NAMES
