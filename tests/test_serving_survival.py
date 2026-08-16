"""Memory-truthfulness of the serving calibration: the footprint budget, the
machine-global lock, and the child-process isolation boundary.

Three Jetsam kills on 2026-08-15 (66.7, 69.4 and 39.5 GB Python footprints on
a 36 GB machine) forced this machinery. The budget reads phys_footprint - the
number Jetsam actually kills on: one dead run self-reported 2.4 GB RSS while
dying at 39.5 GB footprint, so RSS survives only as a secondary print and no
gate reads it.
"""

import ast
import inspect
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import calibrate_quant_serving as harness
from calibrate_quant_serving import (
    DEFAULT_BUDGET_GB,
    EXIT_BUDGET_REFUSAL,
    EXIT_CHILD_DEATH,
    EXIT_LOCK_HELD,
    EXIT_LOW_MEMORY,
    EXIT_NO_DEVICE,
    BudgetExceeded,
    BudgetGuard,
    LowMemoryRefusal,
    Orphaned,
    available_memory_gb,
    build_parser,
    machine_ram_gb,
    phys_footprint_gb,
    refuse,
    require_available_memory,
)
from machine_state import MEASUREMENT_LOCK_PATH, MeasurementLock

BENCH_DIR = Path(__file__).resolve().parents[1] / "bench"


# ---------------------------------------------------------------------------
# T1: the footprint reader. It must report THIS process's phys_footprint -
# what Jetsam kills on - not RSS, and not another process's number.
# ---------------------------------------------------------------------------
def test_footprint_reader_reports_this_process_truthfully():
    current, peak = phys_footprint_gb()
    assert 0.0 < current < machine_ram_gb()
    assert peak >= current * 0.99, "lifetime high-water cannot sit under now"
    ballast = np.ones(64 * 1024 * 1024, dtype=np.float64)  # 512 MB, touched
    grown, grown_peak = phys_footprint_gb()
    assert grown - current > 0.4, "the reader must see our own allocation"
    assert grown_peak >= grown * 0.99
    del ballast


def test_machine_ram_reading_is_plausible():
    ram = machine_ram_gb()
    assert 8.0 <= ram <= 1024.0


# ---------------------------------------------------------------------------
# T1b: what the progress lines SAY. The reader being truthful is not enough if
# the narration prints the other number. Three SIGKILLs were narrated by lines
# reporting RSS, which under-read a 52 GB footprint as 8 GB, so the log showed
# an idle-looking run right up to the kill.
# ---------------------------------------------------------------------------
def test_no_progress_line_prints_rss_alone():
    """Every print goes through footprint_line().

    Checked against the syntax tree, not the text. A text scan gets this wrong
    twice over: counting `rss_line()` reads 2 when the change is complete (the
    definition line contains its own name, and footprint_line keeps one call to
    carry RSS as its tail), and grepping lines that hold both `print(` and
    `rss_line()` flags a docstring that QUOTES the old line. The tree answers
    the question actually being asked: does any print statement, anywhere,
    still reach rss_line?
    """
    tree = ast.parse(inspect.getsource(harness))
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                    and inner.func.id == "rss_line"):
                offenders.append(node.lineno)
    assert offenders == [], offenders

    # and the demotion is real, not a rename: rss_line still has exactly one
    # caller, footprint_line, which is what keeps RSS visible as a second number
    callers = [n.lineno for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "rss_line"]
    assert len(callers) == 1, callers


def test_footprint_line_still_carries_rss_second():
    """RSS is not deleted, it is demoted. A line that dropped it would lose the
    comparison that shows the two readings diverging."""
    line = harness.footprint_line()
    assert line.startswith("footprint ")
    assert "rss " in line
    assert line.index("footprint") < line.index("rss")


def test_the_final_summary_reports_the_footprint_peak_not_ru_maxrss():
    """The closing line was labelled `peak footprint:` and printed ru_maxrss -
    a mislabel, and the single most misleading number in the whole log."""
    line = harness.summary_line({"footprint_peak_gb": 22.4, "rss_peak_gb": 2.4})
    assert line.startswith("peak footprint: 22.4")
    assert "rss 2.4" in line
    assert line.index("footprint") < line.index("rss")


# ---------------------------------------------------------------------------
# T1: the budget guard. Refusal, never silent grid shrinking.
# ---------------------------------------------------------------------------
def test_budget_guard_refuses_over_budget_and_names_the_cell():
    guard = BudgetGuard(24.0, reader=lambda: (25.1, 26.0))
    with pytest.raises(BudgetExceeded) as exc:
        guard.check("probe lm_head")
    assert exc.value.cell == "probe lm_head"
    assert exc.value.footprint_gb == 25.1
    assert exc.value.budget_gb == 24.0
    assert "probe lm_head" in str(exc.value)


def test_budget_guard_passes_under_budget():
    guard = BudgetGuard(24.0, reader=lambda: (10.0, 12.0))
    assert guard.check("grid lm_head normal-0.02 s0") == 10.0


# ---------------------------------------------------------------------------
# T1c: the orphan hole. A child outlives a killed parent, keeps the GPU, and
# keeps allocating - while the machine lock the PARENT held has already been
# released by its death, so the next harness starts on top of it. The child
# must notice and stop. getppid() is the signal: when the parent dies the
# child is reparented, so the value it saw at startup stops being true.
# ---------------------------------------------------------------------------
def test_a_child_stops_at_the_next_record_once_its_parent_is_gone():
    guard = BudgetGuard(24.0, reader=lambda: (1.0, 1.0),
                        parent_pid=os.getppid() + 100000)  # never our real parent
    with pytest.raises(Orphaned) as exc:
        guard.check("grid lm_head normal-0.02 s0")
    assert "lm_head" in str(exc.value), "the cell it died on must be named"


def test_a_child_whose_parent_is_alive_keeps_going():
    guard = BudgetGuard(24.0, reader=lambda: (1.0, 1.0), parent_pid=os.getppid())
    assert guard.check("grid lm_head normal-0.02 s0") == 1.0


def test_the_parent_side_guard_has_no_parent_to_lose():
    """The same guard runs in the parent's own in-process path, where an
    orphan check would be meaningless - and would fire on any process whose
    own parent exits, which is every detached run."""
    guard = BudgetGuard(24.0, reader=lambda: (1.0, 1.0))
    assert guard.check("probe lm_head") == 1.0


def test_refusal_exit_codes_are_distinct_and_leave_the_existing_ones_alone():
    codes = {EXIT_BUDGET_REFUSAL, EXIT_LOCK_HELD, EXIT_LOW_MEMORY,
             EXIT_CHILD_DEATH, EXIT_NO_DEVICE}
    assert len(codes) == 5
    assert not codes & {0, 1, 2}, (
        "0 and 1 already mean attested/measured-and-stopped, and 2 is "
        "argparse's own exit - a gate sharing it is unreadable")


# ---------------------------------------------------------------------------
# T1: the refusal path. Checkpoint written, live cell named, distinct code,
# and NO partial results JSON - the output file means a finished run only.
# ---------------------------------------------------------------------------
def _fingerprint(provenance="derived", shapes=(("tiny", (8, 64)),)):
    return harness.checkpoint_fingerprint(provenance, shapes)


# ---------------------------------------------------------------------------
# The checkpoint fingerprint must carry CODE identity, not just the question
# the run is asking. Without it, tonight's run resumed past STEP 0 - the only
# end-to-end bit-exactness gate against a change in device arithmetic - on a
# checkpoint written by code from before the buffer pool existed.
# ---------------------------------------------------------------------------
def test_the_fingerprint_carries_the_identity_of_the_measuring_code():
    fingerprint = _fingerprint()
    assert fingerprint["code"], "a fingerprint with no code identity resumes " \
                                "across an arithmetic change"
    assert fingerprint == _fingerprint(), "the identity must be stable"
    hashed = {p.name for p in harness.CODE_IDENTITY_PATHS}
    assert "device.py" in hashed, "the runner is what dispatches every record"
    assert "calibrate_quant_serving.py" in hashed, "this harness is measuring"
    for path in harness.CODE_IDENTITY_PATHS:
        assert path.exists(), f"{path} is hashed but not there"


def test_a_change_in_the_measuring_code_invalidates_a_resume(tmp_path,
                                                             monkeypatch):
    """The buffer-pool change, in miniature: edit a module the records pass
    through and the checkpoint written before it stops being reusable."""
    source = tmp_path / "device.py"
    source.write_text("EPSILON = 1\n")
    monkeypatch.setattr(harness, "CODE_IDENTITY_PATHS", (source,))
    before = _fingerprint()
    checkpoint = tmp_path / "partial.json"
    harness.write_checkpoint(checkpoint, before,
                             {"continuity": "64 records reproduced exactly"})
    assert harness.read_checkpoint(checkpoint, before)

    source.write_text("EPSILON = 2\n")  # one byte of arithmetic changes
    after = _fingerprint()
    assert after["code"] != before["code"]
    assert harness.read_checkpoint(checkpoint, after) == {}, \
        "STEP 0 must be re-measured under changed code, never resumed past"


def test_continuity_only_measures_step_zero_and_stops(sandboxed_paths, capsys):
    """--continuity-only exists so the anchor can be run ALONE in the
    foreground after a change to the arithmetic path, without spending the
    45 minutes the full grid costs."""
    calls, steps = [], {}

    def spawn(kind, cell, task):
        calls.append(kind)
        return {"ok": True, "detail": "64 records reproduced exactly"}, {}

    args = build_parser().parse_args(["--continuity-only"])
    code = harness._run_measured_steps(
        args, (("tiny", (8, 64)),), spawn, {}, steps, [],
        lambda step, payload: steps.__setitem__(step, payload), "derived", "")
    assert code == 0
    assert calls == ["continuity"], "nothing beyond STEP 0 may be measured"
    assert steps["continuity"] == "64 records reproduced exactly"
    out = capsys.readouterr().out
    assert "STEP 0" in out and "REPRODUCED" in out


def test_continuity_only_reports_a_failed_anchor_as_a_stop(sandboxed_paths,
                                                           capsys):
    def spawn(kind, cell, task):
        return {"ok": False, "detail": "record 3 member device-dequant-loop"}, {}

    args = build_parser().parse_args(["--continuity-only"])
    steps = {}
    code = harness._run_measured_steps(
        args, (("tiny", (8, 64)),), spawn, {}, steps, [],
        lambda step, payload: steps.__setitem__(step, payload), "derived", "")
    assert code == 1
    assert "STOP" in capsys.readouterr().out


@pytest.fixture
def sandboxed_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "CHECKPOINT_PATH", tmp_path / "partial.json")
    monkeypatch.setattr(harness, "OUT_PATH", tmp_path / "adequacy.json")
    return tmp_path


def test_budget_refusal_writes_checkpoint_names_cell_distinct_code_no_results(
        sandboxed_paths, capsys):
    fingerprint = _fingerprint()
    steps = {"continuity": "64 records reproduced exactly"}
    exc = BudgetExceeded("probe lm_head", 25.1, 24.0)
    code = refuse(EXIT_BUDGET_REFUSAL, str(exc), fingerprint, steps)
    assert code == EXIT_BUDGET_REFUSAL
    assert harness.read_checkpoint(sandboxed_paths / "partial.json",
                                   fingerprint) == steps
    assert not (sandboxed_paths / "adequacy.json").exists()
    out = capsys.readouterr().out
    assert "probe lm_head" in out
    assert "REFUS" in out.upper()


def test_budget_refusal_preserves_resume(sandboxed_paths):
    """After an injected refusal, --resume validates the fingerprint and
    reuses the finished steps; a different run identity reuses nothing."""
    fingerprint = _fingerprint()
    steps = {"continuity": "64 records reproduced exactly"}
    refuse(EXIT_BUDGET_REFUSAL, "over budget at cell probe lm_head",
           fingerprint, steps)
    assert harness.read_checkpoint(sandboxed_paths / "partial.json",
                                   fingerprint) == steps
    other = _fingerprint(provenance="pinned-artifact:elsewhere")
    assert harness.read_checkpoint(sandboxed_paths / "partial.json", other) == {}


def test_tiny_budget_invocation_is_a_refusal_loop(sandboxed_paths):
    """A budget under the interpreter's own footprint refuses on the FIRST
    check of every invocation: the checkpoint survives unchanged and no
    forward progress happens. That loop is the designed behavior, not a bug."""
    fingerprint = _fingerprint()
    finished = {"continuity": "64 records reproduced exactly"}
    harness.write_checkpoint(sandboxed_paths / "partial.json", fingerprint,
                             finished)
    for _ in range(2):  # two successive tiny-budget "runs"
        saved = harness.read_checkpoint(sandboxed_paths / "partial.json",
                                        fingerprint)
        assert saved == finished, "resume reuses the finished step each time"
        guard = BudgetGuard(0.001)  # the REAL reader: any python is over this
        with pytest.raises(BudgetExceeded) as exc:
            guard.check("continuity")
        refuse(EXIT_BUDGET_REFUSAL, str(exc.value), fingerprint, saved)
    assert harness.read_checkpoint(sandboxed_paths / "partial.json",
                                   fingerprint) == finished
    assert not (sandboxed_paths / "adequacy.json").exists()


# ---------------------------------------------------------------------------
# T1: the budget hook fires BEFORE any cell work starts. session=None proves
# it: touching the session would explode, so a raise from the guard means the
# cell was never entered.
# ---------------------------------------------------------------------------
def test_probe_checks_the_budget_before_touching_the_cell():
    guard = BudgetGuard(1e-9, reader=lambda: (1.0, 1.0))
    with pytest.raises(BudgetExceeded) as exc:
        harness.probe_batch_regimes([("tiny", (64, 64))], session=None,
                                    verbose=False, guard=guard)
    assert "tiny" in exc.value.cell


def test_grid_checks_the_budget_before_touching_the_iteration():
    guard = BudgetGuard(1e-9, reader=lambda: (1.0, 1.0))
    with pytest.raises(BudgetExceeded) as exc:
        harness.measure_serving([("tiny", (64, 64))], (1,), session=None,
                                verbose=False, guard=guard)
    assert "tiny" in exc.value.cell


# ---------------------------------------------------------------------------
# T1: the budget CLI. Default well under machine RAM, override accepted,
# nonsense rejected.
# ---------------------------------------------------------------------------
def test_budget_cli_default_is_well_under_machine_ram():
    args = build_parser().parse_args([])
    assert args.budget_gb == DEFAULT_BUDGET_GB
    assert 0.0 < DEFAULT_BUDGET_GB <= machine_ram_gb() * 0.75


def test_budget_cli_override_accepted():
    assert build_parser().parse_args(["--budget-gb", "20.5"]).budget_gb == 20.5


@pytest.mark.parametrize("bad", ["nonsense", "-3", "0", "inf", "nan"])
def test_budget_cli_nonsense_rejected(bad, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--budget-gb", bad])
    capsys.readouterr()


def test_probe_cli_flags_parse_and_reject_nonsense(capsys):
    args = build_parser().parse_args(
        ["--attribution-probe", "--probe-shape", "lm_head",
         "--probe-iterations", "3", "--wall-cap-s", "540"])
    assert args.attribution_probe and not args.stability_probe
    assert args.probe_shape == "lm_head"
    assert args.probe_iterations == 3
    assert args.wall_cap_s == 540.0
    for bad in (["--probe-iterations", "0"], ["--probe-shape", "q_proj"],
                ["--wall-cap-s", "-1"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--stability-probe"] + bad)
    capsys.readouterr()


# ---------------------------------------------------------------------------
# T2: the machine-wide measurement lock. It is ONE lock for every heavy
# measurement harness in this repo, not one per harness: the 03:29 collapse
# was the serving calibration and the pricing probe running at once, so a
# per-harness lock would have permitted exactly the event it exists to stop.
# It is fcntl.flock on a fixed path, so the kernel releases it when the holder
# dies - there is no pid to read, no staleness to judge, and no window between
# creating the file and saying who owns it.
# ---------------------------------------------------------------------------
def _acquire_in_child(path: Path, hold_s: float = 0.0) -> subprocess.Popen:
    """A separate process that takes the lock and holds it until killed."""
    script = ("import sys, time; from pathlib import Path;"
              "sys.path.insert(0, sys.argv[2]);"
              "from machine_state import MeasurementLock;"
              "lock = MeasurementLock('probe', Path(sys.argv[1]));"
              "ok, detail = lock.acquire();"
              "print(int(ok), detail, flush=True);"
              "time.sleep(float(sys.argv[3]))")
    return subprocess.Popen(
        [sys.executable, "-c", script, str(path), str(BENCH_DIR), str(hold_s)],
        stdout=subprocess.PIPE, text=True)


def test_the_lock_is_one_machine_wide_lock_every_harness_shares():
    assert MEASUREMENT_LOCK_PATH.is_absolute()
    repo_root = Path(__file__).resolve().parents[1]
    assert not str(MEASUREMENT_LOCK_PATH).startswith(str(repo_root)), \
        "a worktree-keyed lock is blind to a second checkout of a harness"
    assert "calibrate_quant_serving" not in MEASUREMENT_LOCK_PATH.name, \
        "a per-harness lock lets a second harness run: the 03:29 collapse"
    assert harness.MEASUREMENT_LOCK_PATH is MEASUREMENT_LOCK_PATH, \
        "the harness must lock on the shared path, not a copy of its own"


def test_lock_second_acquire_refuses_and_names_the_holder(tmp_path):
    path = tmp_path / "measurement.lock"
    held = MeasurementLock("first", path)
    assert held.acquire()[0]
    try:
        ok, detail = MeasurementLock("second", path).acquire()
        assert not ok
        assert str(os.getpid()) in detail and "first" in detail
    finally:
        held.release()


def test_a_live_foreign_holder_refuses_across_processes(tmp_path):
    """The two-harness collision for real, across a process boundary."""
    path = tmp_path / "measurement.lock"
    child = _acquire_in_child(path, hold_s=30.0)
    try:
        assert child.stdout.readline().startswith("1 "), "the child must hold"
        ok, detail = MeasurementLock("ours", path).acquire()
        assert not ok
        assert str(child.pid) in detail
    finally:
        child.kill()
        child.wait()


def test_a_holder_dying_releases_the_lock_with_nothing_to_reclaim(tmp_path):
    """The kernel is the reclaim path. A SIGKILLed holder - the exact death
    the three Jetsam kills produced - leaves a lock the next run can take,
    with no pid to read and no staleness rule to get wrong."""
    path = tmp_path / "measurement.lock"
    child = _acquire_in_child(path, hold_s=30.0)
    assert child.stdout.readline().startswith("1 ")
    child.kill()
    child.wait()
    ours = MeasurementLock("after the kill", path)
    try:
        assert ours.acquire()[0], "a dead holder's lock must be free"
    finally:
        ours.release()


def test_a_live_holder_mid_write_is_never_read_as_stale(tmp_path):
    """The race the pid file had: it was created with O_EXCL and the pid was
    written a moment later, so a concurrent reader saw an empty file, failed
    to parse it, called a LIVE holder stale and took its lock. Here the
    contents are diagnostics only - the flock decides - so an empty or
    garbage file while a holder lives still refuses."""
    path = tmp_path / "measurement.lock"
    held = MeasurementLock("first", path)
    assert held.acquire()[0]
    try:
        for content in ("", "not-a-pid", "999999999"):
            path.write_text(content)
            ok, detail = MeasurementLock("second", path).acquire()
            assert not ok, f"a live holder was stolen from on content {content!r}"
            assert detail, "a refusal must still say something usable"
    finally:
        held.release()


def test_lock_release_frees_it_for_the_next_run(tmp_path):
    path = tmp_path / "measurement.lock"
    first = MeasurementLock("first", path)
    assert first.acquire()[0]
    first.release()
    second = MeasurementLock("second", path)
    try:
        assert second.acquire()[0]
    finally:
        second.release()


def test_release_without_the_lock_never_frees_a_foreign_holder(tmp_path):
    path = tmp_path / "measurement.lock"
    held = MeasurementLock("first", path)
    assert held.acquire()[0]
    try:
        MeasurementLock("never acquired", path).release()  # must be a no-op
        assert not MeasurementLock("third", path).acquire()[0]
    finally:
        held.release()


def test_only_one_of_many_simultaneous_acquirers_wins(tmp_path):
    """Contention with no holder to serialize them: whatever the interleaving,
    the kernel hands the lock to exactly one."""
    path = tmp_path / "measurement.lock"
    children = [_acquire_in_child(path, hold_s=3.0) for _ in range(6)]
    try:
        verdicts = [c.stdout.readline().startswith("1 ") for c in children]
        assert sum(verdicts) == 1, f"{sum(verdicts)} winners, expected exactly 1"
    finally:
        for child in children:
            child.kill()
            child.wait()


# ---------------------------------------------------------------------------
# T2: the machine-available-memory refusal before each large cell. Pattern
# from bench/machine_state.py: sample, refuse, name the reason - never push
# a doomed allocation into compression and Jetsam.
# ---------------------------------------------------------------------------
def test_available_memory_reading_is_plausible():
    available = available_memory_gb()
    assert 0.0 < available <= machine_ram_gb()


def test_low_available_memory_refuses_before_the_large_cell():
    with pytest.raises(LowMemoryRefusal) as exc:
        require_available_memory(24.0, "probe lm_head 151936x2560",
                                 reader=lambda: 3.5)
    assert exc.value.cell == "probe lm_head 151936x2560"
    assert exc.value.available_gb == 3.5
    assert exc.value.needed_gb == 24.0
    assert "probe lm_head" in str(exc.value)


def test_ample_available_memory_passes():
    assert require_available_memory(24.0, "cell", reader=lambda: 30.0) == 30.0


# ---------------------------------------------------------------------------
# T3: the child serialization boundary. The continuity anchor compares records
# EXACTLY, so every float must cross the parent-child boundary bit for bit -
# shortest-roundtrip finite doubles, and an explicit NaN/Infinity policy.
# ---------------------------------------------------------------------------
def _bits(x: float) -> bytes:
    return struct.pack("<d", x)


TRICKY_FLOATS = [0.1, 2.0 / 3.0, 0.30000000000000004, math.pi, 1.0 + 2.0 ** -52,
                 1e-300, 5e-324, 2.2250738585072014e-308,  # subnormal + min normal
                 1.7976931348623157e308, 0.0, -0.0,
                 float(np.float32(0.1)), float(np.float16(0.1))]


def test_child_records_round_trip_bit_exactly(tmp_path):
    record = {"members": {f"m{i}": v for i, v in enumerate(TRICKY_FLOATS)},
              "np64": np.float64(0.1) * np.float64(3.0),
              "np32": np.float32(3.14159),
              "batch": 16, "proj": "lm_head", "heldout_draw": False}
    harness.write_child_result(tmp_path / "result.json",
                               {"payload": {"records": [record]}})
    back = harness.read_child_result(tmp_path / "result.json")
    got = back["payload"]["records"][0]
    for key, ours in record["members"].items():
        assert _bits(got["members"][key]) == _bits(ours), key
    assert _bits(got["np64"]) == _bits(float(record["np64"]))
    assert _bits(got["np32"]) == _bits(float(record["np32"]))
    assert got["batch"] == 16 and got["proj"] == "lm_head"
    assert got["heldout_draw"] is False


def test_child_nan_and_infinity_policy_is_explicit(tmp_path):
    """err() on an overflowed fp16 fault path can be inf or nan; both must
    survive the boundary rather than crash or silently become null."""
    harness.write_child_result(tmp_path / "result.json", {
        "payload": {"inf": math.inf, "ninf": -math.inf, "nan": math.nan}})
    back = harness.read_child_result(tmp_path / "result.json")["payload"]
    assert back["inf"] == math.inf
    assert back["ninf"] == -math.inf
    assert math.isnan(back["nan"])


def test_child_result_write_is_atomic_and_absent_reads_as_none(tmp_path):
    harness.write_child_result(tmp_path / "result.json", {"payload": 1})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.json"]
    assert harness.read_child_result(tmp_path / "absent.json") is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text('{"payload": ')
    assert harness.read_child_result(corrupt) is None


# ---------------------------------------------------------------------------
# T3: the parent's child runner. A dead child marks its cell and stops the
# run cleanly - no half-recorded cell, no onward measuring.
# ---------------------------------------------------------------------------
@pytest.fixture
def child_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "CHILD_DIR", tmp_path)
    return tmp_path


def _cmd(script: str) -> list:
    return [sys.executable, "-c", script]


_WRITES_RESULT = """
import json, sys
out = sys.argv[sys.argv.index("--child-out") + 1]
json.dump({"payload": {"records": [], "exact": True},
           "footprint_gb": {"current": 2.0, "peak": %s}}, open(out, "w"))
"""


def test_child_success_returns_payload_and_footprint(child_dir):
    payload, meta = harness.spawn_measurement(
        "grid tiny 8x64 normal-0.02 s0", {"kind": "grid"}, budget_gb=24.0,
        command=_cmd(_WRITES_RESULT % "3.5"))
    assert payload == {"records": [], "exact": True}
    assert meta == {"footprint_gb": {"current": 2.0, "peak": 3.5}}


def test_child_peak_over_budget_refuses_onward(child_dir):
    """The parent enforces the budget on the child's reported footprint: a
    child that survived while exceeding it still stops the run."""
    with pytest.raises(BudgetExceeded) as exc:
        harness.spawn_measurement(
            "grid lm_head 151936x2560 normal-0.02 s0", {"kind": "grid"},
            budget_gb=24.0, command=_cmd(_WRITES_RESULT % "31.0"))
    assert exc.value.footprint_gb == 31.0
    assert "lm_head" in exc.value.cell


def test_killed_child_marks_the_cell_and_stops(child_dir):
    with pytest.raises(harness.ChildRefusal) as exc:
        harness.spawn_measurement(
            "grid tiny 8x64 normal-0.02 s0", {"kind": "grid"}, budget_gb=24.0,
            command=_cmd("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"))
    assert exc.value.cell == "grid tiny 8x64 normal-0.02 s0"
    assert exc.value.exit_for_parent == EXIT_CHILD_DEATH
    assert "SIGKILL" in exc.value.reason


def test_child_budget_self_refusal_propagates_as_budget_exit(child_dir):
    with pytest.raises(harness.ChildRefusal) as exc:
        harness.spawn_measurement(
            "probe lm_head 151936x2560", {"kind": "probe"}, budget_gb=24.0,
            command=_cmd(f"raise SystemExit({EXIT_BUDGET_REFUSAL})"))
    assert exc.value.exit_for_parent == EXIT_BUDGET_REFUSAL


def test_child_without_metal_gets_its_own_code(child_dir):
    with pytest.raises(harness.ChildRefusal) as exc:
        harness.spawn_measurement(
            "continuity", {"kind": "continuity"}, budget_gb=24.0,
            command=_cmd(f"raise SystemExit({EXIT_NO_DEVICE})"))
    assert exc.value.exit_for_parent == EXIT_NO_DEVICE
    assert "Metal" in exc.value.reason


def test_a_child_dying_in_argument_parsing_is_a_death_not_a_missing_gpu(
        child_dir):
    """Exit 2 is argparse's own, and import errors reach the parent the same
    way. Reading either as "no usable Metal device" would send the coordinator
    looking at the GPU for a typo."""
    with pytest.raises(harness.ChildRefusal) as exc:
        harness.spawn_measurement(
            "grid tiny 8x64", {"kind": "grid"}, budget_gb=24.0,
            command=[sys.executable, str(BENCH_DIR / "calibrate_quant_serving.py"),
                     "--budget-gb", "not-a-number"])
    assert exc.value.exit_for_parent == EXIT_CHILD_DEATH
    assert "exit 2" in exc.value.reason


def test_child_exit_zero_without_result_is_a_death(child_dir):
    with pytest.raises(harness.ChildRefusal) as exc:
        harness.spawn_measurement("grid tiny", {"kind": "grid"},
                                  budget_gb=24.0, command=_cmd("pass"))
    assert exc.value.exit_for_parent == EXIT_CHILD_DEATH
    assert "no result" in exc.value.reason


def test_the_child_is_spawned_into_its_own_session(child_dir, monkeypatch):
    """start_new_session is what makes the group kill target the child alone.

    Pinned because the two halves are strictly ordered: without the session,
    the child shares the parent's process group and `killpg(child.pid)` would
    resolve to the parent's own group. Dropping this flag would not fail
    loudly - it would make the reaper lethal.
    """
    seen = {}
    real_popen = harness.subprocess.Popen

    def spy(argv, **kwargs):
        seen.update(kwargs)
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(harness.subprocess, "Popen", spy)
    harness.spawn_measurement("cell", {"kind": "noop"}, 24.0,
                              command=_cmd(_WRITES_RESULT % "3.5"))
    assert seen.get("start_new_session") is True


def test_the_reaper_never_signals_the_parents_own_group(monkeypatch):
    """The one line that could kill the whole foreground job.

    os.getpgid(child) would return OUR group if the session flag were ever
    dropped, so the reaper passes the child's own pid and refuses to signal a
    pgid equal to our own. Asserted directly, because the only other way to
    learn it is to lose the session.
    """
    killed = []
    monkeypatch.setattr(harness.os, "killpg",
                        lambda pgid, sig: killed.append(pgid))

    class _Ours:
        pid = os.getpgrp()          # pretend the child leads OUR group
        def poll(self): return None
        def wait(self, timeout=None): return 0

    harness._reap(_Ours())
    assert killed == [], "the reaper signalled the group it is standing in"


def test_a_parent_that_stops_does_not_leave_the_child_running(child_dir,
                                                              monkeypatch):
    """The hole itself. A child outliving its parent keeps the GPU and keeps
    allocating, while the machine lock the parent held was released by its
    death - so the next harness starts on top of it.

    This also pins a behaviour change the Popen switch introduced: `subprocess
    .run` killed its child on timeout, `proc.wait(timeout=)` does not. Without
    the reaper the wall-cap path would leak a live child on every refusal.
    """
    spawned = []
    real_popen = harness.subprocess.Popen

    def spy(argv, **kwargs):
        proc = real_popen(argv, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(harness.subprocess, "Popen", spy)
    with pytest.raises(harness.ChildRefusal):
        harness.spawn_measurement("cell", {"kind": "noop"}, 24.0,
                                  command=_cmd("import time\ntime.sleep(60)\n"),
                                  wall_cap_s=0.5)

    assert len(spawned) == 1
    assert spawned[0].poll() is not None, (
        "the wall-capped child is still running: the refusal returned and "
        "left it holding the machine")


def test_wall_capped_child_is_a_death_naming_the_cap(child_dir):
    with pytest.raises(harness.ChildRefusal) as exc:
        harness.spawn_measurement(
            "probe lm_head", {"kind": "probe"}, budget_gb=24.0,
            command=_cmd("import time; time.sleep(30)"), wall_cap_s=0.5)
    assert exc.value.exit_for_parent == EXIT_CHILD_DEATH
    assert "wall cap" in exc.value.reason


def test_child_refusal_marks_the_cell_in_the_checkpoint(sandboxed_paths, capsys):
    fingerprint = _fingerprint()
    steps = {"continuity": "64 records reproduced exactly"}
    exc = harness.ChildRefusal("grid lm_head 151936x2560 normal-0.02 s0",
                               EXIT_CHILD_DEATH, "signal 9 (SIGKILL)")
    code = harness.refuse_child(exc, fingerprint, steps)
    assert code == EXIT_CHILD_DEATH
    kept = harness.read_checkpoint(sandboxed_paths / "partial.json", fingerprint)
    assert kept["continuity"] == steps["continuity"], "finished steps survive"
    assert kept["refused_cell"]["cell"] == exc.cell
    assert "SIGKILL" in kept["refused_cell"]["reason"]
    assert "lm_head" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# T3, end to end on the real device: a spawned child must reproduce the
# in-process records BIT for BIT across the process + serialization boundary.
# Skipped wholesale without PyObjC Metal, matching the runner tests.
# ---------------------------------------------------------------------------
Metal = pytest.importorskip("Metal")

from kernelverify.runners.device import MetalDevice  # noqa: E402
from kernelverify.schemas.quant_device import DeviceMemberSession  # noqa: E402

try:
    _DEVICE = MetalDevice()
except RuntimeError:
    _DEVICE = None

needs_metal = pytest.mark.skipif(_DEVICE is None,
                                 reason="no Metal device on this machine")


def _assert_bit_identical(ours, theirs, path="records"):
    if isinstance(ours, dict):
        assert isinstance(theirs, dict) and set(ours) == set(theirs), path
        for key in ours:
            _assert_bit_identical(ours[key], theirs[key], f"{path}.{key}")
    elif isinstance(ours, (list, tuple)):
        assert len(ours) == len(theirs), path
        for i, (a, b) in enumerate(zip(ours, theirs)):
            _assert_bit_identical(a, b, f"{path}[{i}]")
    elif isinstance(ours, float):
        assert isinstance(theirs, float), path
        if math.isnan(ours) or math.isnan(theirs):
            assert math.isnan(ours) and math.isnan(theirs), path
        else:
            assert _bits(ours) == _bits(theirs), path
    else:
        assert ours == theirs, path


@needs_metal
def test_device_dispatch_does_not_retain_buffers_across_runs():
    """The convicted holder of the 30-60 GB gap, pinned at unit scale: the
    runner's autoreleased command buffers kept every bound Metal buffer alive
    for the life of the process - measured tonight at +0.546 GB per 96-call
    sweep at kv_proj and +0.83 GB per dispatch at lm_head. N dispatches of a
    ~4 MB case must not grow the footprint by anything near N * case bytes."""
    from kernelverify.schemas.quant_contract import QuantContract, canonical_quantize
    session = DeviceMemberSession(_DEVICE)
    d_out, d_in = 1024, 2048
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((d_out, d_in)) * 0.02).astype(np.float16)
    artefact = canonical_quantize(w, QuantContract(bits=3, group_size=64))
    x = rng.standard_normal((4, d_in)).astype(np.float32)
    session.run("device-dequant-loop", x, artefact, "leak warmup")
    before, _ = phys_footprint_gb()
    n = 24
    for i in range(n):
        session.run("device-dequant-loop", x, artefact, f"leak {i}")
    grown, _ = phys_footprint_gb()
    per_case_gb = (artefact.q.astype(np.float16).nbytes + x.nbytes) / 1e9
    growth = grown - before
    assert growth < n * per_case_gb * 0.5, (
        f"{growth:.3f} GB across {n} dispatches of a {per_case_gb:.4f} GB "
        f"case: per-dispatch buffer retention is back")


@needs_metal
def test_releasing_the_buffer_pool_gives_the_memory_back():
    """The other half of the pool's memory story, at readable scale.

    A written MTLBuffer dropped plainly keeps its pages charged to this
    process for as long as it lives, so a release between specs would cost
    memory rather than save it: three cycles of 0.5 GB would land as +1.5 GB.
    Marking each buffer purgeable-empty first is what makes the number come
    back (+0.00 GB measured for four dropped 0.54 GB buffers), and this test
    is what would catch that step being dropped.
    """
    _DEVICE.release_pool()
    cycle_bytes, indices = 128 * 1024 * 1024, (910, 911, 912, 913)
    before, _ = phys_footprint_gb()
    for _cycle in range(3):
        for index in indices:
            buffer, _held = _DEVICE.pooled_buffer(index, cycle_bytes)
            np.frombuffer(buffer.contents().as_buffer(cycle_bytes),
                          dtype=np.uint8)[...] = 1  # dirty every page
        _DEVICE.release_pool()
    growth, _ = phys_footprint_gb()
    one_cycle_gb = len(indices) * cycle_bytes / 1e9
    assert growth - before < one_cycle_gb, (
        f"{growth - before:.2f} GB kept after three released "
        f"{one_cycle_gb:.2f} GB cycles: released buffers are retained again")


@needs_metal
def test_attribution_probe_measures_stages_and_sweep_delta(tmp_path, capsys):
    """T4's attribution instrument at a tiny shape: every stage snapshotted,
    the device-member-sweep delta recorded per iteration, JSON durable."""
    report = harness.attribution_probe(
        "tiny", 64, 64, iterations=2, guard=BudgetGuard(24.0), deadline=None,
        out_path=tmp_path / "attr.json", probe_batch_max=2)
    stages = [(s["iteration"], s["stage"]) for s in report["stages"]]
    assert stages[0] == (0, "baseline")
    for i in (1, 2):
        assert (i, "built") in stages
        assert (i, "cpu-pairwise") in stages
        assert (i, "mx-heldout") in stages
        assert (i, "freed") in stages
        assert (i, "mx-cache-cleared") in stages
    assert any(s.startswith("device-sweep") for _, s in stages)
    assert len(report["sweep_delta_gb"]) == 2
    assert len(report["iteration_end_gb"]) == 2
    assert len(report["growth_per_iteration_gb"]) == 1
    on_disk = harness.read_child_result(tmp_path / "attr.json")
    assert on_disk["shape"] == ["tiny", 64, 64]
    assert all(s["footprint_gb"] > 0.0 for s in on_disk["stages"])
    assert "DEVICE-SWEEP DELTA" in capsys.readouterr().out


@needs_metal
def test_real_child_grid_iteration_round_trips_bit_exactly(child_dir):
    """The whole isolation boundary, for real: one tiny grid iteration in a
    spawned child, then the identical iteration in-process; a single moved
    bit would poison the continuity anchor's exact comparisons."""
    task = {"kind": "grid", "name": "tiny", "d_out": 128, "d_in": 128,
            "draw": "normal-0.02", "seed": 0, "batches": [1],
            "verbose": False}
    payload, meta = harness.spawn_measurement(
        "grid tiny 128x128 normal-0.02 s0", task, budget_gb=24.0,
        wall_cap_s=300.0)
    assert payload["exact"] is True
    assert meta["footprint_gb"]["peak"] > 0.0
    assert meta["mx_gb"]["peak"] >= 0.0, "the child reports its mx share too"
    session = DeviceMemberSession(_DEVICE)
    ours = harness.grid_iteration("tiny", 128, 128, "normal-0.02", 0, (1,),
                                  session)
    assert len(payload["records"]) == len(ours["records"]) == 8
    _assert_bit_identical(payload["records"], ours["records"])
