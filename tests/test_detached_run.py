import json
import sys
from pathlib import Path
from conftest import requires_metal
import detached_run as dr
from memory_guard import EXIT_BUDGET_REFUSAL, EXIT_LOCK_HELD, EXIT_NOT_IDLE, EXIT_PRECONDITION


def _fake_harness(tmp_path, code: int):
    p = tmp_path / "fake_harness.py"
    p.write_text(f"import sys; sys.exit({code})\n")
    return p


def test_exit_code_protocol_maps_lock_held_back_to_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "wait_for_strong_idle", lambda deadline: True)
    monkeypatch.setattr(dr, "STATUS_DIR", tmp_path)
    outcome = dr.judge_exit_code(EXIT_LOCK_HELD)
    assert outcome == ("waiting", False), "a held lock is not an attempt consumed"


def test_exit_code_protocol_gives_up_on_budget_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "STATUS_DIR", tmp_path)
    assert dr.judge_exit_code(EXIT_BUDGET_REFUSAL) == ("gave-up", True)
    assert dr.judge_exit_code(EXIT_PRECONDITION) == ("gave-up", True), "pins/zone mismatch cannot heal by waiting"
    assert dr.judge_exit_code(EXIT_NOT_IDLE) == ("waiting", False)
    assert dr.judge_exit_code(0) == ("done", True)
    assert dr.judge_exit_code(1) == ("stopped", True), "a deliberate return 1 is a verdict"
    assert dr.judge_exit_code(1, stderr_tail="...\nTraceback (most recent call last):\n  File x\nValueError: boom") == ("crashed", True), \
        "an uncaught exception also exits 1; the traceback tells them apart (CEO ruling 1A)"


def test_a_crashing_harness_leaves_its_traceback_in_the_status_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "wait_for_strong_idle", lambda deadline: True)
    monkeypatch.setattr(dr, "STATUS_DIR", tmp_path)
    p = tmp_path / "boom.py"; p.write_text("raise ValueError('boom')\n")
    assert dr.main(["--harness", str(p), "--protocol", "exit-code"]) == 4
    status = json.loads((tmp_path / "detached_status-boom.json").read_text())
    assert status["state"] == "crashed" and "ValueError: boom" in status["stderr_tail"]
    assert dr.judge_exit_code(2) == ("crashed", True), "argparse's 2 is a typo, not a busy machine (memory_guard's ruling)"
    assert dr.judge_exit_code(99) == ("crashed", True)


def test_status_file_is_per_harness(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "STATUS_DIR", tmp_path)
    dr.write_status("waiting", harness_stem="serve_sub4bit")
    assert (tmp_path / "detached_status-serve_sub4bit.json").exists()
    assert json.loads((tmp_path / "detached_status-serve_sub4bit.json").read_text())["state"] == "waiting"


def test_the_runner_never_takes_the_machine_lock(tmp_path, monkeypatch):
    """Ruling 2A: harnesses own the lock. A runner that locked would make every
    self-locking harness refuse with EXIT_LOCK_HELD and loop to the deadline."""
    monkeypatch.setattr(dr, "wait_for_strong_idle", lambda deadline: True)
    monkeypatch.setattr(dr, "STATUS_DIR", tmp_path)
    assert "MeasurementLock" not in Path(dr.__file__).read_text()
    code = dr.main(["--harness", str(_fake_harness(tmp_path, 0)), "--protocol", "exit-code"])
    assert code == 0


def test_a_harness_holding_the_lock_makes_the_detached_run_wait(tmp_path, monkeypatch):
    # the fake harness exits EXIT_LOCK_HELD, as a real one would when a foreground run holds the lock;
    # the strong-idle wait returns True once, then False (deadline) - the deadline lives inside that function
    idle = iter([True, False])
    monkeypatch.setattr(dr, "wait_for_strong_idle", lambda deadline: next(idle))
    monkeypatch.setattr(dr, "STATUS_DIR", tmp_path)
    code = dr.main(["--harness", str(_fake_harness(tmp_path, EXIT_LOCK_HELD)), "--protocol", "exit-code"])
    assert code == 3                                     # gave up waiting; the lock-held attempt was never consumed
    status = json.loads((tmp_path / "detached_status-fake_harness.json").read_text())
    assert status["state"] == "gave-up" and status["attempts_consumed"] == 0


def test_baselines_protocol_is_the_default_and_unchanged(monkeypatch):
    args = dr.parse_args([])
    assert args.protocol == "baselines"
    assert args.harness == dr.ROOT / "bench" / "measure_baselines.py"


import subprocess


def test_start_script_derives_label_log_and_args_from_the_harness():
    root = Path(dr.ROOT)
    default = subprocess.run(["zsh", str(root / "bench/start_binding_run.sh"), "--print-plist"],
                             capture_output=True, text=True, cwd=root).stdout
    assert "com.kernelverify.detached-measure_baselines" in default
    assert "detached_run.py" in default and "--protocol" not in default   # baselines is the default protocol
    ab = subprocess.run(["zsh", str(root / "bench/start_binding_run.sh"), "--print-plist",
                         "bench/serve_sub4bit.py", "--", "--ab"], capture_output=True, text=True, cwd=root).stdout
    assert "com.kernelverify.detached-serve_sub4bit" in ab
    assert "<string>--harness</string>" in ab and "<string>--protocol</string>" in ab and "<string>exit-code</string>" in ab
    assert "<string>--ab</string>" in ab and "detached-serve_sub4bit.log" in ab
    escaped = subprocess.run(["zsh", str(root / "bench/start_binding_run.sh"), "--print-plist",
                              "bench/serve_sub4bit.py", "--", "--corpus", "a&b<c>"], capture_output=True, text=True, cwd=root).stdout
    assert "<string>a&amp;b&lt;c&gt;</string>" in escaped, "arguments are XML-escaped (CEO ruling 2A)"


def test_measure_baselines_refuses_a_held_lock_before_the_idle_gate(monkeypatch):
    import measure_baselines

    monkeypatch.setattr(sys, "argv", ["measure_baselines.py"])
    monkeypatch.setattr(
        measure_baselines.MeasurementLock,
        "acquire",
        lambda self: (False, "held by pid 1 (test)"),
    )
    monkeypatch.setattr(
        measure_baselines.machine_state,
        "fingerprint",
        lambda: (_ for _ in ()).throw(
            AssertionError("the idle path ran before the lock refused")
        ),
    )

    assert measure_baselines.main() == EXIT_LOCK_HELD


@requires_metal
def test_spike_timed_path_refuses_a_held_lock_before_loading(monkeypatch):
    """The REAL module, really imported. The first version of this test
    ast-lifted main() out of the file and exec'd it against a namespace that
    supplied MeasurementLock, EXIT_LOCK_HELD and every heavy function itself
    - so deleting the module's actual MeasurementLock import left it green
    while the harness would die on its first line under launchd. A test that
    provides every name it checks for proves only that its own stubs work.

    Gated on Metal for that same reason, not because locking needs a GPU:
    importing the real module reaches mlx_lm.models.qwen3 and so mlx.nn, and
    that import ABORTS the interpreter where no device can be created. Ungated
    it took the whole collection down, this file and every other, which is how
    it reached the supposedly GPU-free subset.
    """
    import spike_mlx_e2e as spike

    monkeypatch.setattr(
        spike.MeasurementLock, "acquire",
        lambda self: (False, "held by pid 1 (test)"))
    monkeypatch.setattr(
        spike, "install_patch",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the patch installed before the lock refused")))
    monkeypatch.setattr(
        spike, "load_model",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the model loaded before the lock refused")))

    assert spike.main(["--mde"]) == EXIT_LOCK_HELD


def test_numbered_nonretryable_environment_and_child_exits_are_explicit():
    # All three are crash-class per the plan's pre-registered table: the
    # gave-up set is (EXIT_BUDGET_REFUSAL, EXIT_PRECONDITION) and nothing
    # else. 7 spent one commit in gave-up because a missing device felt
    # permanent; the table wins, because widening it silently is exactly the
    # drift a pre-registration exists to stop, and crashed is the state that
    # sends the operator to the stderr tail.
    assert dr.judge_exit_code(6) == ("crashed", True)
    assert dr.judge_exit_code(7) == ("crashed", True)
    assert dr.judge_exit_code(10) == ("crashed", True)


def test_the_wait_heartbeat_keeps_the_status_file_alive(tmp_path, monkeypatch):
    """A nine-hour wait must not read "just started" the whole way through.

    The status file is the one artifact the operator card says to read, and
    its updated_at is how a live runner still polling is told apart from a
    dead one. The generalisation pass dropped the per-poll write because
    write_status gained a stem argument the wait does not have; the heartbeat
    hook restores it, so this pins both halves: main wires the hook, and the
    hook writes the CURRENT attempt count, not a reset.
    """
    monkeypatch.setattr(dr, "STATUS_DIR", tmp_path)
    captured = {}

    def _wait(deadline):
        captured["hook"] = dr._WAIT_HEARTBEAT
        return False  # straight to gave-up; the hook is what we came for

    monkeypatch.setattr(dr, "wait_for_strong_idle", _wait)
    assert dr.main(["--harness", "bench/serve_sub4bit.py",
                    "--protocol", "exit-code"]) == 3

    assert captured["hook"] is not None, "main never wired the heartbeat"
    captured["hook"](["on battery"])
    status = json.loads(
        (tmp_path / "detached_status-serve_sub4bit.json").read_text())
    assert status["state"] == "waiting"
    assert status["blockers"] == ["on battery"], (
        "the heartbeat must carry the LIVE blocker, not a placeholder")


def test_the_wait_itself_calls_the_heartbeat_every_blocked_poll(monkeypatch):
    beats = []
    monkeypatch.setattr(dr, "_WAIT_HEARTBEAT", lambda blockers: beats.append(blockers))
    monkeypatch.setattr(dr, "strong_idle_blockers", lambda: ["on battery"])
    monkeypatch.setattr(dr.time, "sleep", lambda s: None)
    clock = iter(range(100))
    monkeypatch.setattr(dr.time, "time", lambda: next(clock))
    assert dr.wait_for_strong_idle(deadline=5) is False
    assert len(beats) >= 3, (
        f"{len(beats)} heartbeats over a multi-poll wait: the status file "
        f"froze again")
    assert all(b == ["on battery"] for b in beats)


def test_the_harness_child_runs_unbuffered(monkeypatch):
    """The child's stdout is the launchd log, not a tty, so without -u Python
    block-buffers at 8 KB: tail shows nothing mid-run, and a Jetsam SIGKILL
    discards the very lines that localise which cell was live."""
    seen = {}

    def _run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(dr.subprocess, "run", _run)
    args = dr.parse_args(["--harness", "bench/serve_sub4bit.py",
                          "--protocol", "exit-code"])
    dr._run_harness(args)
    assert seen["argv"][0] == sys.executable
    assert seen["argv"][1] == "-u", (
        "the harness child must not buffer its progress lines")


def test_a_typoed_flag_or_missing_harness_never_arms(tmp_path):
    """`--print-plsit` (one transposed letter) used to be accepted as a
    HARNESS PATH: it armed a real launchd job, printed "armed", and the
    machine waited up to 12 hours to run a file that does not exist. Both
    rejections must fire before launchctl OR plist printing is reached."""
    root = dr.ROOT
    typo = subprocess.run(
        ["zsh", str(root / "bench/start_binding_run.sh"), "--print-plsit"],
        capture_output=True, text=True, cwd=root)
    assert typo.returncode == 2
    assert "unknown flag" in typo.stderr
    assert "armed" not in typo.stdout

    missing = subprocess.run(
        ["zsh", str(root / "bench/start_binding_run.sh"), "--print-plist",
         "bench/no_such_harness.py"],
        capture_output=True, text=True, cwd=root)
    assert missing.returncode == 2
    assert "no such harness" in missing.stderr
    assert "<plist" not in missing.stdout, (
        "printing a plist for a nonexistent harness is as wrong as arming it")


@requires_metal
def test_spec_verify_spike_refuses_a_held_lock_before_loading(monkeypatch):
    """The same contract as the mlx_e2e spike above, on the second spike.

    Gated on Metal for the import, not for the locking: `spike_spec_verify`
    reaches mlx.nn through serve_sub4bit at module scope, and that import
    ABORTS the interpreter where no device can be created, taking the whole
    collection down rather than failing one test.
    """
    import spike_spec_verify as spike

    monkeypatch.setattr(
        spike.MeasurementLock, "acquire",
        lambda self: (False, "held by pid 1 (test)"))
    monkeypatch.setattr(
        spike, "load_model",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the model loaded before the lock refused")))
    monkeypatch.setattr(
        spike, "require_idle",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the idle path ran before the lock refused")))

    assert spike.main([]) == EXIT_LOCK_HELD
