import argparse
import ast
import json, sys
from pathlib import Path
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


def test_spike_timed_path_refuses_a_held_lock_before_loading(monkeypatch):
    source = (dr.ROOT / "bench" / "spike_mlx_e2e.py").read_text()
    tree = ast.parse(source)
    [main_node] = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]

    class HeldLock:
        def __init__(self, owner):
            self.owner = owner

        def acquire(self):
            return False, "held by pid 1 (test)"

    def never_load():
        raise AssertionError("the model loaded before the lock refused")

    namespace = {
        "argparse": argparse,
        "MeasurementLock": HeldLock,
        "EXIT_LOCK_HELD": EXIT_LOCK_HELD,
        "install_patch": lambda: None,
        "load_model": never_load,
        "smoke": lambda *args: 0,
        "mde": lambda *args: 0,
        "ab": lambda *args: 0,
        "__doc__": "spike test",
    }
    module = ast.Module(body=[main_node], type_ignores=[])
    exec(compile(module, "spike_mlx_e2e.py", "exec"), namespace)

    assert namespace["main"](["--mde"]) == EXIT_LOCK_HELD


def test_numbered_nonretryable_environment_and_child_exits_are_explicit():
    assert dr.judge_exit_code(6) == ("crashed", True)
    assert dr.judge_exit_code(7) == ("gave-up", True)
    assert dr.judge_exit_code(10) == ("crashed", True)
