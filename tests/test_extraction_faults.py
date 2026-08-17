"""Fault injection for the extraction subprocess driver.

The parent-side driver (mlx_arm.py) is the least-exercised critical path: its
job is precisely to survive a worker that crashes, hangs, dies mid-batch, or
writes garbage, and to convert each failure into a named error instead of a
verdict. These tests inject each fault through a fake worker module - no MLX
and no Metal device needed, so they run everywhere.
"""

import textwrap

import pytest

from kernelverify.extraction import mlx_arm
from kernelverify.extraction.mlx_arm import (
    ExtractionError,
    capture_specialization,
    run_live,
)
from kernelverify.extraction.surface import LiveCall, MLXKernelSurface

# Whole-module: every test here drives a real extraction worker against a live surface,
# so the gpu marker is module-level.
pytestmark = pytest.mark.gpu

SURFACE = MLXKernelSurface(
    name="fake", source="out[0] = inp[0];",
    input_names=("inp",), output_names=("out",),
)
CALL = LiveCall(inputs={}, output_shapes=(((1,), "float32"),))


def fake_worker(monkeypatch, tmp_path, body: str) -> None:
    """Install `body` as the worker module the driver will spawn."""
    (tmp_path / "kvtest_fake_worker.py").write_text(textwrap.dedent(body))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(mlx_arm, "WORKER_MODULE", "kvtest_fake_worker")


def test_a_worker_that_crashes_before_reporting_is_a_named_error(
        monkeypatch, tmp_path):
    fake_worker(monkeypatch, tmp_path, """
        import sys
        sys.stderr.write("boom: fake worker exploded")
        sys.exit(1)
    """)
    with pytest.raises(ExtractionError, match="died before reporting.*boom"):
        capture_specialization(SURFACE, CALL)


def test_a_dump_with_no_source_block_is_refused_not_parsed(
        monkeypatch, tmp_path):
    """A worker that reports success but produces an unparseable dump must be
    an extraction failure, never an empty-source certificate input."""
    fake_worker(monkeypatch, tmp_path, """
        import json, sys
        request = json.load(sys.stdin)
        capture = request["capture"]
        with open(capture["dump_path"], "w") as sink:
            sink.write("not a generated-source dump at all")
        with open(capture["result_path"], "w") as sink:
            json.dump({"ok": True, "outputs": [], "mlx_version": "fake"}, sink)
    """)
    with pytest.raises(ExtractionError, match="expected exactly one"):
        capture_specialization(SURFACE, CALL)


def test_a_live_arm_dying_mid_batch_names_its_unreported_cases(
        monkeypatch, tmp_path):
    """Crash isolation with no verdict authority: the reported case survives,
    the unreported one comes back as an error naming the death."""
    fake_worker(monkeypatch, tmp_path, """
        import json, sys
        json.load(sys.stdin)
        print(json.dumps({"case": 0, "outputs": []}), flush=True)
        sys.stderr.write("segfault impression")
        sys.exit(139)
    """)
    first, second = run_live(SURFACE, [CALL, CALL])
    assert first.ok
    assert not second.ok and "died before this case" in second.error
    assert "segfault impression" in second.error


def test_a_hanging_worker_is_killed_and_every_case_says_so(
        monkeypatch, tmp_path):
    fake_worker(monkeypatch, tmp_path, """
        import time
        time.sleep(30)
    """)
    results = run_live(SURFACE, [CALL, CALL], timeout=1.0)
    assert all(not r.ok for r in results)
    assert all("killed after" in r.error for r in results)


def test_garbage_on_the_wire_is_skipped_not_fatal(monkeypatch, tmp_path):
    """A worker interleaving junk lines with valid events (a stray print from
    a dependency) must not poison the batch."""
    fake_worker(monkeypatch, tmp_path, """
        import json, sys
        json.load(sys.stdin)
        print("stray debug line, not JSON", flush=True)
        print(json.dumps({"case": 0, "outputs": []}), flush=True)
        print(json.dumps({"done": True}), flush=True)
    """)
    [result] = run_live(SURFACE, [CALL])
    assert result.ok
