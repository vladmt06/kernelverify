"""Bind measurement inputs and isolate each unit of work in a fresh process.

Comparable harnesses must agree about the bytes they measured and must not
leave work running after a parent stops.
Those guarantees belong here so a second harness cannot silently implement
them differently.
Scientific readings, experiment-specific plans and verdicts stay with the
harness whose pre-registration gives them meaning.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

from memory_guard import (
    BudgetGuard,
    EXIT_BUDGET_REFUSAL,
    EXIT_CHILD_DEATH,
    EXIT_LOW_MEMORY,
    EXIT_NO_DEVICE,
    EXIT_ORPHANED,
    EXIT_PRECONDITION,
    require_available_memory,
)


MLX_VERSION = "0.32.0"
MLX_LM_VERSION = "0.31.3"
ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "qwen3-4b-4bit-g64"
MODEL_DIR = ROOT / "bench" / ".models" / MODEL_NAME
MODEL_FILE = MODEL_DIR / "model.safetensors"
PIN_MANIFEST = MODEL_DIR.parent / "PINNED-HASHES.txt"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RunInvalid(RuntimeError):
    """The records cannot support the registered comparison."""


class PreconditionFailed(RuntimeError):
    """A permanent prerequisite failed before training could be measured."""


class ChildRefusal(RuntimeError):
    """A fresh arm process stopped with a shared refusal code."""

    def __init__(self, cell: str, returncode: int, reason: str):
        super().__init__(f"{cell}: {reason}")
        self.cell = cell
        self.returncode = returncode
        self.reason = reason


class NoDevice(RuntimeError):
    """No usable Metal device exists in the measurement child."""


def _positive_samples(samples: Sequence[float], name: str) -> list[float]:
    values = list(samples)
    if not values:
        raise RunInvalid(f"{name} has no sample")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value <= 0 for value in values):
        raise RunInvalid(f"{name} samples must be positive and finite")
    return values


def _spread_pct(samples: Sequence[float]) -> float:
    return (max(samples) - min(samples)) / statistics.median(samples) * 100


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wrapper_sha256(wrapper: Path) -> str:
    """The wrapper fingerprint, from the wrapper's own implementation.

    Three places compare this digest - the preflight, the child's re-check
    and the wrapper's own evidence object - and any disagreement between
    them refuses every run at the evidence gate. So there is one
    implementation and the other two import it, rather than three copies of
    a hashing algorithm kept in step by tests.

    Imported here rather than at module scope because this file's decision
    rules must stay importable wherever the wrapper is not installed.
    """
    from metalrunner.measurement import tree_sha256

    return tree_sha256(wrapper)


def _tree_sha256(root: Path, files: Sequence[Path] | None = None) -> str:
    selected = (
        sorted(files, key=lambda path: str(path.relative_to(root)))
        if files is not None else sorted(
            path for path in root.rglob("*")
            if (path.is_file() and "__pycache__" not in path.parts
                and path.suffix != ".pyc" and not path.name.startswith("."))
        )
    )
    if not selected:
        raise PreconditionFailed(f"nothing hashable under {root}")
    digest = hashlib.sha256()
    for path in selected:
        relative = str(path.relative_to(root)).encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def package_sha256(distribution_name: str) -> str:
    """Hash every installed distribution file except interpreter caches."""
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as error:
        raise PreconditionFailed(
            f"required distribution {distribution_name!r} is not installed"
        ) from error
    files = []
    for entry in distribution.files or ():
        path = Path(distribution.locate_file(entry))
        if (path.is_file() and "__pycache__" not in path.parts
                and path.suffix != ".pyc"):
            files.append((str(entry), path))
    if not files:
        raise PreconditionFailed(
            f"distribution {distribution_name!r} has no hashable files"
        )
    digest = hashlib.sha256()
    for relative, path in sorted(files):
        name = relative.encode()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def stack_record() -> dict:
    records = {}
    for key, distribution_name, expected in (
        ("mlx", "mlx", MLX_VERSION),
        ("mlx_lm", "mlx-lm", MLX_LM_VERSION),
    ):
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError as error:
            raise PreconditionFailed(
                f"required distribution {distribution_name!r} is not installed"
            ) from error
        if version != expected:
            raise PreconditionFailed(
                f"{distribution_name} is {version}, required {expected}"
            )
        records[key] = {
            "version": version,
            "package_sha256": package_sha256(distribution_name),
        }
    return records


def _verified_model_manifest(model_dir: Path, manifest: Path) -> dict:
    if not manifest.is_file():
        raise PreconditionFailed(f"pin manifest is missing: {manifest}")
    entries = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, path = line.split(maxsplit=1)
        relative = path.removeprefix("bench/.models/")
        if relative.startswith(f"{MODEL_NAME}/"):
            entries[relative.removeprefix(f"{MODEL_NAME}/")] = digest
    actual = {
        str(path.relative_to(model_dir))
        for path in model_dir.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    }
    if actual != set(entries):
        raise PreconditionFailed(
            "model files differ from the pin manifest: "
            f"unlisted={sorted(actual - set(entries))}, "
            f"missing={sorted(set(entries) - actual)}"
        )
    for relative, expected in entries.items():
        observed = _file_sha256(model_dir / relative)
        if observed != expected:
            raise PreconditionFailed(
                f"pin mismatch for {MODEL_NAME}/{relative}"
            )
    return entries


def _data_record(path: Path) -> dict:
    directory = path.resolve()
    if not directory.is_dir():
        raise PreconditionFailed(
            f"training data must be a local directory: {directory}"
        )
    files = [
        directory / f"{split}.jsonl"
        for split in ("train", "valid", "test")
        if (directory / f"{split}.jsonl").is_file()
    ]
    if not files or files[0].name != "train.jsonl":
        raise PreconditionFailed(f"training data has no train.jsonl: {directory}")
    for file in files:
        rows = [line for line in file.read_text().splitlines() if line.strip()]
        if not rows:
            raise PreconditionFailed(f"dataset split is empty: {file}")
        try:
            for row in rows:
                if not isinstance(json.loads(row), Mapping):
                    raise ValueError("row is not an object")
        except (json.JSONDecodeError, ValueError) as error:
            raise PreconditionFailed(f"invalid JSONL in {file}: {error}") from error
    return {
        "directory": str(directory),
        "files": [file.name for file in files],
        "sha256": _tree_sha256(directory, files),
    }


def _absent_module(name: str, find_module) -> bool:
    """Is the named measurement module missing, however it is missing?

    `find_spec` returns None for an absent top-level name and RAISES for a
    dotted name whose parent package does not resolve, so a plan naming
    "metalrunnr.lora" would leave a bare traceback and exit 1 rather than a
    numbered refusal. The detached runner reads exit 1 with a traceback as a
    crash and spends a retry on it, when the answer for a misspelled plan is a
    permanent refusal that should not be retried at all.
    """
    try:
        return find_module(name) is None
    except (ImportError, ValueError, TypeError):
        return True


def preflight_inputs(
    plan: Mapping[str, object],
    *,
    root: Path = ROOT,
    machine: Mapping[str, object],
    package_records: Mapping[str, object] | None = None,
    find_module=importlib.util.find_spec,
) -> dict:
    """Bind every permanent input before a child may load the model."""
    if machine.get("chip") != "Apple M3 Pro":
        raise PreconditionFailed(
            f"sprint-1 is registered for Apple M3 Pro, got {machine.get('chip')!r}"
        )
    memory_bytes = machine.get("memory_bytes")
    if (not isinstance(memory_bytes, int)
            or round(memory_bytes / 2**30) != 36):
        raise PreconditionFailed(
            "sprint-1 is registered for the 36 GB machine"
        )
    model_dir = root / "bench" / ".models" / MODEL_NAME
    model_file = model_dir / "model.safetensors"
    manifest = model_dir.parent / "PINNED-HASHES.txt"
    if not model_file.is_file():
        raise PreconditionFailed(f"pinned model file is missing: {model_file}")
    model_manifest = _verified_model_manifest(model_dir, manifest)
    observed_model = model_manifest.get("model.safetensors")
    if observed_model is None:
        raise PreconditionFailed("pin manifest omits model.safetensors")
    stack = dict(package_records if package_records is not None else stack_record())
    expected_versions = {"mlx": MLX_VERSION, "mlx_lm": MLX_LM_VERSION}
    for package, expected_version in expected_versions.items():
        record = stack.get(package)
        if (not isinstance(record, Mapping)
                or record.get("version") != expected_version
                or not record.get("package_sha256")):
            raise PreconditionFailed(
                f"stack record does not bind {package} {expected_version}"
            )
    if "data" not in plan and "bands" not in plan:
        raise PreconditionFailed(
            "a run plan must pin its training data, by `data` or by `bands`")
    if plan["kept_candidates"] and _absent_module(
            plan["measurement_module"], find_module):
        raise PreconditionFailed(
            f"measurement module {plan['measurement_module']!r} is unavailable"
        )
    wrapper = root / "metalrunner"
    if not wrapper.is_dir():
        raise PreconditionFailed(f"metalrunner wrapper is missing: {wrapper}")
    wrapper_digest = _wrapper_sha256(wrapper)
    return {
        "base_model": {
            "directory": str(model_dir.resolve()),
            "file": str(model_file.resolve()),
            "sha256": observed_model,
        },
        "model_manifest": model_manifest,
        # One caller pins one dataset and one pins two, because the profile's
        # two registered widths come from two bands of one corpus. Both are
        # bound the same way and the harness that asked for two says which is
        # which; a caller that named neither is a caller with no data at all.
        "data": (_data_record(Path(plan["data"])) if "data" in plan else None),
        "data_bands": ({name: _data_record(Path(directory))
                        for name, directory in sorted(plan["bands"].items())}
                       if "bands" in plan else None),
        "stack": stack,
        "wrapper_sha256": wrapper_digest,
        "plan_sha256": _json_sha256(plan),
        "measurement_module": plan["measurement_module"],
    }


def _signal_child(proc, signum: int) -> None:
    try:
        own_group = os.getpgid(proc.pid) == proc.pid
    except (ProcessLookupError, PermissionError):
        return
    try:
        if own_group:
            os.killpg(proc.pid, signum)
        else:
            os.kill(proc.pid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def _reap(proc) -> None:
    if proc.poll() is not None:
        return
    _signal_child(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        _signal_child(proc, signal.SIGKILL)


def spawn_child(task: Mapping[str, object], work_dir: str | Path, *,
                wall_cap_s: float, popen_factory=None, clock_ns=None,
                reap=None, child_entrypoint: str | Path | None = None) -> dict:
    """A process-start timestamp and child adapter timestamp bound job wall."""
    popen = subprocess.Popen if popen_factory is None else popen_factory
    clock = time.perf_counter_ns if clock_ns is None else clock_ns
    reap_process = _reap if reap is None else reap
    entrypoint = (
        Path(__file__).with_name("train_lora_e2e.py")
        if child_entrypoint is None else Path(child_entrypoint)
    )
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    task_path = work / f"task-{nonce}.json"
    result_path = work / f"result-{nonce}.json"
    stamped = {**task, "parent_pid": os.getpid()}
    task_path.write_text(json.dumps(stamped, sort_keys=True))
    argv = [
        sys.executable,
        "-u",
        str(entrypoint.resolve()),
        "--child-task",
        str(task_path),
        "--child-out",
        str(result_path),
    ]
    launched_ns = clock()
    proc = popen(argv, start_new_session=True)
    timed_out = False
    try:
        try:
            proc.wait(timeout=wall_cap_s)
        except subprocess.TimeoutExpired:
            timed_out = True
    finally:
        reap_process(proc)
        task_path.unlink(missing_ok=True)
    completed_ns = clock()
    cell = str(task.get("cell", task.get("kind", "child")))
    if timed_out:
        raise ChildRefusal(
            cell, EXIT_CHILD_DEATH, f"hit the {wall_cap_s:.0f}s wall cap"
        )
    if proc.returncode != 0:
        raise ChildRefusal(cell, proc.returncode, f"child exited {proc.returncode}")
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ChildRefusal(
            cell, EXIT_CHILD_DEATH, f"child left no valid result: {error}"
        ) from error
    finally:
        result_path.unlink(missing_ok=True)
    if task.get("kind") == "train":
        written_ns = result.get("adapter_written_monotonic_ns")
        if (not isinstance(written_ns, int)
                or not launched_ns <= written_ns <= completed_ns):
            raise ChildRefusal(
                cell, EXIT_CHILD_DEATH,
                "adapter completion timestamp is outside child lifetime",
            )
        result["process_launch_monotonic_ns"] = launched_ns
        result["full_job_wall_s"] = (written_ns - launched_ns) / 1e9
    return result


def child_exit_for_parent(returncode: int) -> int:
    """Preserve known shared refusals and classify every other failure."""
    if returncode == 0:
        return 0
    shared = {
        EXIT_BUDGET_REFUSAL,
        EXIT_LOW_MEMORY,
        EXIT_NO_DEVICE,
        EXIT_PRECONDITION,
        EXIT_ORPHANED,
    }
    return returncode if returncode in shared else EXIT_CHILD_DEATH


def _atomic_json(path: Path, record: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True))
    os.replace(temporary, path)


class _ChildGuard:
    def __init__(self, budget_gb: float, parent_pid: int):
        self.budget_gb = budget_gb
        self._budget = BudgetGuard(budget_gb, parent_pid=parent_pid)

    def check(self, cell: str) -> float:
        current = self._budget.check(cell)
        require_available_memory(max(self.budget_gb - current, 0.0), cell)
        return current


def _check_child_inputs(task: Mapping[str, object], observed_stack: dict) -> None:
    plan = task.get("plan")
    provenance = task.get("provenance")
    if not isinstance(plan, Mapping) or not isinstance(provenance, Mapping):
        raise PreconditionFailed("child task has no plan or provenance")
    if _json_sha256(plan) != provenance.get("plan_sha256"):
        raise PreconditionFailed("child run plan differs from parent preflight")
    if observed_stack != provenance.get("stack"):
        raise PreconditionFailed("child package hashes differ from preflight")
    base = provenance.get("base_model")
    if not isinstance(base, Mapping) or not Path(base.get("file", "")).is_file():
        raise PreconditionFailed("child base model evidence is missing")
    model_dir = Path(base["directory"])
    model_manifest = _verified_model_manifest(
        model_dir, model_dir.parent / "PINNED-HASHES.txt"
    )
    if model_manifest != provenance.get("model_manifest"):
        raise PreconditionFailed("child model artifact manifest changed")
    if model_manifest.get("model.safetensors") != base.get("sha256"):
        raise PreconditionFailed("child base model sha256 differs from preflight")
    observed_data = _data_record(Path(plan["data"]))
    if observed_data != provenance.get("data"):
        raise PreconditionFailed("child training data differs from preflight")
    if _wrapper_sha256(ROOT / "metalrunner") != provenance.get(
            "wrapper_sha256"):
        raise PreconditionFailed("child wrapper hash differs from preflight")
