"""Shared building blocks for the P1 driver.

Two oracles produce the *same* per-iteration record schema:

* ``daemon`` — the canonical path: the gpuemu daemon generates inputs and
  validates submitted outputs (gpuemu-py + a running daemon). Used on the GPU
  image for real runs.
* ``local`` — dependency-light dev path (numpy only): a seeded schema-based
  mini-fuzzer generates inputs, the op's *fp64 reference script* (the very
  script the daemon would run) computes the reference, and we compare with the
  same tolerances and mirror gpuemu's error/ULP statistics.

Comparison semantics mirror crates/gpuemu-daemon/src/validator.rs (abs tolerance,
NaN/Inf, and the ErrorStats distribution).
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

_NP = {"float16": np.float16, "float32": np.float32, "float64": np.float64,
       "int32": np.int32, "int64": np.int64}


# --------------------------------------------------------------------------
# Corpus loading
# --------------------------------------------------------------------------
class CorpusOp:
    def __init__(self, meta: dict, corpus_dir: Path):
        self.meta = meta
        self.name = meta["name"]
        self.source = meta.get("source", "unknown")
        self.benchmark_verdict = meta.get("benchmark_verdict", "unknown")
        self.input_names = meta["input_names"]
        self.dtypes = meta.get("dtypes", ["float32"])
        self.tolerances = meta.get("tolerances", {})
        self.op_schema = meta.get("op_schema")
        self.ref_script = str((corpus_dir / meta["reference"]).resolve())
        self._kernel_path = str((corpus_dir / meta["kernel"]).resolve())
        self._run: Optional[Callable] = None

    @property
    def run(self) -> Callable[[Dict[str, np.ndarray]], np.ndarray]:
        if self._run is None:
            spec = importlib.util.spec_from_file_location(f"k_{self.name}", self._kernel_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._run = mod.run
        return self._run

    def tol(self, dtype: str) -> float:
        return float(self.tolerances.get(dtype, 1e-5))


def load_corpus(corpus_dir: Path, names: Optional[List[str]] = None) -> List[CorpusOp]:
    ops = []
    for meta_path in sorted(corpus_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if names and meta["name"] not in names:
            continue
        ops.append(CorpusOp(meta, corpus_dir))
    return ops


# --------------------------------------------------------------------------
# Reference script invocation (gpuemu protocol)
# --------------------------------------------------------------------------
def _encode(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "strides": list(a.strides),
            "dtype": str(a.dtype), "data": base64.b64encode(a.tobytes()).decode()}


def run_reference(ref_script: str, inputs: Dict[str, np.ndarray]) -> np.ndarray:
    payload = {"inputs": {k: _encode(v) for k, v in inputs.items()}, "kwargs": {}}
    p = subprocess.run([sys.executable, ref_script], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"reference {ref_script} failed: {p.stderr.strip()}")
    o = json.loads(p.stdout)
    arr = np.frombuffer(base64.b64decode(o["data"]), dtype=np.dtype(o["dtype"]))
    return arr.reshape(o["shape"])


# --------------------------------------------------------------------------
# ULP distance (mirrors validator.rs)
# --------------------------------------------------------------------------
def _ulp(o: np.ndarray, r: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype == np.float32:
        ui, sign, big = np.uint32, np.uint32(0x80000000), np.uint64(0xFFFF_FFFF_FFFF_FFFF)
    elif dtype == np.float64:
        ui, sign, big = np.uint64, np.uint64(0x8000000000000000), np.uint64(0xFFFF_FFFF_FFFF_FFFF)
    elif dtype == np.float16:
        ui, sign, big = np.uint16, np.uint16(0x8000), np.uint64(0xFFFF_FFFF_FFFF_FFFF)
    else:
        return np.zeros(o.shape, np.uint64)

    def key(x):
        b = np.ascontiguousarray(x.astype(dtype)).view(ui)
        neg = (b >> (b.dtype.itemsize * 8 - 1)) != 0
        return np.where(neg, (~b), (b | sign)).astype(np.uint64)

    ko, kr = key(o), key(r)
    d = np.where(ko >= kr, ko - kr, kr - ko)
    finite = np.isfinite(o.astype(np.float64)) & np.isfinite(r.astype(np.float64))
    return np.where(finite, d, big)


def error_stats(o: np.ndarray, r: np.ndarray, dtype: np.dtype, tol: float) -> dict:
    of, rf = o.astype(np.float64).ravel(), r.astype(np.float64).ravel()
    abs_err = np.abs(of - rf)
    abs_err = np.where(np.isfinite(abs_err), abs_err, np.inf)
    nz = rf != 0.0
    rel = np.abs(of[nz] - rf[nz]) / np.abs(rf[nz]) if np.any(nz) else np.array([0.0])
    ulp = _ulp(o, r, dtype).ravel()
    finite_abs = abs_err[np.isfinite(abs_err)]
    pct = lambda p: float(np.percentile(finite_abs, p)) if finite_abs.size else 0.0
    return {
        "count": int(abs_err.size),
        "num_exceeding": int(np.sum(abs_err > tol)),
        "max_abs": float(np.max(abs_err)) if abs_err.size else 0.0,
        "mean_abs": float(np.mean(abs_err)) if abs_err.size else 0.0,
        "p50_abs": pct(50), "p90_abs": pct(90), "p99_abs": pct(99),
        "max_rel": float(np.max(rel)) if rel.size else 0.0,
        "mean_rel": float(np.mean(rel)) if rel.size else 0.0,
        "max_ulp": int(np.max(ulp)) if ulp.size else 0,
        "mean_ulp": float(np.mean(ulp)) if ulp.size else 0.0,
    }


def compare(output: np.ndarray, reference: np.ndarray, tol: float) -> dict:
    """Mirror validator.rs: shape, abs tolerance, NaN/Inf, + error distribution."""
    if list(output.shape) != list(reference.shape):
        return {"passed": False, "failure_kind": "shape",
                "max_abs_err": float("inf"), "max_rel_err": float("inf"),
                "max_ulp": None, "error_stats": None}
    stats = error_stats(output, reference, output.dtype, tol)
    of = output.astype(np.float64)
    failure_kind = None
    if np.any(np.isnan(of)):
        failure_kind = "nan"
    elif np.any(np.isinf(of)):
        failure_kind = "inf"
    elif stats["num_exceeding"] > 0:
        failure_kind = "tolerance"
    return {
        "passed": failure_kind is None,
        "failure_kind": failure_kind,
        "max_abs_err": stats["max_abs"],
        "max_rel_err": stats["max_rel"],
        "max_ulp": stats["max_ulp"],
        "error_stats": stats,
    }


# --------------------------------------------------------------------------
# Local schema-based mini-fuzzer (dev path; not bit-identical to the Rust fuzzer)
# --------------------------------------------------------------------------
def sample_case(op: CorpusOp, dtype: str, rng: np.random.Generator) -> dict:
    """Return {'shapes': {name: shape}, 'repr_shape': [...]}. Uses op_schema if present."""
    schema = op.op_schema
    if schema:
        dims = {d["name"]: int(rng.choice(d["candidates"])) for d in schema["dims"]}
        shapes = {t["name"]: [dims[n] for n in t["dims"]] for t in schema["inputs"]}
        out = schema.get("output")
        repr_shape = [dims[n] for n in out["dims"]] if out else next(iter(shapes.values()))
    else:
        shape = [int(rng.choice([1, 2, 8])), int(rng.choice([1, 7, 128])),
                 int(rng.choice([1, 3, 256]))]
        shapes = {n: shape for n in op.input_names}
        repr_shape = shape
    return {"shapes": shapes, "repr_shape": repr_shape}


def make_inputs(op: CorpusOp, shapes: Dict[str, list], dtype: str,
                rng: np.random.Generator) -> Dict[str, np.ndarray]:
    npdt = _NP[dtype]
    inputs = {}
    for name in op.input_names:
        shp = shapes[name]
        # mirror the Rust fuzzer's value range: uniform [-10, 10]
        data = (rng.random(int(np.prod(shp)) if shp else 1) * 2.0 - 1.0) * 10.0
        inputs[name] = data.reshape(shp).astype(npdt)
    return inputs


# --------------------------------------------------------------------------
# Daemon lifecycle (canonical oracle) — start a real gpuemu daemon, isolated.
# --------------------------------------------------------------------------
def find_daemon_binary() -> Optional[str]:
    env = os.environ.get("GPUEMU_DAEMON")
    if env and Path(env).exists():
        return env
    # sibling gpuemu repo: gpuemu-paper/drivers/_p1lib.py -> .../Code/gpuemu
    code_dir = Path(__file__).resolve().parents[2]
    candidates = []
    for base in (code_dir / "gpuemu", Path.cwd()):
        for prof in ("release", "debug"):
            p = base / "target" / prof / "gpuemu-daemon"
            if p.exists():
                candidates.append(p)
    if not candidates:
        return None
    # Pick the most recently built binary so local rebuilds always win over a
    # stale release/debug artifact.
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


class DaemonManager:
    """Run a fresh, isolated gpuemu daemon for the duration of a `with` block.

    Isolation: the daemon's socket/db live under a temp HOME (default_socket_path
    is derived from $HOME), so we never collide with or clobber a real ~/.gpuemu.
    The daemon loads gpuemu.toml from `corpus_dir` (its cwd).
    """

    def __init__(self, corpus_dir, binary: Optional[str] = None):
        self.corpus_dir = str(corpus_dir)
        self.binary = binary or find_daemon_binary()
        self.home = tempfile.mkdtemp(prefix="gpuemu-home-")
        self.socket_path = os.path.join(self.home, ".gpuemu", "gpuemu.sock")
        self.log_path = os.path.join(self.home, "daemon.log")
        self.proc = None

    def __enter__(self) -> "DaemonManager":
        if not self.binary:
            raise RuntimeError(
                "gpuemu-daemon binary not found. Build it (cargo build -p gpuemu-daemon) "
                "or set GPUEMU_DAEMON to its path."
            )
        os.makedirs(os.path.join(self.home, ".gpuemu"), exist_ok=True)
        env = dict(os.environ, HOME=self.home)
        log = open(self.log_path, "w")
        self.proc = subprocess.Popen([self.binary], cwd=self.corpus_dir, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
        self._wait_ready()
        return self

    def client(self):
        from gpuemu_py.client import Client
        return Client(socket_path=self.socket_path, timeout_ms=60000)

    def _wait_ready(self, timeout: float = 20.0) -> None:
        from gpuemu_py.client import Client
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = Path(self.log_path).read_text()[-800:]
                raise RuntimeError(f"daemon exited early (rc={self.proc.returncode}):\n{tail}")
            try:
                with Client(socket_path=self.socket_path, timeout_ms=1500) as c:
                    c.ping()
                    return
            except Exception as e:
                last = e
                time.sleep(0.3)
        raise RuntimeError(f"daemon not ready after {timeout}s: {last}")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
