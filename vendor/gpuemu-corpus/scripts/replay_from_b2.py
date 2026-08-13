#!/usr/bin/env python3
"""Replay a failure from an object-store run record.

Fetches `<run_id>/results.jsonl` from B2 (using the same S3 env vars the
harness uses), picks failing rows that carry a `repro_info` payload, and
re-runs each via the daemon: regenerate inputs from the snapshot, ask the
kernel under test to produce its output, submit, and confirm the same
verdict is reached. This validates the *reproducibility* leg of the cloud
pipeline — same run id, same data, same verdict — without re-provisioning
GPUs.

Usage:
    python3 scripts/replay_from_b2.py <run_id>
        [--corpus ../corpus] [--kernels softmax_llm_buggy] [--max 5]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drivers"))

from gpuemu_lab.config import Credentials  # noqa: E402
from gpuemu_lab.storage import S3Store  # noqa: E402
import _p1lib as lib  # noqa: E402


def _decode_repro_inputs(repro: dict, input_names: list) -> dict:
    """Re-materialize the exact inputs that produced the original failure.

    Prefers the binary `input_snapshot` (bit-identical reproduction) when
    present; otherwise we cannot reliably regenerate from seed alone here
    without the Rust fuzzer in-process, so we skip that case.
    """
    snap_b64 = repro.get("input_snapshot")
    if not snap_b64:
        raise RuntimeError("repro_info has no input_snapshot; would need the "
                           "daemon's Reproduce endpoint to regenerate from seed")
    blob = base64.b64decode(snap_b64)
    return _parse_snapshot(blob, input_names)


def _parse_snapshot(blob: bytes, expected_names: list) -> dict:
    """Mirror of fuzzer::deserialize_input_snapshot (binary format).

    Layout: count:u16 | for each input: name_len:u16, name:bytes,
    shape_len:u16, shape:u32[], strides_len:u16, strides:u32[],
    dtype:u8, data_len:u32, data:bytes.
    Returns {name: np.ndarray}.
    """
    DTYPES = ["float16", "bfloat16", "float32", "float64",
              "int8", "int16", "int32", "int64",
              "uint8", "uint16", "uint32", "uint64", "bool"]
    NP_DTYPE = {"float16": "float16", "bfloat16": "float16",   # bf16 -> proxy
                "float32": "float32", "float64": "float64",
                "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
                "uint8": "uint8", "uint16": "uint16", "uint32": "uint32",
                "uint64": "uint64", "bool": "bool_"}
    o = 0
    (count,) = np.frombuffer(blob[o:o+2], dtype="<u2"); o += 2
    out = {}
    for _ in range(int(count)):
        (nlen,) = np.frombuffer(blob[o:o+2], dtype="<u2"); o += 2
        name = blob[o:o+nlen].decode("utf-8"); o += nlen
        (slen,) = np.frombuffer(blob[o:o+2], dtype="<u2"); o += 2
        shape = np.frombuffer(blob[o:o+4*slen], dtype="<u4").tolist(); o += 4*slen
        (tlen,) = np.frombuffer(blob[o:o+2], dtype="<u2"); o += 2
        _strides = np.frombuffer(blob[o:o+4*tlen], dtype="<u4").tolist(); o += 4*tlen
        dtype_idx = blob[o]; o += 1
        dtype_name = DTYPES[dtype_idx] if dtype_idx < len(DTYPES) else "float32"
        (dlen,) = np.frombuffer(blob[o:o+4], dtype="<u4"); o += 4
        data = np.frombuffer(blob[o:o+int(dlen)], dtype=NP_DTYPE[dtype_name]); o += int(dlen)
        arr = data.reshape(tuple(shape)) if shape else data.reshape(())
        out[name] = arr.copy()
    if expected_names:
        missing = [n for n in expected_names if n not in out]
        if missing:
            raise RuntimeError(f"snapshot missing expected inputs: {missing}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parents[1] / "corpus"))
    ap.add_argument("--kernels", default="", help="optional filter (comma-sep)")
    ap.add_argument("--max", type=int, default=5, help="max failures to replay")
    args = ap.parse_args()

    creds = Credentials.from_env()
    if not creds.have_storage():
        print("missing S3 env:", ", ".join(creds.missing_for_cloud()), file=sys.stderr)
        return 2

    print(f"[replay] fetching gpuemu/{args.run_id}/results.jsonl from B2...", flush=True)
    store = S3Store(creds)
    with tempfile.TemporaryDirectory() as tmp:
        store.get_dir(args.run_id, Path(tmp))
        rj = Path(tmp) / "results.jsonl"
        if not rj.exists():
            print(f"no results.jsonl under {args.run_id}", file=sys.stderr)
            return 3
        rows = [json.loads(l) for l in rj.read_text().splitlines() if l.strip()]

    failures = [r for r in rows if not r.get("passed")
                and (r.get("repro_info") or {}).get("input_snapshot")]
    if args.kernels:
        wanted = {k for k in args.kernels.split(",") if k}
        failures = [r for r in failures if r.get("kernel") in wanted]
    failures = failures[: args.max]
    if not failures:
        print(f"no replayable failures in {args.run_id}")
        return 0
    print(f"[replay] {len(failures)} failure(s) selected for replay", flush=True)

    ops_by_name = {op.name: op for op in lib.load_corpus(Path(args.corpus))}
    matches = mismatches = 0
    with lib.DaemonManager(args.corpus) as daemon:
        with daemon.client() as client:
            for r in failures:
                kname = r["kernel"]
                op = ops_by_name.get(kname)
                if op is None:
                    print(f"  [skip] {kname}: not in current corpus")
                    continue
                inputs = _decode_repro_inputs(r["repro_info"], op.input_names)
                try:
                    output = op.run(inputs)
                except Exception as e:
                    print(f"  [skip] {kname} seed={r['seed']}: kernel error {e!s:.80}")
                    continue
                res = client.submit_output(kname, inputs, output, r["seed"])
                orig_passed = bool(r.get("passed"))
                if res.passed == orig_passed:
                    matches += 1
                    mark = "MATCH"
                else:
                    mismatches += 1
                    mark = "DIVERGES"
                print(f"  [{mark}] {kname:24s} seed={r['seed']} "
                      f"orig.passed={orig_passed} replay.passed={res.passed}")

    print(f"\n[replay] {matches} verdicts match, {mismatches} diverge.")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
