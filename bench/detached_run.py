#!/usr/bin/env python3
"""Detached binding run: wait for a strong-idle window, then measure.

Why this exists: the contention gate keeps naming the interactive sessions as
the contention. The 2026-08-14 14:57Z run had 6 of 10 model rows rejected on
dispersion while the idle gate itself passed, and the kernels lane measured
2.2x-9.7x spreads from Terminal and VS Code sharing the GPU at load 1.79.
The sessions corrupt samples even below every threshold the gate can check,
so the fix is not a better gate, it is a machine with no sessions on it, and
the operator can only close them all if the harness runs with no terminal
attached. Start it with bench/start_binding_run.sh (a one-shot launchd job);
this module is the process launchd runs. ADR 0010 records the decision.

The measurement itself is bench/measure_baselines.py, unchanged: same gates,
same schema, same append-only JSONL. This wrapper only decides *when* to
start, retries when repeats disagree, and records what happened at
bench/.baselines/detached_status.json so the operator can check the outcome
without scrolling a log.

Strong idle is deliberately stronger than the harness's own pre-check: the
plain idle gate must pass on several consecutive samples half a minute apart,
and the five-minute load must be under the threshold too. load1 recovers
within a minute of the operator closing apps while indexers and page-outs are
still settling; load5 under the threshold means the machine has actually been
quiet, not just momentarily.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import machine_state

ROOT = Path(__file__).resolve().parent.parent
BASELINES = ROOT / "bench" / ".baselines"
STATUS = BASELINES / "detached_status.json"
HARNESS = ROOT / "bench" / "measure_baselines.py"

STRONG_IDLE_SAMPLES = 5      # consecutive clean samples before starting
SAMPLE_INTERVAL_S = 30       # between clean samples (2 min of held quiet)
POLL_INTERVAL_S = 60         # between checks while the machine is busy
WAIT_LIMIT_S = 12 * 3600     # give up if no quiet window appears at all
MAX_ATTEMPTS = 3             # full re-runs when repeats disagree


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def write_status(state: str, **extra) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"state": state, "updated_at": now(), **extra},
                                 indent=1) + "\n")


def strong_idle_blockers() -> list[str]:
    """One sample of the start condition. Empty means clean."""
    chk = machine_state.idle_check()
    if chk["idle"] and chk["load5"] > chk["load_threshold"]:
        return [f"load5 {chk['load5']:.2f} over threshold "
                f"{chk['load_threshold']:.2f} (machine not yet settled)"]
    return list(chk["blockers"])


def wait_for_strong_idle(deadline: float) -> bool:
    """Block until STRONG_IDLE_SAMPLES consecutive clean samples, or timeout."""
    streak, last_reported = 0, None
    while time.time() < deadline:
        blockers = strong_idle_blockers()
        if not blockers:
            streak += 1
            log(f"idle sample {streak}/{STRONG_IDLE_SAMPLES} clean")
            if streak >= STRONG_IDLE_SAMPLES:
                return True
            time.sleep(SAMPLE_INTERVAL_S)
            continue
        if streak:
            log("idle streak broken")
        streak = 0
        # Only log when the reason changes; a 12-hour wait must stay readable.
        if blockers != last_reported:
            log(f"waiting, not idle: {'; '.join(blockers)}")
            last_reported = blockers
        write_status("waiting", blockers=blockers)
        time.sleep(POLL_INTERVAL_S)
    return False


def newest_run_rows() -> tuple[str | None, list[dict]]:
    """run_id and rows of the most recently appended run, across date files."""
    files = sorted(BASELINES.glob("*.jsonl"))
    if not files:
        return None, []
    rows = [json.loads(l) for l in files[-1].read_text().splitlines() if l.strip()]
    if not rows:
        return None, []
    run_id = rows[-1]["run_id"]
    return run_id, [r for r in rows if r["run_id"] == run_id]


def main() -> int:
    log(f"detached runner started, pid {os.getpid()}, "
        f"waiting up to {WAIT_LIMIT_S // 3600} h for a strong-idle window")
    write_status("waiting", blockers=["just started"])
    deadline = time.time() + WAIT_LIMIT_S
    attempt = 0
    run_id, rows = None, []

    while attempt < MAX_ATTEMPTS:
        if not wait_for_strong_idle(deadline):
            log(f"RESULT: GAVE UP, no strong-idle window in {WAIT_LIMIT_S // 3600} h; "
                "re-run bench/start_binding_run.sh to rearm")
            write_status("gave-up", waited_h=WAIT_LIMIT_S // 3600)
            return 3

        attempt += 1
        prev_run_id, _ = newest_run_rows()
        log(f"strong idle held; attempt {attempt}/{MAX_ATTEMPTS} starting")
        write_status("measuring", attempt=attempt)
        t0 = time.time()
        code = subprocess.run([sys.executable, str(HARNESS)], cwd=ROOT).returncode
        minutes = (time.time() - t0) / 60

        if code == 2:
            # The harness's own pre-check saw a machine our last sample missed.
            # Not a measurement attempt; go back to waiting.
            attempt -= 1
            log("harness refused to start (machine went busy); back to waiting")
            continue

        run_id, rows = newest_run_rows()
        if run_id == prev_run_id or not rows:
            # Exit 1 with no new rows is a crash, not a rejection; a rejection
            # always appends its rows as evidence. Retrying will not help.
            log(f"RESULT: CRASHED, harness exit {code} with no rows appended; "
                "see the traceback above")
            write_status("crashed", exit_code=code, attempt=attempt)
            return 4

        rejected = [r for r in rows if not r["binding"]]
        if code == 0:
            log(f"RESULT: BINDING run {run_id}, {len(rows)} rows, "
                f"{minutes:.0f} min measuring")
            write_status("binding", run_id=run_id, rows=len(rows),
                         attempt=attempt, minutes=round(minutes))
            return 0

        log(f"attempt {attempt}: {len(rejected)} of {len(rows)} rows rejected "
            f"after {minutes:.0f} min:")
        for r in rejected:
            log(f"  {r['row_id'].split('/', 1)[1]}: "
                f"{'; '.join(r['binding_blockers'])}")

    log(f"RESULT: NOT BINDING after {MAX_ATTEMPTS} attempts; last run {run_id} "
        f"has {len([r for r in rows if r['binding']])} of {len(rows)} rows binding")
    write_status("not-binding", run_id=run_id, rows=len(rows),
                 binding_rows=len([r for r in rows if r["binding"]]),
                 attempts=MAX_ATTEMPTS)
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        log(f"RESULT: CRASHED in the runner itself: {e!r}")
        write_status("crashed", error=repr(e))
        raise
