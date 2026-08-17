#!/usr/bin/env python3
"""Detached long run: wait for a strong-idle window, then run one harness.

Why this exists: the contention gate keeps naming the interactive sessions as
the contention. The 2026-08-14 14:57Z run had 6 of 10 model rows rejected on
dispersion while the idle gate itself passed, and the kernels lane measured
2.2x-9.7x spreads from Terminal and VS Code sharing the GPU at load 1.79.
The sessions corrupt samples even below every threshold the gate can check,
so the fix is a machine with no sessions on it. The operator can only close
them all if the harness runs with no terminal attached. Start it with
bench/start_binding_run.sh; this module is the process launchd runs.
ADR 0010 records the decision.

The baselines protocol preserves measure_baselines.py's append-only retry
semantics. The exit-code protocol runs any other long harness and interprets
the project's numbered refusal codes. Each harness owns the machine-wide
measurement lock. This runner only decides when to start, so a child can
acquire that lock itself.

Strong idle is deliberately stronger than each harness's own pre-check: the
plain idle gate must pass on several consecutive samples half a minute apart,
and the five-minute load must be under the threshold too. load1 recovers
within a minute of the operator closing apps while indexers and page-outs are
still settling; load5 under the threshold means the machine has actually been
quiet, not just momentarily.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import machine_state
from memory_guard import (
    EXIT_BUDGET_REFUSAL,
    EXIT_CHILD_DEATH,
    EXIT_LOCK_HELD,
    EXIT_LOW_MEMORY,
    EXIT_NO_DEVICE,
    EXIT_NOT_IDLE,
    EXIT_PRECONDITION,
)

ROOT = Path(__file__).resolve().parent.parent
BASELINES = ROOT / "bench" / ".baselines"
STATUS_DIR = BASELINES
HARNESS = ROOT / "bench" / "measure_baselines.py"

# I5 owns the constant in memory_guard. I2 still names its already-allocated
# value explicitly so an orphaned child is never mistaken for an unknown code.
EXIT_ORPHANED = 10

STRONG_IDLE_SAMPLES = 5      # consecutive clean samples before starting
SAMPLE_INTERVAL_S = 30       # between clean samples (2 min of held quiet)
POLL_INTERVAL_S = 60         # between checks while the machine is busy
WAIT_LIMIT_S = 12 * 3600     # give up if no quiet window appears at all
MAX_ATTEMPTS = 3             # full baseline re-runs when repeats disagree


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def write_status(state: str, harness_stem: str, **extra) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    attempts_consumed = extra.pop("attempts_consumed", 0)
    status = STATUS_DIR / f"detached_status-{harness_stem}.json"
    status.write_text(json.dumps({
        "state": state,
        "updated_at": now(),
        "attempts_consumed": attempts_consumed,
        **extra,
    }, indent=1) + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw:
        separator = raw.index("--")
        runner_args, harness_args = raw[:separator], raw[separator + 1:]
    else:
        runner_args, harness_args = raw, []

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harness", type=Path, default=HARNESS)
    parser.add_argument("--protocol", choices=("baselines", "exit-code"),
                        default="baselines")
    args = parser.parse_args(runner_args)
    args.harness_args = harness_args
    return args


def judge_exit_code(code: int, stderr_tail: str = "") -> tuple[str, bool]:
    """Return the runner state and whether this invocation spent an attempt."""
    if code == 0:
        return "done", True
    if code == 1:
        return ("crashed" if "Traceback" in stderr_tail else "stopped"), True
    if code in (EXIT_NOT_IDLE, EXIT_LOCK_HELD, EXIT_LOW_MEMORY):
        return "waiting", False
    if code in (EXIT_BUDGET_REFUSAL, EXIT_NO_DEVICE, EXIT_PRECONDITION):
        return "gave-up", True
    if code in (EXIT_CHILD_DEATH, EXIT_ORPHANED):
        return "crashed", True
    return "crashed", True


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
        time.sleep(POLL_INTERVAL_S)
    return False


def newest_run_rows() -> tuple[str | None, list[dict]]:
    """run_id and rows of the most recently appended run, across date files."""
    files = sorted(BASELINES.glob("*.jsonl"))
    if not files:
        return None, []
    rows = [json.loads(line) for line in files[-1].read_text().splitlines()
            if line.strip()]
    if not rows:
        return None, []
    run_id = rows[-1]["run_id"]
    return run_id, [row for row in rows if row["run_id"] == run_id]


def _stderr_tail(stderr: str) -> str:
    return "\n".join(stderr.splitlines()[-40:])


def _run_harness(args: argparse.Namespace) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(args.harness), *args.harness_args],
        cwd=ROOT,
        stderr=subprocess.PIPE,
        text=True,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    harness_stem = args.harness.stem
    log(f"detached runner started for {harness_stem}, pid {os.getpid()}, "
        f"waiting up to {WAIT_LIMIT_S // 3600} h for a strong-idle window")
    write_status("waiting", harness_stem, blockers=["just started"])
    deadline = time.time() + WAIT_LIMIT_S
    attempts_consumed = 0
    run_id, rows = None, []
    last_stderr_tail = ""

    while attempts_consumed < MAX_ATTEMPTS:
        if not wait_for_strong_idle(deadline):
            log(f"RESULT: GAVE UP, no strong-idle window in "
                f"{WAIT_LIMIT_S // 3600} h; re-run "
                "bench/start_binding_run.sh to rearm")
            write_status("gave-up", harness_stem,
                         attempts_consumed=attempts_consumed,
                         waited_h=WAIT_LIMIT_S // 3600)
            return 3

        prev_run_id = None
        if args.protocol == "baselines":
            prev_run_id, _ = newest_run_rows()
        log(f"strong idle held; attempt {attempts_consumed + 1}/"
            f"{MAX_ATTEMPTS} starting")
        write_status("measuring", harness_stem,
                     attempts_consumed=attempts_consumed,
                     attempt=attempts_consumed + 1)
        t0 = time.time()
        proc = _run_harness(args)
        minutes = (time.time() - t0) / 60
        last_stderr_tail = _stderr_tail(proc.stderr)
        if proc.stderr:
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n",
                  file=sys.stderr, flush=True)

        outcome, consumed = judge_exit_code(proc.returncode, last_stderr_tail)
        if consumed:
            attempts_consumed += 1

        status_extra = {
            "attempts_consumed": attempts_consumed,
            "exit_code": proc.returncode,
        }
        if proc.returncode != 0:
            status_extra["stderr_tail"] = last_stderr_tail

        if outcome == "waiting":
            log(f"harness refused to start with exit {proc.returncode}; "
                "back to waiting")
            write_status("waiting", harness_stem, **status_extra)
            continue
        if outcome == "gave-up":
            log(f"RESULT: GAVE UP, harness refusal exit {proc.returncode}")
            write_status("gave-up", harness_stem, **status_extra)
            return 3
        if outcome == "crashed":
            log(f"RESULT: CRASHED, harness exit {proc.returncode}")
            write_status("crashed", harness_stem, **status_extra)
            return 4

        if args.protocol == "exit-code":
            log(f"RESULT: {outcome.upper()}, harness exit {proc.returncode}, "
                f"{minutes:.0f} min")
            write_status(outcome, harness_stem, minutes=round(minutes),
                         **status_extra)
            return 0 if outcome == "done" else 1

        run_id, rows = newest_run_rows()
        if run_id == prev_run_id or not rows:
            # A rejection always appends its rows as evidence. No rows means
            # the baselines harness failed before it could make a verdict.
            log(f"RESULT: CRASHED, harness exit {proc.returncode} with no "
                "rows appended; see stderr above")
            write_status("crashed", harness_stem, **status_extra)
            return 4

        rejected = [row for row in rows if not row["binding"]]
        if proc.returncode == 0:
            log(f"RESULT: BINDING run {run_id}, {len(rows)} rows, "
                f"{minutes:.0f} min measuring")
            write_status("binding", harness_stem, run_id=run_id,
                         rows=len(rows), minutes=round(minutes),
                         **status_extra)
            return 0

        log(f"attempt {attempts_consumed}: {len(rejected)} of {len(rows)} "
            f"rows rejected after {minutes:.0f} min:")
        for row in rejected:
            log(f"  {row['row_id'].split('/', 1)[1]}: "
                f"{'; '.join(row['binding_blockers'])}")
        write_status("waiting", harness_stem, run_id=run_id, rows=len(rows),
                     **status_extra)

    log(f"RESULT: NOT BINDING after {MAX_ATTEMPTS} attempts; last run "
        f"{run_id} has {len([row for row in rows if row['binding']])} of "
        f"{len(rows)} rows binding")
    write_status(
        "not-binding",
        harness_stem,
        attempts_consumed=attempts_consumed,
        run_id=run_id,
        rows=len(rows),
        binding_rows=len([row for row in rows if row["binding"]]),
        stderr_tail=last_stderr_tail,
    )
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        log(f"RESULT: CRASHED in the runner itself: {exc!r}")
        parsed = parse_args()
        write_status("crashed", parsed.harness.stem, error=repr(exc))
        raise
