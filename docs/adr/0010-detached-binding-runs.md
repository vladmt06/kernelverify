# ADR 0010: Detached binding runs - the sessions are the contention

Date: 2026-08-14
Status: Accepted

## Context

The baseline harness refuses to call a row binding when the machine was busy, unplugged, or its repeats disagreed (ADR 0007, commits 8a9bee8 and e2234b0).
Those gates are working, and what they keep rejecting is the measurement environment itself.

Two measurements forced this decision:

- The 2026-08-14 14:57Z run (`bench/.baselines/2026-08-14.jsonl`, run `20260814T145747Z-6bc2d222`) had the idle gate pass at both ends, and still lost 6 of 10 model rows to the dispersion gate, with spreads of 12.7% to 131.5% against the 10% limit.
- The kernels lane measured 2.2x-9.7x spreads on an AC-powered machine at load average 1.79, caused by Terminal at 36% and VS Code at 25% CPU driving the display stack.

The interactive sessions corrupt GPU timing from below every threshold the gates can observe.
macOS offers no unprivileged GPU utilisation reading, so the contention gate's CPU proxy cannot be sharpened into seeing it (`bench/machine_state.py`).
Per this repo's working rules, the fix for that is not a bigger gate; it is removing the cause.
A machine with zero interactive sessions cannot be produced by a harness that needs a terminal to stay open, so the harness must be startable in a way that survives every session being closed.

## Decision 1: runs start as a one-shot launchd job, never from a terminal

`bench/start_binding_run.sh` generates a plist into `bench/.cache/` and bootstraps it into the user's gui launchd domain.
The job runs `bench/detached_run.py` under `caffeinate -i`, so the machine will not idle-sleep while waiting overnight, while the display still may (a dark display quiets WindowServer, which is the point).
`ProcessType` is `Interactive`, so launchd does not down-schedule the benchmark the way it does background jobs.
The plist deliberately never enters `~/Library/LaunchAgents`, so it cannot re-fire at login; re-running the script replaces any previous instance.
Output appends to `bench/.baselines/detached.log`, and `bench/.baselines/detached_status.json` always holds the current state, so the outcome is readable without scrolling.

`nohup` from a closing terminal would also survive, but it starts inside a terminal session and inherits its environment; the launchd job never has a controlling terminal at any point in its life, which is the productized form of the condition being created.

## Decision 2: a strong-idle start condition, separate from the bindingness gates

The runner starts a pass only after 5 consecutive clean samples 30 seconds apart, where clean means the plain idle gate passes and the five-minute load is also under the threshold.
load1 recovers within a minute of closing apps while indexers and page-outs are still settling; load5 under the threshold means the machine has actually been quiet, not just momentarily.
This is a start condition, not a bindingness condition: every row is still judged by the unchanged gates in `measure_baselines.py`, so a detached start can never make a bad number bind.

## Decision 3: bounded append-only retries

A pass whose repeats disagree is re-run after re-reaching strong idle, up to 3 passes; every pass appends its rows to the JSONL, because a rejection is evidence, not an absence.
A harness refusal (exit 2, machine went busy between samples) does not consume an attempt.
A pass that exits non-zero without appending rows is a crash, and the runner stops rather than retries, because a deterministic failure re-run three times is noise in the log, not persistence.

## Result

The first detached run is armed as of this ADR; its outcome will be appended here with the measured pass duration.

## Pre-registered next step

If a zero-session detached run still loses rows to dispersion across all 3 passes, the pre-registered suspect is power management rather than session contention, and the upgrade is sampling machine state around each sample rather than only at run ends, to localize which sample the transient landed in.
Adopt it only when such a run actually happens, per the ADR 0002/0003 discipline.
