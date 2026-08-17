# Detached runs: operator card

One command arms a long harness, and the machine runs it after you leave it alone.

## Start a run

1. Plug the Mac into AC power.
2. In Terminal, change to the repository: `cd /Users/vlad/kernelverify`.
3. Arm the required harness with one of these commands:

       bench/start_binding_run.sh
       bench/start_binding_run.sh bench/calibrate_quant_serving.py
       bench/start_binding_run.sh bench/serve_sub4bit.py -- --ab

4. When it prints `armed`, quit every app, including Terminal, Chrome, VS Code, and Slack.
5. Leave the lid open and walk away.

Arguments after `--` are passed to the selected harness.
The default command runs `measure_baselines.py` with its baselines retry protocol.
Every other harness uses the project's numbered exit-code protocol.

## Check a run

Each harness writes its own log and status file, so runs cannot overwrite another harness's outcome.

    tail -3 /Users/vlad/kernelverify/bench/.baselines/detached-<harness-stem>.log
    cat /Users/vlad/kernelverify/bench/.baselines/detached_status-<harness-stem>.json

For example, `serve_sub4bit.py` writes `detached-serve_sub4bit.log` and `detached_status-serve_sub4bit.json`.
The status records how many real attempts were consumed and includes the last 40 stderr lines after every non-zero harness exit.

| Status | Meaning |
|---|---|
| `done` or `binding` | The harness completed successfully. |
| `stopped` or `not-binding` | The harness measured and deliberately returned a non-success verdict. |
| `waiting` | The machine is busy, another harness holds the lock, or available memory is temporarily low. |
| `gave-up` | The 12-hour idle deadline expired or a permanent precondition refused the run. |
| `crashed` | The harness failed, and `stderr_tail` carries the end of its traceback or error output. |

## What happens while you are gone

- The runner waits for five clean idle samples in a row and for the five-minute load to settle.
- The selected harness takes the one machine-wide measurement lock before its own idle gate and measuring body.
- A transient idle, lock, or memory refusal returns to waiting without consuming an attempt.
- A budget or permanent precondition refusal gives up, while interpreter, child, and orphan failures report a crash.
- The baselines protocol retries rejected rows up to three measured passes and appends every pass as evidence.
