# Binding run: operator card

One command arms it; the machine then measures itself once you leave it alone.

## Do this

1. Plug the Mac into AC power.
2. In Terminal: `cd /Users/vlad/kernelverify && bench/start_binding_run.sh`
3. When it prints `armed`, quit every app: Terminal itself, Chrome, VS Code, Slack, all of it.
   Leave the lid open; the display may sleep, the machine will not.
4. Walk away for about an hour.

## How you know it bound

Reopen Terminal later and run:

    tail -3 /Users/vlad/kernelverify/bench/.baselines/detached.log

| Last `RESULT` line | Meaning |
|---|---|
| `RESULT: BINDING run <id>, 14 rows` | done; the rows are in `bench/.baselines/<date>.jsonl` under that run id |
| `RESULT: NOT BINDING ...` | three passes ran and repeats still disagreed; the rejected rows and reasons are right above in the log |
| `RESULT: GAVE UP ...` | no quiet window appeared within 12 h; rerun step 2 |
| `RESULT: CRASHED ...` | the harness broke; the traceback is above it in the log |
| no `RESULT` line yet | still waiting or measuring; `bench/.baselines/detached_status.json` says which |

## What it does while you are gone

- Waits until the machine has been continuously quiet for 2 minutes: the idle gate passing on 5 samples in a row, 5-minute load settled, on AC, no busy processes.
- Then runs the full harness: ceilings first, then every llama.cpp and MLX row, interleaved round-robin.
- One pass takes roughly 25 minutes; if any row's repeats disagree by more than 10%, it waits for quiet again and re-runs, up to 3 passes.
- Every pass is appended to the JSONL either way; a rejected row is recorded evidence, not a hole.
