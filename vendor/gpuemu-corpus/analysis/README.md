# Analysis

Scripts/notebooks that turn raw `results.jsonl` (under `../data/<run_id>/`) into the
figures and tables in `../papers/`. Each paper's headline result must be regenerable
from raw data by a single command — the per-paper "science gate".

- `summarize.py` — quick pass/fail + failure-kind breakdown for any run (works on smoke data).
