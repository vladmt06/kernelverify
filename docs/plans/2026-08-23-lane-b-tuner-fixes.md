<!-- Provenance: written 2026-08-23 by a Lane B fixer agent in plan mode, so
nothing in it had been executed when it was written. Copied into the repo so a
fresh clone carries it; paths made repo-relative, and worktrees are sibling
directories of the repo root. Referenced from docs/PLAN.md, Lane B row. -->

# Lane B fixer plan: kv-anymac-mr, both review findings

Plan mode is active, so nothing below has been executed yet: no file has
been edited, no test has been run, nothing has been committed. This is the
exact plan I will carry out once released from plan mode. Both findings
were independently confirmed against the actual code in
`the `kv-anymac-mr` worktree` (branch `lane/anymac-metalrunner`, clean working
tree at commit `4c0fd8b`); I re-derived both root causes by reading the
real files rather than trusting the finding text alone.

## Finding 2 first: metalrunner/tuner.py:276, a real accounting bug

### Root cause, confirmed by reading the code

`metalrunner/tuner.py`, inside `ensure_tuned`, around line 128-148 (`_cells`)
and 235-280:

- `_cells(batches)` builds one tuple per `(shape_name, m)` pair in
  `batches`, with **no dedup** by the `key = f"{shape_name}:M{bucket}"` it
  computes. Two raw `batches` entries that share a bucket (an ordinary
  repeated step batch size like `[128, 128]`, or two different raw Ms that
  `m_bucket` folds into one bucket) produce two tuples with the same key.
- The per-cell loop has `if key in verdicts: continue` (line 236) - this
  exists to let a resumed tune skip cells it already decided, but it also
  silently absorbs an in-run duplicate the same way, so `verdicts` and the
  derived `knob_map` end up correctly deduplicated (one verdict per key).
- The report-building code (lines 269-280) computes `knob_map` from
  `verdicts` (correct, deduplicated) but computes `cells_considered` and
  `knob_census["stock"]` from `len(cells)` - the raw, non-deduplicated list.

Result: `cells_considered` and `knob_census` can overstate the true number
of per-cell decisions, and `knob_census` can name a "stock" cell that has
no matching entry in `verdicts` or `knob_map` at all. `knob_map` itself
(what routing reads) is unaffected - this is purely a receipt/audit-trail
accounting bug, exactly as the finding says.

### Fix (root cause, no guard)

Replace the two `len(cells)` uses with `len(verdicts)` (equivalently
`len(knob_map)`, same size), so the report counts what was actually
decided:

```python
    knob_map = {key: v["knobs"] for key, v in verdicts.items()}
    kept = sum(1 for value in knob_map.values() if value != "stock")
    # len(verdicts), not len(cells): two raw cells can share one key (a
    # repeated batch M, or two Ms the same m_bucket collapses together),
    # and `if key in verdicts: continue` above already folds those into one
    # verdict. The report must count what verdicts/knob_map actually hold,
    # not the raw, pre-dedup cell list.
    cells_considered = len(verdicts)
    report = {
        "cache_hit": False,
        "deferred": False,
        "candidate_sha256": candidate,
        "operation": OPERATION,
        "cells_considered": cells_considered,
        "knob_census": {"kept": kept, "stock": cells_considered - kept},
        "verdicts": verdicts,
        "route_to_stock": kept == 0,
    }
```

This is a 2-line semantic change (one new local variable, two call sites
updated). No other line in `ensure_tuned` moves. It does not touch
`knob_map`, `entry`, or anything routing reads.

### Regression check against existing tests (I traced every batches= value)

Every existing test in `tests/test_metalrunner_tuner.py` uses `batches` with
no repeated bucket key (`[128]` alone, or `[128, 512]` with the `shapes`
fixture's identity `m_bucket`), so `len(cells) == len(verdicts)` in every
one of them today. The fix is a no-op for all of them; none needs editing,
per the brief's "if an existing test fails, the code is wrong, not the
test" rule staying satisfied by construction here.

### New test (fails before the fix, passes after)

Add to `tests/test_metalrunner_tuner.py`, in the "5. Keep decision" area,
mirroring the reviewer's exact repro (`batches=[128, 128]` under the
`shapes` fixture's identity `m_bucket`, so both raw Ms collapse onto the
key `"S1:M128"`):

```python
# ---------------------------------------------------------------------------
# The report's counts must match verdicts/knob_map, never the raw,
# pre-dedup cell list: a repeated raw M in `batches` is ordinary (a real
# training run's per-step batch size routinely repeats), and the cell it
# produces a second time must not be double-counted once `if key in
# verdicts: continue` has already folded it into one verdict.
# ---------------------------------------------------------------------------
def test_a_repeated_batch_m_that_collapses_to_one_cell_is_counted_once(
        tmp_path, shapes):
    path = tmp_path / "certified-v1.json"
    outcome = tuner.ensure_tuned(
        model_path="/m", bits=BITS, group_size=GROUP_SIZE, chip=CHIP,
        mlx_version=MLX_VERSION, batches=[128, 128], store_path=path,
        hardware=_hardware(
            verify_cell=lambda shape, m, knobs: True,
            time_cell=lambda shape, m, knobs: _timings([1.0, 1.0], [0.9, 0.9])))

    assert set(outcome.report["verdicts"]) == {"S1:M128"}
    assert outcome.entry["knob_map"] == {"S1:M128": {}}
    assert outcome.report["cells_considered"] == 1
    assert outcome.report["knob_census"] == {"kept": 1, "stock": 0}
    census_total = sum(outcome.report["knob_census"].values())
    assert census_total == len(outcome.entry["knob_map"]), (
        "the report's own census must add up to the entries it actually "
        "produced")
```

Before the fix: `cells_considered == 2`, `knob_census == {"kept": 1,
"stock": 1}`, `census_total == 2 != len(knob_map) == 1` - the test fails on
the `cells_considered == 1` assertion (and would also fail the census/total
assertions). After the fix: all four assertions pass. I will run this
specific test both before and after editing `tuner.py` to confirm red then
green, not just reason about it.

## Finding 1: tests/test_metalrunner_lora.py, the missing cache-hit-through-main() test

### Why this is a real, confirmed gap

I read every test in `tests/test_metalrunner_lora.py` that touches
`tuner.ensure_tuned`. Three exist:

- `test_ensure_tuned_is_called_before_routing_decide_and_its_report_lands_in_the_receipt`
  - monkeypatches `tuner.ensure_tuned` entirely.
- `test_a_raising_tuner_is_reported_as_a_deferral_and_the_run_still_completes`
  - monkeypatches it to raise.
- `test_certified_includes_a_matching_local_store_entry`
  - also monkeypatches `tuner.ensure_tuned` to defer immediately, then
    separately proves a pre-populated local_store entry reaches
    `routing.decide`/`routing.eligible` - it never lets the real
    `ensure_tuned` touch the local_store cache at all.

None drives `main()` with the real `tuner.ensure_tuned`. I then read
`tuner.ensure_tuned`'s own body: the cache check
(`local_store.load(...)` at line 207) runs and returns before the
`hardware is None` check (line 214). `metalrunner/lora.py`'s `main()` calls
`tuner.ensure_tuned(..., hardware=None)` unconditionally (line 156-158) -
so a real, matching local_store entry produces a genuine cache hit through
`main()` today, with no code change needed anywhere. This confirms the
finding's narrower claim exactly (cache-hit is reachable; the brief's other
two scenarios, battery and forced-stock, are not, because `hardware=None`
returns before either check - the report's excuse holds for those two, not
for cache-hit).

### Fix: this is a test-coverage gap only, not a code bug - no production change

I will add one new test to `tests/test_metalrunner_lora.py`, in the "Task 5
wiring" section, right after `test_certified_includes_a_matching_local_store_entry`.
Unlike every existing tuner-wiring test in this file, it does **not**
monkeypatch `tuner.ensure_tuned` at all - only `routing.chip` and
`local_store.DEFAULT_PATH`, the same two seams
`test_certified_includes_a_matching_local_store_entry` already uses. I
deliberately give the model an **unverified** quantization format
(8-bit group-32, the same "unverified format" value already used by
`test_an_unverified_format_is_named_in_the_decline`) rather than 4-bit
group-64. This isolates the test to exactly the cache-hit claim: with an
unsupported format, `routing.eligible` returns `()` regardless of the
local entry, so nothing is routed and `measurement.install` is a no-op -
no need to fake `measurement.OPERATIONS` the way the sibling test does,
because this test is not trying to prove routing, only that the real
`ensure_tuned` reaches a cache hit through `main()`.

```python
def test_a_second_run_cache_hits_through_the_real_unmonkeypatched_tuner(
        stub_trainer, tmp_path, monkeypatch, capsys):
    """Brief line 167, 'second run cache-hits', driven through main() with
    the REAL tuner.ensure_tuned - not monkeypatched, unlike every other
    tuner-wiring test above. tuner.ensure_tuned's own cache check
    (local_store.load) runs before its hardware-is-None check, so a
    pre-populated matching local_store entry produces a genuine cache hit
    here even though lora.py wires hardware=None. The quantization is
    deliberately an unverified format (8-bit group-32) so nothing is
    eligible to route: this test proves only the cache-hit claim, not
    routing, and needs no fake measurement.OPERATIONS row for that."""
    from metalrunner import local_store, routing, tuner
    from metalrunner.versions import require_verified_stack

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    bits, group_size = 8, 32
    (model_dir / "config.json").write_text(json.dumps(
        {"quantization": {"bits": bits, "group_size": group_size}}))

    store_path = tmp_path / "certified-v1.json"
    monkeypatch.setattr(local_store, "DEFAULT_PATH", store_path)
    monkeypatch.setattr(routing, "chip", lambda: "Apple M3 Pro")

    mlx_version = require_verified_stack().mlx
    candidate = tuner.candidate_sha256(bits, group_size)
    entry = {
        "candidate_sha256": candidate,
        "operation": tuner.OPERATION,
        "chip": "Apple M3 Pro", "bits": bits, "group_size": group_size,
        "pricing_recording_sha256": "e" * 64,
        "schema": 2, "provenance": "local-tune", "mlx_version": mlx_version,
        "knob_map": {"S1:M128": {}}, "tuned_at": "t",
        "evidence_sha256": "e" * 64,
    }
    local_store.save(entry)

    code = main(["--model", str(model_dir), "--train",
                "--adapter-path", str(tmp_path / "adapters")])
    assert code == 0

    out = capsys.readouterr().out
    assert "cache hit, using the kernel already tuned on this chip" in out

    record = json.loads(
        (tmp_path / "adapters" / "metalrunner-receipt.json").read_text())
    assert record["tuning"]["cache_hit"] is True
    assert record["tuning"]["candidate_sha256"] == candidate
    # Unverified format: nothing was eligible, nothing routed. This test is
    # about the cache hit only, not routing.
    assert record["measurement"]["routed_calls"] == 0
```

I traced this line by line against the real `main()` and confirmed it
produces a receipt with `tuning.cache_hit == True` with no other code
change: `read_quantization` returns `(8, 32)`; `tuner.ensure_tuned`'s cache
check finds the saved entry (chip, mlx_version, candidate_sha256 all
match) and returns before ever looking at `hardware`; `routing.decide` and
`routing.eligible` both decline everything because 8-bit group-32 is not
the verified format, so `measurement.install` receives an empty candidate
list and is a no-op; `routing.render`'s `_tuning_summary` prints the exact
cache-hit line because `tuning.get("cache_hit")` is checked before
`deferred`.

### Proving the test is load-bearing (not a coincidence)

Since this is a pure test addition with no code fix behind it, "fails
before, passes after" does not apply in the usual sense - there is no
buggy line to flip. Instead I will prove the test is meaningful by
temporarily reordering `ensure_tuned`'s own checks (swap the cache check
and the `hardware is None` check) on a throwaway basis, rerunning only this
new test, confirming it fails, then reverting the temporary reorder before
committing. That demonstrates the test actually depends on the real
cache-before-hardware-check ordering rather than passing for an unrelated
reason.

## Execution order and verification

1. `metalrunner/tuner.py` fix + its new test in
   `tests/test_metalrunner_tuner.py` - commit 1.
   - Run `tests/test_metalrunner_tuner.py` alone before the fix (confirm
     the new test fails on `cells_considered`/`knob_census`), apply the
     fix, rerun (confirm green).
   - Then both full-suite commands from the brief.
   - `git add metalrunner/tuner.py tests/test_metalrunner_tuner.py`
   - `git commit` (no em dashes, describes the accounting bug and the fix).
2. New test only in `tests/test_metalrunner_lora.py` - commit 2.
   - Run the new test alone first to confirm it passes against the real,
     unmonkeypatched `ensure_tuned` (this finding has no code bug, so there
     is no red state to start from - I substitute the reorder-and-revert
     check described above to prove it is load-bearing).
   - Then both full-suite commands again.
   - `git add tests/test_metalrunner_lora.py`
   - `git commit`.
3. Run, from `the `kv-anymac-mr` worktree`:
   - `.venv/bin/python -m pytest -q`
   - `KV_FORCE_NO_METAL=1 .venv/bin/python -m pytest -q -m gpu`
   and record the true pass/fail/error/skip counts from actual output, not
   estimated ones. The implementer's report already documents a pre-existing,
   unrelated baseline of 8 failed / 50 errors on the full suite (missing
   `bench/.models/` artifacts and a missing salt file, both outside
   `metalrunner/`), confirmed by stashing Lane B's changes and rerunning; I
   will re-confirm that baseline is unchanged by my two commits rather than
   assume it.
4. Append a fix report to
   `.superpowers/sdd/keen-inventing-ritchie/laneB-report.md`
   describing both fixes, the new tests, and the true suite counts.

## Both findings: assessed as real

Neither finding is a false positive. Finding 2 is a genuine, currently-live
accounting bug in `tuner.py`'s report (confirmed by reading `_cells`, the
per-cell loop, and the report-building code directly, and by tracing that
no existing test's `batches` value exercises the duplicate-key path).
Finding 1 correctly identifies that the shipped report's excuse ("hardware
is None short-circuits before cache/battery/forced-stock") is true for
battery and forced-stock but factually wrong for cache-hit, because the
cache check in `tuner.ensure_tuned` runs before the hardware-is-None check
- confirmed by reading the exact line order in `ensure_tuned`.

## What I have NOT done (plan mode)

No file under `the `kv-anymac-mr` worktree` has been edited. No test has been
run. No commit exists. No report has been appended. Everything above is
the plan to execute once plan mode is lifted.
