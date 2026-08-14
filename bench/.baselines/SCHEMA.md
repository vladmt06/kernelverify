# Baseline JSONL schema, version 3

Version 3 changes what a sampling group is.
A v2 group was one harness invocation: every spec in the run shared one group and was sampled in one all-specs round-robin, so the two arms of any A/B sat a full rotation - minutes - apart.
A v3 group is one workload cell: exactly the specs that realise one workload (same kind, matmul width, and prefill length), with the arms alternating back to back every round.
A v3 group is therefore one real A/B session by construction, never a relabel of a run-wide rotation, and `sampling.group` is `<run_id>/<cell label>` rather than the bare run id.
Version 3 also adds `model.logical_name` (required on measurement rows, one canonical spelling from a constrained set) and constrains `roofline.binding_resource` to exactly `compute`, `memory`, `unknown`.
Version 2 added `sampling` (how the row was interleaved) and the dispersion fields on `result`, both of which gate what a consumer may do with a row.
Version 1 rows were produced before those gates existed, were never consumed, and were discarded rather than left in the record claiming a bindingness they had not been checked for.
An earlier revision of this document said version 2 in its header while the field table below still said 1; the table is part of the contract, so it now states the version it documents and must move with the header.

Produced by `bench/measure_baselines.py`, consumed by the matrix renderer.
One JSON object per line, append-only, one file per UTC date: `<YYYY-MM-DD>.jsonl`.
Rows are never edited or deleted; a re-measurement is a new row with a new `run_id`.

The producer validates every row against this contract before writing it (`validate_row`), so a missing field fails the measurement run rather than silently blanking a published column.

## Two different questions: may this row be published, and may it be compared

They have different answers and different fields, so a consumer must not use one to decide the other.

**Publishable as an absolute number** needs `provenance_tier` in (`owner-run`, `rental-run`) and `binding: true`.

**Comparable against another row** additionally needs both rows to carry `sampling.interleaved: true` and the same `sampling.group`.
This is not pedantry: the kernels lane ran an A/B whose every dispatch was already past 5 ms, and it still reversed sign against an interleaved rerun, turning a 1.45x win into 0.77-0.94x.
One fixed shape drifted 131.7 to 93.1 us between runs minutes apart.
An A/B in separate passes measures the clock, not the kernels, so batching to a millisecond does not rescue it.
Rows from different `group`s may both be binding and still not be comparable to each other.

## The two fields that decide whether a row may be published as fact

`provenance_tier` is one of `owner-run`, `rental-run`, `community-unattested`.
Only the first two are measured by someone the project controls.
Unattested rows are claims and must be rendered separately, never merged into a binding result.

`binding` is a boolean, with `binding_blockers` listing every reason it is false.
A row fails to bind when the machine was busy, on battery, in low power mode or thermally warned at either end of the run, when the fastest sample fell below the 1 ms timing floor, or when the repeats disagreed by more than `result.max_spread_pct`.

That last gate is not implied by the first: a run reported load average 1.93 from start to finish while one spec's third sample came in at 22.07 against 99.38 and 90.06.
A one-minute load average cannot see a transient that lands inside a single sample, so an idle machine is not evidence that a measurement succeeded, and `result.samples` is kept on every row so a consumer can check rather than trust.
`binding: false` rows are kept deliberately: they are still valid for ratios measured in the same run, and they are the evidence for why a number was rejected.

A published absolute number requires `provenance_tier` in (`owner-run`, `rental-run`) **and** `binding: true`.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | 3 |
| `run_id` | str | one per harness invocation, `<utc timestamp>-<8 hex>` |
| `row_id` | str | `<run_id>/<spec id>`, unique across all files |
| `measured_at` | str | ISO 8601, UTC |
| `provenance_tier` | str | see above |
| `binding` | bool | see above |
| `binding_blockers` | list[str] | empty iff `binding` |
| `machine` | object | chip, hw_model, cores, performance_cores, efficiency_cores, memory_bytes, os, os_build |
| `idle_before`, `idle_after` | object | load1/5/15, load_threshold, power {source, low_power_mode, thermal_warning}, idle, blockers |
| `stack` | object | name, version, plus build_flags and path where they exist; `mlx-lm` rows also carry `mlx_version` |
| `model` | object or null | name, `logical_name`, path, `sha256_32`, quant. Null on ceiling rows. `logical_name` (v3) is the cross-stack identity of what ran, one canonical spelling from a constrained set (currently `qwen3-4b`); `name` is the per-stack artefact and differs across stacks for one logical model |
| `measurement` | object | kind, matmul_width, n_prompt, n_gen, width_mechanism |
| `result` | object | metric, median, spread_pct, reps, samples, min_sample_ms, floor_ms, below_timing_floor |
| `sampling` | object | interleaved, rotation, rounds, group, group_members. Required on every non-ceiling row. v3: `group` is one workload cell (`<run_id>/<cell label>`), `group_members` are its arms only, `rotation` is `arm-alternation` |
| `roofline` | object | present on every non-ceiling row, see below |

### `measurement.kind`

| Kind | Meaning |
|---|---|
| `ceiling` | a machine ceiling, not a model run. `model` is null |
| `decode` | single-stream generation, matmul width 1 |
| `prefill` | prompt processing at `n_prompt` tokens |
| `matmul_width` | the weight matmul's n dimension swept as a first-class row |

`measurement.width_mechanism` is `prompt-width` on llama.cpp and `batch-size` on MLX.
Both widen the same matmul dimension, but they are not the same workload: prompt width is one causal sequence, batch size is independent streams.
Render the mechanism next to the width or the two stacks will look more comparable than they are.

### `roofline`

| Field | Meaning |
|---|---|
| `bytes_per_pass` | from the model's own tensor table, never file size |
| `byte_model` | `tensor-table` |
| `denominator_name`, `denominator_gbs` | which bandwidth ceiling was divided by, and its value |
| `achieved_gbs`, `bandwidth_utilisation_pct` | achieved bandwidth and its share of that ceiling |
| `binding_resource` | exactly one of `compute`, `memory`, `unknown` (validated at v3); which ceiling actually binds this row. `unknown` marks rows no parameter count exists to place |
| `achieved_gflops`, `roofline_ceiling_gflops`, `roofline_utilisation_pct` | compute-side placement, null on MLX rows |
| `arithmetic_intensity_flop_per_byte`, `ridge_flop_per_byte` | position relative to the ridge |

**Pick the column by `binding_resource`.**
A prefill row read as a bandwidth percentage looks like a 1% catastrophe when it is a 60% compute result.
`flops_model` appears on MLX rows to say why the compute columns are null: the checkpoint gives bytes but no unambiguous parameter count, and a guessed one would be a fabricated published number.

## Ceilings

Ceiling rows carry `measurement.name` in `bandwidth_read`, `bandwidth_copy`, `fp16_fma`, `fp32_fma`, with `result.metric` of `gbs` or `gflops`.
Use `bandwidth_read` for decode-dominated rows: decode streams weights and barely writes, so the copy figure understates the ceiling and overstates utilisation by about 5% on this machine.

## Notes for the renderer

- v3: rows from one `sampling.group` are the arms of one workload cell, measured back to back with the arms alternating every round, so they are comparable to each other even when the machine drifted underneath them.
  v2: a group was a whole run's all-specs round-robin; the renderer additionally keys comparison cells on the workload, which is what made v2 groups safe to consume.
  In both versions, rows from different groups are not paired.
- The renderer accepts both versions but refuses any comparison across schema versions: the meaning of a group changed at the v3 boundary, so a cross-version ratio would pair rows measured under two different contracts.
- For v3 rows a comparison additionally requires the same `model.logical_name`; v2 rows predate the field and keep their original gates.
- Refuse to render a comparative claim across rows whose `sampling.interleaved` is false or absent, the same way absolutes are refused when `binding` is false. A contributed row from someone else's harness is the case this exists for.
- `result.spread_pct` is `(max - min) / median` across repeats. Treat a difference smaller than the spread as no difference.
- A `run_id` may contain both binding and non-binding rows only if the machine changed state mid-run; `idle_before` and `idle_after` say which end moved.
