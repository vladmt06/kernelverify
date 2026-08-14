# Baseline JSONL schema, version 1

Produced by `bench/measure_baselines.py`, consumed by the matrix renderer.
One JSON object per line, append-only, one file per UTC date: `<YYYY-MM-DD>.jsonl`.
Rows are never edited or deleted; a re-measurement is a new row with a new `run_id`.

The producer validates every row against this contract before writing it (`validate_row`), so a missing field fails the measurement run rather than silently blanking a published column.

## The two fields that decide whether a row may be published as fact

`provenance_tier` is one of `owner-run`, `rental-run`, `community-unattested`.
Only the first two are measured by someone the project controls.
Unattested rows are claims and must be rendered separately, never merged into a binding result.

`binding` is a boolean, with `binding_blockers` listing every reason it is false.
A row fails to bind when the machine was busy, on battery, in low power mode or thermally warned at either end of the run, or when the fastest sample fell below the 1 ms timing floor.
`binding: false` rows are kept deliberately: they are still valid for ratios measured in the same run, and they are the evidence for why a number was rejected.

A published absolute number requires `provenance_tier` in (`owner-run`, `rental-run`) **and** `binding: true`.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | 1 |
| `run_id` | str | one per harness invocation, `<utc timestamp>-<8 hex>` |
| `row_id` | str | `<run_id>/<spec id>`, unique across all files |
| `measured_at` | str | ISO 8601, UTC |
| `provenance_tier` | str | see above |
| `binding` | bool | see above |
| `binding_blockers` | list[str] | empty iff `binding` |
| `machine` | object | chip, hw_model, cores, performance_cores, efficiency_cores, memory_bytes, os, os_build |
| `idle_before`, `idle_after` | object | load1/5/15, load_threshold, power {source, low_power_mode, thermal_warning}, idle, blockers |
| `stack` | object | name, version, plus build_flags and path where they exist; `mlx-lm` rows also carry `mlx_version` |
| `model` | object or null | name, path, `sha256_32`, quant. Null on ceiling rows |
| `measurement` | object | kind, matmul_width, n_prompt, n_gen, width_mechanism |
| `result` | object | metric, median, spread_pct, reps, samples, min_sample_ms, floor_ms, below_timing_floor |
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
| `binding_resource` | `memory` or `compute`; which ceiling actually binds this row |
| `achieved_gflops`, `roofline_ceiling_gflops`, `roofline_utilisation_pct` | compute-side placement, null on MLX rows |
| `arithmetic_intensity_flop_per_byte`, `ridge_flop_per_byte` | position relative to the ridge |

**Pick the column by `binding_resource`.**
A prefill row read as a bandwidth percentage looks like a 1% catastrophe when it is a 60% compute result.
`flops_model` appears on MLX rows to say why the compute columns are null: the checkpoint gives bytes but no unambiguous parameter count, and a guessed one would be a fabricated published number.

## Ceilings

Ceiling rows carry `measurement.name` in `bandwidth_read`, `bandwidth_copy`, `fp16_fma`, `fp32_fma`, with `result.metric` of `gbs` or `gflops`.
Use `bandwidth_read` for decode-dominated rows: decode streams weights and barely writes, so the copy figure understates the ceiling and overstates utilisation by about 5% on this machine.

## Notes for the renderer

- Rows from one `run_id` were measured interleaved round-robin, so they are comparable to each other even when the machine drifted. Rows from different `run_id`s are not paired.
- `result.spread_pct` is `(max - min) / median` across repeats. Treat a difference smaller than the spread as no difference.
- A `run_id` may contain both binding and non-binding rows only if the machine changed state mid-run; `idle_before` and `idle_after` say which end moved.
