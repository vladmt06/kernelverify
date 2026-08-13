# Experiment drivers

A driver runs on the compute host (local or vast.ai) and writes `results.jsonl`
(+ `summary.json`) into its `--out` directory. The harness collects that directory
into the object store under the run id.

Common CLI contract (all drivers accept these):
`--run-id --paper --iters --kernels --seed --out`

| Driver | Purpose |
|--------|---------|
| `smoke.py` | No-GPU lifecycle check; emits synthetic results in the real schema. |
| `p1_llm_kernels.py` | Fuzz the corpus and contrast gpuemu's verdict with each kernel's benchmark verdict; record failure category + minimal failing case. |

Helpers (not drivers): `_p1lib.py` (corpus loading, oracles, compare/error-stats),
`_capture.py` (PTX/SASS extraction + CUDA-event timing for the P4 artifact track).

### `p1_llm_kernels.py` oracles
- `--oracle local` (default): numpy mini-fuzzer + each op's fp64 reference script,
  compared with the op's tolerances; mirrors gpuemu's validator semantics. No
  daemon / no `pynng` needed — runs anywhere.
- `--oracle daemon`: the canonical path — gpuemu-py `get_test_batch` + `submit_output`
  against a running daemon. Used on the GPU image.
- `--oracle auto`: daemon if importable, else local.

`results.jsonl` row schema (superset; drivers fill what applies): `run_id, paper,
kernel, source, benchmark_verdict, iteration, seed, dtype, layout, shape,
input_shapes, passed, failure_kind, max_abs_err, max_rel_err, max_ulp,
error_stats{count,num_exceeding,max_abs,mean_abs,p50_abs,p90_abs,p99_abs,max_rel,
mean_rel,max_ulp,mean_ulp}, oracle`.

`summary.json` adds per-kernel roll-ups + `extra_bugs_found` (kernels the benchmark
calls "pass" that gpuemu shows to fail). `analysis/p1_table.py <run_id>` renders the
headline table from `results.jsonl`.
