# Kernel corpus

Each kernel lives in its own directory and ships:

- `kernel.py` — the kernel under test (human-written or LLM-generated).
- `ref_fp64.py` — a high-precision (fp64) CPU reference, wired as the gpuemu daemon
  `reference` script. The oracle is independent of the kernel's own precision.
- `meta.json` — provenance: `source` (`human` | benchmark name), the benchmark's own
  `passing` verdict (for P1's contrast), op `schema` (matmul/attention/elementwise/...),
  and input/output names.

Op registrations for the daemon are generated into `gpuemu.toml` from each `meta.json`.

> Populated by task #3. Start with `softmax`, `layernorm`, `matmul`.
