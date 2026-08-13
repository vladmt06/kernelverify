# gpuemu-corpus

A 26-op kernel-correctness corpus, the experiment drivers that exercise it,
and the analysis scripts that produce every table and figure in the gpuemu
research program.

## Contents

| Directory | What it holds |
|---|---|
| `gpuemu_corpus/` | Python package: 16 correct controls + 10 LLM-style buggy variants. Each op has `meta.json`, `ref_fp64.py`, and `kernel.py`. |
| `drivers/` | Experiment drivers cited in the papers' §Method sections: `p1_llm_kernels.py`, `p3_strategies.py`, `p4_artifacts.py`. |
| `analysis/` | Per-paper table and figure generators: `p1_table.py`, `p2_calibrate.py`, `p3_strategy_table.py`, `p4_correlation.py`, `make_figures.py`. |
| `scripts/` | `replay_from_b2.py` (fetch a stored run and re-run analysis), `run_p4_cross_gpu.py` (orchestrate the five-GPU sweep), `vendor_corpus.py` (build the package from a checkout). |

The corpus is the canonical source for the four gpuemu preprints:

- **P1** — *The Correctness Illusion in LLM-Generated GPU Kernels*. [arXiv:2606.20128](https://arxiv.org/abs/2606.20128).
- **P2** — *Operator-Aware Mixed-Precision Tolerance Calibration for Tensor Kernels*. [arXiv:2607.16228](https://arxiv.org/abs/2607.16228).
- **P3** — *Test-Input Generation for Tensor Programs: What Actually Finds Kernel Bugs*. [arXiv:2606.27396](https://arxiv.org/abs/2606.27396).
- **P4** — *Static PTX Metrics Track Structural Kernel Regressions but Miss Semantic Ones*. [arXiv:2607.02541](https://arxiv.org/abs/2607.02541).

## Install

```bash
pip install git+https://github.com/sarkar-dipankar/gpuemu-corpus.git
```

For local development:

```bash
git clone https://github.com/sarkar-dipankar/gpuemu-corpus.git
cd gpuemu-corpus
pip install -e .
```

The corpus is loadable from Python without the daemon. To run the validator
side, install the Rust daemon from
[`Skelf-Research/gpuemu`](https://github.com/Skelf-Research/gpuemu).

## Reproduce a paper

Each run record cited in the papers lives on Backblaze B2 at
`sarkar-dipankar-research/gpuemu/<run-id>/`. The replay script fetches the
record and re-runs the analysis locally; no GPU required.

```bash
# P1: bug-class verdict table + cross-GPU consistency
python3 scripts/replay_from_b2.py --run-id run-20260611-080050-a70b33
python3 analysis/p1_table.py --data-dir data

# P2: per-(op, dtype) tolerance calibration
python3 scripts/replay_from_b2.py --run-id run-20260611-085027-0b8ddc
python3 analysis/p2_calibrate.py --data-dir data --corpus gpuemu_corpus/data

# P3: seven-strategy ablation
python3 scripts/replay_from_b2.py --run-id run-20260611-101922-4dcac1
python3 analysis/p3_strategy_table.py --data-dir data \
    --run-id run-20260611-101922-4dcac1

# P4: static PTX metrics paired with measured perf
python3 scripts/replay_from_b2.py --run-id run-20260611-142511-884321
python3 analysis/p4_correlation.py --data-dir data \
    --run-id run-20260611-142511-884321
```

To regenerate every figure used in the papers:

```bash
python3 analysis/make_figures.py \
    --p1 run-20260611-080050-a70b33 \
    --p3 run-20260611-101922-4dcac1 \
    --p4 run-20260611-142511-884321
```

The full set of run records and their GPU class is in P4 Table 5.

## Run a new experiment

A live GPU is required for this path. The drivers send each kernel to the
gpuemu daemon, capture static PTX metrics and CUDA-event-timed perf, and
write one `results.jsonl` row per (kernel, dtype, iter).

```bash
# Bring up the daemon (separate terminal)
gpuemu daemon start --config gpuemu.toml

# P1: every kernel under the standard oracle
python3 drivers/p1_llm_kernels.py --run-id my-p1-run --iters 8

# P3: seven generation strategies × all kernels × 8 iters
python3 drivers/p3_strategies.py --run-id my-p3-run --iters 8

# P4: static metrics + perf timing
python3 drivers/p4_artifacts.py --run-id my-p4-run --iters 50
```

Register the corpus with the daemon first:

```python
from pathlib import Path
from gpuemu_corpus import register_all

register_all(client=None, toml_path=Path("gpuemu.toml"))
```

## Corpus details

The 16 correct controls are 13 Triton kernels (`softmax_triton`,
`gelu_triton`, `silu_triton`, `rmsnorm_triton`, `l2norm_triton`,
`leaky_relu_triton`, `relu_triton`, `sigmoid_triton`, `tanh_triton`,
`elu_triton`, `matmul_triton`, `attention_triton`,
`flash_attention_triton`) plus 3 numpy baselines (`softmax`, `layernorm`,
`matmul`).

The 10 buggy variants seed documented LLM-transcription patterns:
`softmax_llm_buggy`, `softmax_triton_buggy`, `gelu_triton_buggy`
(drops the leading `0.5` factor), `silu_triton_buggy`,
`rmsnorm_triton_buggy`, `l2norm_triton_buggy`, `leaky_relu_triton_buggy`,
`matmul_triton_buggy` (`acc=` instead of `acc+=`),
`attention_triton_buggy` (missing `1/sqrt(D)` scale), and
`flash_attention_triton_buggy` (missing `acc * alpha` rescale on max
update).

Each op ships:

- `meta.json` — name, source, benchmark verdict, input names, reference path, kernel path, dtypes, per-op tolerances, and `op_schema` for the fuzzer's shape generator.
- `ref_fp64.py` — the high-precision fp64 reference.
- `kernel.py` — the kernel under test.

## License

Dual-licensed under MIT OR Apache-2.0.
