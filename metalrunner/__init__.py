"""Verified kernels for QLoRA fine-tuning on Apple Silicon.

The user-facing package. `python -m metalrunner.lora` takes every argument
`mlx_lm.lora` takes, checks that the installed mlx and mlx-lm are the ones
its kernels were verified against, reports what it routed and what it
declined, runs mlx-lm's own trainer, and writes a receipt beside the
adapter.

The verifier itself is `kernelverify`, which this package depends on and
which keeps its own name: one is the library that decides what is correct,
the other is what a person installs and runs.
"""

__version__ = "0.1.0"
