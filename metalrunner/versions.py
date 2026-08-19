"""What stack this was verified against, and the refusal when it is not that.

Everything metalrunner does rests on knowing exactly where mlx-lm's trainer
puts its seams. Those seams are not a public API: they are module-level
functions this package reaches into, and a release that moves one of them
would leave metalrunner swapping a kernel into a place that no longer means
what it meant when the kernel was verified.

So the pin is checked before anything is installed, and a mismatch refuses
the whole run rather than degrading. That direction is deliberate. Silently
falling back to stock would be friendlier and would also mean a user could
believe they were running verified kernels while running none, which is the
one failure this project exists to prevent.

Version strings alone are not enough, because a patch release can move a
function without moving the version a user sees. The files this package
reads are hashed too, so a changed seam is caught even when the version
string did not move.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path

# The stack sprint 1 verified against, recorded 2026-08-19.
MLX_VERSION = "0.32.0"
MLX_LM_VERSION = "0.31.3"

# The mlx-lm files metalrunner depends on the internals of. `lora.py` for the
# parser and the argument merging main() performs without a hook, and the
# trainer for the step it wraps.
SEAM_HASHES = {
    "lora.py": "3f188fc6aef80efcb9938678af0548588122ed25845053cc555aece0ad2da5e7",
    "tuner/trainer.py": "ee33ebdbd20a184108541cb490d08085485e71a82ffd6d68d7d216029ecd28fe",
    "tuner/utils.py": "166eaf5e5f923113bed43614a5fb7319795fa0cac5a7fa319ea54e5f0045b553",
}


class UnverifiedStack(RuntimeError):
    """The installed mlx or mlx-lm is not the one this was verified against."""


@dataclass(frozen=True)
class Stack:
    mlx: str
    mlx_lm: str
    seams: dict


def observed(mlx_lm_root: Path | None = None) -> Stack:
    """What is actually installed, read rather than assumed."""
    import mlx.core as mx
    import mlx_lm

    root = Path(mlx_lm.__file__).parent if mlx_lm_root is None else mlx_lm_root
    seams = {}
    for name in SEAM_HASHES:
        path = root / name
        seams[name] = (hashlib.sha256(path.read_bytes()).hexdigest()
                       if path.exists() else "missing")
    return Stack(mlx=mx.__version__,
                 mlx_lm=importlib.metadata.version("mlx-lm"),
                 seams=seams)


def differences(stack: Stack) -> list[str]:
    """Every way this stack is not the verified one, in plain words."""
    problems = []
    if stack.mlx != MLX_VERSION:
        problems.append(f"mlx is {stack.mlx}, verified against {MLX_VERSION}")
    if stack.mlx_lm != MLX_LM_VERSION:
        problems.append(
            f"mlx-lm is {stack.mlx_lm}, verified against {MLX_LM_VERSION}")
    for name, expected in SEAM_HASHES.items():
        found = stack.seams.get(name, "missing")
        if found != expected:
            problems.append(
                f"mlx_lm/{name} has changed since it was verified "
                f"({found[:12]} against {expected[:12]})")
    return problems


def require_verified_stack() -> Stack:
    """The gate every entry point passes through before touching anything."""
    stack = observed()
    problems = differences(stack)
    if problems:
        raise UnverifiedStack(
            "metalrunner will not patch a stack it has not verified:\n  "
            + "\n  ".join(problems)
            + "\n\nNothing was changed. Run mlx_lm.lora directly, or install "
              f"mlx {MLX_VERSION} and mlx-lm {MLX_LM_VERSION}.")
    return stack
