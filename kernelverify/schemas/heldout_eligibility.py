"""Per-cell held-out eligibility for the quantization contract (ADR 0014).

Why this module exists
----------------------
The serving calibration admits two held-out implementations as "correct code
we did not write": `block-tiled` and `mlx-on-device` (MLX's own
`quantized_matmul`). Admissibility, though, is a property of the KERNEL an
implementation dispatches, and MLX dispatches by (batch, dtype): at batch 1
it runs `affine_qmv` / `affine_qmv_fast`, whose per-thread sub-sum of eight
activations is evaluated in half precision at fp16 activations, and at batch
12+ it runs `affine_qmm_t`, whose dequantized weight tile lives in
`threadgroup T*` - half at fp16. Contract clause C1 (working precision
floor, `kernelverify/tolerance/contract.py`) rules any narrower-than-fp32
intermediate OUT of contract, so in those cells `mlx-on-device` is not an
admissible implementation and its error is not evidence about the floor.

The evidence bar for an entry here is STRUCTURAL (ruling D2, ADR 0014): a
verified reading of the shipped kernel source showing a reduced-precision
intermediate. Mechanism proofs - like the bit-exact B1 emulation in
`bench/.cache/b1-cliff-investigation.md` - are corroboration, never a
prerequisite, because an outcome-driven bar ("exclude it once it fires")
would let the labelling chase the numbers.

This table is versioned contract DATA, not harness code: excluding a
held-out changes what the verifier promises, so the exact contents are
pinned by `tests/test_heldout_eligibility.py` and any edit is a contract
edit with an ADR behind it.

The hash guard
--------------
Every exclusion was ruled against ONE kernel source: the `quantized.h`
shipped inside the mlx_metal 0.32.0 wheel (its sha256 below matches the
wheel RECORD entry). If MLX fixes the half-precision sub-sum, the exclusion
must not outlive the bug, so `exclusion_for` verifies the installed wheel
before applying any exclusion and FAILS LOUDLY on mismatch: a changed
source hash is a stale ruling even at an unchanged version string, and an
unreadable source falls back to requiring the exact ruled MLX version.
In-contract lookups never touch the guard - it protects exclusions, not
lookups.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path

ELIGIBILITY_VERSION = 1

# The ruling basis: the mlx_metal 0.32.0 wheel's Metal kernel source.
# sha256 of mlx/include/mlx/backend/metal/kernels/quantized.h, equal to the
# wheel RECORD entry TaUr9O5ogWWmW4TFKl9Ogu_K5_aejHTZ7j4AvvRjyZ8 (b64url of
# the same digest), verified 2026-08-15 by the B1 investigation.
MLX_VERSION_RULED = "0.32.0"
MLX_QUANTIZED_KERNELS_SHA256 = (
    "4da52bf4ee688165a65b84c52a5f4e82efcae7f69e8c74d9ee3e00bef463c99f")
_KERNELS_RELPATH = "include/mlx/backend/metal/kernels/quantized.h"


class StaleEligibilityError(RuntimeError):
    """The installed MLX is not the one the exclusion was ruled against:
    applying the exclusion anyway could wave a fixed kernel through, so the
    consumer stops here and the ruling is re-examined instead."""


@dataclass(frozen=True)
class Exclusion:
    heldout: str
    batch: int
    dtype: str
    kernel: str     # the MLX kernel dispatched at this cell
    violation: str  # the structural C1 violation, stated from the source
    adr: str
    mlx_version: str
    kernel_source_sha256: str


def _exclusion(batch: int, kernel: str, violation: str) -> Exclusion:
    return Exclusion(
        heldout="mlx-on-device", batch=batch, dtype="float16",
        kernel=kernel, violation=violation, adr="ADR 0014",
        mlx_version=MLX_VERSION_RULED,
        kernel_source_sha256=MLX_QUANTIZED_KERNELS_SHA256,
    )


# Keyed (heldout name, batch, dtype), exactly the cells the ruling names;
# the keys are derived from the entries so the two can never disagree.
# `block-tiled` appears nowhere: it is eligible everywhere.
EXCLUSIONS = {(entry.heldout, entry.batch, entry.dtype): entry for entry in (
    _exclusion(
        batch=1, kernel="affine_qmv / affine_qmv_fast",
        violation="C1: the per-thread sub-sum of 8 activations is evaluated "
                  "in half precision (all operands are T = half, so the "
                  "seven adds round in fp16) before the fp32 accumulator"),
    _exclusion(
        batch=16, kernel="affine_qmm_t",
        violation="C1: the dequantized weight tile is stored in "
                  "threadgroup T*, i.e. half at fp16 activations, a "
                  "narrower-than-fp32 intermediate"),
)}


def installed_kernel_source() -> Path | None:
    """The installed wheel's quantized.h, or None when mlx (a namespace
    package, so no __file__) is absent or laid out without its headers."""
    try:
        import mlx
    except ImportError:
        return None
    for entry in mlx.__path__:
        candidate = Path(entry) / _KERNELS_RELPATH
        if candidate.exists():
            return candidate
    return None


def installed_mlx_version() -> str | None:
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return None


def verify_ruling_basis(source_path: Path | None = None,
                        mlx_version: str | None = None) -> None:
    """Fail loudly unless the installed MLX is the ruled-on one.

    The source hash is the primary guard: a readable file that hashes
    differently is stale REGARDLESS of the version string, because a patched
    wheel keeps its version. Only an unreadable source falls back to the
    version equality, and nothing verifiable at all is itself a failure.
    """
    path = installed_kernel_source() if source_path is None else source_path
    if path is not None and path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != MLX_QUANTIZED_KERNELS_SHA256:
            raise StaleEligibilityError(
                f"{path} hashes {digest}, not the ruled-on "
                f"{MLX_QUANTIZED_KERNELS_SHA256}: the exclusions were ruled "
                f"against mlx {MLX_VERSION_RULED}'s kernel source and must "
                f"not be applied to a different one; re-examine ADR 0014 "
                f"against the installed MLX")
        return
    version = installed_mlx_version() if mlx_version is None else mlx_version
    if version != MLX_VERSION_RULED:
        raise StaleEligibilityError(
            f"mlx kernel source not readable and installed mlx version "
            f"{version!r} is not the ruled-on {MLX_VERSION_RULED!r}: "
            f"refusing to apply a stale exclusion; re-examine ADR 0014 "
            f"against the installed MLX")


def exclusion_for(heldout: str, batch: int, dtype: str,
                  source_path: Path | None = None,
                  mlx_version: str | None = None) -> Exclusion | None:
    """The exclusion at this cell, basis-verified, or None (in contract)."""
    entry = EXCLUSIONS.get((heldout, int(batch), dtype))
    if entry is None:
        return None
    verify_ruling_basis(source_path, mlx_version)
    return entry
