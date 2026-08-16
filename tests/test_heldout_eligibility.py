"""The per-cell held-out eligibility table (ADR 0014).

The table is versioned contract DATA, so its exact contents are pinned here:
a silent edit to an exclusion is a silent edit to what the verifier promises,
and the kernel-source hash guard is what keeps an exclusion from outliving
the MLX code it was ruled against.
"""

import pytest

from kernelverify.schemas.heldout_eligibility import (
    ELIGIBILITY_VERSION,
    EXCLUSIONS,
    MLX_QUANTIZED_KERNELS_SHA256,
    MLX_VERSION_RULED,
    StaleEligibilityError,
    exclusion_for,
)

B1 = ("mlx-on-device", 1, "float16")
B16 = ("mlx-on-device", 16, "float16")


# ---------------------------------------------------------------------------
# The table pin: exactly the ruled exclusions, nothing more, nothing less
# ---------------------------------------------------------------------------
def test_table_pins_exactly_the_ruled_exclusions():
    assert ELIGIBILITY_VERSION == 1
    assert set(EXCLUSIONS) == {B1, B16}


def test_each_exclusion_carries_its_ruling_and_its_basis():
    for entry in EXCLUSIONS.values():
        assert entry.adr == "ADR 0014"
        assert entry.mlx_version == MLX_VERSION_RULED == "0.32.0"
        assert entry.kernel_source_sha256 == MLX_QUANTIZED_KERNELS_SHA256
        assert len(entry.kernel_source_sha256) == 64
        assert "C1" in entry.violation


def test_the_b1_and_b16_entries_name_the_kernels_they_were_ruled_against():
    assert "affine_qmv" in EXCLUSIONS[B1].kernel
    assert "affine_qmm_t" in EXCLUSIONS[B16].kernel


def test_block_tiled_is_eligible_everywhere():
    assert not any(heldout == "block-tiled" for heldout, _, _ in EXCLUSIONS)
    for batch in (1, 2, 8, 16):
        for dtype in ("float32", "float16"):
            assert exclusion_for("block-tiled", batch, dtype) is None


def test_fp32_and_the_clean_batches_are_not_excluded():
    for batch in (1, 2, 8, 16):
        assert exclusion_for("mlx-on-device", batch, "float32") is None
    for batch in (2, 8):
        assert exclusion_for("mlx-on-device", batch, "float16") is None


# ---------------------------------------------------------------------------
# The hash guard: an exclusion applies only against the source it was ruled
# on; anything else fails loudly rather than waving an MLX fix through
# ---------------------------------------------------------------------------
def test_exclusion_applies_against_the_installed_wheel():
    entry = exclusion_for(*B1)
    assert entry is not None and "affine_qmv" in entry.kernel
    assert exclusion_for(*B16) is not None


def test_changed_kernel_source_fails_loudly_even_at_the_same_version(tmp_path):
    patched = tmp_path / "quantized.h"
    patched.write_text("// an MLX fix the exclusion must not outlive\n")
    with pytest.raises(StaleEligibilityError):
        exclusion_for(*B1, source_path=patched, mlx_version=MLX_VERSION_RULED)


def test_missing_source_falls_back_to_the_mlx_version(tmp_path):
    absent = tmp_path / "absent.h"
    entry = exclusion_for(*B1, source_path=absent,
                          mlx_version=MLX_VERSION_RULED)
    assert entry is not None
    with pytest.raises(StaleEligibilityError):
        exclusion_for(*B1, source_path=absent, mlx_version="0.33.0")


def test_in_contract_lookup_never_touches_the_basis(tmp_path):
    """A cell with no exclusion must answer None even where the basis is
    unverifiable: the guard protects exclusions, not lookups."""
    absent = tmp_path / "absent.h"
    assert exclusion_for("mlx-on-device", 2, "float16",
                         source_path=absent, mlx_version="9.9.9") is None
