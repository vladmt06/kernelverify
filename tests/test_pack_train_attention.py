"""The training-attention pack's geometry and cell arithmetic, with no device."""

from __future__ import annotations

import pytest

from kernelverify.pack import train_attention


def test_the_geometry_is_qwen3_4bs_and_the_group_ratio_is_derived():
    assert train_attention.N_Q_HEADS == 32
    assert train_attention.N_KV_HEADS == 8
    assert train_attention.GQA_GROUP == 4
    assert train_attention.HEAD_DIM == 128


def test_the_scale_is_the_value_qwen3_hands_to_the_seam():
    """`Attention.__init__` computes `head_dim**-0.5`. A kernel that scales by
    anything else is wrong in a way no shape check would catch."""
    assert train_attention.SCALE == 128 ** -0.5
    assert train_attention.SCALE == pytest.approx(0.08838834764831845, abs=0.0)


def test_the_two_registered_cells_are_the_two_pinned_bands():
    assert train_attention.cell_key(4, 65) == "B4:T128"
    assert train_attention.cell_key(2, 1057) == "B2:T2048"
    assert set(train_attention.CELLS) == {"B4:T128", "B2:T2048"}


def test_t_bucket_rounds_up_so_a_configuration_is_never_credited_upward():
    assert train_attention.t_bucket(1) == 128
    assert train_attention.t_bucket(65) == 128
    assert train_attention.t_bucket(128) == 128
    assert train_attention.t_bucket(129) == 256
    assert train_attention.t_bucket(1057) == 2048


def test_t_bucket_caps_rather_than_growing_without_bound():
    assert train_attention.t_bucket(4096) == train_attention.T_BUCKET_CAP
    assert train_attention.t_bucket(10 ** 6) == train_attention.T_BUCKET_CAP


def test_t_bucket_refuses_a_step_with_no_width():
    with pytest.raises(ValueError):
        train_attention.t_bucket(0)


def test_cell_key_refuses_an_unregistered_batch():
    with pytest.raises(ValueError):
        train_attention.cell_key(8, 128)


def test_an_unregistered_batch_is_none_rather_than_the_nearest_cell():
    """None routes to stock. A near miss must never be answered with a
    configuration measured at a different batch."""
    assert train_attention.cell_name(8, 128) is None
    assert train_attention.cell_name(3, 128) is None
    assert train_attention.cell_name(4, 65) == "B4:T128"


def test_geometry_ok_refuses_a_model_that_is_not_the_one_verified():
    assert train_attention.geometry_ok(32, 8, 128)
    # Llama-3.2-3B: 24 query heads, 8 key-value heads, head dimension 128.
    assert not train_attention.geometry_ok(24, 8, 128)
    # The same head counts at a different head dimension is a different kernel.
    assert not train_attention.geometry_ok(32, 8, 64)
    # Multi-head rather than grouped-query: the group sum has nothing to sum.
    assert not train_attention.geometry_ok(32, 32, 128)


# ---------------------------------------------------------------------------
# The forward kernel's source and knob arithmetic, all of it device-free
# ---------------------------------------------------------------------------
def test_the_source_carries_no_narrow_intermediate_at_either_storage_type():
    """Contract clause C1, attested on the text. `T` is bound to the storage
    type at dispatch, so the source that gets compiled is the substituted one
    and both bindings have to be checked."""
    from kernelverify.compiler.lint import lint

    source = train_attention.fwd_door_source({"RQ": 8, "SGROUPS": 1, "BKEY": 32})
    for binding in ("half", "bfloat16_t"):
        report = lint(source, types={"T": binding})
        assert report.ok, f"{binding}: {report.reason}"


def test_the_substitution_leaves_no_knob_name_in_the_source():
    """A token that survives substitution is a compile error at best and a
    silently different kernel at worst."""
    source = train_attention.fwd_door_source({"RQ": 4, "SGROUPS": 2, "BKEY": 32})
    for token in ("RQ", "SGROUPS", "BKEY", "DHEAD", "SCALE_LIT",
                  "T_LEN", "HEADS_Q", "HEADS_KV"):
        assert token not in source, token


def test_the_scale_reaches_the_kernel_as_the_value_qwen3_passes():
    source = train_attention.fwd_door_source({"RQ": 4, "SGROUPS": 1, "BKEY": 32})
    assert "0.08838834764831845f" in source


def test_the_head_dimension_reaches_the_kernel_as_a_literal():
    """It sizes threadgroup arrays and bounds unrolled loops, so it cannot be a
    shape read at run time the way the width is."""
    source = train_attention.fwd_door_source({"RQ": 4, "SGROUPS": 1, "BKEY": 32},
                                             head_dim=64)
    assert "constexpr uint DPL   = 64 / LANES;" in source
    assert "q_shape[2]" in source        # the width stays an expression


def test_every_knob_setting_divides_the_lane_count_the_kernel_indexes_by():
    """A lane owns whole keys in one phase and whole output columns in the
    other, so both counts must divide 32. A setting that does not is not slow,
    it is wrong, and it must never reach the device."""
    axes = train_attention.fwd_axes()
    assert all(bkey % 32 == 0 for bkey in axes["BKEY"])
    assert train_attention.HEAD_DIM % 32 == 0


def test_the_threadgroup_arithmetic_counts_every_array_the_kernel_declares():
    """Two tiles at storage width, the query rows and the probabilities at
    fp32. This number is what decides a setting's launchability, so it is
    checked against the hand count rather than trusted."""
    knobs = {"RQ": 8, "SGROUPS": 1, "BKEY": 32}
    by_hand = (2 * 128 * 32 * 2) + (8 * 128 * 4) + (8 * 32 * 4)
    assert train_attention.fwd_threadgroup_bytes(knobs, 128) == by_hand == 21504
    # The setting a 32 KB device refuses, and by how much.
    assert train_attention.fwd_threadgroup_bytes(
        {"RQ": 4, "SGROUPS": 4, "BKEY": 64}, 128) == 45056


def test_the_launch_covers_every_query_row_including_a_partial_tile():
    knobs = {"RQ": 8, "SGROUPS": 1, "BKEY": 32}
    grid, threadgroup = train_attention.fwd_launch(knobs, 2, 32, 65)
    assert threadgroup == (32, 1, 1)
    assert grid[0] == 9 * 32          # ceil(65 / 8) query tiles, 32 lanes each
    assert grid[1] == 1
    assert grid[2] == 2 * 32          # one threadgroup column per (batch, head)
    assert train_attention.fwd_threads(knobs) == 32


def test_the_routed_dtypes_are_the_two_the_storage_arithmetic_assumes():
    """The threadgroup byte count multiplies by two bytes per staged element,
    so admitting a four-byte dtype would under-count what the kernel stages."""
    assert train_attention.ROUTED_DTYPES == ("bfloat16", "float16")
    assert train_attention.STORAGE_BYTES == 2
