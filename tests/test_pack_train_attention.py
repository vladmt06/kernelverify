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
    type the answer goes back to, so the source that gets compiled is the
    substituted one and both bindings have to be checked."""
    from kernelverify.compiler.lint import lint

    source = train_attention.fwd_door_source({"SGROUPS": 2, "BKEY": 16})
    for binding in ("half", "bfloat16_t"):
        report = lint(source, types={"T": binding})
        assert report.ok, f"{binding}: {report.reason}"


def test_the_substitution_leaves_no_knob_name_in_the_source():
    """A token that survives substitution is a compile error at best and a
    silently different kernel at worst."""
    source = train_attention.fwd_door_source({"SGROUPS": 4, "BKEY": 32})
    for token in ("SGROUPS", "BKEY", "DHEAD", "SCALE_LIT", "T_PAD",
                  "HEADS_Q", "HEADS_KV"):
        assert token not in source, token


def test_the_scale_reaches_the_kernel_as_the_value_qwen3_passes():
    source = train_attention.fwd_door_source({"SGROUPS": 2, "BKEY": 16})
    assert "0.08838834764831845f" in source


def test_the_true_width_is_an_input_and_the_padded_one_is_a_shape():
    """The door pads the operands, so the shape the kernel sees is the padded
    width and the true width cannot be read back out of it. Making the true
    width a literal instead would recompile the kernel on nearly every batch,
    since mlx-lm's width moves with the data."""
    source = train_attention.fwd_door_source({"SGROUPS": 2, "BKEY": 16})
    assert "const uint t_len = tlen[0];" in source
    assert "const uint t_pad = q_shape[2];" in source
    assert "tlen" in train_attention.FWD_INPUT_NAMES


def test_the_head_dimension_reaches_the_kernel_as_a_literal():
    """It sizes fragment arrays and bounds unrolled loops, so it cannot be a
    shape read at run time the way the width is."""
    source = train_attention.fwd_door_source({"SGROUPS": 2, "BKEY": 16},
                                             head_dim=64)
    assert "constexpr uint DB    = 64 / 8u;" in source
    assert "q_shape[2]" in source        # the padded width stays an expression


def test_every_knob_setting_divides_the_fragment_the_kernel_indexes_by():
    """A matrix fragment is eight by eight, so a key tile is a whole number of
    fragments and so is the head dimension. A setting that is not is not slow,
    it is wrong, and it must never reach the device."""
    axes = train_attention.fwd_axes()
    assert all(bkey % 8 == 0 for bkey in axes["BKEY"])
    assert train_attention.HEAD_DIM % 8 == 0


def test_this_kernel_stages_nothing_and_says_so():
    """Threadgroup memory is where an earlier design put the operand tiles, and
    the staging cost more than the matmuls it fed. The knob filter still asks,
    because launchability is asked the same way of every operation."""
    assert train_attention.fwd_threadgroup_bytes({"SGROUPS": 8, "BKEY": 64}) == 0
    assert train_attention.fwd_threads({"SGROUPS": 8}) == 256


def test_the_row_multiple_covers_both_tiles_a_fragment_load_reads():
    """A fragment load reads eight rows whether or not eight rows exist, so the
    operands are padded to whole tiles. The query tile and the key tile must
    both divide the padding, so it is the larger of the two."""
    assert train_attention.fwd_row_multiple({"SGROUPS": 1, "BKEY": 64}) == 64
    assert train_attention.fwd_row_multiple({"SGROUPS": 8, "BKEY": 16}) == 64
    assert train_attention.fwd_row_multiple({"SGROUPS": 2, "BKEY": 16}) == 16


def test_the_launch_covers_every_query_row_including_a_partial_tile():
    knobs = {"SGROUPS": 2, "BKEY": 16}
    grid, threadgroup = train_attention.fwd_launch(knobs, 2, 32, 80)
    assert threadgroup == (32, 2, 1)
    assert grid[0] == 5 * 32          # ceil(80 / 16) query tiles, 32 lanes each
    assert grid[1] == 2
    assert grid[2] == 2 * 32          # one threadgroup column per (batch, head)


def test_the_routed_dtypes_are_the_two_the_seam_can_hand_us():
    """bfloat16 is what the shipped 4-bit artifacts carry and what the seam was
    measured handing over; float16 is admitted because the same source serves
    it and its tolerance is the sharper of the two."""
    assert train_attention.ROUTED_DTYPES == ("bfloat16", "float16")
