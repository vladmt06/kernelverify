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
