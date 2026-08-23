"""The training-matmul pack's shape and bucket arithmetic, with no device."""

from __future__ import annotations

import pytest

from kernelverify.pack import train_qmm


def test_the_six_registered_shapes_are_the_ones_a_step_dispatches():
    assert set(train_qmm.TRAIN_SHAPES) == {"S1", "S2", "S3", "S4", "S5", "S6"}
    # k_proj and v_proj, the shape a five-entry table missed.
    assert train_qmm.TRAIN_SHAPES["S6"] == (1024, 2560)
    # The tied head, the largest by two orders of magnitude.
    assert train_qmm.TRAIN_SHAPES["S5"] == (151936, 2560)


def test_every_shape_is_a_whole_number_of_quantization_groups():
    """A 64-wide group must not straddle two rows, or scales index wrongly."""
    for name, (_d_out, d_in) in train_qmm.TRAIN_SHAPES.items():
        assert d_in % train_qmm.GROUP_SIZE == 0, name


def test_m_bucket_rounds_up_so_a_configuration_is_never_credited_upward():
    assert train_qmm.m_bucket(1) == 64
    assert train_qmm.m_bucket(64) == 64
    assert train_qmm.m_bucket(65) == 128
    assert train_qmm.m_bucket(128) == 128
    assert train_qmm.m_bucket(1057) == 2048


def test_m_bucket_caps_rather_than_growing_without_bound():
    assert train_qmm.m_bucket(4224) == train_qmm.M_BUCKET_CAP
    assert train_qmm.m_bucket(10 ** 6) == train_qmm.M_BUCKET_CAP


def test_m_bucket_refuses_a_step_with_no_tokens():
    with pytest.raises(ValueError):
        train_qmm.m_bucket(0)


def test_cell_key_pairs_a_shape_with_a_bucket():
    assert train_qmm.cell_key("S5", 65) == "S5:M128"
    assert train_qmm.cell_key("S1", 1057) == "S1:M2048"


def test_cell_key_refuses_an_unregistered_shape():
    with pytest.raises(KeyError):
        train_qmm.cell_key("S9", 128)


def test_shape_name_recognises_a_registered_call_site():
    assert train_qmm.shape_name(151936, 2560) == "S5"
    assert train_qmm.shape_name(1024, 2560) == "S6"


def test_an_unregistered_call_site_is_none_rather_than_the_nearest_shape():
    """None routes to stock. A near miss must never be answered with a
    configuration measured somewhere else."""
    assert train_qmm.shape_name(1025, 2560) is None
    assert train_qmm.shape_name(2560, 2560) is None
