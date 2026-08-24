"""The training-attention gate, everywhere it can be pinned without a device.

What needs the GPU - that every launchable knob setting verifies, that the
kernel writes every declared cell, that it beats the composed path - lives in
the gate itself and in tests/test_pack_train_attention_live.py. What lives here
is the arithmetic the gate rules with, and the shape of the comparison it
makes, because a credited ratio computed the optimistic way looks exactly like
one computed the conservative way until the samples disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

pack_train_attention = pytest.importorskip("pack_train_attention")


def test_the_credited_ratio_is_the_worst_pairing_the_samples_permit():
    """Smallest numerator over largest denominator, which is what every
    pricing verdict in this repository is computed with. The optimistic
    reading, median over median, would report 2.0 on these samples."""
    composed = [10.0, 20.0, 30.0]
    ours = [5.0, 10.0, 15.0]
    assert pack_train_attention.ratio_lo(composed, ours) == 10.0 / 15.0


def test_a_single_sample_pair_reduces_to_the_plain_ratio():
    assert pack_train_attention.ratio_lo([9.0], [3.0]) == 3.0


def test_the_two_fills_differ_so_an_unwritten_cell_cannot_hide():
    """The screen's whole mechanism: a cell no store reached holds the fill, so
    the two passes disagree there. Equal fills would make the screen vacuous."""
    assert pack_train_attention.FILL_A != pack_train_attention.FILL_B


def test_the_timed_cells_are_the_two_the_pack_registers():
    from kernelverify.pack import train_attention as ta

    timed = {ta.cell_key(batch, width)
             for batch, width in pack_train_attention.TIMED_CELLS}
    assert timed == set(ta.CELLS)


def test_the_verified_widths_straddle_every_tile_edge_the_kernel_has():
    """A tiled kernel's faults live where a tile ends: one row short, exact,
    one row past. The gate has to carry all three for each width the fragment
    and key tiles can take."""
    widths = set(pack_train_attention.VERIFY_WIDTHS)
    for edge in (8, 16, 32, 64, 128):
        assert {edge - 1, edge, edge + 1} & widths, edge


def test_the_verified_group_ratios_cover_more_than_the_shipped_one():
    """A kernel that indexes the group wrongly is silently correct at a ratio
    of one, so the gate has to run more ratios than the model has."""
    ratios = {n_q // n_kv for n_q, n_kv in pack_train_attention.VERIFY_HEADS}
    assert {1, 2, 4, 8} <= ratios


def test_both_storage_dtypes_are_verified_not_just_the_shipping_one():
    from kernelverify.pack import train_attention as ta

    assert set(pack_train_attention.VERIFY_DTYPES) == set(ta.ROUTED_DTYPES)
