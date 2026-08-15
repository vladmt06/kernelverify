"""The routed windows: derived from the committed recording, pinned as data.

The table is the whole routing decision, so the properties worth pinning are
the ones that make a silent change impossible: the exact derived cells against
hand-written literals, and the three things the cells are a claim ABOUT - the
recording they came from, the kernel source they priced, and the launch config
they ran at. Any of the three moving must fail here rather than ship a routing
table that describes a kernel nobody measured.

No MLX needed: the table is pure data derived from a committed JSON file, so
this module runs on non-Mac CI as well.
"""

import hashlib

import pytest

from kernelverify.pack import routed_windows as rw

# The six Qwen3-4B decode dispatch shapes at the block's 3-bit arm, written
# out by hand from bench/results/qmv-boundary-pricing-2026-08-15.json. Five
# shapes win M = 5..9; lm_head wins one width further because its 151936-row
# weight matrix is large enough that MLX's second pass still costs more than
# our single pass at M = 10.
EXPECTED_TABLE = {
    (3, 4096, 2560): {5, 6, 7, 8, 9},      # q_proj
    (3, 1024, 2560): {5, 6, 7, 8, 9},      # k_proj/v_proj
    (3, 2560, 4096): {5, 6, 7, 8, 9},      # o_proj
    (3, 9728, 2560): {5, 6, 7, 8, 9},      # gate_proj/up_proj
    (3, 2560, 9728): {5, 6, 7, 8, 9},      # down_proj
    (3, 151936, 2560): {5, 6, 7, 8, 9, 10},  # lm_head (M=4 excluded, D2)
}


def test_derived_table_matches_the_hand_written_cells():
    """The literals ARE the ruled decision; re-deriving must reproduce them."""
    assert {k: set(v) for k, v in rw.ROUTED_WINDOWS.items()} == EXPECTED_TABLE


def test_windows_are_sets_so_a_future_gap_is_representable():
    """Stored as the SET of winning widths, never as (lo, hi): a recording
    with a hole in the middle must be expressible, not a crash or a lie."""
    assert all(isinstance(v, frozenset) for v in rw.ROUTED_WINDOWS.values())


def test_lm_head_m4_is_excluded_by_ruling_d2():
    """lm_head M=4 measured WIN and is still NOT routed: taking it would
    widen below the pre-registered 5..11 window, which the pre-registration
    does not permit. It enters only through the widening rule, after a run
    that prices it and a gate that covers it."""
    assert 4 in rw.EXCLUDED_CELLS[(3, 151936, 2560)]
    assert 4 not in rw.ROUTED_WINDOWS[(3, 151936, 2560)]


def test_recording_sha256_still_describes_the_recording():
    digest = hashlib.sha256(rw.RECORDING_PATH.read_bytes()).hexdigest()
    assert digest == rw.RECORDING_SHA256


def test_kernel_source_sha256_still_describes_the_kernel():
    """The windows are a claim about THIS kernel body. Editing the MSL and
    keeping the table is the failure this pin exists to make loud."""
    mslsource = pytest.importorskip("kernelverify.pack.wide_qmv").WIDE_QMV_MSL
    assert hashlib.sha256(mslsource.encode()).hexdigest() == rw.KERNEL_SOURCE_SHA256


def test_pinned_launch_config_matches_the_shipped_one():
    """A window priced at R = 4 says nothing about the same cell launched at
    R = 2, so the pinned per-M R must be the one wide_qmv will launch."""
    wide_qmv = pytest.importorskip("kernelverify.pack.wide_qmv")
    assert all(wide_qmv.rows_per_simdgroup(m) == r
               for m, r in rw.ROWS_PER_SIMDGROUP.items())


def test_provenance_names_the_recording_and_its_hash():
    assert rw.RECORDING_DATE in rw.PROVENANCE
    assert rw.RECORDING_SHA256[:12] in rw.PROVENANCE


def test_an_unpriced_key_has_an_empty_window_not_a_default():
    """No evidence reads as no routing, never as the old 5..11 default."""
    assert rw.window_for(4, 4096, 2560) == frozenset()   # 4-bit is unpriced
    assert rw.window_for(3, 777, 2560) == frozenset()    # unknown shape


def test_priced_shapes_is_the_denominator_for_every_bit_width():
    assert set(rw.PRICED_SHAPES) == {3}
    assert len(rw.PRICED_SHAPES[3]) == 6


def test_prose_states_the_scope_and_cites_the_recording():
    """The certificate emitter's scope line comes from here, so it must name
    the width, the shapes, and the file the claim rests on."""
    wide = rw.dispatch_boundary_prose(3, 5)
    assert "6 of the 6 priced dispatch shapes" in wide
    narrow = rw.dispatch_boundary_prose(3, 10)
    assert "1 of the 6 priced dispatch shapes" in narrow and "151936x2560" in narrow
    assert "none of the 6" in rw.dispatch_boundary_prose(3, 1)
    assert "4-bit is unpriced" in rw.dispatch_boundary_prose(4, 5)
    assert all(rw.PROVENANCE in p or "unpriced" in p
               for p in (wide, narrow, rw.dispatch_boundary_prose(4, 5)))
