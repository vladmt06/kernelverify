"""bench/interleave.py: the shared interleaved-timing engine.

Skipped wholesale when MLX is unavailable, matching the other bench-module
suites; the engine imports mlx.core at module scope.
"""

import pytest

pytest.importorskip("mlx.core")

import interleave  # noqa: E402

from kernelverify.runners.compare import DEFAULT_SPREAD_LIMIT  # noqa: E402


def test_the_pack_gates_canary_limit_matches_the_live_comparator_today():
    """Two settings of decision D6's reference-arm max/min, deliberately not
    one symbol.

    MAX_CANARY_SPREAD is the threshold above which a pack gate withholds a
    certificate; DEFAULT_SPREAD_LIMIT is the threshold above which the live
    worker comparator rejects a round. D6 sets the limit per class - 1.5x for
    kernel arms, tighter for steadier quantities - so binding one to the other
    would let a tightened comparator silently re-gate every pack certificate.
    They agree at 1.5 today, and this test is where a divergence surfaces.
    """
    assert interleave.MAX_CANARY_SPREAD == 1.5
    assert DEFAULT_SPREAD_LIMIT == 1.5
    assert interleave.MAX_CANARY_SPREAD == DEFAULT_SPREAD_LIMIT


def test_the_pack_gates_all_read_the_one_canary_limit():
    """The engine exists so an amended timing rule cannot stay old in one gate;
    each gate must take the limit from it rather than keep a private copy."""
    import pack_kv_attention
    import pack_moe_dispatch
    import pack_wide_qmv

    for gate in (pack_kv_attention, pack_moe_dispatch, pack_wide_qmv):
        assert gate.MAX_CANARY_SPREAD is interleave.MAX_CANARY_SPREAD
