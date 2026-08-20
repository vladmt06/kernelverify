"""bench/interleave.py: the shared interleaved-timing engine.

Skipped wholesale when MLX is unavailable, matching the other bench-module
suites; the engine imports mlx.core at module scope.
"""

import pytest

pytest.importorskip("mlx.core")

import interleave  # noqa: E402

from conftest import requires_metal  # noqa: E402
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


# ---------------------------------------------------------------------------
# The n-arm sampler, and the two-arm wrapper that must not move
#
# These carry `requires_metal` because the engine's whole job is to DISPATCH:
# every one of them runs real eval-and-synchronize work sized to
# MIN_SAMPLE_MS. That is not about needing a device to pass - MLX would fall
# back to its CPU stream - it is the second half of the marker, which excuses
# a test from the safe subset. This machine has one GPU and one machine-wide
# measurement lock, so a `pytest -m "not gpu"` run during a 33-minute A/B must
# not dispatch these against it.
# ---------------------------------------------------------------------------
def _recorder():
    """Builders that record the order they were dispatched in, so the sampling
    order is checked rather than assumed. Each returns a tiny real array so
    the engine's eval and sync run for real on MLX's own stream."""
    import mlx.core as mx

    seen = []

    def make(label):
        def build_one(i):
            seen.append(label)
            return mx.array([float(i)]) + 1.0
        return build_one

    return seen, make


@requires_metal
def test_every_arm_is_sampled_in_every_round():
    seen, make = _recorder()
    got = interleave.interleaved_arms(
        {name: make(name) for name in ("x", "y", "z")}, rounds=4)
    assert set(got) == {"x", "y", "z"}
    assert all(len(times) == 4 for times in got.values()), \
        "each arm gets one sample per round, so the rounds are comparable"
    assert all(t > 0 for times in got.values() for t in times)


def _blocks(seen):
    """The dispatch order, one entry per contiguous run of an arm's builds.

    ``dispatch`` calls ``build_one`` once per copy, so a single arm's turn
    appears as many identical entries; collapsing them gives the arm order
    the engine actually ran, which is the thing under test."""
    out = []
    for label in seen:
        if not out or out[-1] != label:
            out.append(label)
    return out


@requires_metal
def test_the_arm_order_rotates_so_no_arm_always_runs_first():
    """Allocation and cache effects land on whichever arm runs first, so a
    fixed order would charge them to the same arm every round."""
    seen, make = _recorder()
    order = ("x", "y", "z")
    rounds = 3
    interleave.interleaved_arms({name: make(name) for name in order},
                                rounds=rounds)
    # The timed rounds are the LAST rounds * len(order) turns; everything
    # before them is the one warm-up build per arm and the calibration
    # dispatches, which all run on the first arm.
    timed = _blocks(seen)[-rounds * len(order):]
    ran = [tuple(timed[i * len(order):(i + 1) * len(order)])
           for i in range(rounds)]
    assert ran == [("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y")], ran
    assert {row[0] for row in ran} == set(order), "every arm leads once"


@requires_metal
def test_the_two_arm_wrapper_keeps_its_fixed_order():
    """Every published pack certificate was measured with B following A, so
    the wrapper must not inherit the n-arm path's rotation."""
    seen, make = _recorder()
    a_times, b_times = interleave.interleaved_samples(
        make("a"), make("b"), rounds=3)
    assert len(a_times) == len(b_times) == 3
    assert _blocks(seen)[-6:] == ["a", "b", "a", "b", "a", "b"], \
        "arm A leads every round, exactly as it did before the n-arm path"


@requires_metal
def test_one_calibrated_batch_serves_every_arm():
    """Per-arm batches would divide by different copy counts and make the
    per-round times incomparable, which is what sampling together is for."""
    import mlx.core as mx

    counts = {"a": 0, "b": 0}

    def make(label, cost):
        def build_one(i):
            counts[label] += 1
            return mx.zeros((cost, cost)) + float(i)
        return build_one

    rounds = 2
    got = interleave.interleaved_arms(
        {"a": make("a", 4), "b": make("b", 64)}, rounds=rounds)
    # One warm-up build per arm, then arm A alone pays the calibration
    # dispatches, then both arms run the SAME copy count every round.
    copies = (counts["b"] - 1) / rounds
    assert copies == int(copies) and copies >= 8, copies
    assert counts["a"] > 1 + rounds * copies, \
        "arm A pays the calibration dispatches on top of its timed rounds"
    assert (counts["a"] - 1 - rounds * copies) % 1 == 0
    assert len(got["a"]) == len(got["b"]) == rounds


@requires_metal
def test_an_empty_arm_set_refuses():
    with pytest.raises(ValueError, match="at least one arm"):
        interleave.interleaved_arms({}, rounds=3)


@requires_metal
def test_a_round_count_below_one_refuses():
    seen, make = _recorder()
    with pytest.raises(ValueError, match="at least 1"):
        interleave.interleaved_arms({"a": make("a")}, rounds=0)


@requires_metal
def test_the_guard_runs_once_per_round_before_any_arm():
    seen, make = _recorder()
    cells = []
    interleave.interleaved_arms({"a": make("a"), "b": make("b")}, rounds=3,
                                guard=lambda cell: cells.append(cell))
    assert cells == ["round 1/3", "round 2/3", "round 3/3"]


@requires_metal
def test_a_guard_refusal_stops_the_run():
    seen, make = _recorder()

    def refuse(cell):
        raise RuntimeError(f"no memory at {cell}")

    with pytest.raises(RuntimeError, match="round 1/2"):
        interleave.interleaved_arms({"a": make("a")}, rounds=2, guard=refuse)
