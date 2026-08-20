"""The Day 1 profile's instrument: what it measures, and what it refuses to.

The arithmetic needs no device: spans, counts, shares and every refusal are
read off fake logs, so the rules can be argued with and tested anywhere.

The mechanism itself does need one, and that was measured rather than assumed.
Setting MLX's default device to the CPU does NOT make it device-independent:
in a sandbox with no Metal device at all, importing `mlx.nn` aborts the
process outright and `mx.random.seed` raises. So every test below that touches
MLX is gated, even the ones that then run on the CPU stream. Running them on
the CPU stream is still worth doing - it keeps them off the machine's one GPU
and its measurement lock - but it does not remove the requirement.

The tests that matter most are the refusals. A log that double counts a nested
region, or one missing a mark that never fired, still produces a number; the
number is simply wrong, and a wrong share aims the sprint at the wrong
operation. So the arithmetic is pushed against overlapping spans, spans that
never close, spans that close having never opened, and stamps from outside the
step they claim to describe.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import profile_instrument as pi  # noqa: E402
from conftest import requires_metal  # noqa: E402
from decode_rules import RunInvalid  # noqa: E402


# What identifies the workload a pass ran. Every fake pass here shares one, so
# a test that mixes two is doing so on purpose.
CTX = {"cell": "B", "width": 161, "batch": 4}
OTHER = {"cell": "B", "width": 161, "batch": 1}


def entry(region, end, direction, stamp):
    return (region, end, direction, stamp)


def forward(region, start, end):
    return [entry(region, pi.ENTER, pi.FORWARD, start),
            entry(region, pi.EXIT, pi.FORWARD, end)]


def backward(region, start, end):
    """Backward runs the span in reverse: the exit mark fires first."""
    return [entry(region, pi.EXIT, pi.BACKWARD, start),
            entry(region, pi.ENTER, pi.BACKWARD, end)]


# ---------------------------------------------------------------------------
# Reading the log
# ---------------------------------------------------------------------------
def test_a_forward_span_runs_enter_to_exit():
    read = pi.decompose(forward("attn-core", 1.0, 1.5), 0.0, 2.0, context=CTX)
    assert read["regions"]["attn-core"][pi.FORWARD] == pytest.approx(0.5)
    assert read["regions"]["attn-core"][pi.BACKWARD] == 0.0


def test_a_backward_span_runs_exit_to_enter():
    """The gradient rules fire in reverse order, so the exit mark opens the
    backward span. Reading the log as if it were sorted into enter-then-exit
    pairs would produce a negative span."""
    read = pi.decompose(backward("attn-core", 1.0, 1.8), 0.0, 2.0, context=CTX)
    assert read["regions"]["attn-core"][pi.BACKWARD] == pytest.approx(0.8)


def test_every_occurrence_of_a_region_adds_to_its_span():
    """Attention runs once per layer, so a region's span is the sum of many."""
    entries = (forward("attn-core", 1.0, 1.1) + forward("attn-core", 1.2, 1.5)
               + forward("attn-core", 1.6, 1.7))
    read = pi.decompose(entries, 0.0, 2.0, context=CTX)
    assert read["regions"]["attn-core"][pi.FORWARD] == pytest.approx(0.5)
    assert read["counts"]["attn-core"][pi.FORWARD] == 3


def test_the_remainder_is_what_no_region_accounted_for():
    entries = forward("qmm", 1.0, 1.25) + backward("qmm", 1.5, 1.75)
    read = pi.decompose(entries, 1.0, 3.0, context=CTX)
    assert read["covered"] == pytest.approx(0.5)
    assert read["elapsed"] == pytest.approx(2.0)
    assert read["remainder"] == pytest.approx(1.5)


def test_regions_are_totalled_across_both_directions():
    entries = forward("qmm", 1.0, 1.25) + backward("qmm", 1.5, 2.0)
    read = pi.decompose(entries, 0.0, 3.0, context=CTX)
    assert read["totals"]["qmm"] == pytest.approx(0.75)


def test_a_region_with_no_backward_reports_zero_rather_than_a_span():
    """Under LoRA the blocks below the first adapted one have no trainable
    parameter beneath them and genuinely no backward. That is the workload,
    not a fault, and it must not read as a span."""
    read = pi.decompose(forward("attn-core", 1.0, 1.5), 0.0, 2.0, context=CTX)
    assert read["counts"]["attn-core"] == {pi.FORWARD: 1, pi.BACKWARD: 0}
    assert read["regions"]["attn-core"][pi.BACKWARD] == 0.0


def test_counts_are_reported_per_direction():
    entries = (forward("qmm", 1.0, 1.1) + forward("qmm", 1.2, 1.3)
               + backward("qmm", 1.4, 1.5))
    assert pi.counts(entries) == {"qmm": {pi.FORWARD: 2, pi.BACKWARD: 1}}


def test_a_step_with_no_time_cannot_be_decomposed():
    with pytest.raises(RunInvalid, match="no measured time"):
        pi.decompose(forward("qmm", 0.0, 0.0), 1.0, 1.0, context=CTX)


# ---------------------------------------------------------------------------
# Refusals: a log that does not describe the step it claims to
# ---------------------------------------------------------------------------
def test_a_nested_region_refuses_rather_than_counting_twice():
    """The four patch points do not nest today. A future one that did would
    add its time to both itself and its container, which reads as a large
    share and would aim the sprint at the wrong operation."""
    entries = [entry("outer", pi.ENTER, pi.FORWARD, 1.0),
               entry("inner", pi.ENTER, pi.FORWARD, 1.1),
               entry("inner", pi.EXIT, pi.FORWARD, 1.2),
               entry("outer", pi.EXIT, pi.FORWARD, 1.3)]
    with pytest.raises(RunInvalid, match="counted twice"):
        pi.decompose(entries, 0.0, 2.0, context=CTX)


def test_a_close_with_nothing_open_refuses():
    with pytest.raises(RunInvalid, match="nothing opened"):
        pi.decompose([entry("qmm", pi.EXIT, pi.FORWARD, 1.0)], 0.0, 2.0, context=CTX)


def test_a_span_that_never_closes_refuses():
    with pytest.raises(RunInvalid, match="never closed"):
        pi.decompose([entry("qmm", pi.ENTER, pi.FORWARD, 1.0)], 0.0, 2.0, context=CTX)


def test_a_mark_from_outside_the_step_refuses():
    """A log stitched from two runs would otherwise decompose cleanly."""
    with pytest.raises(RunInvalid, match="outside the step"):
        pi.decompose(forward("qmm", 1.0, 5.0), 0.0, 2.0, context=CTX)


def test_a_close_from_a_different_region_refuses():
    entries = [entry("a", pi.ENTER, pi.FORWARD, 1.0),
               entry("b", pi.EXIT, pi.FORWARD, 1.1)]
    with pytest.raises(RunInvalid, match="counted twice|opened by"):
        pi.decompose(entries, 0.0, 2.0, context=CTX)


# ---------------------------------------------------------------------------
# Shares: the numerator is this pass, the denominator is another
# ---------------------------------------------------------------------------
def test_a_share_divides_by_a_denominator_from_another_pass():
    """Dividing by this pass's own total would deflate a lightly marked
    region and inflate a heavily marked one, which is exactly the comparison
    the selection rule makes."""
    read = pi.decompose(forward("attn-core", 1.0, 1.4), 0.0, 4.0, context=CTX)
    assert pi.candidate_share(read, "A", denominator={"total": 2.0, "context": CTX}) == pytest.approx(0.2)


def test_a_candidate_built_from_two_regions_sums_them():
    entries = forward("head-matmul", 1.0, 1.2) + forward("cross-entropy",
                                                          1.3, 1.6)
    read = pi.decompose(entries, 0.0, 4.0, context=CTX)
    assert pi.candidate_share(read, "L", denominator={"total": 1.0, "context": CTX}) == pytest.approx(0.5)


def test_the_overlapping_candidates_both_claim_the_output_head():
    """L and Q share the head, which is why Amendment 5 forbids composing
    their gains."""
    assert "head-matmul" in pi.CANDIDATE_REGIONS["L"]
    assert "head-matmul" in pi.CANDIDATE_REGIONS["Q"]


def test_a_missing_region_refuses_rather_than_reading_as_zero():
    read = pi.decompose(forward("attn-core", 1.0, 1.4), 0.0, 4.0, context=CTX)
    with pytest.raises(RunInvalid, match="did not measure them"):
        pi.candidate_share(read, "L", denominator={"total": 1.0, "context": CTX})


def test_an_unregistered_candidate_refuses():
    read = pi.decompose(forward("attn-core", 1.0, 1.4), 0.0, 4.0, context=CTX)
    with pytest.raises(RunInvalid, match="not a candidate"):
        pi.candidate_share(read, "Z", denominator={"total": 1.0, "context": CTX})


def test_a_share_needs_a_positive_denominator():
    read = pi.decompose(forward("attn-core", 1.0, 1.4), 0.0, 4.0, context=CTX)
    with pytest.raises(RunInvalid, match="positive step total"):
        pi.candidate_share(read, "A", denominator={"total": 0.0, "context": CTX})


# ---------------------------------------------------------------------------
# Shapes, which the collapse rule weights by
# ---------------------------------------------------------------------------
def test_shapes_are_counted_per_direction_not_merely_per_shape():
    """The collapse rule weights a credited ratio by how often each shape ran
    in each direction, and under LoRA those two counts differ. A tally that
    knew the shape and not the direction could not be joined with one that
    knew the direction and not the shape."""
    recorder = pi.Recorder()
    for _ in range(3):
        recorder.note_shape("qmm", (4096, 2560), pi.FORWARD)
    recorder.note_shape("qmm", (4096, 2560), pi.BACKWARD)
    recorder.note_shape("qmm", (1024, 2560), pi.FORWARD)
    assert recorder.shapes["qmm"] == {
        "4096x2560": {pi.FORWARD: 3, pi.BACKWARD: 1},
        "1024x2560": {pi.FORWARD: 1, pi.BACKWARD: 0}}


def test_a_shape_counted_in_one_direction_reads_zero_in_the_other():
    """Zero is a statement: a shape whose backward never fired ran in a block
    with nothing trainable beneath it. Absent would be a different claim."""
    recorder = pi.Recorder()
    recorder.note_shape("qmm", (8, 8), pi.FORWARD)
    assert recorder.shapes["qmm"]["8x8"][pi.BACKWARD] == 0


def test_the_shape_tally_is_exactly_what_the_collapse_rule_weights_by():
    """Read straight into the rule, so the two cannot drift apart: what the
    instrument counts is what the ratio is weighted by."""
    import profile_rules as rules

    recorder = pi.Recorder()
    for _ in range(7):
        recorder.note_shape("qmm", (4096, 2560), pi.FORWARD)
    for _ in range(3):
        recorder.note_shape("qmm", (4096, 2560), pi.BACKWARD)
    tally = recorder.shapes["qmm"]["4096x2560"]
    collapsed = rules.collapse_ratio_lo({"S1": {
        "forward": {"count": tally[pi.FORWARD], "numerator": [2.0],
                    "denominator": [1.0]},
        "backward": {"count": tally[pi.BACKWARD], "numerator": [4.0],
                     "denominator": [1.0]}}})
    assert collapsed["shapes"]["S1"]["forward"]["count"] == 7
    assert collapsed["shapes"]["S1"]["backward"]["count"] == 3


# ---------------------------------------------------------------------------
# Installing the marks on the real names
# ---------------------------------------------------------------------------
@requires_metal
def test_an_unknown_region_refuses_and_installs_nothing():
    with pytest.raises(RunInvalid, match="no such region"):
        pi.install(["attn-core", "nonsense"], pi.Recorder())


@requires_metal
def test_installing_and_removing_leaves_every_name_as_it_was():
    import mlx.nn
    import mlx_lm.models.qwen3 as qwen3

    before = {
        "sdpa": qwen3.scaled_dot_product_attention,
        "qmm": mlx.nn.QuantizedLinear.__call__,
        "head": mlx.nn.QuantizedEmbedding.as_linear,
        "ce": mlx.nn.losses.cross_entropy,
    }
    installation = pi.install(list(pi.REGIONS), pi.Recorder())
    try:
        assert qwen3.scaled_dot_product_attention is not before["sdpa"]
        assert mlx.nn.QuantizedLinear.__call__ is not before["qmm"]
    finally:
        installation.remove()
    assert qwen3.scaled_dot_product_attention is before["sdpa"]
    assert mlx.nn.QuantizedLinear.__call__ is before["qmm"]
    assert mlx.nn.QuantizedEmbedding.as_linear is before["head"]
    assert mlx.nn.losses.cross_entropy is before["ce"]
    assert installation.foreign_on_removal == []


@requires_metal
def test_a_partial_install_rolls_back(monkeypatch):
    """One bad seam must not leave the others in place: a half-instrumented
    process would time some regions and silently drop the rest."""
    import mlx.nn
    import mlx_lm.models.qwen3 as qwen3

    before = qwen3.scaled_dot_product_attention
    broken = pi.Region("broken",
                       pi.seams.Seam("mlx.nn", "NoSuchThing.__call__"),
                       "first_argument")
    monkeypatch.setitem(pi.REGIONS, "broken", broken)
    with pytest.raises(pi.seams.SeamRefusal):
        pi.install(["attn-core", "broken"], pi.Recorder())
    assert qwen3.scaled_dot_product_attention is before
    assert mlx.nn.QuantizedLinear.__call__.__module__ == "mlx.nn.layers.quantized"


# ---------------------------------------------------------------------------
# The mechanism itself, on MLX's CPU stream
# ---------------------------------------------------------------------------
@pytest.fixture()
def on_cpu():
    """Run the marks on the CPU stream.

    Not because it removes the need for a device - it does not, see the module
    docstring - but because it keeps these off the machine's one GPU, which a
    binding measurement may be holding.
    """
    import mlx.core as mx

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield mx
    finally:
        mx.set_default_device(previous)


@requires_metal
def test_a_mark_changes_no_value(on_cpu):
    """The whole instrument rests on this: if the marked run and the plain run
    disagree on the loss or the gradients, the two are timing different work
    and no comparison between them means anything."""
    mx = on_cpu
    mx.random.seed(0)
    weight = mx.random.normal((32, 32))
    inputs = mx.random.normal((8, 32))
    mx.eval(weight, inputs)

    def plain(w):
        return mx.sum(mx.tanh(inputs @ w))

    recorder = pi.Recorder()
    enter = pi.make_mark(recorder, "region", pi.ENTER)
    leave = pi.make_mark(recorder, "region", pi.EXIT)

    def marked(w):
        return mx.sum(leave(mx.tanh(enter(inputs @ w))))

    plain_loss, plain_grad = mx.value_and_grad(plain)(weight)
    marked_loss, marked_grad = mx.value_and_grad(marked)(weight)
    mx.eval(plain_loss, plain_grad, marked_loss, marked_grad)
    assert bool(mx.array_equal(plain_loss, marked_loss))
    assert bool(mx.array_equal(plain_grad, marked_grad))


@requires_metal
def test_the_marks_fire_in_both_directions_and_in_order(on_cpu):
    mx = on_cpu
    mx.random.seed(0)
    weight = mx.random.normal((16, 16))
    inputs = mx.random.normal((4, 16))
    mx.eval(weight, inputs)

    recorder = pi.Recorder()
    enter = pi.make_mark(recorder, "region", pi.ENTER)
    leave = pi.make_mark(recorder, "region", pi.EXIT)

    def marked(w):
        return mx.sum(leave(mx.tanh(enter(inputs @ w))))

    loss, grad = mx.value_and_grad(marked)(weight)
    mx.eval(loss, grad)

    assert [(end, direction) for _, end, direction, _ in recorder.entries] == [
        (pi.ENTER, pi.FORWARD), (pi.EXIT, pi.FORWARD),
        (pi.EXIT, pi.BACKWARD), (pi.ENTER, pi.BACKWARD),
    ]
    assert pi.counts(recorder.entries) == {
        "region": {pi.FORWARD: 1, pi.BACKWARD: 1}}


@requires_metal
def test_a_region_no_gradient_reaches_logs_a_forward_and_no_backward(on_cpu):
    """The detectable signature of a region that is simply not in the
    backward, which is what every non-adapted block looks like."""
    mx = on_cpu
    mx.random.seed(0)
    weight = mx.random.normal((16, 16))
    constant = mx.random.normal((4, 16))
    mx.eval(weight, constant)

    recorder = pi.Recorder()
    stranded = pi.make_mark(recorder, "stranded", pi.ENTER)

    def marked(w):
        stranded(constant)          # nothing downstream reads this
        return mx.sum(mx.tanh(constant @ w))

    loss, grad = mx.value_and_grad(marked)(weight)
    mx.eval(loss, grad)
    assert pi.counts(recorder.entries)["stranded"] == {pi.FORWARD: 1,
                                                       pi.BACKWARD: 0}


@requires_metal
def test_a_real_span_is_positive_and_inside_the_step(on_cpu):
    """The arithmetic and the mechanism meeting: a log the marks actually
    produced, decomposed by the same function the harness will use."""
    import time

    mx = on_cpu
    mx.random.seed(0)
    weight = mx.random.normal((64, 64))
    inputs = mx.random.normal((32, 64))
    mx.eval(weight, inputs)

    recorder = pi.Recorder()
    enter = pi.make_mark(recorder, "region", pi.ENTER)
    leave = pi.make_mark(recorder, "region", pi.EXIT)

    def marked(w):
        return mx.sum(leave(mx.tanh(enter(inputs @ w))))

    mx.synchronize()
    started = time.perf_counter()
    loss, grad = mx.value_and_grad(marked)(weight)
    mx.eval(loss, grad)
    mx.synchronize()
    ended = time.perf_counter()

    read = pi.decompose(recorder.entries, started, ended, context=CTX)
    assert read["regions"]["region"][pi.FORWARD] > 0.0
    assert read["regions"]["region"][pi.BACKWARD] > 0.0
    assert read["remainder"] >= 0.0
    assert read["covered"] <= read["elapsed"]


def test_a_denominator_from_a_different_workload_refuses():
    """The numerator and the denominator come from different passes by
    design, so nothing else would notice that one measured batch 4 and the
    other batch 1. The result would be a ratio of two workloads."""
    read = pi.decompose(forward("attn-core", 1.0, 1.4), 0.0, 4.0, context=CTX)
    with pytest.raises(RunInvalid, match="two workloads"):
        pi.candidate_share(read, "A",
                           denominator={"total": 2.0, "context": OTHER})


def test_a_bare_number_is_not_a_denominator():
    """A number carries no evidence of where it came from, so it cannot be
    checked against the pass it is dividing."""
    read = pi.decompose(forward("attn-core", 1.0, 1.4), 0.0, 4.0, context=CTX)
    with pytest.raises(RunInvalid, match="must carry the context"):
        pi.candidate_share(read, "A", denominator=2.0)


def test_the_decomposition_carries_the_context_it_ran_in():
    read = pi.decompose(forward("attn-core", 1.0, 1.4), 0.0, 4.0, context=CTX)
    assert read["context"] == CTX


def test_clearing_a_warmup_keeps_what_identifies_the_run():
    """The context is not timing. Dropping it with the warm-up entries would
    let the checked share become an unchecked one."""
    recorder = pi.Recorder(context=dict(CTX))
    recorder.entries.append(entry("qmm", pi.ENTER, pi.FORWARD, 1.0))
    recorder.note_shape("qmm", (8, 8), pi.FORWARD)
    recorder.clear()
    assert recorder.entries == []
    assert recorder.shapes == {}
    assert recorder.context == CTX
