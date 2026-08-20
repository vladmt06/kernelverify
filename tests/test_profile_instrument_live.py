"""The instrument inside mlx-lm's own training step, on a real model.

Everything else about the instrument is pinned without a device. This file
pins the one claim that cannot be: that marks installed on the shipped names
fire inside the trainer's real step, in both directions, and change nothing
about what that step computes.

The identity check is the load-bearing one. If the marked run and the plain
run disagree on the loss or on any gradient, then the two are not timing the
same work and no share taken from the marked run says anything about the
shipped step. It is asserted over every gradient array rather than a summary,
because a summary can agree while the arrays beneath it do not.

Qwen3-0.6B stands in for the pinned 4B here. The structure the instrument
reads - a projection per layer, attention per layer, one output head, and a
backward that reaches only the adapted blocks - is the same in both, and this
one loads in seconds.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import profile_instrument as pi  # noqa: E402
from conftest import requires_metal  # noqa: E402

MODEL = (Path(__file__).resolve().parents[1] / "bench" / ".models"
         / "qwen3-0.6b-4bit-g64")
ADAPTED_LAYERS = 4
BATCH, TOKENS = 2, 128

# What identifies the workload these passes ran. Both the plain and the marked
# step here use the same batch, so they share one context.
CTX = {"model": "qwen3-0.6b-4bit-g64", "batch": BATCH, "tokens": TOKENS}


@pytest.fixture(scope="module")
def trained_step():
    """One real training step, plain and instrumented, from one loaded model.

    Built once for the module: loading the model and warming both paths costs
    far more than the step being measured, and every test here reads the same
    pair of runs rather than provoking new ones.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.trainer import default_loss
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load

    was_compiling = True
    mx.disable_compile()
    try:
        model, _ = load(str(MODEL))
        model.freeze()
        linear_to_lora_layers(model, ADAPTED_LAYERS,
                              {"rank": 8, "scale": 20.0, "dropout": 0.0})
        model.train()

        mx.random.seed(0)
        tokens = mx.random.randint(0, 1000, (BATCH, TOKENS + 1))
        lengths = mx.repeat(mx.array([[TOKENS // 2, TOKENS]], dtype=mx.int32),
                            BATCH, axis=0)
        mx.eval(tokens, lengths)

        value_and_grad = nn.value_and_grad(
            model, lambda m: default_loss(m, tokens, lengths)[0])

        def step():
            loss, grad = value_and_grad(model)
            mx.eval(loss, grad)
            return loss, dict(tree_flatten(grad))

        step()
        plain_loss, plain_grad = step()

        recorder = pi.Recorder()
        installation = pi.install(list(pi.REGIONS), recorder)
        try:
            step()
            recorder.clear()
            marked_loss, marked_grad = step()
        finally:
            installation.remove()

        yield {
            "mx": mx, "depth": len(model.model.layers), "recorder": recorder,
            "plain_loss": plain_loss, "plain_grad": plain_grad,
            "marked_loss": marked_loss, "marked_grad": marked_grad,
            "foreign": installation.foreign_on_removal,
        }
    finally:
        if was_compiling:
            mx.enable_compile()


@requires_metal
def test_the_marked_step_computes_the_same_loss(trained_step):
    mx = trained_step["mx"]
    assert bool(mx.array_equal(trained_step["plain_loss"],
                               trained_step["marked_loss"]))


@requires_metal
def test_the_marked_step_computes_the_same_gradients(trained_step):
    """Every array, not a summary: a summary can agree while the arrays
    beneath it do not."""
    mx = trained_step["mx"]
    plain, marked = trained_step["plain_grad"], trained_step["marked_grad"]
    assert set(plain) == set(marked)
    assert plain, "the model produced no gradients, so nothing was compared"
    differing = [name for name in plain
                 if not bool(mx.array_equal(plain[name], marked[name]))]
    assert differing == []


@requires_metal
def test_every_region_fires_forward_once_per_place_it_appears(trained_step):
    """The count is what proves a seam was reached. Attention runs once per
    layer and the output head once per step, and both are read from the model
    rather than assumed."""
    counts = pi.counts(trained_step["recorder"].entries)
    depth = trained_step["depth"]
    assert counts["attn-core"][pi.FORWARD] == depth
    assert counts["head-matmul"][pi.FORWARD] == 1
    assert counts["cross-entropy"][pi.FORWARD] == 1
    # Seven quantized projections per block: q, k, v, o, gate, up, down.
    assert counts["qmm"][pi.FORWARD] == 7 * depth


@requires_metal
def test_the_backward_reaches_only_the_adapted_blocks(trained_step):
    """Under LoRA the blocks below the first adapted one have no trainable
    parameter beneath them and no backward at all. That is the workload, and
    a profile that assumed a backward per layer would overstate every share
    built on one."""
    counts = pi.counts(trained_step["recorder"].entries)
    assert counts["attn-core"][pi.BACKWARD] == ADAPTED_LAYERS
    assert counts["head-matmul"][pi.BACKWARD] == 1
    assert counts["cross-entropy"][pi.BACKWARD] == 1
    assert 0 < counts["qmm"][pi.BACKWARD] < counts["qmm"][pi.FORWARD]


@requires_metal
def test_the_shapes_the_instrument_records_include_the_key_value_projections(
        trained_step):
    """The registered shape list S1 to S5 omits k_proj and v_proj, which are
    a distinct shape and are counted in candidate Q's share. Amendment 5 adds
    them; this is the measurement that says they exist."""
    shapes = trained_step["recorder"].shapes["qmm"]
    depth = trained_step["depth"]
    assert sum(shapes.values()) == 7 * depth
    # Five distinct projection shapes, not four: q and o differ, gate/up and
    # down differ, and k/v are their own.
    assert len(shapes) == 5


@requires_metal
def test_the_log_decomposes_into_spans_inside_the_step(trained_step):
    """The arithmetic and the live mechanism meeting: a log the marks really
    produced, read by the function the harness will use."""
    entries = trained_step["recorder"].entries
    stamps = [stamp for _, _, _, stamp in entries]
    read = pi.decompose(entries, min(stamps) - 1e-6, max(stamps) + 1e-6,
                        context=CTX)
    for region in pi.REGIONS:
        assert read["totals"][region] > 0.0
    assert read["remainder"] >= 0.0
    assert read["covered"] < read["elapsed"]


@requires_metal
def test_every_candidate_can_be_scored_from_one_combined_pass(trained_step):
    """Not how the deciding cell runs - it splits a pass per candidate - but
    the cells that decide nothing use one combined pass, so each candidate's
    regions must all be present in it."""
    entries = trained_step["recorder"].entries
    stamps = [stamp for _, _, _, stamp in entries]
    read = pi.decompose(entries, min(stamps) - 1e-6, max(stamps) + 1e-6,
                        context=CTX)
    for candidate in pi.CANDIDATE_REGIONS:
        share = pi.candidate_share(
            read, candidate,
            denominator={"total": read["elapsed"], "context": CTX})
        assert 0.0 < share < 1.0


@requires_metal
def test_the_names_are_left_exactly_as_they_were_found(trained_step):
    """A profile that leaves a mark installed would silently instrument
    whatever runs next in the same interpreter."""
    import mlx.nn
    import mlx_lm.models.qwen3 as qwen3

    assert trained_step["foreign"] == []
    assert qwen3.scaled_dot_product_attention.__module__ == "mlx_lm.models.base"
    assert mlx.nn.QuantizedLinear.__call__.__module__ == "mlx.nn.layers.quantized"
    assert mlx.nn.losses.cross_entropy.__module__ == "mlx.nn.losses"


@requires_metal
def test_gradient_checkpointing_replays_the_forward_and_inflates_its_count():
    """Why the profile must refuse `--grad-checkpoint` rather than tolerate it.

    Checkpointing trades memory for time by discarding activations and running
    a block's forward AGAIN during the backward. The marks survive it - loss
    and gradients are unchanged - but the replayed forward fires forward marks,
    so a region's forward count and forward span quietly absorb work that
    belongs to the backward. Measured here: attention fires 30 times forward
    on a 28-block model, and the projections 210 times where an unchecked step
    fires 196.

    Nothing built on those counts would notice. The share would look the same
    shape and mean something different, and the collapse rule weights a
    credited ratio by exactly these counts. So the harness refuses the flag,
    and this is the measurement that says why.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load

    # `grad_checkpoint` replaces the layer CLASS's `__call__` and never puts
    # it back, so without this the checkpointing stays installed for every
    # test that loads a Qwen3 model afterwards in the same interpreter. It was
    # caught by the profile harness's own count check, which read 44 attention
    # forwards on a 28-block model in a full-suite run and 28 in an isolated
    # one - which is exactly the failure that check exists to make loud.
    block_type = None
    original_call = None
    mx.disable_compile()
    try:
        model, _ = load(str(MODEL))
        model.freeze()
        linear_to_lora_layers(model, 2, {"rank": 8, "scale": 20.0,
                                         "dropout": 0.0})
        model.train()
        depth = len(model.model.layers)
        block_type = type(model.model.layers[0])
        original_call = block_type.__call__
        grad_checkpoint(model.model.layers[0])

        mx.random.seed(0)
        tokens = mx.random.randint(0, 1000, (2, 65))
        lengths = mx.repeat(mx.array([[32, 64]], dtype=mx.int32), 2, axis=0)
        mx.eval(tokens, lengths)
        value_and_grad = nn.value_and_grad(
            model, lambda m: default_loss(m, tokens, lengths)[0])

        def step():
            loss, grad = value_and_grad(model)
            mx.eval(loss, grad)
            return loss, dict(tree_flatten(grad))

        plain_loss, plain_grad = step()
        recorder = pi.Recorder(context=dict(CTX))
        installation = pi.install(list(pi.REGIONS), recorder)
        try:
            step()
            recorder.clear()
            marked_loss, marked_grad = step()
        finally:
            installation.remove()
    finally:
        if block_type is not None and original_call is not None:
            block_type.__call__ = original_call
        mx.enable_compile()

    # The marks are still identity, so this is not a correctness failure.
    assert bool(mx.array_equal(plain_loss, marked_loss))
    assert all(bool(mx.array_equal(plain_grad[k], marked_grad[k]))
               for k in plain_grad)

    # It is a counting failure, which is worse, because it is silent.
    counts = pi.counts(recorder.entries)
    assert counts["attn-core"][pi.FORWARD] > depth
    assert counts["qmm"][pi.FORWARD] > 7 * depth

    # And the checkpointing is gone again. Asserted here rather than trusted,
    # because the leak is invisible in this file and only shows up as wrong
    # counts in whatever test loads a Qwen3 model next.
    assert block_type.__call__ is original_call
    assert block_type.__call__.__module__.startswith("mlx_lm.models")
