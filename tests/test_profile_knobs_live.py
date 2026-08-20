"""The dials inside mlx-lm's own training step, on a real model.

Everything about the knob arithmetic is pinned without a device. This file
pins the claims that cannot be, and they are the ones that would fail
silently rather than loudly:

    a dial that does nothing produces a clean fit through a horizontal line,
    which looks better than a bad fit rather than worse;

    an arm that shares another arm's compiled graph reports that other arm's
    time, and nothing in the numbers says so;

    a seam left installed during a timed round puts its own cost inside the
    measurement, and a seam removed before an arm traces leaves that arm
    measuring stock.

Each of those is asserted directly here rather than inferred from a share.

Qwen3-0.6B stands in for the pinned 4B. The structure the dials touch - a
quantized projection per layer, one tied head, one loss - is the same in
both, and this one loads in seconds.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import profile_knobs as pk  # noqa: E402
from conftest import requires_metal  # noqa: E402
from decode_rules import RunInvalid  # noqa: E402

MODEL = (Path(__file__).resolve().parents[1] / "bench" / ".models"
         / "qwen3-0.6b-4bit-g64")
ADAPTED_LAYERS = 4
BATCH, WIDTH = 2, 97


@pytest.fixture(scope="module")
def rig():
    """One loaded model, one settled optimizer, one fixed batch.

    The optimizer is settled BEFORE anything is built, which is clause 25's
    second requirement: Adam allocates its moments on the first update, and
    that growth forces a second trace which could otherwise land after a
    seam has been removed.
    """
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load

    model, _ = load(str(MODEL))
    model.freeze()
    linear_to_lora_layers(model, ADAPTED_LAYERS,
                          {"rank": 8, "scale": 20.0, "dropout": 0.0})
    model.train()
    optimizer = optim.Adam(learning_rate=1e-5)
    pk.settle_optimizer(model, optimizer)

    mx.random.seed(0)
    tokens = mx.random.randint(0, 1000, (BATCH, WIDTH))
    lengths = mx.repeat(mx.array([[WIDTH // 2, WIDTH - 1]], dtype=mx.int32),
                        BATCH, axis=0)
    mx.eval(tokens, lengths)
    state = [model.state, optimizer.state, mx.random.state]
    return {"model": model, "optimizer": optimizer, "state": state,
            "batch": (tokens, lengths)}


def _arm(rig, knob, phi, prepared, *, scaffold=False, label=None):
    build = knob.scaffold if scaffold else knob.arm
    arm = pk.Arm(label=label or f"{knob.candidate}@{phi}", phi=phi,
                 seams=build(prepared, phi))
    return pk.prepare_arm(rig["model"], rig["optimizer"], rig["state"],
                          rig["batch"], arm, warmups=1)


@requires_metal
def test_the_optimizer_has_its_state_before_any_arm_is_built(rig):
    """Clause 25's second requirement, checked rather than assumed."""
    from mlx.utils import tree_flatten

    leaves = dict(tree_flatten(rig["optimizer"].state))
    assert leaves, "the optimizer reports no state at all"
    assert any(name.endswith(".m") or name.endswith(".v") for name in leaves), (
        "Adam's moments are absent, so the first update would still grow the "
        "captured tree and force a second trace")


@requires_metal
def test_a_compiled_arm_traces_exactly_once_and_then_stops(rig):
    knob = pk.KNOBS["P3"]
    prepared = knob.prepare(rig["model"])
    compiled = _arm(rig, knob, 0.5, prepared)
    assert compiled.traced_by_warmup == 1, (
        f"the arm traced {compiled.traced_by_warmup} times during one "
        f"warm-up, so something is still growing the captured state")
    pk.one_step(compiled, rig["batch"])
    assert not compiled.retraced()


@requires_metal
def test_two_arms_at_different_settings_do_not_share_one_graph(rig):
    """The compile-cache trap, asserted on times rather than on trust.

    MLX keys its cache on the underlying callable. If both arms somehow shared
    one traced graph they would report the same work, so a real difference in
    time is the evidence that each arm kept its own.
    """
    knob = pk.KNOBS["P3"]
    prepared = knob.prepare(rig["model"])
    full = _arm(rig, knob, 1.0, prepared, label="full")
    quarter = _arm(rig, knob, 0.25, prepared, label="quarter")

    samples = pk.timed_rounds([full, quarter], rig["batch"], rounds=3)
    import statistics
    wide = statistics.median(samples["full"])
    narrow = statistics.median(samples["quarter"])
    assert narrow < wide, (
        f"the quarter-width arm took {narrow:.4f}s against the full arm's "
        f"{wide:.4f}s, so the dial moved nothing and the two arms are "
        f"probably running one graph")


@requires_metal
def test_the_timed_rounds_run_with_no_seam_installed(rig):
    """A leaked installation would cost time inside the measurement.

    Checked by reading the seam's live binding during the timed region, not
    by trusting that `prepare_arm` removed what it installed.
    """
    from metalrunner import seams

    knob = pk.KNOBS["P3"]
    prepared = knob.prepare(rig["model"])
    seam = next(iter(knob.arm(prepared, 0.5)))
    original = seams.current(seam)

    compiled = _arm(rig, knob, 0.5, prepared)
    assert seams.current(seam) is original, (
        "the seam is still replaced after the arm was built, so every timed "
        "round would pay the installer's own cost")
    pk.one_step(compiled, rig["batch"])
    assert seams.current(seam) is original


@requires_metal
def test_a_retrace_during_a_timed_round_refuses_the_run(rig):
    """Clause 25's general guard, provoked deliberately.

    A compiled step retraces when its input signature changes, so timing an
    arm against a batch of a different width forces exactly the fault the
    counter exists to catch.
    """
    import mlx.core as mx

    knob = pk.KNOBS["P3"]
    prepared = knob.prepare(rig["model"])
    compiled = _arm(rig, knob, 0.5, prepared)

    mx.random.seed(1)
    other = (mx.random.randint(0, 1000, (BATCH, WIDTH + 32)),
             mx.repeat(mx.array([[WIDTH // 2, WIDTH]], dtype=mx.int32),
                       BATCH, axis=0))
    mx.eval(*other)

    with pytest.raises(RunInvalid, match="re-traced during timed rounds"):
        pk.timed_rounds([compiled], other, rounds=1)


@requires_metal
def test_the_full_setting_is_bit_identical_to_stock(rig):
    """The check that separates a dial from a different computation.

    Compared over the whole gradient tree and not only the loss, because a
    dial that changed only a gradient would pass a loss-only check and would
    then be measuring work the shipped step never does.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.trainer import default_loss
    from metalrunner import seams

    knob = pk.KNOBS["Q"]
    prepared = knob.prepare(rig["model"])
    value_and_grad = nn.value_and_grad(rig["model"], default_loss)

    def evaluate(seam_map):
        installation = seams.Installation()
        try:
            for seam, wrap in seam_map.items():
                installation.install(seam, wrap)
            (loss, _toks), grad = value_and_grad(rig["model"], *rig["batch"])
            mx.eval(loss, grad)
            return loss, dict(tree_flatten(grad))
        finally:
            installation.remove()

    stock_loss, stock_grad = evaluate({})
    dial_loss, dial_grad = evaluate(knob.arm(prepared, 1.0))

    assert mx.array_equal(stock_loss, dial_loss), (
        "the arm at full setting does not compute stock's loss")
    assert sorted(stock_grad) == sorted(dial_grad)
    differing = [name for name in stock_grad
                 if not mx.array_equal(stock_grad[name], dial_grad[name])]
    assert not differing, (
        f"the arm at full setting agrees with stock on the loss but not on "
        f"{len(differing)} gradients, starting with {differing[:3]}")


@requires_metal
def test_the_dial_does_not_change_how_often_the_seam_is_called(rig):
    """A retune knob must not delete a dispatch, which a stub would.

    This counts PYTHON CALLS through the seam, which is what a deletion
    would break. It does not prove no Metal kernel was added, and Amendment 5
    says so rather than calling it a launch count.
    """
    from metalrunner import seams

    knob = pk.KNOBS["P3"]
    prepared = knob.prepare(rig["model"])

    def count_at(phi):
        installation = seams.Installation()
        try:
            for seam, wrap in knob.arm(prepared, phi).items():
                installation.install(seam, wrap)
            step, _traces = pk.build_step(rig["model"], rig["optimizer"],
                                          rig["state"])
            pk.one_step(pk.CompiledArm(label=str(phi), phi=phi, step=step,
                                       state=rig["state"], traces=[],
                                       traced_by_warmup=0), rig["batch"])
            return dict(installation.counts)
        finally:
            installation.remove()

    assert count_at(1.0) == count_at(0.25)


@requires_metal
def test_every_seam_the_knobs_name_is_reached_at_least_once(rig):
    """A dial installed on a name nothing calls would read as a flat line."""
    from metalrunner import seams

    for name in ("Q", "L", "P3"):
        knob = pk.KNOBS[name]
        prepared = knob.prepare(rig["model"])
        installation = seams.Installation()
        try:
            for seam, wrap in knob.arm(prepared, 0.5).items():
                installation.install(seam, wrap)
            step, _traces = pk.build_step(rig["model"], rig["optimizer"],
                                          rig["state"])
            pk.one_step(pk.CompiledArm(label=name, phi=0.5, step=step,
                                       state=rig["state"], traces=[],
                                       traced_by_warmup=0), rig["batch"])
            counts = dict(installation.counts)
        finally:
            installation.remove()
        silent = [seam for seam, count in counts.items() if count == 0]
        assert not silent, (
            f"candidate {name} installs {silent}, which nothing called, so "
            f"that region's contribution would read as zero")


@requires_metal
def test_the_scaffold_arm_runs_the_operation_at_full_size(rig):
    """Clause 5's scaffold-only arm, checked against stock's own value.

    It applies the dial's index arithmetic and discards it, so it must still
    compute what stock computes at any setting. An arm that changed the value
    would be pricing something other than the scaffold.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.tuner.trainer import default_loss
    from metalrunner import seams

    knob = pk.KNOBS["P3"]
    prepared = knob.prepare(rig["model"])
    value_and_grad = nn.value_and_grad(rig["model"], default_loss)

    def loss_with(seam_map):
        installation = seams.Installation()
        try:
            for seam, wrap in seam_map.items():
                installation.install(seam, wrap)
            (loss, _toks), _grad = value_and_grad(rig["model"], *rig["batch"])
            mx.eval(loss)
            return loss
        finally:
            installation.remove()

    stock = loss_with({})
    for phi in (1.0, 0.25):
        assert mx.array_equal(stock, loss_with(knob.scaffold(prepared, phi))), (
            f"the scaffold-only arm at phi={phi} changed the loss, so it is "
            f"not measuring only the scaffold")


@requires_metal
def test_every_patched_class_is_put_back(rig):
    """A leaked class-level patch already cost a whole test session once.

    mlx-lm and MLX both hold methods on classes, and a replacement written to
    a class outlives the test that wrote it, so this asserts the binding is
    the original object rather than merely callable.
    """
    from metalrunner import seams

    for name in ("Q", "L", "P3"):
        knob = pk.KNOBS[name]
        prepared = knob.prepare(rig["model"])
        seam_map = knob.arm(prepared, 0.5)
        originals = {seam: seams.current(seam) for seam in seam_map}

        installation = seams.Installation()
        try:
            for seam, wrap in seam_map.items():
                installation.install(seam, wrap)
        finally:
            installation.remove()

        for seam, original in originals.items():
            assert seams.current(seam) is original, (
                f"{seam} was left replaced after candidate {name}'s arm")
        assert not installation.foreign_on_removal
