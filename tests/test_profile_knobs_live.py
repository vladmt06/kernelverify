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
    prepared = knob.prepare(rig["model"], WIDTH)
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
    prepared = knob.prepare(rig["model"], WIDTH)
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
    prepared = knob.prepare(rig["model"], WIDTH)
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
    prepared = knob.prepare(rig["model"], WIDTH)
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
    prepared = knob.prepare(rig["model"], WIDTH)
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
    prepared = knob.prepare(rig["model"], WIDTH)

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
        prepared = knob.prepare(rig["model"], WIDTH)
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
    prepared = knob.prepare(rig["model"], WIDTH)
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
        prepared = knob.prepare(rig["model"], WIDTH)
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


# --- which attention path MLX actually takes, and when ----------------------
#
# Section 4.2 nominates MLX's fused attention as candidate A's floor, and the
# plan's own investigation asked whether stock's training forward already
# calls it, which would put A's forward ratio at 1.0 and leave the whole case
# for candidate A in the backward. These settle it by reading the graph MLX
# built rather than by reading MLX's source, and they are tests rather than a
# probe's printout because both answers are load-bearing and an MLX upgrade
# must break them loudly.


def _primitives(*outputs) -> list[str]:
    """The names of the operations MLX put in the graph behind `outputs`.

    `export_to_dot` is the only way from Python to see which primitive MLX
    chose, and choosing is exactly what is in question here: `fast.
    scaled_dot_product_attention` is one primitive when it is fused and about
    twenty when it is composed out of matmuls and a softmax.
    """
    import io
    import re

    import mlx.core as mx

    buffer = io.StringIO()
    mx.export_to_dot(buffer, *outputs)
    return re.findall(r'label ="([A-Za-z]+)"', buffer.getvalue())


def _operands(head_dim, queries=8, keys=2, length=97):
    import mlx.core as mx

    q = mx.random.normal((1, queries, length, head_dim))
    k = mx.random.normal((1, keys, length, head_dim))
    v = mx.random.normal((1, keys, length, head_dim))
    mx.eval(q, k, v)
    return q, k, v


FUSED = "ScaledDotProductAttention"
FUSED_AT = (64, 80, 128)
COMPOSED_AT = (32, 48, 72, 96, 160, 256)


@requires_metal
def test_mlx_fuses_attention_at_three_head_dimensions_and_no_others():
    """The discontinuity a four-point head-dimension ladder would cross.

    It binds the FLOOR and not the share: a floor arm calls the fused entry
    point, so a 128/96/64/32 ladder would fit one line through two different
    implementations, and a clean fit through a discontinuity is worse than a
    bad one because it looks fine.
    """
    import mlx.core as mx

    for head_dim in FUSED_AT:
        q, k, v = _operands(head_dim)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=head_dim ** -0.5, mask="causal")
        assert _primitives(out) == [FUSED], (
            f"head dimension {head_dim} was expected to take the fused path "
            f"and did not")

    for head_dim in COMPOSED_AT:
        q, k, v = _operands(head_dim)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=head_dim ** -0.5, mask="causal")
        assert FUSED not in _primitives(out), (
            f"head dimension {head_dim} took the fused path, so the boundary "
            f"is not where the floor's ladder was built to avoid it")


@requires_metal
def test_a_gradient_trace_composes_attention_at_every_head_dimension():
    """Stock's training attention is composed in BOTH directions.

    This is what settles the plan's open investigation. mlx-lm's training path
    passes no cache, so it reaches `mx.fast.scaled_dot_product_attention`, and
    section 4.2 reads that as stock already being at its own floor. It is not:
    inside a gradient trace MLX composes the call, so the fused entry point is
    a genuine floor for candidate A rather than a description of stock.
    """
    import mlx.core as mx

    for head_dim in FUSED_AT:
        q, k, v = _operands(head_dim)

        def attention(q, k, v):
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=head_dim ** -0.5, mask="causal").sum()

        value, grads = mx.vjp(attention, [q, k, v], [mx.array(1.0)])
        assert FUSED not in _primitives(*value, *grads), (
            f"at head dimension {head_dim} the gradient trace kept the fused "
            f"path, so stock's training forward IS its own floor and "
            f"candidate A's case rests on the backward alone")


@requires_metal
def test_a_gradient_trace_composes_attention_it_is_not_differentiating():
    """The decomposition is a property of the TRACE, not of the operands.

    It matters because mlx-lm adapts only the last few blocks, so most of the
    model's attention calls have no gradient reaching them at all. If the
    choice were per-array, stock would be fused in those blocks and composed
    in the adapted ones, and one dial would be measuring two implementations.
    Both calls below are inside one trace and only one is differentiated.
    """
    import mlx.core as mx

    head_dim = 128
    q, k, v = _operands(head_dim)
    frozen = _operands(head_dim)

    def two_calls(q, k, v):
        differentiated = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=head_dim ** -0.5, mask="causal")
        constant = mx.fast.scaled_dot_product_attention(
            *frozen, scale=head_dim ** -0.5, mask="causal")
        return differentiated.sum() + constant.sum()

    value, grads = mx.vjp(two_calls, [q, k, v], [mx.array(1.0)])
    traced = _primitives(*value, *grads)
    assert FUSED not in traced, (
        "a call on constants inside a gradient trace kept the fused path, so "
        "stock's attention is fused in the unadapted blocks and composed in "
        "the adapted ones, and one dial spans two implementations")
    assert traced.count("Softmax") == 2, (
        f"expected both calls composed and found {traced.count('Softmax')} "
        f"softmaxes")

    outside = _primitives(
        mx.fast.scaled_dot_product_attention(
            q, k, v, scale=head_dim ** -0.5, mask="causal"),
        mx.fast.scaled_dot_product_attention(
            *frozen, scale=head_dim ** -0.5, mask="causal"))
    assert outside.count(FUSED) == 2, (
        "the same two calls outside a gradient trace did not fuse, so this "
        "test proves nothing about tracing")


# --- the three attention dials clause 15 registers --------------------------


def _attention_prepared(rig, dial, width=WIDTH):
    knob = pk.ATTENTION_KNOBS[dial]
    return knob, knob.prepare(rig["model"], width)


def _loss_and_gradients(rig, seam_map):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.trainer import default_loss

    from metalrunner import seams

    value_and_grad = nn.value_and_grad(rig["model"], default_loss)
    installation = seams.Installation()
    try:
        for seam, wrap in seam_map.items():
            installation.install(seam, wrap)
        (loss, _toks), grad = value_and_grad(rig["model"], *rig["batch"])
        mx.eval(loss, grad)
        return loss, dict(tree_flatten(grad)), dict(installation.counts)
    finally:
        installation.remove()


@requires_metal
def test_every_attention_dial_is_bit_identical_to_stock_at_full_setting(rig):
    """The check that separates a dial from a different computation.

    For the length dial this is also the check that its hand-built mask IS
    what the string "causal" means, which is otherwise a claim about MLX's
    reading of that string rather than a measured fact.
    """
    import mlx.core as mx

    stock_loss, stock_grad, _counts = _loss_and_gradients(rig, {})
    for dial in pk.ATTENTION_DIALS:
        knob, prepared = _attention_prepared(rig, dial)
        loss, grad, _ = _loss_and_gradients(rig, knob.arm(prepared, 1.0))
        assert mx.array_equal(stock_loss, loss), (
            f"dial {dial} at the full setting does not compute stock's loss")
        differing = [name for name in stock_grad
                     if not mx.array_equal(stock_grad[name], grad[name])]
        assert not differing, (
            f"dial {dial} at the full setting agrees with stock on the loss "
            f"but not on {len(differing)} gradients, starting with "
            f"{differing[:3]}")


@requires_metal
def test_every_attention_dial_holds_its_call_count_across_settings(rig):
    """A dial must shrink the operation, never delete one of its instances."""
    for dial in pk.ATTENTION_DIALS:
        knob, prepared = _attention_prepared(rig, dial)
        counts = [_loss_and_gradients(rig, knob.arm(prepared, phi))[2]
                  for phi in pk.PHIS]
        assert all(count == counts[0] for count in counts), (
            f"dial {dial} calls attention a different number of times at "
            f"different settings: {counts}")
        assert all(value > 0 for value in counts[0].values()), (
            f"dial {dial} installs a seam nothing called: {counts[0]}")


@requires_metal
def test_every_attention_scaffold_arm_computes_stocks_own_loss(rig):
    """Clause 5's scaffold-only arm, at both ends of every dial's ladder."""
    import mlx.core as mx

    stock_loss, _grad, _counts = _loss_and_gradients(rig, {})
    for dial in pk.ATTENTION_DIALS:
        knob, prepared = _attention_prepared(rig, dial)
        for phi in (1.0, min(pk.PHIS)):
            loss, _g, _c = _loss_and_gradients(rig, knob.scaffold(prepared, phi))
            assert mx.array_equal(stock_loss, loss), (
                f"dial {dial}'s scaffold arm at phi={phi} changed the loss, "
                f"so it is not measuring only the scaffold")


@requires_metal
def test_every_attention_dial_places_four_realisable_settings(rig):
    """Clause 15's eligibility test, measured rather than assumed."""
    for dial in pk.ATTENTION_DIALS:
        _knob, prepared = _attention_prepared(rig, dial)
        settings = pk.realisable_settings(prepared["attention"])
        assert len(settings) >= pk.MIN_DIAL_SETTINGS, (
            f"dial {dial} places only {len(settings)} distinct settings on "
            f"the ladder, so clause 15 refuses it whatever its price")
        assert settings == pk.PHIS


@requires_metal
def test_the_length_dial_refuses_a_step_of_a_different_width(rig):
    """A mask built for the wrong width masks the wrong score matrix.

    The failure it replaces is a broadcast error deep inside MLX, which says
    nothing about which of two widths was the wrong one.
    """
    knob, prepared = _attention_prepared(rig, "kv-length", width=WIDTH + 32)
    with pytest.raises(RunInvalid, match="masks were built for"):
        _loss_and_gradients(rig, knob.arm(prepared, 1.0))


@requires_metal
def test_the_attention_ablation_arm_also_zeroes_the_key_and_value_backward(rig):
    """What the residue clause 18 credits actually rests on, stated exactly.

    The arm keeps the keys and values alive through a reduction multiplied by
    zero, so a lazy graph cannot drop their projections in the FORWARD and
    charge that work to attention. What the trick cannot preserve is the
    backward: multiplying by zero makes the cotangent reaching the key and
    value projections zero, so their own backward computes zeros where stock
    computes a matmul.

    So the ablated step is missing attention AND part of two projections'
    backward, and the residue read off it is an upper bound rather than a
    measurement. Pinned here because it is invisible in the timings and the
    residue is credited straight into a rewrite candidate's share.
    """
    import mlx.core as mx

    stock_loss, stock_grad, _none = _loss_and_gradients(rig, {})
    knob, prepared = _attention_prepared(rig, "kv-length")
    _dialled, _g, dialled_counts = _loss_and_gradients(rig, knob.arm(prepared, 1.0))
    loss, grad, counts = _loss_and_gradients(rig, knob.ablate(prepared))

    assert not mx.array_equal(stock_loss, loss), (
        "the ablated arm computes stock's loss, so it removed nothing")
    assert sorted(stock_grad) == sorted(grad), (
        "the ablated arm changed which parameters have a gradient at all, so "
        "it removed more of the model than attention")
    assert counts == dialled_counts, (
        f"the ablated arm reaches attention {counts} times against a dialled "
        f"arm's {dialled_counts}, so it is not a like-for-like replacement")

    queries = [name for name in grad if "q_proj" in name]
    assert queries and all(bool(mx.any(grad[name] != 0)) for name in queries), (
        "the query projections have no gradient under the ablated arm, so it "
        "cut the path the arm is supposed to pass through")
    zeroed = [name for name in grad
              if ("k_proj" in name or "v_proj" in name)
              and not bool(mx.any(grad[name] != 0))]
    assert len(zeroed) == 4 * ADAPTED_LAYERS, (
        f"{len(zeroed)} key and value adapters carry a zero gradient under "
        f"the ablated arm and {4 * ADAPTED_LAYERS} were expected, one pair of "
        f"low-rank factors for each of the two projections in every adapted "
        f"layer; a different count means the arm's reach changed")


@requires_metal
def test_the_fused_entry_point_cannot_be_reached_from_inside_the_real_step(rig):
    """Clause 19's floor for candidate A, checked where clause 19 puts it.

    Clause 19 requires a floor to be installed at the same seam as the knob
    and dialled inside the same step, so that the share and the ratio are the
    same kind of quantity. Candidate A's floor is MLX's fused attention, and
    this installs exactly that at exactly that seam.

    What comes back is the composed path, because the step is a gradient
    trace. So a floor arm built to clause 19's letter would run stock's own
    computation and report a ratio of 1.0 - a number that looks measured and
    is an artefact of where it was measured.

    The layer inspected is the first, which LoRA does not adapt, so nothing
    differentiates its attention and it is still composed.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.tuner.trainer import default_loss

    import profile_instrument as pi
    from metalrunner import seams

    def capture(outputs):
        def wrap(_original):
            def call(queries, keys, values, *args, **kwargs):
                out = mx.fast.scaled_dot_product_attention(
                    queries, keys, values, scale=kwargs["scale"],
                    mask=kwargs["mask"])
                outputs.append(out)
                return out
            return call
        return wrap

    def first_attention(under_gradient):
        outputs = []
        installation = seams.Installation()
        try:
            installation.install(pi.REGIONS["attn-core"].seam, capture(outputs))
            if under_gradient:
                nn.value_and_grad(rig["model"], default_loss)(
                    rig["model"], *rig["batch"])
            else:
                default_loss(rig["model"], *rig["batch"])
        finally:
            installation.remove()
        return _primitives(outputs[0])

    plain = first_attention(under_gradient=False)
    assert plain.count(FUSED) == 1 and "Softmax" not in plain, (
        "the fused entry point did not fuse even outside a gradient trace, so "
        "this test says nothing about tracing")

    traced = first_attention(under_gradient=True)
    assert FUSED not in traced, (
        "the fused entry point survived mlx-lm's own gradient trace, so a "
        "floor arm installed at this seam would measure the fused kernel and "
        "clause 19 can be applied to candidate A as written")
    assert traced.count("Softmax") == 1
