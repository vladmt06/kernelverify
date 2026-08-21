"""The ceiling sweep on the real device, where its two claims can fail.

Both are claims about what MLX actually does, and both would fail SILENTLY
rather than loudly. A backward driven by the wrong cotangent returns a clean
number that is not the cost a real loss produces. A floor arm that still
dispatched a quantized matmul returns a ratio near 1.0 that looks measured.

The shapes here are the registered ones, so these arms are the bench's own
arms rather than a miniature of them. They are synthetic matmuls of seconds
each, which is what makes that affordable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import ceiling_sweep as cs  # noqa: E402
import profile_rules as rules  # noqa: E402
from conftest import requires_metal  # noqa: E402
from decode_rules import RunInvalid  # noqa: E402

# The smallest registered shape at the shorter width, so a live test costs
# milliseconds. What is being checked is which primitive MLX chose and which
# cotangent it was handed, and neither depends on the shape being large.
SHAPE, WIDTH = "S6", "short"


def _primitives(*outputs) -> list[str]:
    import io
    import re

    import mlx.core as mx

    buffer = io.StringIO()
    mx.export_to_dot(buffer, *outputs)
    return re.findall(r'label ="([A-Za-z]+)"', buffer.getvalue())


@pytest.fixture(scope="module")
def built():
    return cs._operands(SHAPE, WIDTH, batch=1)


@requires_metal
def test_the_stock_arm_dispatches_a_quantized_matmul_and_the_dense_one_does_not(built):
    """A ceiling needs the thing and the thing it is a ceiling for.

    A dense arm that still dispatched `QuantizedMatmul` would report a ratio
    near 1.0 that looked measured, which is exactly the fault Amendment 6
    clause 27 found in candidate A's floor and the reason candidate A has no
    floor at all.
    """
    activation = built["activation"]
    stock = _primitives(built[cs.STOCK](activation))
    dense = _primitives(built[cs.DENSE](activation))
    assert "QuantizedMatmul" in stock
    assert "QuantizedMatmul" not in dense
    assert "Matmul" in dense


@requires_metal
def test_a_backward_is_a_single_quantized_matmul_fed_the_cotangent(built):
    """Amendment 10 clause 44 read this off the built graph and registered it,
    so it is pinned here rather than left as a claim in a document."""
    import mlx.core as mx

    activation = built["activation"]
    call = built[cs.STOCK]
    output = call(activation)
    mx.eval(output)
    cotangent = cs._dense_cotangent(output)
    _value, grads = mx.vjp(call, [activation], [cotangent])
    names = _primitives(*grads)
    assert names.count("QuantizedMatmul") == 1, names


@requires_metal
def test_the_gradients_evaluated_alone_hold_no_forward(built):
    """Amendment 11 clause 48, and it is the whole of why no subtraction runs.

    Asked only for the gradients, MLX's lazy graph never builds the forward:
    a quantized matmul's gradients hold exactly ONE primitive. So the
    gradients alone ARE the backward, and subtracting a separately timed
    forward from them over-subtracts.
    """
    import mlx.core as mx

    activation = built["activation"]
    call = built[cs.STOCK]
    output = call(activation)
    mx.eval(output)
    cotangent = cs._dense_cotangent(output)
    _value, grads = mx.vjp(call, [activation], [cotangent])
    names = _primitives(*grads)
    assert names == ["QuantizedMatmul"], names


@requires_metal
def test_the_backward_arm_agrees_with_a_standalone_call_to_the_primitive(built):
    """The corroboration clause 48 rests on.

    Two independent constructions of the same quantity: MLX's own vjp, and a
    direct call to the primitive its graph says the vjp dispatches. They agree
    where the registered construction is right and diverge where it is not.
    """
    import mlx.core as mx

    activation = built["activation"]
    call = built[cs.STOCK]
    output = call(activation)
    mx.eval(output)
    cotangent = cs._dense_cotangent(output)
    layer = built["stock_layer"]
    biases = layer.get("biases")

    alone = min(cs._time(
        lambda: mx.vjp(call, [activation], [cotangent])[1],
        warmups=3, rounds=7))
    standalone = min(cs._time(
        lambda: mx.quantized_matmul(
            cotangent, layer["weight"], scales=layer["scales"], biases=biases,
            transpose=False, group_size=layer.group_size, bits=layer.bits),
        warmups=3, rounds=7))
    assert alone == pytest.approx(standalone, rel=0.25), (
        f"the vjp's gradients read {alone:.4f} ms and a standalone call to "
        f"the primitive its graph dispatches read {standalone:.4f} ms")


@requires_metal
def test_the_two_cotangents_agree_and_clause_forty_four_s_factor_is_withdrawn(built):
    """Amendment 11 clause 47.

    Clause 44 reported a factor of about two hundred between a `sum()`-driven
    backward and a dense-cotangent one. Measured the same way on both sides
    they agree closely; the factor was a subtracted quantity set beside an
    unsubtracted one. This asserts the RELATION rather than a figure, because
    the figure moves between runs and the relation does not.
    """
    import mlx.core as mx

    activation = built["activation"]
    call = built[cs.STOCK]
    output = call(activation)
    mx.eval(output)
    cotangent = cs._dense_cotangent(output)

    by_sum = min(cs._time(
        lambda: mx.grad(lambda x: call(x).sum())(activation),
        warmups=3, rounds=7))
    by_dense = min(cs._time(
        lambda: mx.vjp(call, [activation], [cotangent])[1],
        warmups=3, rounds=7))
    assert by_dense == pytest.approx(by_sum, rel=0.25), (
        f"the dense-cotangent backward read {by_dense:.4f} ms and the "
        f"`sum()`-driven one {by_sum:.4f} ms; clause 47 withdraws the claim "
        f"that these differ by a factor of about two hundred")


@requires_metal
def test_the_cotangent_is_built_to_the_output_s_own_shape(built):
    """Clause 44's refusal, which is what makes the registration checkable
    rather than a convention somebody can drift away from."""
    import mlx.core as mx

    output = built[cs.STOCK](built["activation"])
    mx.eval(output)
    cotangent = cs._dense_cotangent(output)
    assert cotangent.shape == output.shape
    assert cotangent.size == output.size > 1


@requires_metal
def test_a_broadcast_cotangent_is_refused(monkeypatch):
    """The shape a `sum()`-driven backward would present at this boundary."""
    import mlx.core as mx

    monkeypatch.setattr(mx.random, "normal",
                        lambda *_a, **_k: mx.array(1.0))
    with pytest.raises(RunInvalid, match="DENSE cotangent"):
        cs._dense_cotangent(mx.zeros((4, 8)))


@requires_metal
def test_the_operands_are_the_registered_shape_at_the_registered_width():
    """A bench measuring something other than what the kill rule reads would
    produce a verdict about a workload nobody registered."""
    import mlx.core as mx

    out_dims, in_dims = rules.SHAPES[SHAPE]
    built = cs._operands(SHAPE, WIDTH, batch=4)
    activation = built["activation"]
    assert activation.shape == (
        4 * rules.WIDTHS[WIDTH]["operation_width"], in_dims)
    for implementation in cs.IMPLEMENTATIONS:
        output = built[implementation](activation)
        mx.eval(output)
        assert output.shape == (activation.shape[0], out_dims)


@requires_metal
def test_one_shape_at_one_width_reduces_through_clause_eight():
    """The whole path, end to end, on real timings and real counts: four arms
    timed, summed within a round, and reduced to what `kill_q` reads."""
    built = cs._operands(SHAPE, WIDTH, batch=1)
    activation = built["activation"]
    measured = {}
    for implementation in cs.IMPLEMENTATIONS:
        import mlx.core as mx

        call = built[implementation]
        output = call(activation)
        mx.eval(output)
        cotangent = cs._dense_cotangent(output)
        measured[f"{implementation}:{cs.FORWARD}"] = cs._time(
            lambda call=call: call(activation), warmups=2, rounds=3)
        measured[f"{implementation}:{cs.BACKWARD}"] = cs._time(
            lambda call=call, cotangent=cotangent: mx.vjp(
                call, [activation], [cotangent])[1], warmups=2, rounds=3)
    entry = cs.shape_at_width(measured, {cs.FORWARD: 36, cs.BACKWARD: 16})
    assert entry.ratio > 0
    assert entry.numerator_high > 0
    assert entry.denominator_low > 0
