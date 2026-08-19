"""Can a Metal kernel we wrote carry gradients through mlx-lm's real path?

The whole training product rests on one mechanism. A raw Metal kernel has no
gradient at all: MLX refuses outright, by name, and that refusal is pinned
below because it is the reason the wrapper is mandatory rather than tidy.
The route that does work is `mx.custom_function` with a `.vjp` registered on
it, which is MLX's documented mechanism and the one mlx-lm's own losses use.

Working in isolation is not enough. mlx-lm's trainer does three more things
to whatever sits inside a layer, and each is a place a hand-written gradient
can quietly stop being called:

  - `nn.value_and_grad` over the module, which is how the loss is taken;
  - `mx.checkpoint` wrapped onto the layer type itself, which recomputes the
    forward during the backward pass, so the vjp runs against a re-executed
    kernel rather than a retained one;
  - `mx.compile` around the whole step, which rewrites the graph.

So the last test drives all three at once, in the same order and the same
shape `mlx_lm/tuner/trainer.py` uses, and compares against the identical
computation in stock MLX. A version of MLX that breaks any of these breaks
the product, and this is where that would show up.

The kernel here is a deliberately naive matmul. It is not fast and is not
meant to be; it exists because a matmul's vjp is the real gradient shape
(one matmul per primal) rather than an elementwise special case.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from conftest import requires_metal

SOURCE = """
    uint gid = thread_position_in_grid.x;
    uint n = b_shape[1];
    uint k = a_shape[1];
    float acc = 0.0f;
    for (uint i = 0; i < k; ++i) {
        acc += float(a[(gid / n) * k + i]) * float(b[i * n + gid % n]);
    }
    out[gid] = acc;
"""


def _raw(a, b):
    """The kernel with no gradient attached, dispatched directly."""
    kernel = mx.fast.metal_kernel(name="spike_naive_matmul",
                                  input_names=["a", "b"], output_names=["out"],
                                  source=SOURCE)
    m, n = a.shape[0], b.shape[1]
    return kernel(inputs=[a, b], output_shapes=[(m, n)],
                  output_dtypes=[mx.float32], grid=(m * n, 1, 1),
                  threadgroup=(min(m * n, 256), 1, 1))[0]


@mx.custom_function
def matmul(a, b):
    return _raw(a, b)


@matmul.vjp
def _(primals, cotangent, _output):
    a, b = primals
    return _raw(cotangent, b.T), _raw(a.T, cotangent)


@pytest.fixture()
def operands():
    mx.random.seed(0)
    return mx.random.normal((8, 16)), mx.random.normal((16, 4))


def _grad_of_sum(fn, a, b):
    return mx.grad(lambda x: mx.sum(fn(x, b)))(a)


# ---------------------------------------------------------------------------
# The mechanism
# ---------------------------------------------------------------------------
@requires_metal
def test_the_forward_is_the_matmul_it_claims_to_be(operands):
    """Nothing below means anything if the forward is wrong, so it is checked
    against MLX's own matmul first."""
    a, b = operands
    assert float(mx.max(mx.abs(_raw(a, b) - (a @ b)))) < 1e-4


@requires_metal
def test_a_raw_metal_kernel_has_no_gradient_and_says_so(operands):
    """The refusal that makes the wrapper mandatory, pinned by its own words.
    If a future MLX implements this, the test goes red and the wrapper becomes
    a choice rather than a requirement, which is worth being told about."""
    a, b = operands
    with pytest.raises(ValueError, match="Not implemented for CustomKernel"):
        mx.eval(_grad_of_sum(_raw, a, b))


@requires_metal
def test_the_registered_vjp_produces_mlxs_own_gradient(operands):
    a, b = operands
    ours = _grad_of_sum(matmul, a, b)
    stock = _grad_of_sum(lambda x, y: x @ y, a, b)
    mx.eval(ours, stock)
    assert float(mx.max(mx.abs(ours - stock))) < 1e-4


@requires_metal
def test_the_gradient_survives_compile(operands):
    a, b = operands
    ours = mx.compile(mx.grad(lambda x: mx.sum(matmul(x, b))))(a)
    stock = _grad_of_sum(lambda x, y: x @ y, a, b)
    mx.eval(ours, stock)
    assert float(mx.max(mx.abs(ours - stock))) < 1e-4


@requires_metal
def test_the_gradient_survives_checkpointing(operands):
    """Checkpointing re-executes the forward during the backward pass, so this
    is the test that the vjp works against a recomputed kernel."""
    a, b = operands
    ours = mx.grad(mx.checkpoint(lambda x: mx.sum(matmul(x, b))))(a)
    stock = _grad_of_sum(lambda x, y: x @ y, a, b)
    mx.eval(ours, stock)
    assert float(mx.max(mx.abs(ours - stock))) < 1e-4


# ---------------------------------------------------------------------------
# The composition mlx-lm actually builds
# ---------------------------------------------------------------------------
class _Ours(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.w = weight

    def __call__(self, x):
        return matmul(x, self.w)


class _Stock(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.w = weight

    def __call__(self, x):
        return x @ self.w


def _checkpoint_the_type(layer):
    """`grad_checkpoint` from mlx_lm/tuner/trainer.py, reproduced: it replaces
    __call__ on the TYPE, so every instance of that layer is checkpointed."""
    original = type(layer).__call__

    def checkpointed(model, *args, **kwargs):
        def inner(params, *inner_args, **inner_kwargs):
            model.update(params)
            return original(model, *inner_args, **inner_kwargs)

        return mx.checkpoint(inner)(model.trainable_parameters(),
                                    *args, **kwargs)

    type(layer).__call__ = checkpointed
    return original


@requires_metal
def test_the_whole_trainer_composition_reaches_our_vjp(operands):
    """value_and_grad over the module, checkpointed at the type, compiled as
    a step: all three at once, which is the arrangement the trainer builds and
    the only one that proves the gradient is really ours end to end."""
    a, b = operands
    ours, stock = _Ours(b), _Stock(b)
    original = _checkpoint_the_type(ours)
    try:
        value_and_grad = nn.value_and_grad(ours, lambda m, x: mx.sum(m(x)))

        @mx.compile
        def step(x):
            return value_and_grad(ours, x)

        value, grad = step(a)
        ref_value, ref_grad = nn.value_and_grad(
            stock, lambda m, x: mx.sum(m(x)))(stock, a)
        mx.eval(value, grad, ref_value, ref_grad)
    finally:
        type(ours).__call__ = original

    assert abs(float(value) - float(ref_value)) < 1e-3
    assert float(mx.max(mx.abs(grad["w"] - ref_grad["w"]))) < 1e-3
