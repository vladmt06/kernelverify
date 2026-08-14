"""The admissible-implementation contract, and a generator for its population.

Why this module exists
----------------------
ADR 0004 ships `tolerance = max(base_tol, K * ensemble_floor)` with K = 1.5,
where the floor is the worst deviation of three hand-written correct
implementations. K itself was chosen as the smallest value on a grid that
produced no false positive over a handful of held-out variants. That is the
same weakness the project already refuses to accept for the fault catalogue: a
number fitted to a small self-authored population, with nothing stating what it
would have to be to be safe.

Anchoring K means writing down, first, exactly which implementations the
verifier promises not to flag - the contract - and then measuring the worst
deviation any implementation in that declared class can commit. K stops being a
fitted constant and becomes a measured property of a stated promise.

The contract
------------
A candidate kernel is ADMISSIBLE, meaning the verifier undertakes never to flag
it, if and only if it computes the operator's specified formula under the
following freedoms, and no others.

C1. Working precision floor.
    Every intermediate is computed in IEEE binary32 or wider. Inputs are
    upcast from the storage dtype and the result is rounded to the storage
    dtype exactly once, at the end. Any narrower intermediate is OUT of
    contract. This clause is what keeps a precision fault a fault: the fp16
    score matrix of `attention[scores_dtype=float16]` is out of contract by
    construction, so no amount of measured tolerance can legitimise it.

C2. Evaluation-order freedom.
    The specified expression may be evaluated in any association order, and
    any reduction (sum, dot product, max) in any order or permutation of its
    summands: sequential, pairwise, blocked at any width, magnitude-sorted, or
    compensated. Reassociation of a non-reduction expression is included, so
    `k*(x + c*x**3)` and `k*x + (k*c)*x**3` are both admissible.

C3. Tiling freedom.
    Streaming and tiled formulations that are exact in exact arithmetic are
    admissible at any tile width, including the online-softmax (flash)
    formulation with its rescale, and padded-block formulations whose padding
    lanes are neutral for the reduction.

C4. Stability requirement.
    A reduction of exponentials must be shifted by a maximum taken over its
    own summands, either the whole-row maximum or the running maximum of an
    online formulation. An unshifted exponential sum is OUT of contract even
    though it is exact in real arithmetic, because it overflows: admitting it
    would make the admissible class unbounded in error and the tolerance
    derived from it vacuous.

C5. The formula is the specified one.
    The operator is the one the corpus fp64 reference computes, not any
    mathematically defensible relative of it. The exact-erf GELU is a correct
    function and an inadmissible implementation of a tanh-approximation GELU.

C6. Specification constants are fixed.
    Epsilons, scale exponents and coefficients belong to the specification,
    not to the implementer.

What is deliberately not in the contract
----------------------------------------
Narrower-than-fp32 intermediates (C1), unshifted exponential sums (C4), and
alternative formulas (C5) are all excluded. Each exclusion is a promise the
verifier is allowed to break: it may flag such a kernel. That is the point -
a contract wide enough to admit everything would force a tolerance that
absolves everything.

Using this module
-----------------
`contract_implementations(op, ...)` returns labelled callables, each a correct
implementation of `op` drawn from the class above. `bench/calibrate_k.py`
measures the worst deviation over that population, per case, against the same
fp64 references the verifier uses, and reports the K that class demands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from kernelverify.reference.kernels import (
    L2NORM_EPS,
    RMSNORM_EPS,
    next_pow2,
)

# Bumping this invalidates any cached calibration, the way the catalogue
# fingerprint invalidates cached verdicts.
CONTRACT_VERSION = "contract-v2"

# Tile widths a legitimate kernel may choose (C3). 8 is the smallest width the
# corpus kernels ever emit; 256 spans the largest reduction in the schemas.
TILE_WIDTHS = (8, 16, 32, 64, 128, 256)

# Every corpus operator the contract is stated for.
OPERATORS = (
    "gelu_triton",
    "silu_triton",
    "leaky_relu_triton",
    "softmax_triton",
    "rmsnorm_triton",
    "l2norm_triton",
    "matmul_triton",
    "attention_triton",
    "flash_attention_triton",
)


# ---------------------------------------------------------------------------
# C2: reduction orders
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Order:
    """One admissible way to evaluate a reduction, as a labelled sum."""

    label: str
    fn: Callable  # (fp32 array, axis) -> fp32 array with `axis` removed


def _to_front(a: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(a, axis, 0)


def _sum_pairwise(a, axis):
    return a.sum(axis=axis, dtype=np.float32)


def _sum_sequential(a, axis):
    return np.add.accumulate(a, axis=axis).take(-1, axis=axis)


def _sum_reversed(a, axis):
    return _sum_sequential(np.flip(a, axis=axis), axis)


def _sum_by_magnitude(a, axis, descending):
    key = -np.abs(a) if descending else np.abs(a)
    order = np.argsort(key, axis=axis, kind="stable")
    return _sum_sequential(np.take_along_axis(a, order, axis=axis), axis)


def _sum_blocked(a, axis, width):
    """Pairwise within tiles of `width`, sequential across tiles (C3)."""
    moved = _to_front(a, axis)
    n = moved.shape[0]
    if width >= n:
        return _sum_pairwise(a, axis)
    pad = (-n) % width
    if pad:
        moved = np.concatenate(
            [moved, np.zeros((pad,) + moved.shape[1:], np.float32)], axis=0
        )
    tiles = moved.reshape(-1, width, *moved.shape[1:]).sum(axis=1, dtype=np.float32)
    return _sum_sequential(tiles, 0)


def _sum_kahan(a, axis):
    """Compensated summation: admissible, and more accurate than the reference."""
    moved = _to_front(a, axis)
    total = np.zeros(moved.shape[1:], np.float32)
    carry = np.zeros(moved.shape[1:], np.float32)
    for term in moved:
        adjusted = term - carry
        provisional = total + adjusted
        carry = (provisional - total) - adjusted
        total = provisional
    return total


def _sum_wide(a, axis):
    """Accumulation in binary64: 'or wider' in C1, so admissible."""
    return a.astype(np.float64).sum(axis=axis).astype(np.float32)


def _sum_permuted(a, axis, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(a.shape[axis])
    return _sum_sequential(np.take(a, order, axis=axis), axis)


def reduction_orders(n_random: int, seed: int) -> list[Order]:
    """Every admissible summation order this experiment samples.

    The deterministic members are chosen to bracket the class rather than to
    represent it uniformly: magnitude-descending sequential is the classic
    worst order for a sum of like-signed terms, Kahan and binary64 are the
    classic best, and the tile widths are the orders a real kernel actually
    emits. The random permutations sample the interior, which is where an
    order nobody thought to hand-write would sit.
    """
    orders = [
        Order("pairwise", _sum_pairwise),
        Order("sequential", _sum_sequential),
        Order("reversed", _sum_reversed),
        Order("magnitude-desc", lambda a, ax: _sum_by_magnitude(a, ax, True)),
        Order("magnitude-asc", lambda a, ax: _sum_by_magnitude(a, ax, False)),
        Order("kahan", _sum_kahan),
        Order("binary64", _sum_wide),
    ]
    orders += [
        Order(f"blocked-{w}", (lambda w: lambda a, ax: _sum_blocked(a, ax, w))(w))
        for w in TILE_WIDTHS
    ]
    orders += [
        Order(f"perm-{i}", (lambda s: lambda a, ax: _sum_permuted(a, ax, s))(seed + i))
        for i in range(n_random)
    ]
    return orders


# ---------------------------------------------------------------------------
# Per-family admissible implementations
#
# Each builder returns callables with the kernel signature used everywhere
# else in the project: inputs dict in storage dtype -> output in storage dtype.
# ---------------------------------------------------------------------------
def _f32(a):
    return a.astype(np.float32)


def _rows(x):
    return np.ascontiguousarray(x).reshape(-1, x.shape[-1]).astype(np.float32), x.shape


# -- elementwise -------------------------------------------------------------
# No reduction exists, so the only freedom is C2 reassociation of the specified
# expression plus C1's licence to compute wider than binary32.
def _gelu_variants() -> list[tuple[str, Callable]]:
    kappa = np.float32(0.7978845608028654)
    coeff = np.float32(0.044715)

    def horner(inputs):  # inner = k*(x + c*x^3), grouped as k*x*(1 + c*x^2)
        x = inputs["input"]
        xf = _f32(x)
        inner = kappa * xf * (np.float32(1.0) + coeff * xf * xf)
        return (np.float32(0.5) * xf * (np.float32(1.0) + np.tanh(inner))).astype(x.dtype)

    def distributed(inputs):  # inner = k*x + (k*c)*x^3
        x = inputs["input"]
        xf = _f32(x)
        inner = kappa * xf + (kappa * coeff) * xf * xf * xf
        return (np.float32(0.5) * xf * (np.float32(1.0) + np.tanh(inner))).astype(x.dtype)

    def split_half(inputs):  # 0.5*x + 0.5*x*tanh(inner)
        x = inputs["input"]
        xf = _f32(x)
        inner = kappa * (xf + coeff * xf * xf * xf)
        half = np.float32(0.5) * xf
        return (half + half * np.tanh(inner)).astype(x.dtype)

    def sigmoid_form(inputs):  # 0.5*(1+tanh(z)) is exactly sigmoid(2z)
        x = inputs["input"]
        xf = _f32(x)
        inner = kappa * (xf + coeff * xf * xf * xf)
        with np.errstate(over="ignore"):
            return (xf / (np.float32(1.0) + np.exp(np.float32(-2.0) * inner))).astype(x.dtype)

    def wide(inputs):
        x = inputs["input"]
        xd = x.astype(np.float64)
        inner = np.float64(0.7978845608028654) * (xd + np.float64(0.044715) * xd ** 3)
        return (0.5 * xd * (1.0 + np.tanh(inner))).astype(x.dtype)

    return [("horner", horner), ("distributed", distributed),
            ("split-half", split_half), ("sigmoid-form", sigmoid_form),
            ("binary64", wide)]


def _silu_variants() -> list[tuple[str, Callable]]:
    def tanh_form(inputs):  # x*sigmoid(x) == 0.5*x*(1+tanh(x/2)), exactly
        x = inputs["input"]
        xf = _f32(x)
        return (np.float32(0.5) * xf * (np.float32(1.0) + np.tanh(np.float32(0.5) * xf))).astype(x.dtype)

    def reciprocal_form(inputs):
        x = inputs["input"]
        xf = _f32(x)
        return (xf * (np.float32(1.0) / (np.float32(1.0) + np.exp(-xf)))).astype(x.dtype)

    def wide(inputs):
        x = inputs["input"]
        xd = x.astype(np.float64)
        return (xd / (1.0 + np.exp(-xd))).astype(x.dtype)

    return [("tanh-form", tanh_form), ("reciprocal-form", reciprocal_form),
            ("binary64", wide)]


def _leaky_relu_variants() -> list[tuple[str, Callable]]:
    alpha = np.float32(0.01)

    def multiplied(inputs):
        x = inputs["input"]
        xf = _f32(x)
        return (xf * np.where(xf > 0.0, np.float32(1.0), alpha)).astype(x.dtype)

    def maximum_form(inputs):  # max(x, alpha*x) for 0 < alpha < 1
        x = inputs["input"]
        xf = _f32(x)
        return np.maximum(xf, alpha * xf).astype(x.dtype)

    def wide(inputs):
        x = inputs["input"]
        xd = x.astype(np.float64)
        return np.where(xd > 0.0, xd, 0.01 * xd).astype(x.dtype)

    return [("multiplied", multiplied), ("maximum-form", maximum_form),
            ("binary64", wide)]


# -- row reductions ----------------------------------------------------------
def _rmsnorm_variant(order: Order) -> Callable:
    def impl(inputs):
        x = inputs["input"]
        flat, shape = _rows(x)
        mean_sq = order.fn(flat * flat, 1) / np.float32(flat.shape[1])
        return (flat / np.sqrt(mean_sq + np.float32(RMSNORM_EPS))[:, None]).reshape(
            shape
        ).astype(x.dtype)

    return impl


def _l2norm_variant(order: Order) -> Callable:
    def impl(inputs):
        x = inputs["input"]
        flat, shape = _rows(x)
        norm = np.sqrt(order.fn(flat * flat, 1) + np.float32(L2NORM_EPS))
        return (flat / norm[:, None]).reshape(shape).astype(x.dtype)

    return impl


def _softmax_variant(order: Order, pad_to_pow2: bool) -> Callable:
    """Shifted softmax whose denominator is summed in `order`.

    `pad_to_pow2` reproduces the block-padded load the Triton kernels emit:
    the tail lanes hold -inf, contribute exactly zero, and change only the
    shape of the summation tree, which is admissible under C3.
    """

    def impl(inputs):
        x = inputs["input"]
        flat, shape = _rows(x)
        n_cols = flat.shape[1]
        block = next_pow2(n_cols) if pad_to_pow2 else n_cols
        if block != n_cols:
            padded = np.full((flat.shape[0], block), -np.inf, np.float32)
            padded[:, :n_cols] = flat
        else:
            padded = flat
        shifted = padded - padded.max(axis=1, keepdims=True)
        e = np.exp(shifted)
        y = e[:, :n_cols] / order.fn(e, 1)[:, None]
        return y.reshape(shape).astype(x.dtype)

    return impl


def _softmax_online(block: int) -> Callable:
    """The online (flash-style) softmax: running max, rescaled normaliser (C3/C4)."""

    def impl(inputs):
        x = inputs["input"]
        flat, shape = _rows(x)
        n_rows, n_cols = flat.shape
        running_max = np.full(n_rows, -np.inf, np.float32)
        normaliser = np.zeros(n_rows, np.float32)
        for start in range(0, n_cols, block):
            tile = flat[:, start:start + block]
            new_max = np.maximum(running_max, tile.max(axis=1))
            alpha = np.exp(running_max - new_max)
            normaliser = normaliser * alpha + np.exp(tile - new_max[:, None]).sum(
                axis=1, dtype=np.float32
            )
            running_max = new_max
        y = np.exp(flat - running_max[:, None]) / normaliser[:, None]
        return y.reshape(shape).astype(x.dtype)

    return impl


# -- matmul ------------------------------------------------------------------
def _matmul_variant(order: Order) -> Callable:
    """C = A @ B with the K reduction evaluated in `order`.

    The product tensor is built in slabs over N so peak memory stays at
    M*K*64 floats rather than M*K*N.
    """

    def impl(inputs):
        a, b = inputs["a"], inputs["b"]
        af, bf = _f32(a), _f32(b)
        m, n = af.shape[0], bf.shape[1]
        out = np.empty((m, n), np.float32)
        for n0 in range(0, n, 64):
            slab = af[:, :, None] * bf[None, :, n0:n0 + 64]
            out[:, n0:n0 + 64] = order.fn(slab, 1)
        return out.astype(a.dtype)

    return impl


def _matmul_blas(inputs):
    """The vendor BLAS kernel: blocked and vectorised in some unspecified order."""
    a, b = inputs["a"], inputs["b"]
    return (_f32(a) @ _f32(b)).astype(a.dtype)


def _matmul_wide(inputs):
    a, b = inputs["a"], inputs["b"]
    return (a.astype(np.float64) @ b.astype(np.float64)).astype(a.dtype)


# -- attention ---------------------------------------------------------------
def _attention_variant(order: Order, key_perm_seed: int | None = None) -> Callable:
    """Attention with the score dot product, denominator and PV sum in `order`.

    `key_perm_seed` additionally permutes the key/value rows, which reorders
    both the softmax denominator and the PV accumulation, and is admissible
    because the output is a sum over keys.
    """

    def impl(inputs):
        q, k, v = inputs["q"], inputs["k"], inputs["v"]
        qf, kf, vf = _f32(q), _f32(k), _f32(v)
        if key_perm_seed is not None:
            perm = np.random.default_rng(key_perm_seed).permutation(kf.shape[0])
            kf, vf = kf[perm], vf[perm]
        dim = qf.shape[1]
        scale = np.float32(float(dim) ** -0.5)
        scores = order.fn(qf[:, None, :] * kf[None, :, :], 2) * scale
        shifted = scores - scores.max(axis=1, keepdims=True)
        e = np.exp(shifted)
        p = e / order.fn(e, 1)[:, None]
        out = order.fn(p[:, :, None] * vf[None, :, :], 1)
        return out.astype(q.dtype)

    return impl


def _attention_wide(inputs):
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    qd, kd, vd = (t.astype(np.float64) for t in (q, k, v))
    scores = (qd @ kd.T) / np.sqrt(qd.shape[1])
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return ((e / e.sum(axis=1, keepdims=True)) @ vd).astype(q.dtype)


def _attention_blas(inputs):
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    qf, kf, vf = _f32(q), _f32(k), _f32(v)
    scores = (qf @ kf.T) * np.float32(float(qf.shape[1]) ** -0.5)
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return ((e / e.sum(axis=1, keepdims=True)) @ vf).astype(q.dtype)


def _flash_variant(block_n_cap: int) -> Callable:
    from kernelverify.reference.kernels import flash_attention

    def impl(inputs):
        return flash_attention(inputs, block_n_cap=block_n_cap)

    return impl


# ---------------------------------------------------------------------------
# The population, per corpus operator
# ---------------------------------------------------------------------------
def _prioritise(members: list[tuple[str, Callable]], leads: list[str]):
    """Order the population: named leads, then the random orders, then the rest.

    The random permutations come before the remaining deterministic members
    because the remaining deterministic members cannot bound the class: Kahan
    and binary64 are strictly more accurate than the reference, ascending
    magnitude is the best order rather than a bad one, and the leftover tile
    widths sit between pairwise and sequential, both of which the leads
    already carry. A random permutation is the only member left that can be
    worse than everything above it.
    """
    index = {label: fn for label, fn in members}
    head = [(label, index[label]) for label in leads]
    taken = set(leads)
    random_members = [(l, f) for l, f in members
                      if l not in taken and ("perm-" in l)]
    rest = [(l, f) for l, f in members
            if l not in taken and ("perm-" not in l)]
    return head + random_members + rest


def _named(label: str, fn: Callable) -> Callable:
    """Give a member a `__name__` that says which member it is.

    Most members are closures over a reduction order, so they all inherit the
    name of the factory that built them - eight distinct attention members all
    called `impl`. Anything that fingerprints an ensemble by reflection then
    sees one name where there are eight, and swapping a tile width or a
    permutation seed leaves the fingerprint byte-identical. That is precisely
    the failure the verdict-cache tag exists to prevent, so the members carry
    their own identity rather than relying on the caller to know it.

    The wrapper is a new object every call, so nothing shared - including the
    reference kernels themselves - has its name rewritten underneath it.
    """
    def member(inputs):
        return fn(inputs)

    member.__name__ = label
    member.__qualname__ = label
    return member


def contract_implementations(op: str, *, n_random: int = 6,
                             seed: int = 20260814) -> list[tuple[str, Callable]]:
    """Admissible implementations of `op`, in ensemble-priority order, named."""
    return [(label, _named(label, fn))
            for label, fn in _population(op, n_random=n_random, seed=seed)]


def _population(op: str, *, n_random: int, seed: int) -> list[tuple[str, Callable]]:
    """The population itself, before the members are given their names.

    Every member is a correct implementation of the specified operator; the
    only thing that varies is a freedom the contract explicitly grants. The
    list is deterministic given (op, n_random, seed) so a calibration run is
    reproducible.

    The ORDER is load-bearing, because `contract_ensemble` ships a prefix of
    this list as the verifier's floor and a prefix is only as good as what it
    puts first. The priority is structural diversity before order diversity:

    1. the reference implementation, so a one-member ensemble is the status
       quo ante;
    2. a structurally different formulation of the same operator - tiled
       against untiled, online against two-pass - because a different
       algorithm shape moves the error further than a different summation
       order within one shape;
    3. tile-width extremes, since the width is a free parameter of the
       shipped algorithm and the corpus kernels vary it;
    4. the deterministic adversarial orders, sequential and magnitude-sorted;
    5. seeded random permutations, which are what actually bound the class
       on the reduction-heavy operators and which no fixed list reaches.
    """
    orders = reduction_orders(n_random, seed)
    by_label = {o.label: o for o in orders}

    if op == "gelu_triton":
        from kernelverify.reference.kernels import gelu
        return [("reference", gelu)] + _gelu_variants()
    if op == "silu_triton":
        from kernelverify.reference.kernels import silu
        return [("reference", silu)] + _silu_variants()
    if op == "leaky_relu_triton":
        from kernelverify.reference.kernels import leaky_relu
        return [("reference", leaky_relu)] + _leaky_relu_variants()

    if op in ("rmsnorm_triton", "l2norm_triton"):
        build = _rmsnorm_variant if op == "rmsnorm_triton" else _l2norm_variant
        members = [(f"sum:{o.label}", build(o)) for o in orders]
        return _prioritise(members, ["sum:pairwise", "sum:sequential",
                                   "sum:magnitude-desc", "sum:blocked-32"])

    if op == "softmax_triton":
        members = [(f"pow2-pad,sum:{o.label}", _softmax_variant(o, True)) for o in orders]
        members += [(f"unpadded,sum:{o.label}", _softmax_variant(o, False)) for o in orders]
        members += [(f"online-{w}", _softmax_online(w)) for w in TILE_WIDTHS]
        return _prioritise(members, ["pow2-pad,sum:pairwise", "online-32",
                                   "unpadded,sum:pairwise",
                                   "pow2-pad,sum:sequential",
                                   "pow2-pad,sum:magnitude-desc", "online-8",
                                   "online-256"])

    if op == "matmul_triton":
        members = [("blas", _matmul_blas), ("binary64", _matmul_wide)]
        members += [(f"k-sum:{o.label}", _matmul_variant(o)) for o in orders]
        return _prioritise(members, ["blas", "k-sum:sequential",
                                   "k-sum:magnitude-desc", "k-sum:blocked-32"])

    if op in ("attention_triton", "flash_attention_triton"):
        members = [("blas", _attention_blas), ("binary64", _attention_wide)]
        members += [(f"sum:{o.label}", _attention_variant(o)) for o in orders]
        members += [
            (f"keyperm-{i}", _attention_variant(by_label["sequential"],
                                                key_perm_seed=seed + 500 + i))
            for i in range(n_random)
        ]
        members += [(f"flash-{w}", _flash_variant(w)) for w in TILE_WIDTHS]
        # The flash op's reference is tiled, the plain op's is not; each leads
        # with its own and takes the other as its first structural neighbour.
        lead = ["flash-32", "blas"] if op == "flash_attention_triton" else ["blas", "flash-32"]
        return _prioritise(members, lead + ["flash-8", "flash-256",
                                          "sum:sequential", "sum:magnitude-desc"])

    raise KeyError(f"no contract population defined for operator {op}")


def contract_ensemble(op: str, budget: int, *, seed: int = 20260814
                      ) -> list[tuple[str, Callable]]:
    """The first `budget` admissible implementations, for use as a tolerance floor.

    A shipped verifier cannot afford the whole population per case, so it
    takes a prefix. `bench/calibrate_k.py` measures how much of the class a
    prefix of each size actually covers, which is what decides the budget.
    """
    population = contract_implementations(op, n_random=max(budget, 8), seed=seed)
    return population[:budget]
