"""Parameterised CPU kernels: correct by default, mutable by keyword.

Every kernel here computes the correct result when called with no keyword
arguments. Each keyword is a seam where a realistic transcription fault can be
introduced, which is what makes this module the substrate for both halves of
the project:

- `bench/cpu_ports.py` pins specific keywords to reproduce the 10 faults the
  gpuemu corpus seeds by hand.
- `kernelverify/mutation/` walks the same keywords to synthesise new faults
  that nobody hand-wrote, which is the only way to score an oracle without
  overfitting to a fault list someone published.

Having one source for both means the synthetic fault population provably
contains the real ones, rather than being a parallel invention that merely
resembles them.

Numerical conventions, which match the Triton kernels these mirror:

- Arithmetic in float32, rounded to the input dtype on return.
- `tl.load(..., mask=m, other=V)` is modelled by materialising the padded block
  and filling out-of-range lanes with V, because that padding is load-bearing
  for the tail-masking faults.
- `BLOCK = triton.next_power_of_2(n)` is reproduced exactly.
"""

from __future__ import annotations

import numpy as np

GELU_KAPPA = 0.7978845608028654  # sqrt(2/pi)
GELU_COEFF = 0.044715
RMSNORM_EPS = 1e-5
L2NORM_EPS = 1e-12
LEAKY_RELU_ALPHA = 0.01
SOFTMAX_LLM_BLOCK = 128


def next_pow2(n: int) -> int:
    """Mirror triton.next_power_of_2."""
    p = 1
    while p < n:
        p *= 2
    return p


def _rows(x: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    return np.ascontiguousarray(x).reshape(-1, x.shape[-1]).astype(np.float32), x.shape


def _pad_block(flat: np.ndarray, block: int, fill: float) -> np.ndarray:
    rows, n_cols = flat.shape
    padded = np.full((rows, block), fill, dtype=np.float32)
    padded[:, :n_cols] = flat
    return padded


def _stable_tanh(inner: np.ndarray) -> np.ndarray:
    """The manual stable tanh the GELU kernels use, reproduced as written.

    exp overflows to inf for |inner| > ~44; 2/(inf+1) is 0 and the result is
    the correct saturated ±1, matching the Triton kernel's behaviour, so the
    overflow warning is suppressed rather than the formula changed.
    """
    abs_inner = np.where(inner >= 0.0, inner, -inner)
    sign = np.where(inner >= 0.0, 1.0, -1.0).astype(np.float32)
    with np.errstate(over="ignore"):
        return (sign * (1.0 - 2.0 / (np.exp(2.0 * abs_inner) + 1.0))).astype(np.float32)


# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------
def gelu(inputs, *, leading_scale=0.5, kappa=GELU_KAPPA, coeff=GELU_COEFF, variant="tanh"):
    """Tanh-approximation GELU.

    Seams: `leading_scale` (dropping it is the corpus GELU fault), `kappa` and
    `coeff` (misremembered constants from the published formula), and `variant`
    ("erf" implements the exact-erf GELU, which is a correct function but not
    the one the tanh reference specifies - the two differ by up to ~1e-3, a
    near-tolerance systematic drift rather than a gross error).
    """
    x = inputs["input"]
    xf = x.astype(np.float32)
    if variant == "erf":
        import torch

        erf = torch.erf(torch.from_numpy(np.ascontiguousarray(xf) / np.float32(np.sqrt(2.0))))
        return (leading_scale * xf * (1.0 + erf.numpy())).astype(x.dtype)
    inner = kappa * (xf + coeff * xf * xf * xf)
    return (leading_scale * xf * (1.0 + _stable_tanh(inner))).astype(x.dtype)


def silu(inputs, *, beta=1.0):
    """SiLU / Swish. Seam: `beta`, the Swish family parameter."""
    x = inputs["input"]
    xf = x.astype(np.float32)
    return (xf / (1.0 + np.exp(-beta * xf))).astype(x.dtype)


def leaky_relu(inputs, *, alpha=LEAKY_RELU_ALPHA, boundary_ge=False):
    """LeakyReLU.

    Seams: `alpha`, which differs by framework, and `boundary_ge`, writing the
    comparison as x >= 0 instead of x > 0. The boundary seam should be an
    equivalent mutant: both branches produce exactly 0 at x = 0.
    """
    x = inputs["input"]
    xf = x.astype(np.float32)
    positive = xf >= 0.0 if boundary_ge else xf > 0.0
    return np.where(positive, xf, alpha * xf).astype(x.dtype)


# ---------------------------------------------------------------------------
# Row reductions
# ---------------------------------------------------------------------------
def rmsnorm(inputs, *, use_sqrt=True, eps=RMSNORM_EPS, divisor_offset=0, eps_outside=False):
    """RMSNorm over the last axis.

    Seams: `use_sqrt` (the corpus fault), `eps` (magnitude misremembered),
    `divisor_offset` (off-by-one in the element count of the mean), and
    `eps_outside` (sqrt(m) + eps instead of sqrt(m + eps), a placement fault
    whose signal is tiny at ordinary magnitudes).
    """
    x = inputs["input"]
    flat, shape = _rows(x)
    n_cols = flat.shape[1] + divisor_offset
    mean_sq = (flat * flat).sum(axis=1, keepdims=True) / max(n_cols, 1)
    if use_sqrt:
        denom = np.sqrt(mean_sq) + eps if eps_outside else np.sqrt(mean_sq + eps)
    else:
        denom = mean_sq + eps
    return (flat / denom).reshape(shape).astype(x.dtype)


def l2norm(inputs, *, use_sqrt=True, eps=L2NORM_EPS):
    """L2 normalisation over the last axis. Seam: `use_sqrt` (the corpus fault)."""
    x = inputs["input"]
    flat, shape = _rows(x)
    sum_sq = (flat * flat).sum(axis=1, keepdims=True)
    denom = np.sqrt(sum_sq + eps) if use_sqrt else (sum_sq + eps)
    return (flat / denom).reshape(shape).astype(x.dtype)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------
def softmax(inputs, *, pad_fill=-np.inf, subtract_max=True, exp_base2=False,
            read_past_row=False):
    """Block-tiled softmax over the last axis.

    Seams: `pad_fill`, the value out-of-range lanes load as. The correct value
    is -inf so padding contributes exp(-inf)=0; the corpus fault uses 0.0 so
    every padding lane contributes exp(0 - max) to the denominator.
    `subtract_max` is the numerical-stability step, whose removal is a
    different and well known fault. `exp_base2` computes 2^x instead of e^x,
    the classic tl.exp2 confusion where the log2(e) factor is forgotten.
    `read_past_row` widens the load mask by one lane (col <= n_cols instead of
    col < n_cols), so the first padding lane reads the next row's first
    element instead of the fill value - contamination across rows, which no
    fault in the published corpus exercises. The final row reads 0.0, standing
    in for whatever sits past the buffer.
    """
    x = inputs["input"]
    flat, shape = _rows(x)
    n_cols = flat.shape[1]
    padded = _pad_block(flat, next_pow2(n_cols), pad_fill)
    if read_past_row and padded.shape[1] > n_cols:
        neighbour = np.roll(flat[:, 0], -1)
        neighbour[-1] = 0.0
        padded[:, n_cols] = neighbour
    shifted = padded - padded.max(axis=1, keepdims=True) if subtract_max else padded
    e = np.exp2(shifted) if exp_base2 else np.exp(shifted)
    y = e / e.sum(axis=1, keepdims=True)
    return y[:, :n_cols].reshape(shape).astype(x.dtype)


def softmax_padded(inputs, *, block=SOFTMAX_LLM_BLOCK, mask_tail=True):
    """Softmax whose reduction is padded up to a multiple of `block`.

    Seam: `mask_tail`. When False the padded zeros stay in the denominator,
    which is the corpus's `softmax_llm_buggy` fault.
    """
    x = inputs["input"]
    xf = x.astype(np.float32)
    n_cols = xf.shape[-1]
    padded_cols = ((n_cols + block - 1) // block) * block
    if padded_cols != n_cols:
        fill = -np.inf if mask_tail else 0.0
        pad = np.full(xf.shape[:-1] + (padded_cols - n_cols,), fill, np.float32)
        xp = np.concatenate([xf, pad], axis=-1)
    else:
        xp = xf
    e = np.exp(xp - xp.max(axis=-1, keepdims=True))
    y = e / e.sum(axis=-1, keepdims=True)
    return y[..., :n_cols].astype(x.dtype)


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------
def matmul(inputs, *, accumulate=True, k_offset=0, accum_dtype="float32"):
    """C = A @ B.

    Seams: `accumulate` False reproduces `acc=` instead of `acc+=`, so only the
    final K step survives. `k_offset` truncates the reduction, an off-by-one in
    the loop bound. `accum_dtype` "float16" accumulates at the input precision
    instead of fp32, the standard mixed-precision mistake - its error *grows*
    with K, the inverse of every dilution-class fault, so it is invisible at
    exactly the small shapes where other faults are loudest.
    """
    a, b = inputs["a"], inputs["b"]
    af, bf = a.astype(np.float32), b.astype(np.float32)
    n_k = af.shape[1]
    if not accumulate:
        last = n_k - 1
        return np.outer(af[:, last], bf[last, :]).astype(a.dtype)
    limit = max(1, min(n_k, n_k + k_offset))
    if accum_dtype == "float16":
        acc = np.zeros((af.shape[0], bf.shape[1]), np.float16)
        for k_index in range(limit):
            step = np.outer(af[:, k_index], bf[k_index, :]).astype(np.float16)
            acc = (acc + step).astype(np.float16)
        return acc.astype(a.dtype)
    return (af[:, :limit] @ bf[:limit, :]).astype(a.dtype)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
def attention(inputs, *, scale_power=-0.5, subtract_max=True, post_scale=False,
              scores_dtype="float32"):
    """Scaled dot-product attention.

    Seam: `scale_power`, the exponent applied to D in the score scale. The
    correct value is -0.5, giving 1/sqrt(D). The corpus fault is 0.0, dropping
    the scale entirely; -1.0 is the equally common confusion with 1/D. Keeping
    this as an exponent rather than a constant means the fault stays wrong at
    every D instead of only at the one D where a fixed constant happens to be
    right. `post_scale` applies the scale to the output instead of to the
    scores, a transposition of operation order rather than a wrong constant.
    `scores_dtype` "float16" keeps the score matrix in half precision, the
    precision-canary fault a conditioning-aware tolerance must never absolve.
    """
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    qf, kf, vf = (t.astype(np.float32) for t in (q, k, v))
    dim = qf.shape[1]
    applied = np.float32(float(dim) ** scale_power)
    scores = (qf @ kf.T) * (np.float32(1.0) if post_scale else applied)
    if scores_dtype != "float32":
        scores = scores.astype(scores_dtype).astype(np.float32)
    shifted = scores - scores.max(axis=1, keepdims=True) if subtract_max else scores
    e = np.exp(shifted)
    out = (e / e.sum(axis=1, keepdims=True)) @ vf
    if post_scale:
        out = out * applied
    return out.astype(q.dtype)


def flash_attention(inputs, *, rescale_acc=True, rescale_norm=True, scale_power=-0.5,
                    block_n_cap=32, init_max_zero=False):
    """Flash attention v1: tiled keys and values with an online softmax.

    Seams: `rescale_acc` False drops `acc *= alpha` on a max update, which is
    the corpus fault. `rescale_norm` False drops the same rescale on the
    normaliser, a symmetric fault nobody has published. `scale_power` mirrors
    the plain-attention seam. `block_n_cap` changes the tile width, which moves
    where any tiling-dependent fault starts to express. `init_max_zero` starts
    the running max at 0 instead of -inf; the online softmax stays exact in
    real arithmetic under any initial max, so this fault is purely numerical -
    it only expresses when a whole row's scores sit far enough below zero that
    exp underflows, giving 0/0.
    """
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    qf, kf, vf = (t.astype(np.float32) for t in (q, k, v))
    n_keys, dim = kf.shape
    n_queries = qf.shape[0]
    block_n = max(8, next_pow2(min(n_keys, block_n_cap)))
    applied = np.float32(float(dim) ** scale_power)

    initial = 0.0 if init_max_zero else -np.inf
    running_max = np.full(n_queries, initial, dtype=np.float32)
    normaliser = np.zeros(n_queries, dtype=np.float32)
    acc = np.zeros((n_queries, dim), dtype=np.float32)

    for start in range(0, n_keys, block_n):
        idx = start + np.arange(block_n)
        in_range = idx < n_keys
        k_tile = np.zeros((block_n, dim), np.float32)
        v_tile = np.zeros((block_n, dim), np.float32)
        k_tile[in_range] = kf[idx[in_range]]
        v_tile[in_range] = vf[idx[in_range]]

        scores = (qf @ k_tile.T) * applied
        scores = np.where(in_range[None, :], scores, -np.inf)

        new_max = np.maximum(running_max, scores.max(axis=1))
        alpha = np.exp(running_max - new_max)
        probs = np.exp(scores - new_max[:, None])
        normaliser = normaliser * alpha + probs.sum(axis=1) if rescale_norm else normaliser + probs.sum(axis=1)
        contribution = probs @ v_tile
        acc = acc * alpha[:, None] + contribution if rescale_acc else acc + contribution
        running_max = new_max

    return (acc / normaliser[:, None]).astype(q.dtype)


# Operator name as used by the corpus schemas -> parameterised implementation.
KERNELS = {
    "gelu": gelu,
    "silu": silu,
    "leaky_relu": leaky_relu,
    "rmsnorm": rmsnorm,
    "l2norm": l2norm,
    "softmax": softmax,
    "softmax_padded": softmax_padded,
    "matmul": matmul,
    "attention": attention,
    "flash_attention": flash_attention,
}
