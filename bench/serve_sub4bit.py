"""Batched-decode serving on sub-4-bit weights: the four-arm harness.

Pre-registered in docs/research/2026-08-15-sub4bit-serve-findings.md
(tickets T4/T5/T8 of the 2026-08-15 pivot plan, rulings D3.1/D3.2/D3.3/
D3.6/D5); sections 1-7 of that doc are written before any measurement,
and no timing runs outside a coordinated quiet window (T8).

The kernel under test is kernelverify.pack.wide_qmv, routed at decode
time under the model's quantized linear projections. The interception is
scoped to the MODEL'S OWN layer objects: each nn.QuantizedLinear leaf in
the installed model tree is swapped for a wrapper holding the original.
Nothing global is touched - no class attribute, no module function, no
MLX namespace edit - so oracle calls (direct mx.quantized_matmul) and any
second model instance can never route through the kernel under test;
tests/test_serve_sub4bit.py holds that proof.

Arms (D3.2, D3.6):
  1 ours     3-bit model, patch installed, routing per should_dispatch
  2 stock3   3-bit model, no patch (the incumbent; primary ratio is 1/2)
  3 stock4   4-bit model, no patch (the composed-capability comparator)
  4 control  3-bit model, patch installed, routing FORCED to stock:
             eligibility is evaluated identically and then discarded, so
             arm4 - arm2 is the patch's host cost with no kernel in it.
             The 2026-08-15 kv_attention spike died from exactly this
             cost going unmeasured; this arm exists so it never can again.

Routing is delegated entirely to should_dispatch, so the harness inherits
the pack's boundary and never encodes one of its own. Since 2026-08-15 that
boundary is per shape and per bit width (kernelverify/pack/routed_windows.py,
ADR 0015), so every routing question here is asked with the whole cell -
tile width, bits, d_out, d_in - and the answer can differ between two
projections in the same layer. Prefill-shaped calls (L > 1) fall back to
stock by design and are whitelisted by reason.

Modes:
  --smoke   wiring correctness, no timing: pins verified, dispatch counts
            exact (the per-shape sum over the wrapped sites, which is 0 at
            every B the table routes nowhere), zero hard fallbacks,
            forced-stock arm bit-identical.
  --mde     the pre-registered arithmetic: interception cost via arm 4 at
            B=1, per-op times at the true projection shapes, arm-2
            per-token times and noise floors. Quiet window only.
  --ab      the pre-registered four-arm grid over B. Quiet window only,
            and only after the coordinator's go (T8).
  --ppl     the T5 quality pair: both artifacts, stock, one corpus,
            one number pair. Not a timing; never concurrent with one.

Exit vocabulary, all of it memory_guard's and none of it this module's:
0 attested, 1 measured-and-stopped, 3/4/5 the memory refusals (budget, lock
held, machine out of room), 8 a permanent precondition (pins, pinned zone,
corpus digest) that retrying cannot heal, 9 the machine not being quiet,
which is transient and is exactly what waiting fixes. 2 is left to the
interpreter, where argparse and a failed import already put it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import machine_state  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from machine_state import MeasurementLock  # noqa: E402

# The refusal vocabulary, single-sourced: this harness numbers nothing itself,
# because per-harness numbering is how two "protected" runs collapsed the
# machine at 03:29 on 2026-08-15 (memory_guard's docstring has the dissection).
from memory_guard import (  # noqa: E402
    EXIT_BUDGET_REFUSAL,
    EXIT_LOCK_HELD,
    EXIT_LOW_MEMORY,
    EXIT_NOT_IDLE,
    EXIT_PRECONDITION,
    BudgetExceeded,
    BudgetGuard,
    LowMemoryRefusal,
    budget_gb_arg,
    require_available_memory,
)

from kernelverify.pack.wide_qmv import (  # noqa: E402
    SUPPORTED_BITS,
    build,
    launch_config,
    should_dispatch,
)

MODELS_ROOT = Path("/Users/vlad/kernelverify/bench/.models")
MODEL_3BIT = "qwen3-4b-3bit-g64"
MODEL_4BIT = "qwen3-4b-4bit-g64"
CORPUS = Path("/Users/vlad/kernelverify/bench/.corpus/ppl.txt")
# The corpus is registered by CONTENT, not by path (findings doc, sections 7
# and 8): the file is gitignored and lives per worktree, so only the digest
# can say that two runs scored the same text.
PPL_CORPUS_SHA256 = ("173c87a53759e0201f33e0ccf978e510"
                     "c2042d7f2cb78229d9a50d79b9e7dd08")
GROUP_SIZE = 64

# One name for the one machine-wide lock this harness takes.
LOCK_NAME = "serve_sub4bit"
# The same number the serving calibration budgets at, and deliberately its own
# literal rather than an import: the two harnesses describe different
# allocations and a shared symbol would tie one's budget to the other's
# measurements.
AB_BUDGET_GB = 24.0

# Pre-registered measurement parameters (findings doc, section 3).
PROMPT_T = 512
GEN_TOKENS = 128
B_GRID = [1, 4, 5, 6, 8, 11, 12, 16]
# The bit width of both pinned artifacts' projections, and the only width the
# pack's routing table was priced at (ADR 0015, ruling D1: 4-bit is unpriced
# and routes nowhere until its own pricing run lands).
PINNED_BITS = 3
# The pre-registered win zone, per intercepted dispatch shape (d_out, d_in).
# Routing still delegates to should_dispatch; this map exists so the timing
# modes REFUSE to run when the pack's boundary disagrees with what was
# registered, instead of silently rescoping the claim (findings doc,
# section 3, and its 2026-08-15 amendment).
#
# The entries are literals deliberately. Derived from routed_windows at
# import they could never disagree with the pack, and a check that cannot
# fail registers nothing. Each one is window_for(3, d_out, d_in) = {5..9}
# intersected with B_GRID, read off the 2026-08-15 boundary pricing run
# (bench/results/qmv-boundary-pricing-2026-08-15.json, ADR 0015). B = 11 sat
# in the zone under the uniform 5..11 default and is out of it at all five
# now, though not for one reason: it is a measured LOSS at three of them
# (q_proj, o_proj, down_proj) and REFUSED, meaning undecided, at the other
# two. It is a WIN at none, and only a WIN routes a cell.
#
# lm_head (151936x2560) is absent on purpose, not by oversight. The patch
# wraps nn.QuantizedLinear leaves only, and Qwen3-4B's lm_head is tied
# embeddings, so it is never intercepted; a zone entry for a shape this
# harness cannot route would register a claim nothing here enforces.
PINNED_ZONE = {
    (4096, 2560): frozenset({5, 6, 8}),    # q_proj
    (1024, 2560): frozenset({5, 6, 8}),    # k_proj, v_proj
    (2560, 4096): frozenset({5, 6, 8}),    # o_proj
    (9728, 2560): frozenset({5, 6, 8}),    # gate_proj, up_proj
    (2560, 9728): frozenset({5, 6, 8}),    # down_proj
}
ROUNDS = 5
PPL_WINDOW = 1024
PPL_WINDOWS = 96
MAX_SPREAD_PCT = machine_state.MAX_SPREAD_PCT

# The seven per-layer projections of Qwen3-4B as (d_in, d_out, count) over
# their five distinct shapes: k and v share one, gate and up share another
# (MDE, section 5).
PROJ_SHAPES = [(2560, 4096, 1), (2560, 1024, 2), (4096, 2560, 1),
               (2560, 9728, 2), (9728, 2560, 1)]
PROJS_PER_LAYER = 7

MIN_SAMPLE_MS = 5.0
WORKING_SET_MB = 512

PROMPT_SEED = (
    "The verification of GPU kernels is a study in mismatched incentives: "
    "the benchmark that ships with a kernel was written by whoever wrote the "
    "kernel, and it passes. What escapes is decided by the pair of bug and "
    "test case, never by the bug alone, so the only honest measure of a test "
    "policy is how it scores against a fault population it did not choose. "
)

_KERNEL = build(mx)


# ---------------------------------------------------------------------------
# the interception layer
# ---------------------------------------------------------------------------
def site_cell(inner: nn.QuantizedLinear) -> tuple[int, int, int]:
    """One layer's routing cell as (bits, d_out, d_in).

    The order is should_dispatch's, and it is worth reading twice: MLX
    stores scales as (d_out, d_in // group_size), so the rows are the OUTPUT
    dimension and the columns have to be scaled back up by the group size to
    recover d_in. Reading the pair the other way round would silently ask
    the routing table about a shape the model does not have, and the table
    answers an unknown shape with "no evidence", so the mistake would look
    like a boundary that simply never routes.
    """
    return (inner.bits, inner.scales.shape[0],
            inner.scales.shape[1] * inner.group_size)


class _RoutedLinear:
    """One wrapped QuantizedLinear. A plain object, deliberately not a
    Module: it exists only inside the installed model tree, so removing
    it is a single attribute restore and nothing outside that tree can
    ever reach it."""

    __slots__ = ("inner", "patch")

    def __init__(self, inner: nn.QuantizedLinear, patch: "Patch"):
        self.inner = inner
        self.patch = patch

    def _ineligible(self, x) -> str | None:
        """None when the kernel takes this call; otherwise the reason.
        The prefill check precedes the M check so a prefill chunk is
        labeled prefill regardless of how the boundary is set."""
        inner = self.inner
        mode = str(getattr(inner, "mode", "affine"))
        if not mode.endswith("affine"):
            return f"mode-{mode}"
        if inner.bits not in SUPPORTED_BITS or inner.group_size != GROUP_SIZE:
            return f"quant-{inner.bits}b-gs{inner.group_size}"
        if "bias" in inner:
            return "bias-term"
        if x.dtype != mx.float16 or inner.scales.dtype != mx.float16:
            return f"dtype-{x.dtype}-{inner.scales.dtype}"
        if x.ndim >= 3 and x.shape[-2] != 1:
            return f"prefill-L{x.shape[-2]}"
        d_in = x.shape[-1]
        if d_in % 64:
            return f"din-{d_in}"
        m = x.size // d_in
        # The whole routing cell, never just the tile width: since ADR 0015
        # two projections in one layer can answer this differently. The
        # reason carries the shape for the same reason.
        d_out = inner.scales.shape[0]
        if not should_dispatch(m, inner.bits, d_out, d_in):
            return f"m-{m}-outside-dispatch-{d_out}x{d_in}"
        return None

    def _fused(self, x):
        inner = self.inner
        d_in = x.shape[-1]
        x2 = x.reshape(-1, d_in)
        m = x2.shape[0]
        d_out = inner.scales.shape[0]
        grid, tg, r = launch_config(d_out, m)
        out = _KERNEL(
            inputs=[x2, inner.weight, inner.scales, inner.biases],
            output_shapes=[(m, d_out)], output_dtypes=[x.dtype],
            grid=grid, threadgroup=tg,
            template=[("T", x.dtype), ("BITS", inner.bits),
                      ("M", m), ("R", r)])[0]
        return out.reshape(*x.shape[:-1], d_out)

    def __call__(self, x):
        patch = self.patch
        why = self._ineligible(x)
        if why is None:
            if patch.mode == "fused":
                patch.calls += 1
                return self._fused(x)
            why = "forced-stock"  # arm 4: same evaluation, answer discarded
        patch.fallbacks[why] = patch.fallbacks.get(why, 0) + 1
        return self.inner(x)


# Reasons that do not mean the arm stopped being the arm (doc, section 4).
_WHITELIST_PREFIXES = ("prefill-", "m-", "forced-stock")


class Patch:
    """The installed interception over one model tree."""

    def __init__(self, model: nn.Module):
        if getattr(model, "_serve_sub4bit_patch", None) is not None:
            raise RuntimeError("patch already installed on this model")
        self.mode = "fused"
        self.calls = 0
        self.fallbacks: dict[str, int] = {}
        self._model = model
        self._sites: list[tuple] = []
        self._wrap(model)
        model._serve_sub4bit_patch = self
        self.n_wrapped = len(self._sites)
        # One routing cell per wrapped site, in the order should_dispatch
        # takes them. Recorded at install and kept after uninstall, like
        # n_wrapped, because the A/B builds its row for a cell after the
        # last arm has already removed the patch.
        self.site_cells = [site_cell(o) for _, _, o in self._sites]

    def _wrap(self, node):
        if isinstance(node, (nn.Module, dict)):
            entries = [(node, k, v) for k, v in node.items()]
        elif isinstance(node, (list, tuple)):
            entries = [(node, i, v) for i, v in enumerate(node)]
        else:
            return
        for parent, key, value in entries:
            if isinstance(value, nn.QuantizedLinear):
                wrapper = _RoutedLinear(value, self)
                self._replace(parent, key, wrapper)
                self._sites.append((parent, key, value))
            elif isinstance(value, (nn.Module, list, tuple, dict)):
                self._wrap(value)

    @staticmethod
    def _replace(parent, key, value):
        if isinstance(parent, nn.Module):
            setattr(parent, key, value)
        else:
            parent[key] = value

    def reset(self):
        self.calls = 0
        self.fallbacks = {}

    def hard_fallbacks(self) -> dict[str, int]:
        return {k: v for k, v in self.fallbacks.items()
                if not k.startswith(_WHITELIST_PREFIXES)}

    def uninstall(self):
        for parent, key, original in self._sites:
            self._replace(parent, key, original)
        self._sites = []
        self._model._serve_sub4bit_patch = None


def install_patch(model: nn.Module) -> Patch:
    return Patch(model)


# ---------------------------------------------------------------------------
# artifact pins
# ---------------------------------------------------------------------------
def verify_pins(root: Path, names: list[str]) -> dict[str, str]:
    """Refuse to run on anything but the coordinator-pinned artifacts.
    This gate is claim-critical: every number in the findings doc binds
    to these hashes, so a mismatch is a refusal, never a warning."""
    pins = root / "PINNED-HASHES.txt"
    if not pins.exists():
        raise RuntimeError(f"no pin manifest at {pins}")
    entries = []
    for line in pins.read_text().splitlines():
        line = line.strip()
        if line:
            digest, rel = line.split(maxsplit=1)
            # The coordinator spells paths from the repo root; accept both
            # that and .models-relative spellings.
            rel = rel.removeprefix("bench/.models/")
            entries.append((digest, rel))
    manifest = {}
    for name in names:
        mine = [(d, r) for d, r in entries if r.startswith(name + "/")]
        if not mine:
            raise RuntimeError(f"{name} has no entries in {pins}")
        # Coverage, not just consistency: an unlisted file (a tokenizer, a
        # config) would load unverified while the recorded numbers claim to
        # bind to the manifest. Hidden files are Finder noise, ignored.
        listed = {r for _, r in mine}
        actual = {f"{name}/{p.relative_to(root / name)}"
                  for p in (root / name).rglob("*")
                  if p.is_file() and not p.name.startswith(".")}
        extra = sorted(actual - listed)
        if extra:
            raise RuntimeError(
                f"{name} holds files outside the pin manifest: {extra}")
        for digest, rel in mine:
            path = root / rel
            if not path.exists():
                raise RuntimeError(f"pinned file missing: {path}")
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            if h.hexdigest() != digest:
                raise RuntimeError(f"pin mismatch for {rel}")
            manifest[rel] = digest
    return manifest


# ---------------------------------------------------------------------------
# the batched decode driver
# ---------------------------------------------------------------------------
def decode_window(model, prompts, gen_tokens: int) -> tuple[float, int]:
    """One batched generation; returns (seconds, steps) for the timed
    window: the gen_tokens - 1 decode steps after the first token.
    Prefill and the first token are excluded, the spike's convention.
    One step is kept in flight, generate_step's own pipelining shape."""
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    logits = model(prompts, cache=cache)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)
    steps = gen_tokens - 1
    t0 = time.perf_counter()
    prev = y
    for _ in range(steps):
        logits = model(prev[:, None], cache=cache)
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        mx.async_eval(nxt)
        mx.eval(prev)
        prev = nxt
    mx.eval(prev)
    dt = time.perf_counter() - t0
    return dt, steps


# ---------------------------------------------------------------------------
# the perplexity proxy (T5, D5)
# ---------------------------------------------------------------------------
def perplexity(model, ids: list[int], window: int = PPL_WINDOW,
               n_windows: int = PPL_WINDOWS) -> float:
    """exp of the mean float32 NLL of positions 2..window over the first
    n_windows non-overlapping windows; the tail is dropped (doc, sec 7)."""
    need = window * n_windows
    if len(ids) < need:
        raise RuntimeError(f"corpus too short: {len(ids)} < {need} tokens")
    total_nll, total_tok = 0.0, 0
    for w in range(n_windows):
        chunk = mx.array(ids[w * window:(w + 1) * window],
                         dtype=mx.int32)[None, :]
        logits = model(chunk).astype(mx.float32)
        pred_logits = logits[0, :-1, :]
        tgt = chunk[0, 1:]
        picked = mx.take_along_axis(pred_logits, tgt[:, None], axis=-1)[:, 0]
        nll = mx.logsumexp(pred_logits, axis=-1) - picked
        total_nll += float(mx.sum(nll))
        total_tok += window - 1
        mx.clear_cache()
    return math.exp(total_nll / total_tok)


# ---------------------------------------------------------------------------
# shared plumbing for the model modes
# ---------------------------------------------------------------------------
def load_model(name: str):
    from mlx_lm import load
    model, tokenizer = load(str(MODELS_ROOT / name))
    # bf16 checkpoints are cast to fp16 for every arm, the spike's rule:
    # same lane width on Metal, and the frozen contract's dtype.
    model.set_dtype(mx.float16)
    return model, tokenizer


def make_prompts(tokenizer, t: int, b: int):
    ids = tokenizer.encode(PROMPT_SEED * 40)
    if len(ids) < t:
        raise RuntimeError(f"prompt seed too short: {len(ids)} < {t}")
    row = mx.array(ids[:t], dtype=mx.int32)[None, :]
    return mx.tile(row, (b, 1))


def provenance(manifest: dict) -> dict:
    import mlx_lm
    return {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
            "pins": manifest, "machine": machine_state.fingerprint()}


class NotIdle(RuntimeError):
    """The machine is not quiet. TRANSIENT: the same run on the same code
    succeeds once whatever is busy stops, so a detached runner waits."""


class PreconditionFailed(RuntimeError):
    """A precondition that waiting cannot heal: the artifacts do not hash to
    their pins, the pack's boundary has moved off its registration, the
    corpus is not the registered one. PERMANENT: a runner gives up and a
    human re-registers or re-pins."""


class ServeGuard:
    """This run's memory checks, in the order price_qmv_boundary takes them.

    Two different quantities, and both are needed: the budget bounds THIS
    process's phys_footprint (the number Jetsam kills on), while the memory
    gate bounds the MACHINE's remaining room against the headroom this run may
    still legitimately grow into - budget minus current, never the whole
    budget, or our own growth would make the check refuse a healthy run.
    """

    def __init__(self, budget_gb: float):
        self.budget = BudgetGuard(budget_gb)
        self.peak_gb = 0.0

    def __call__(self, cell: str) -> float:
        current = self.budget.check(cell)
        self.peak_gb = max(self.peak_gb, current)
        require_available_memory(max(self.budget.budget_gb - current, 0.0),
                                 cell)
        return current


def require_idle(label: str) -> dict:
    state = machine_state.idle_check()
    if not state["idle"]:
        print(f"NOT QUIET ({label}): " + "; ".join(state["blockers"]))
        print("Timing refused (T8): rerun inside a coordinated quiet window.")
        raise NotIdle(f"machine not quiet {label}: "
                      + "; ".join(state["blockers"]))
    return state


def pack_zone(d_out: int, d_in: int) -> frozenset[int]:
    """The grid cells the pack routes at one intercepted shape."""
    return frozenset(b for b in B_GRID
                     if should_dispatch(b, PINNED_BITS, d_out, d_in))


def require_pinned_zone():
    """Refuse unless the pack agrees with the registration at every shape.

    Per shape, because the boundary is per shape: a table that moved at one
    projection and not at the others would pass any check that collapsed the
    grid to a single set.
    """
    disagreements = [
        f"{d_out}x{d_in}: pack routes {sorted(pack_zone(d_out, d_in))}, "
        f"registered {sorted(pinned)}"
        for (d_out, d_in), pinned in PINNED_ZONE.items()
        if pack_zone(d_out, d_in) != pinned]
    if disagreements:
        print(f"REFUSED: the pack's dispatch zone disagrees with the "
              f"pre-registered zone at {len(disagreements)} of "
              f"{len(PINNED_ZONE)} intercepted shapes; timing on a moved "
              "boundary needs a re-registration, not a run")
        for line in disagreements:
            print(f"  {line}")
        raise PreconditionFailed(
            f"pack dispatch zone disagrees with the registration at "
            f"{len(disagreements)} of {len(PINNED_ZONE)} shapes")


def check_idle_after(exit_code: int) -> int:
    after = machine_state.idle_check()
    if not after["idle"]:
        print("WARNING: machine went non-idle during the run: "
              + "; ".join(after["blockers"]))
        return 1
    return exit_code


def spread_pct(vals: list[float]) -> float:
    return (max(vals) - min(vals)) / statistics.median(vals) * 100


def expected_calls(patch: Patch, b: int, steps: int) -> int:
    """The exact fused-call count for a decode window at batch size b.

    Summed per wrapped site rather than multiplied by n_wrapped: the routing
    table can route one projection shape and not another at the same b, and
    a whole-model multiply would then be wrong in both directions at once.
    """
    return steps * sum(1 for cell in patch.site_cells
                       if should_dispatch(b, *cell))


def routed_shapes(patch: Patch, b: int) -> list[str]:
    """The distinct intercepted shapes the pack routes at this batch size."""
    return sorted({f"{d_out}x{d_in}"
                   for bits, d_out, d_in in patch.site_cells
                   if should_dispatch(b, bits, d_out, d_in)})


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def smoke() -> int:
    """Wiring correctness, no timing anywhere: exact dispatch counts on
    both sides of the boundary, zero hard fallbacks, arm 4 inert."""
    model3, tok = load_model(MODEL_3BIT)
    model4, _ = load_model(MODEL_4BIT)
    n_layers = len(model3.layers)
    t, g = 64, 9
    # The numerics check and arm 4 need a B the pack routes at every
    # intercepted shape AND that every shape's registration holds: smoke
    # must never dispatch a shape the certificates do not cover.
    b_zone = next(b for b in B_GRID
                  if all(b in pack_zone(d_out, d_in) and b in pinned
                         for (d_out, d_in), pinned in PINNED_ZONE.items()))
    skipped = []
    ok = True

    patch = install_patch(model3)
    if patch.n_wrapped != PROJS_PER_LAYER * n_layers:
        print(f"FAIL: wrapped {patch.n_wrapped} leaves, expected "
              f"{PROJS_PER_LAYER} x {n_layers}")
        ok = False
    # The registration covers the intercepted shapes and only those. If the
    # patch ever wrapped a sixth shape (an untied lm_head, say) it would
    # route on a cell nothing here registered, so the mismatch is a failure
    # rather than a note.
    wrapped_shapes = sorted({(d_out, d_in)
                             for _, d_out, d_in in patch.site_cells})
    if wrapped_shapes != sorted(PINNED_ZONE):
        print(f"FAIL: wrapped shapes {wrapped_shapes} are not the registered "
              f"shapes {sorted(PINNED_ZONE)}")
        ok = False
    for b in B_GRID:
        disagreeing = [f"{d_out}x{d_in}"
                       for (d_out, d_in), pinned in PINNED_ZONE.items()
                       if (b in pack_zone(d_out, d_in)) != (b in pinned)]
        if disagreeing:
            # Running this cell fused would dispatch an uncertified M (or
            # mask a cell the registration says must dispatch nothing).
            print(f"B={b:>2} SKIPPED: pack boundary disagrees with the "
                  f"registered zone at {', '.join(disagreeing)}")
            skipped.append(b)
            continue
        patch.reset()
        patch.mode = "fused"
        _, steps = decode_window(model3, make_prompts(tok, t, b), g)
        want = expected_calls(patch, b, steps)
        hard = patch.hard_fallbacks()
        line_ok = patch.calls == want and not hard
        ok = ok and line_ok
        print(f"B={b:>2} fused: calls {patch.calls} (want {want}), routed "
              f"shapes {len(routed_shapes(patch, b))}/{len(PINNED_ZONE)}, "
              f"hard fallbacks {hard or 'none'}"
              + ("" if line_ok else "  <-- FAIL"))

    # The routed numerics at the real weights: fused vs stock per
    # projection on layer 0, decode-shaped. Both fp32-accumulate the same
    # math, so anything past a few fp16 ulps of the output magnitude is a
    # wiring mistake (wrong buffer, wrong stride), not arithmetic.
    worst = 0.0
    for parent, key, original in patch._sites[:PROJS_PER_LAYER]:
        wrapper = getattr(parent, key)
        x = mx.random.normal(
            shape=(b_zone, original.scales.shape[1] * GROUP_SIZE)
        ).astype(mx.float16)
        fused, stock = wrapper._fused(x), original(x)
        mx.eval(fused, stock)
        mag = float(mx.abs(stock).max())
        gap = float(mx.abs(fused - stock).max())
        worst = max(worst, gap / (4e-3 * max(1.0, mag)))
    num_ok = worst < 1.0
    ok = ok and num_ok
    print(f"per-op gap on layer-0 weights: worst {worst:.3f} of the 4-ulp "
          f"gate" + ("" if num_ok else "  <-- FAIL"))

    # Arm 4: same wrapper, eligibility evaluated, zero dispatches.
    patch.reset()
    patch.mode = "stock"
    _, steps = decode_window(model3, make_prompts(tok, t, b_zone), g)
    forced = patch.fallbacks.get("forced-stock", 0)
    arm4_ok = patch.calls == 0 and forced == patch.n_wrapped * steps
    ok = ok and arm4_ok
    print(f"arm4 at B={b_zone}: calls {patch.calls} (want 0), forced-stock "
          f"{forced} (want {patch.n_wrapped * steps})"
          + ("" if arm4_ok else "  <-- FAIL"))
    patch.uninstall()

    # Stock arms exist and decode; no counters may move once uninstalled.
    # Mode goes back to fused first: a broken uninstall in stock mode would
    # count forced-stock rather than calls and slip past a calls-only check.
    patch.mode = "fused"
    before_calls, before_fb = patch.calls, dict(patch.fallbacks)
    decode_window(model3, make_prompts(tok, t, b_zone), 3)
    decode_window(model4, make_prompts(tok, t, b_zone), 3)
    ok = (ok and patch.calls == before_calls
          and patch.fallbacks == before_fb)
    if skipped:
        print(f"NOTE: cells {skipped} were skipped; the pack's routing table "
              "and the registered zone must be reconciled, then rerun smoke")
    print("SMOKE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def _op_probe(bits: int, d_in: int, d_out: int, m: int) -> dict:
    """Stock vs fused per-op time at one true projection shape, g64 weights
    rotated over a working set, interleaved, batched past MIN_SAMPLE_MS."""
    weight_bytes = d_out * d_in * bits / 8 + 2 * (d_out * d_in / 64) * 2
    n_sets = max(2, min(64, int(WORKING_SET_MB * 1e6 // weight_bytes) + 1))
    sets = []
    for seed in range(n_sets):
        rng = np.random.default_rng(1000 + seed)
        w = (rng.standard_normal((d_out, d_in)).astype(np.float32)
             * 0.02).astype(np.float16)
        sets.append(mx.quantize(mx.array(w), group_size=GROUP_SIZE,
                                bits=bits))
    mx.eval([a for s in sets for a in s])
    x = mx.random.normal(shape=(m, d_in)).astype(mx.float16)
    mx.eval(x)
    grid, tg, r = launch_config(d_out, m)

    def ours(i):
        wq, sc, bi = sets[i % n_sets]
        return _KERNEL(inputs=[x, wq, sc, bi], output_shapes=[(m, d_out)],
                       output_dtypes=[mx.float16], grid=grid, threadgroup=tg,
                       template=[("T", mx.float16), ("BITS", bits),
                                 ("M", m), ("R", r)])[0]

    def theirs(i):
        wq, sc, bi = sets[i % n_sets]
        return mx.quantized_matmul(x, wq, sc, bi, transpose=True,
                                   group_size=GROUP_SIZE, bits=bits)

    def sample(fn, copies):
        outs = [fn(i) for i in range(copies)]
        t0 = time.perf_counter()
        mx.eval(outs)
        mx.synchronize()
        return time.perf_counter() - t0

    mx.eval(ours(0), theirs(0))
    mx.synchronize()
    copies = 8
    while copies < 4096 and sample(theirs, copies) * 1e3 < MIN_SAMPLE_MS:
        copies *= 2
    a_s, b_s = [], []
    for _ in range(7):
        a_s.append(sample(ours, copies) / copies)
        b_s.append(sample(theirs, copies) / copies)
    return {"stock_us": statistics.median(b_s) * 1e6,
            "fused_us": statistics.median(a_s) * 1e6,
            "stock_spread_pct": round(spread_pct(b_s), 1)}


def mde(manifest: dict, guard) -> int:
    """The numbers section 5 of the findings doc is filled from: the
    interception cost via arm 4 at B=1, per-op times at the true shapes,
    arm-2 per-step times and noise floors per in-zone B."""
    guard("mde-start")          # before the first model, or it guards nothing
    print(json.dumps(provenance(manifest)))
    model3, tok = load_model(MODEL_3BIT)
    n_layers = len(model3.layers)
    zone = [b for b in B_GRID
            if any(b in pack_zone(d_out, d_in) for d_out, d_in in PINNED_ZONE)]

    # Interception cost: arm 4 vs arm 2 at B=1, interleaved rounds.
    prompts = make_prompts(tok, PROMPT_T, 1)
    a4, a2 = [], []
    decode_window(model3, prompts, 4)        # warm the decode graph
    for _ in range(ROUNDS):
        patch = install_patch(model3)
        patch.mode = "stock"
        dt, steps = decode_window(model3, prompts, GEN_TOKENS)
        a4.append(dt / steps)
        patch.uninstall()                    # n_wrapped survives the removal
        dt, steps = decode_window(model3, prompts, GEN_TOKENS)
        a2.append(dt / steps)
    i_ms = (statistics.median(a4) - statistics.median(a2)) * 1e3
    print(json.dumps({"interception_cost_ms_per_step": round(i_ms, 4),
                      "arm4_spread_pct": round(spread_pct(a4), 2),
                      "arm2_spread_pct": round(spread_pct(a2), 2),
                      "wrapped_calls_per_step": patch.n_wrapped}))

    # Per-op savings at the true shapes, then the expected gain per B.
    for b in zone:
        guard(f"B={b}")
        saving_us = 0.0
        rows = []
        for d_in, d_out, count in PROJ_SHAPES:
            # A shape the table does not route at this b contributes no
            # saving, because at that cell the model runs stock there.
            if b not in pack_zone(d_out, d_in):
                rows.append({"shape": f"{d_in}x{d_out}", "count": count,
                             "routed": False})
                continue
            p = _op_probe(PINNED_BITS, d_in, d_out, b)
            saving_us += count * (p["stock_us"] - p["fused_us"])
            rows.append({"shape": f"{d_in}x{d_out}", "count": count,
                         "routed": True, **p})
        dts = []
        for _ in range(ROUNDS):
            dt, steps = decode_window(model3,
                                      make_prompts(tok, PROMPT_T, b),
                                      GEN_TOKENS)
            dts.append(dt / steps)
        t_step_ms = statistics.median(dts) * 1e3
        saving_ms = n_layers * saving_us / 1e3
        gain_pct = 100 * (saving_ms - i_ms) / t_step_ms
        print(json.dumps({
            "B": b, "per_op": rows,
            "arm2_ms_per_step": round(t_step_ms, 3),
            "arm2_noise_floor_pct": round(spread_pct(dts), 2),
            "step_saving_ms": round(saving_ms, 4),
            "expected_gain_pct": round(gain_pct, 2)}))
    return check_idle_after(0)


def primary_verdict(prim_pct: float, noise_pct: float) -> str:
    """Arm 1 against arm 2, read off the doc's section-6 criterion. The
    noise floor is the cell's own canary spread, so a gap it could have
    produced is a null result, not a small win."""
    if abs(prim_pct) <= noise_pct:
        return "null"
    if prim_pct > 0:
        return "win"
    return "regression"


def composed_attribution(comp_pct: float, base_pct: float,
                         noise_pct: float) -> str:
    """Arm 1 against arm 3, one row per outcome cell of the doc's
    section-6 attribution table; arm 2 against arm 3 (base_pct) is what
    separates a win the artifact already had from one the kernel unlocked."""
    if abs(comp_pct) <= noise_pct:
        return "inconclusive"
    if comp_pct <= 0:
        return "negative"
    if base_pct > noise_pct:
        return "artifact-alone"
    return "joint"


def ab(manifest: dict, guard) -> int:
    """The pre-registered four-arm grid (doc, sections 3 and 6): arms
    interleaved [1, 2, 3, 4] in every round, arm 2 the canary, cells over
    the spread limit withheld, dispatch counts asserted exact."""
    guard("ab-start")           # before the first model, or it guards nothing
    print(json.dumps(provenance(manifest)))
    model3, tok = load_model(MODEL_3BIT)
    model4, _ = load_model(MODEL_4BIT)
    exit_code = 0
    for b in B_GRID:
        guard(f"B={b}")
        prompts = make_prompts(tok, PROMPT_T, b)
        arms: dict[str, list[float]] = {"1": [], "2": [], "3": [], "4": []}
        calls_want = calls_got = 0
        hard: dict[str, int] = {}
        # Warm every arm outside the timed rounds: the fused kernel JIT
        # compiles once per M, and that compile must never sit inside a
        # recorded window.
        patch = install_patch(model3)
        decode_window(model3, prompts, 4)
        patch.uninstall()
        decode_window(model3, prompts, 4)
        decode_window(model4, prompts, 4)
        for _ in range(ROUNDS):
            patch = install_patch(model3)
            patch.mode = "fused"
            dt, steps = decode_window(model3, prompts, GEN_TOKENS)
            arms["1"].append(steps / dt)
            calls_want += expected_calls(patch, b, steps)
            calls_got += patch.calls
            for k, v in patch.hard_fallbacks().items():
                hard[k] = hard.get(k, 0) + v
            patch.uninstall()
            guard(f"B={b} arm1")

            dt, steps = decode_window(model3, prompts, GEN_TOKENS)
            arms["2"].append(steps / dt)
            guard(f"B={b} arm2")

            dt, steps = decode_window(model4, prompts, GEN_TOKENS)
            arms["3"].append(steps / dt)
            guard(f"B={b} arm3")

            patch = install_patch(model3)
            patch.mode = "stock"
            dt, steps = decode_window(model3, prompts, GEN_TOKENS)
            arms["4"].append(steps / dt)
            patch.uninstall()
            guard(f"B={b} arm4")

        med = {k: statistics.median(v) for k, v in arms.items()}
        routed = routed_shapes(patch, b)
        row = {"B": b, "gen": GEN_TOKENS, "rounds": ROUNDS,
               "routed_shapes": routed, "in_zone": bool(routed),
               "fused_calls": calls_got, "fused_calls_want": calls_want,
               "hard_fallbacks": hard}
        sp = spread_pct(arms["2"])
        if hard or calls_got != calls_want:
            row["verdict"] = "INVALID: arm 1 stopped being arm 1"
            exit_code = 1
        elif sp > MAX_SPREAD_PCT:
            row["verdict"] = (f"WITHHELD: canary spread {sp:.1f}% > "
                              f"{MAX_SPREAD_PCT}%")
            exit_code = 1
        else:
            # Every claim is qualified by the cell's noise floor: the
            # canary arm's spread (doc, section 6).
            comp_pct = 100 * (med["1"] / med["3"] - 1)
            base_pct = 100 * (med["2"] / med["3"] - 1)
            prim_pct = 100 * (med["1"] / med["2"] - 1)
            row.update({
                "per_stream_tps": {k: round(v, 3) for k, v in med.items()},
                "aggregate_tps": {k: round(b * v, 2)
                                  for k, v in med.items()},
                "spread_pct": {k: round(spread_pct(v), 2)
                               for k, v in arms.items()},
                "noise_floor_pct": round(sp, 2),
                "ratio_ours_stock3": round(med["1"] / med["2"], 4),
                "primary_verdict": primary_verdict(prim_pct, sp),
                "ratio_patchcost": round(med["4"] / med["2"], 4),
                "ratio_composed_vs_4bit": round(med["1"] / med["3"], 4),
                "composed_attribution": composed_attribution(
                    comp_pct, base_pct, sp),
            })
        print(json.dumps(row))
    return check_idle_after(exit_code)


def ppl(corpus: Path) -> int:
    """The T5 pair. Stock models only; the corpus hash is printed first
    so the findings doc records it before either number is read."""
    if not corpus.exists():
        print(f"REFUSED (exit {EXIT_PRECONDITION}): no corpus at {corpus}; "
              "the findings doc pins its sha256 before any number is read")
        return EXIT_PRECONDITION
    text = corpus.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    # Before a model is loaded, not after: printing a digest and scoring
    # whatever was there is how a quality pair ends up bound to the wrong
    # text (findings doc section 7 registers the content, not the path).
    if digest != PPL_CORPUS_SHA256:
        print(f"REFUSED (exit {EXIT_PRECONDITION}): corpus at {corpus} "
              f"hashes {digest}, not the registered {PPL_CORPUS_SHA256}")
        return EXIT_PRECONDITION
    model3, tok3 = load_model(MODEL_3BIT)
    model4, tok4 = load_model(MODEL_4BIT)
    ids3, ids4 = tok3.encode(text), tok4.encode(text)
    if ids3 != ids4:
        print(f"REFUSED (exit {EXIT_PRECONDITION}): the two artifacts "
              "tokenize the corpus differently")
        return EXIT_PRECONDITION
    print(json.dumps({"corpus": str(corpus), "sha256": digest,
                      "tokens": len(ids3), "window": PPL_WINDOW,
                      "windows": PPL_WINDOWS}))
    pair = {"ppl_3bit": round(perplexity(model3, ids3), 4),
            "ppl_4bit": round(perplexity(model4, ids4), 4)}
    print(json.dumps(pair))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--mde", action="store_true")
    mode.add_argument("--ab", action="store_true")
    mode.add_argument("--ppl", action="store_true")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--budget-gb", type=budget_gb_arg,
                        default=AB_BUDGET_GB,
                        help="phys_footprint budget in decimal GB; crossing "
                             f"it refuses and exits {EXIT_BUDGET_REFUSAL} "
                             f"(default {AB_BUDGET_GB})")
    args = parser.parse_args(argv)

    # Pins first, before the lock: a hash mismatch is permanent, so holding
    # the machine while discovering it would deny the slot to a run that
    # could have used it.
    try:
        manifest = verify_pins(MODELS_ROOT, [MODEL_3BIT, MODEL_4BIT])
    except RuntimeError as e:
        print(f"REFUSED (exit {EXIT_PRECONDITION}): {e}")
        return EXIT_PRECONDITION
    if args.smoke:
        return smoke()          # wiring only, no timing: takes no lock

    # The ONE machine-wide lock, taken before anything else samples the
    # machine: two heavy harnesses whose idle gates co-fired is the 03:29
    # collapse itself, so even the idle gate waits behind the lock.
    lock = MeasurementLock(LOCK_NAME)
    acquired, detail = lock.acquire()
    if not acquired:
        print(f"REFUSAL (exit {EXIT_LOCK_HELD}): machine measurement lock "
              f"{detail}; one heavy measurement at a time")
        return EXIT_LOCK_HELD
    try:
        # Perplexity is not a timing, so it skips the idle gate by section
        # 7's own rule - but it loads both artifacts, and the same section
        # forbids it running beside a timing mode, which is what the lock
        # above enforces.
        if args.ppl:
            return ppl(args.corpus)
        require_idle("before")
        require_pinned_zone()
        guard = ServeGuard(args.budget_gb)
        return mde(manifest, guard) if args.mde else ab(manifest, guard)
    except NotIdle as e:
        print(f"REFUSAL (exit {EXIT_NOT_IDLE}): {e}")
        return EXIT_NOT_IDLE
    except PreconditionFailed as e:
        print(f"REFUSAL (exit {EXIT_PRECONDITION}): {e}")
        return EXIT_PRECONDITION
    except BudgetExceeded as e:
        print(f"REFUSAL (exit {EXIT_BUDGET_REFUSAL}): {e}")
        print("run discarded; nothing measured under a crossed budget is a "
              "claim")
        return EXIT_BUDGET_REFUSAL
    except LowMemoryRefusal as e:
        print(f"REFUSAL (exit {EXIT_LOW_MEMORY}): {e}")
        print("another process is eating the machine; free it and re-arm "
              "the slot")
        return EXIT_LOW_MEMORY
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
