"""Timing named regions of a real mlx-lm training step, in both directions.

The problem this solves
----------------------
The Day 1 profile has to say where the step's time goes, counting each
region's backward as well as its forward. MLX gives no per-operation profiler
and no way to read a GPU trace from Python, so the time has to be taken from
inside the step. Two facts, both measured on 2026-08-20 against mlx 0.32.0,
decide the mechanism.

An `mx.eval` inside a compiled step is refused outright, which is why the
profile runs under `mx.disable_compile()` and why Amendment 4 exists. An
`mx.eval` inside a `custom_function` vjp is PERMITTED, fires in true backward
order, and forces the work queued since the previous such eval. That second
fact is the whole instrument: it is the only way found to put a clock inside
an MLX backward.

The mark
--------
A mark is an identity: it takes tensors, evaluates them, records the time and
returns them unchanged. Its gradient rule does the same to the cotangent. So
it computes nothing, changes no value, and adds no operation to the graph
whose result anyone reads; what it adds is a fence, and a fence is what makes
a timestamp mean "everything before this point has finished".

Stock is never recomputed. MLX's own forward runs between a region's two
forward marks and MLX's own backward runs between its two backward marks, so
what is timed is the shipped operation rather than a re-implementation of it.

Reading the log
---------------
Forward, a region's enter mark fires before its exit mark. Backward, the order
reverses: the exit mark's gradient rule fires first, then the region's own
backward runs, then the enter mark's fires. So a forward span is exit minus
enter and a backward span is enter minus exit, and `decompose` knows which
label opens a span in each direction rather than assuming the log is sorted
into pairs.

What an absent backward mark means
----------------------------------
A mark's gradient rule only fires if a gradient actually reaches it. Under
LoRA that is not a formality: mlx-lm adapts only the last `num_layers` blocks,
so every block below the first adapted one has no trainable parameter beneath
it and no backward at all. Its forward marks fire and its backward marks do
not, and that is the truth about the workload rather than a fault in the
instrument. `counts` reports forward and backward separately for exactly this
reason, and a region with no backward is reported as having none rather than
as having a backward of zero.

Why the spans must not overlap
------------------------------
Summing a region that runs inside another would count the inner one twice. The
four patch points do not nest - the attention core sits between the
projections rather than around them, and the output head is a different class
from the projections - but `decompose` checks rather than trusts, because a
silent double count reads as a large share and would aim the sprint at the
wrong operation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from decode_rules import RunInvalid

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metalrunner import seams  # noqa: E402

FORWARD = "fwd"
BACKWARD = "bwd"
ENTER = "enter"
EXIT = "exit"

# Which end of a region opens its span, per direction. Forward runs enter to
# exit; backward runs the same span in reverse, so exit opens it.
_OPENS = {FORWARD: ENTER, BACKWARD: EXIT}


@dataclass
class Recorder:
    """Everything one instrumented pass observed.

    `entries` is the timing log, in the order the marks fired. `shapes` counts
    how often each region ran at each logical shape, which is what the
    collapse rule in `profile_rules` weights a credited ratio by; it is taken
    from the layer the call went through rather than from the model's declared
    geometry, so a share and a ratio cannot end up describing different
    workloads.
    """

    entries: list[tuple[str, str, str, float]] = field(default_factory=list)
    shapes: dict[str, dict[str, int]] = field(default_factory=dict)
    context: dict = field(default_factory=dict)

    def note_shape(self, region: str, shape: tuple[int, int]) -> None:
        by_shape = self.shapes.setdefault(region, {})
        key = f"{shape[0]}x{shape[1]}"
        by_shape[key] = by_shape.get(key, 0) + 1

    def clear(self) -> None:
        """Drop what a warm-up recorded, keeping what identifies the run.

        The context is not timing and must survive: it is what says two passes
        measured the same batch, and clearing it would let a numerator from
        one workload be divided by a denominator from another.
        """
        self.entries.clear()
        self.shapes.clear()


def make_mark(recorder: Recorder, region: str, end: str):
    """An identity that fences and timestamps, in both directions.

    Built per region and per end rather than once, because a
    `mx.custom_function` carries its gradient rule on itself and the rule is
    what has to know which label it is recording.
    """
    import mlx.core as mx

    @mx.custom_function
    def mark(*tensors):
        mx.eval(*tensors)
        recorder.entries.append((region, end, FORWARD, time.perf_counter()))
        return tensors[0] if len(tensors) == 1 else tensors

    @mark.vjp
    def mark_vjp(_primals, cotangents, _output):
        mx.eval(cotangents)
        recorder.entries.append((region, end, BACKWARD, time.perf_counter()))
        return cotangents

    return mark


# ---------------------------------------------------------------------------
# The four patch points
# ---------------------------------------------------------------------------
# Each is a name looked up at call time, so replacing it is reached. Two of
# them live on a class, which is why `seams.Seam` takes a dotted path.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Region:
    """One named part of the step, and where to bracket it."""

    name: str
    seam: seams.Seam
    wrapper: str  # which bracketing shape below applies


REGIONS = {
    "attn-core": Region(
        "attn-core",
        seams.Seam("mlx_lm.models.qwen3", "scaled_dot_product_attention",
                   defined_in="mlx_lm.models.base"),
        "attention"),
    "qmm": Region(
        "qmm",
        seams.Seam("mlx.nn", "QuantizedLinear.__call__",
                   defined_in="mlx.nn.layers.quantized"),
        "quantized_layer"),
    "head-matmul": Region(
        "head-matmul",
        seams.Seam("mlx.nn", "QuantizedEmbedding.as_linear",
                   defined_in="mlx.nn.layers.quantized"),
        "quantized_layer"),
    "cross-entropy": Region(
        "cross-entropy",
        seams.Seam("mlx.nn", "losses.cross_entropy",
                   defined_in="mlx.nn.losses"),
        "first_argument"),
}

# Which regions each candidate's share is built from, section 3.3 as amended.
# `head-matmul` appears twice on purpose: the tied output head IS a quantized
# matmul, so it counts in both, and Amendment 5 records that the two shares
# overlap and must never be composed.
CANDIDATE_REGIONS = {
    "A": ("attn-core",),
    "L": ("head-matmul", "cross-entropy"),
    "Q": ("qmm", "head-matmul"),
}


def _quantized_shape(layer) -> tuple[int, int]:
    """The logical (out, in) of a quantized layer, unpacked from its storage.

    A quantized weight is stored packed, so its second dimension is the packed
    width rather than the number of inputs; the same arithmetic mlx-lm's own
    LoRA wrapper uses recovers the logical one.
    """
    out_dims, packed = layer.weight.shape
    return out_dims, packed * 32 // layer.bits


def _wrap_attention(recorder: Recorder, region: str):
    enter, exit_ = (make_mark(recorder, region, ENTER),
                    make_mark(recorder, region, EXIT))

    def wrap(original):
        def call(queries, keys, values, *args, **kwargs):
            queries, keys, values = enter(queries, keys, values)
            return exit_(original(queries, keys, values, *args, **kwargs))
        return call

    return wrap


def _wrap_quantized_layer(recorder: Recorder, region: str):
    enter, exit_ = (make_mark(recorder, region, ENTER),
                    make_mark(recorder, region, EXIT))

    def wrap(original):
        def call(self, x, *args, **kwargs):
            recorder.note_shape(region, _quantized_shape(self))
            return exit_(original(self, enter(x), *args, **kwargs))
        return call

    return wrap


def _wrap_first_argument(recorder: Recorder, region: str):
    enter, exit_ = (make_mark(recorder, region, ENTER),
                    make_mark(recorder, region, EXIT))

    def wrap(original):
        def call(first, *args, **kwargs):
            return exit_(original(enter(first), *args, **kwargs))
        return call

    return wrap


_WRAPPERS = {"attention": _wrap_attention,
             "quantized_layer": _wrap_quantized_layer,
             "first_argument": _wrap_first_argument}


def install(names, recorder: Recorder) -> seams.Installation:
    """Bracket each named region, or change nothing.

    Installation goes through `metalrunner.seams`, which refuses a seam that
    something else has already replaced. That refusal is what proves the
    profile measured stock: a process carrying metalrunner's own kernels, or
    anyone else's patch, cannot be marked at all.
    """
    unknown = sorted(set(names) - set(REGIONS))
    if unknown:
        raise RunInvalid(f"no such region: {unknown}")
    installation = seams.Installation()
    try:
        for name in names:
            region = REGIONS[name]
            wrap = _WRAPPERS[region.wrapper](recorder, name)
            installation.install(region.seam, wrap)
    except Exception:
        installation.remove()
        raise
    return installation


# ---------------------------------------------------------------------------
# Reading the log: pure arithmetic, no MLX, no device
# ---------------------------------------------------------------------------
def counts(entries) -> dict[str, dict[str, int]]:
    """How often each region's span completed, per direction.

    A region whose backward never fired reports zero there, and zero is a
    statement: under LoRA the blocks below the first adapted one genuinely
    have no backward, while a region that should have one and does not is a
    seam that was installed and never reached.
    """
    tally: dict[str, dict[str, int]] = {}
    for region, end, direction, _ in entries:
        if end != _OPENS[direction]:
            continue
        tally.setdefault(region, {FORWARD: 0, BACKWARD: 0})[direction] += 1
    return tally


def decompose(entries, step_start: float, step_end: float, *,
              context) -> dict:
    """Per-region spans, the unattributed remainder, and what they rest on.

    `context` identifies the workload this pass ran - the cell, the batch, the
    width. It is carried rather than checked here, and `candidate_share`
    refuses to divide across two contexts that differ, which is the only thing
    standing between a split-pass profile and a share whose numerator and
    denominator describe different steps.

    Refuses rather than reporting a plausible number: an unopened close, an
    unclosed open, spans that overlap, or a span reaching outside the step all
    mean the log does not describe the step it claims to.
    """
    elapsed = step_end - step_start
    if elapsed <= 0.0:
        raise RunInvalid("a step with no measured time cannot be decomposed")

    spans: dict[str, dict[str, float]] = {}
    open_span: tuple[str, str, float] | None = None
    for region, end, direction, stamp in sorted(entries, key=lambda e: e[3]):
        if not step_start <= stamp <= step_end:
            raise RunInvalid(
                f"{region}.{end} fired at a time outside the step it "
                f"describes, so the log and the step are not the same run")
        if end == _OPENS[direction]:
            if open_span is not None:
                held, held_direction, _ = open_span
                raise RunInvalid(
                    f"{region}.{end} opened while {held} was still open in "
                    f"{held_direction}: nested regions would be counted twice")
            open_span = (region, direction, stamp)
            continue
        if open_span is None:
            raise RunInvalid(f"{region}.{end} closed a span nothing opened")
        held, held_direction, started = open_span
        if held != region or held_direction != direction:
            raise RunInvalid(
                f"{region}.{end} in {direction} closed a span opened by "
                f"{held} in {held_direction}")
        by_direction = spans.setdefault(region, {FORWARD: 0.0, BACKWARD: 0.0})
        by_direction[direction] += stamp - started
        open_span = None
    if open_span is not None:
        held, held_direction, _ = open_span
        raise RunInvalid(
            f"{held} opened in {held_direction} and never closed, so its span "
            f"has no end")

    covered = sum(total for by_direction in spans.values()
                  for total in by_direction.values())
    return {
        "regions": {region: dict(by_direction)
                    for region, by_direction in sorted(spans.items())},
        "totals": {region: by_direction[FORWARD] + by_direction[BACKWARD]
                   for region, by_direction in sorted(spans.items())},
        "covered": covered,
        "remainder": elapsed - covered,
        "elapsed": elapsed,
        "counts": counts(entries),
        "context": context,
    }


def candidate_share(decomposed, candidate: str, denominator) -> float:
    """One candidate's share f, over a denominator taken from another pass.

    The denominator is the plain step's total rather than this pass's own, so
    a share carries only the cost of its own marks. Marking every region at
    once and dividing by that pass's inflated total would deflate the lightly
    marked regions and inflate the heavily marked one, which is exactly the
    comparison the selection rule makes.

    Because the two numbers come from different passes, `denominator` carries
    the context it was measured in and this refuses unless the two agree. A
    numerator measured on one batch over a denominator measured on another is
    a ratio of two workloads, and nothing downstream could tell.
    """
    if candidate not in CANDIDATE_REGIONS:
        raise RunInvalid(f"not a candidate of this pre-registration: "
                         f"{candidate!r}")
    try:
        total = denominator["total"]
        other = denominator["context"]
    except (TypeError, KeyError):
        raise RunInvalid(
            "a denominator must carry the context it was measured in, as "
            "{'total': seconds, 'context': ...}; a bare number cannot be "
            "checked against the pass it is dividing") from None
    if other != decomposed["context"]:
        raise RunInvalid(
            f"the denominator was measured in {other!r} and this pass ran in "
            f"{decomposed['context']!r}: dividing one by the other would be a "
            f"ratio of two workloads")
    if total <= 0.0:
        raise RunInvalid("a share needs a positive step total to divide by")
    totals = decomposed["totals"]
    missing = [name for name in CANDIDATE_REGIONS[candidate]
               if name not in totals]
    if missing:
        raise RunInvalid(
            f"candidate {candidate} needs {missing} and this pass did not "
            f"measure them; a region absent from the log is not a region "
            f"that took no time")
    return sum(totals[name] for name in CANDIDATE_REGIONS[candidate]) / total
