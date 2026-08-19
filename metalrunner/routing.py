"""What was swapped in, what was not, and why. Printed before training starts.

A user running metalrunner is entitled to know exactly what happened to
their training step, before it runs rather than after. So the first thing
the entry point prints is a routing report: every operation this package
knows how to replace, whether it was replaced on this machine for this
model, and if not, the reason in plain words.

Routing is not a preference. An operation routes only when a kernel exists
that was verified for this chip, this quantization and this operation, and
whose pricing recording says it wins there. Any of those missing is a
decline, and a decline is reported rather than silently passed over.

Today every operation declines, because no training kernel has been kept
yet: the sprint's first operation is selected by a measurement that has not
run. That is the honest state of the product, and printing it is better
than printing nothing. `CERTIFIED` is filled by the keep stage, from the
committed pricing recording, the way the decode routing table already is,
so shipping a kernel is a data change here rather than a code change.
"""

from __future__ import annotations

from dataclasses import dataclass

# The training operations this package knows how to replace. Named here so a
# decline can be specific about what did not happen, rather than silent.
KNOWN_OPERATIONS = (
    ("masked output head and cross-entropy",
     "streams the vocabulary projection and its loss over supervised rows only"),
    ("attention forward and backward",
     "tiled, grouped-query, causal, with a fused backward MLX does not implement"),
    ("quantized matmul at training widths",
     "the projections, retuned for training token counts rather than decode"),
)

# Kept kernels, each with the chip, quantization and operation it was
# certified and priced for. Empty until the first kernel survives pricing;
# the keep stage writes it from the committed recording.
CERTIFIED: tuple = ()

SUPPORTED_FINE_TUNE_TYPES = ("lora",)
SUPPORTED_BITS = (4,)
SUPPORTED_GROUP_SIZE = 64


@dataclass(frozen=True)
class Decision:
    operation: str
    routed: bool
    reason: str


def chip() -> str:
    """The GPU this is running on, read from MLX rather than assumed."""
    import mlx.core as mx

    try:
        return str(mx.device_info()["device_name"])
    except Exception:  # a machine with no Metal device still gets a report
        return "unknown"


def decide(fine_tune_type: str, bits: int | None, group_size: int | None,
           *, on_chip: str | None = None) -> list[Decision]:
    """One decision per known operation, each carrying its own reason.

    `bits` and `group_size` are None when the model's quantization could not
    be read; that is a decline rather than an assumption, because routing a
    kernel verified for 4-bit group-64 onto something else is exactly the
    kind of guess this package exists not to make.
    """
    where = chip() if on_chip is None else on_chip
    decisions = []
    for operation, _description in KNOWN_OPERATIONS:
        decisions.append(Decision(operation, False,
                                  _why_not(operation, fine_tune_type, bits,
                                           group_size, where)))
    return decisions


def _why_not(operation: str, fine_tune_type: str, bits, group_size,
             where: str) -> str:
    if fine_tune_type not in SUPPORTED_FINE_TUNE_TYPES:
        return f"{fine_tune_type} fine-tuning is not supported"
    if bits is None or group_size is None:
        return ("the model's quantization could not be read, and a kernel "
                "verified for one format must not be routed onto another")
    if bits not in SUPPORTED_BITS or group_size != SUPPORTED_GROUP_SIZE:
        return (f"this model is {bits}-bit group-{group_size}; only "
                f"4-bit group-64 is verified")
    for entry in CERTIFIED:
        if entry["operation"] == operation and entry["chip"] == where:
            return "certified and priced here"
    if not CERTIFIED:
        return ("no training kernel has been kept yet, so there is nothing "
                "certified to route")
    return f"no kernel certified for this operation on {where}"


def render(decisions: list[Decision], stack, *, model: str,
           fine_tune_type: str, bits, group_size) -> str:
    """The report, printed before the trainer starts."""
    routed = [d for d in decisions if d.routed]
    lines = [
        "metalrunner routing report",
        f"  chip           {chip()}",
        f"  mlx / mlx-lm   {stack.mlx} / {stack.mlx_lm} (verified)",
        f"  model          {model}",
        f"  quantization   {_quant(bits, group_size)}",
        f"  fine-tune      {fine_tune_type}",
        "",
    ]
    if routed:
        lines.append(f"  routed ({len(routed)}):")
        lines += [f"    {d.operation}: {d.reason}" for d in routed]
        lines.append("")
    lines.append(f"  not routed ({len(decisions) - len(routed)}):")
    lines += [f"    {d.operation}: {d.reason}"
              for d in decisions if not d.routed]
    lines.append("")
    if not routed:
        lines.append("  This run is stock mlx-lm. metalrunner changed nothing "
                     "about the training itself and makes no speed claim; it "
                     "checked the stack and will write a receipt.")
    return "\n".join(lines)


def _quant(bits, group_size) -> str:
    if bits is None or group_size is None:
        return "unknown"
    return f"{bits}-bit group-{group_size}"
