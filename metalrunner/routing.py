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

import os
from dataclasses import dataclass

# The control arm's switch. An end-to-end measurement needs a third arm that
# installs this wrapper and then routes nothing, so that the wrapper's own
# host cost can be measured against stock separately from any kernel's
# effect. It is an environment variable rather than a flag because the
# argument surface is mlx-lm's and must stay identical to it; and it is
# reported in the routing report and recorded in the receipt, so a control
# run can never be mistaken afterwards for a real one.
FORCE_STOCK_ENV = "METALRUNNER_FORCE_STOCK"

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
# the keep stage writes it from the committed recording, and its own tests
# check what it writes against metalrunner.measurement.entry_problems, which
# is the schema these entries answer to.
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


def forced_to_stock() -> bool:
    """Is this process a control arm? Read from the environment each call, so
    a harness can set it per subprocess without importing anything."""
    return os.environ.get(FORCE_STOCK_ENV, "") not in ("", "0")


def eligible(fine_tune_type: str, bits: int | None, group_size: int | None,
             *, on_chip: str | None = None,
             certified: tuple | None = None) -> tuple:
    """The certified entries this run may route, and nothing else.

    Every condition is a decline rather than an assumption. An unreadable
    quantization returns nothing, because routing a kernel verified for one
    format onto another is the guess this package exists not to make; a
    chip that certified nothing returns nothing, for the same reason.

    This is what both callers derive their candidate list from, so the user
    path and the measurement path can never disagree about what is routable
    on this machine.
    """
    entries = CERTIFIED if certified is None else certified
    if fine_tune_type not in SUPPORTED_FINE_TUNE_TYPES:
        return ()
    if bits is None or group_size is None:
        return ()
    if bits not in SUPPORTED_BITS or group_size != SUPPORTED_GROUP_SIZE:
        return ()
    if not entries:
        return ()
    where = chip() if on_chip is None else on_chip
    return tuple(entry for entry in entries
                 if entry["chip"] == where and entry["bits"] == bits
                 and entry["group_size"] == group_size)


def decide(fine_tune_type: str, bits: int | None, group_size: int | None,
           *, on_chip: str | None = None, force_stock: bool | None = None,
           certified: tuple | None = None) -> list[Decision]:
    """One decision per known operation, each carrying its own reason.

    `bits` and `group_size` are None when the model's quantization could not
    be read; that is a decline rather than an assumption, because routing a
    kernel verified for 4-bit group-64 onto something else is exactly the
    kind of guess this package exists not to make.

    Under force-stock every operation declines for that reason and no other,
    so the decline is legible as the control it is rather than looking like
    an ordinary lack of coverage.
    """
    where = chip() if on_chip is None else on_chip
    forced = forced_to_stock() if force_stock is None else force_stock
    routable = {} if forced else {
        entry["operation"]: entry
        for entry in eligible(fine_tune_type, bits, group_size,
                              on_chip=where, certified=certified)
    }
    decisions = []
    for operation, _description in KNOWN_OPERATIONS:
        if forced:
            decisions.append(Decision(
                operation, False,
                f"forced to stock by {FORCE_STOCK_ENV}: this is a control run"))
        elif operation in routable:
            decisions.append(Decision(operation, True,
                                      "certified and priced here"))
        else:
            decisions.append(Decision(
                operation, False,
                _why_not(operation, fine_tune_type, bits, group_size, where,
                         CERTIFIED if certified is None else certified)))
    return decisions


def _why_not(operation: str, fine_tune_type: str, bits, group_size,
             where: str, entries: tuple) -> str:
    """Why this operation is not routed, in the caller's own words.

    The entries are passed in rather than read from the module so the reason
    describes the table the decision was actually made against; a reason
    that named a different table would be a decline nobody could check.
    """
    if fine_tune_type not in SUPPORTED_FINE_TUNE_TYPES:
        return f"{fine_tune_type} fine-tuning is not supported"
    if bits is None or group_size is None:
        return ("the model's quantization could not be read, and a kernel "
                "verified for one format must not be routed onto another")
    if bits not in SUPPORTED_BITS or group_size != SUPPORTED_GROUP_SIZE:
        return (f"this model is {bits}-bit group-{group_size}; only "
                f"4-bit group-64 is verified")
    if not entries:
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
    if forced_to_stock():
        lines.append(f"  CONTROL RUN: {FORCE_STOCK_ENV} is set, so every "
                     "routing decision was forced to stock. This run measures "
                     "the wrapper's own cost and nothing else.")
    elif not routed:
        lines.append("  This run is stock mlx-lm. metalrunner changed nothing "
                     "about the training itself and makes no speed claim; it "
                     "checked the stack and will write a receipt.")
    return "\n".join(lines)


def _quant(bits, group_size) -> str:
    if bits is None or group_size is None:
        return "unknown"
    return f"{bits}-bit group-{group_size}"
