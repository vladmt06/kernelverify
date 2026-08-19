"""Where candidates come from, before any language model is involved.

The loop must never starve. A generator that depends on a model being
reachable is a loop that stops when a key expires, so the first generator is
parametric: one hand-written body with holes in it, swept across every knob
setting the chip can actually launch.

That is a complete generator in its own right, not a placeholder. Tile width,
simdgroups per threadgroup and unroll depth are exactly the knobs a human
tunes by hand, and sweeping them through the same funnel and the same store as
anything else means the loop can be run, measured and trusted before a model
is added to it.

Two things it deliberately does not do. It does not invent bodies, so it can
only search inside what the seed already expresses. And it does not skip a
setting for looking unpromising; `search_space` prunes only what the device
cannot launch, and the clock decides the rest.
"""

from __future__ import annotations

from collections.abc import Iterator

from kernelverify.compiler.loop import Proposal
from kernelverify.compiler.search_space import Limits, Space, enumerate_knobs
from kernelverify.runners.specialize import specialize


def describe(knobs: dict) -> str:
    return " ".join(f"{name}={value}" for name, value in sorted(knobs.items()))


def sweep(template, axes: dict[str, list], limits: Limits, *, threads,
          threadgroup_bytes, launch_for=None,
          origin: str = "knob-sweep") -> tuple[Iterator[Proposal], Space]:
    """Every launchable knob setting of one seed body, as proposals.

    Returns the proposals and the space they came from, so a session can say
    what it enumerated and what the device refused as well as what it tried.
    `launch_for` maps a knob setting to the launch configuration that goes
    with it, for knobs that change the threadgroup rather than the source.
    """
    space = enumerate_knobs(axes, limits, threads=threads,
                            threadgroup_bytes=threadgroup_bytes)

    def proposals() -> Iterator[Proposal]:
        for knobs in space.launchable:
            spec = specialize(template, {name: str(value)
                                         for name, value in knobs.items()})
            extra = {"launch": launch_for(knobs)} if launch_for else {}
            yield Proposal(source=spec.source,
                           origin=f"{origin} {describe(knobs)}", extra=extra)

    return proposals(), space
