"""Which knob settings this machine can actually launch, and why the rest cannot.

A generated kernel body is one thing; the tile widths, simdgroup layout,
unroll factors and threadgroup staging it could be built with are a space,
and most of that space will not run. A threadgroup asking for more threads
than the device allows, or more threadgroup memory than it has, fails at
launch and tells you nothing about the kernel.

Two rules, and deliberately only two. Both are properties of the device, read
from the device:

    threads per threadgroup   <= max_threads_per_threadgroup
    threadgroup memory bytes  <= max_threadgroup_memory

Nothing is dropped for being slow, awkward, or unlikely to win. A search that
quietly removed settings on a hunch would report a clean sweep of a space it
had narrowed by taste, and the number that came out would describe a
different search than the one described. Settings that are merely inefficient
stay in and lose on the clock, which is a measurement rather than an opinion.

Every rejection is kept with its reason, so a session can say it enumerated
240 settings and could launch 96, rather than reporting 96 and calling it the
space.

The device limits are never written down here. They are per chip: this
machine reports 32768 bytes and 1024 threads, and the next one need not, so
they are read through the runner's own probe and a machine with no device
refuses rather than assuming a plausible pair of numbers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass


class NoDevice(RuntimeError):
    """There is no GPU to read limits from, and there is no default pair."""


@dataclass(frozen=True)
class Limits:
    """What this chip can launch, as this chip reports it."""

    max_threads_per_threadgroup: int
    max_threadgroup_memory: int

    @staticmethod
    def probe(runner) -> "Limits":
        device = runner.probe()
        if device is None:
            raise NoDevice(
                "no Metal device: launchability is a property of the chip, so "
                "there is nothing to fall back to")
        return Limits(
            max_threads_per_threadgroup=device.max_threads_per_threadgroup,
            max_threadgroup_memory=device.max_threadgroup_memory)


@dataclass(frozen=True)
class Rejected:
    knobs: dict
    reason: str


@dataclass(frozen=True)
class Space:
    launchable: tuple[dict, ...]
    rejected: tuple[Rejected, ...]

    @property
    def enumerated(self) -> int:
        return len(self.launchable) + len(self.rejected)

    @property
    def census(self) -> dict[str, int]:
        """Counts by reason, plus the survivors, so a report can state what it
        swept and what it never tried."""
        counts = {"launchable": len(self.launchable)}
        for entry in self.rejected:
            counts[entry.reason] = counts.get(entry.reason, 0) + 1
        return counts


def enumerate_knobs(axes: dict[str, list], limits: Limits, *, threads,
                    threadgroup_bytes) -> Space:
    """Walk the whole product of `axes` and split it by what will launch.

    `threads` and `threadgroup_bytes` map one knob setting to the two device
    quantities. Only the operation knows that mapping, so it supplies it;
    everything above this line is the same whichever operation is being tuned.

    The walk is over sorted axes, so two runs enumerate in the same order and
    a session's settings can be compared between runs.
    """
    if not axes:
        raise ValueError("an empty knob space has nothing to search")
    for axis, choices in sorted(axes.items()):
        if not choices:
            raise ValueError(f"knob axis {axis!r} offers no choices")

    names = sorted(axes)
    launchable, rejected = [], []
    for combination in itertools.product(*(axes[name] for name in names)):
        knobs = dict(zip(names, combination))
        reason = _why_not(knobs, limits, threads, threadgroup_bytes)
        if reason is None:
            launchable.append(knobs)
        else:
            rejected.append(Rejected(knobs=knobs, reason=reason))
    return Space(launchable=tuple(launchable), rejected=tuple(rejected))


def _why_not(knobs: dict, limits: Limits, threads, threadgroup_bytes) -> str | None:
    count = threads(knobs)
    if count < 1:
        return "a threadgroup needs at least one thread"
    if count > limits.max_threads_per_threadgroup:
        return (f"threadgroup of {count} threads is over this device's "
                f"{limits.max_threads_per_threadgroup}")

    memory = threadgroup_bytes(knobs)
    if memory < 0:
        return "threadgroup memory cannot be negative"
    if memory > limits.max_threadgroup_memory:
        return (f"{memory} bytes of threadgroup memory is over this device's "
                f"{limits.max_threadgroup_memory}")
    return None
