"""Model-call instrumentation shared by serving harnesses."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import mlx.core as mx


@dataclass
class _CallCounter:
    shapes: list[tuple[int, ...]] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)


@contextmanager
def counting_calls(target, *, sync: bool = False):
    """Count calls to one model while leaving same-class models untouched."""
    model_type = type(target)
    original = model_type.__call__
    counter = _CallCounter()

    def counted(self, inputs, *args, **kwargs):
        if self is not target:
            return original(self, inputs, *args, **kwargs)

        counter.shapes.append(tuple(inputs.shape))
        if not sync:
            return original(self, inputs, *args, **kwargs)

        started = time.perf_counter()
        result = original(self, inputs, *args, **kwargs)
        mx.eval(result)
        counter.seconds.append(time.perf_counter() - started)
        return result

    model_type.__call__ = counted
    try:
        yield counter
    finally:
        model_type.__call__ = original
