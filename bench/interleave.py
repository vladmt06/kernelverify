"""The shared interleaved-timing engine for every bench A/B on the GPU.

One copy of the discipline the pack gates and the e2e spike all follow, so a
timing rule amended in one gate cannot silently stay old in another:

- every sample is a batched dispatch sized to at least MIN_SAMPLE_MS, because
  anything measured below a millisecond is reporting the power manager, not
  the kernel (bench/machine_state.py carries that rule's provenance);
- arms are sampled interleaved within a round, so a power-state excursion
  hits both arms rather than whichever ran first;
- interleaving equalises a clock excursion across the arms but cannot detect
  one, so the reference arm's own spread is the detector: rows whose spread
  exceeds MAX_CANARY_SPREAD are withheld rather than published. Twice a run
  has reported a kernel "win" that was the machine's clock moving under it
  (MLX's own time for one fixed shape moved 2.9x inside a single interleaved
  round while nothing about MLX changed).

The timed region is sacred: `dispatch` builds the lazy outputs, then times
exactly one `mx.eval` plus `mx.synchronize`, the same operations in the same
order as every gate it replaced.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

# Below this, a single sample is measuring the power manager (see
# bench/machine_state.py TIMING_FLOOR_MS for why the floor exists at all;
# this is the batching target that keeps every sample safely above it).
MIN_SAMPLE_MS = 5.0

# The same QUANTITY runners/compare.py rejects rounds on: decision D6's
# reference-arm max/min per round, which is why both read 1.5 today. Held as a
# literal rather than imported from DEFAULT_SPREAD_LIMIT, because D6 sets the
# limit per class - 1.5x for kernel arms, tighter for steadier quantities - so
# tightening the live comparator would otherwise silently re-gate every pack
# certificate this constant publishes. tests/test_interleave.py asserts the two
# are equal today, so a deliberate divergence is a test to update rather than a
# behaviour change nobody sees.
MAX_CANARY_SPREAD = 1.5


def dispatch(build_one, copies: int) -> float:
    """One batched dispatch: `copies` lazy outputs, one timed eval+sync."""
    outs = [build_one(i) for i in range(copies)]
    t0 = time.perf_counter()
    mx.eval(outs)
    mx.synchronize()
    return time.perf_counter() - t0


def calibrate_copies(sample, start: int = 8, cap: int = 4096) -> int:
    """Fewest copies per dispatch that still reach MIN_SAMPLE_MS."""
    copies = start
    while copies < cap and sample(copies) * 1e3 < MIN_SAMPLE_MS:
        copies *= 2
    return copies


def arms_agree(ours, theirs) -> bool:
    """Do the two timed arms compute the same thing, to 5e-3 absolute-or-
    relative against the reference arm's peak? A smoke check on the timing
    comparison's fairness, not an oracle verdict - the gates' correctness
    claims come from kernelverify.pack.verify, never from this."""
    a = np.array(ours).astype(np.float64)
    b = np.array(theirs).astype(np.float64)
    return float(np.max(np.abs(a - b))) <= 5e-3 * max(1.0, float(np.max(np.abs(b))))

def interleaved_arms(builders, rounds: int, guard=None, *,
                     rotate: bool = True) -> dict:
    """Per-round times of n labelled arms, sampled interleaved every round.

    ``builders`` maps a label to a ``build_one(i)`` callable, and its order is
    the arm order; a mapping rather than a list because two arms sharing a
    label would silently overwrite one another's samples.

    Interleaving is load-bearing and NOT sufficient (AGENTS.md): it equalizes
    a clock excursion across the arms but cannot detect one, so the caller
    must still gate on a reference arm's own spread. Every arm is warmed
    first, and ONE dispatch batch is calibrated on the FIRST arm and used by
    all of them, because per-arm batches would make the per-round times
    incomparable, which is the whole point of sampling them together.

    ``rotate`` starts each round at a different arm so no arm always runs
    first, which is where allocation and cache effects land. It defaults on
    for the n-arm path and is turned OFF by the two-arm wrapper below, whose
    fixed order is what every published pack certificate was measured under.

    ``guard``, when given, is called once per round with a cell label and may
    refuse by raising (the pricing probe's memory checks). The seam lives HERE
    because this loop is shared and a diverged sampler copy is the ADR 0004
    two-halves mistake. ``rounds`` is required rather than defaulted: each
    harness pins its own round count as a measurement parameter, and a default
    here would let one of them drift onto a number it never registered.
    """
    labels = list(builders)
    if not labels:
        raise ValueError("interleaved_arms needs at least one arm")
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, not {rounds}")
    mx.eval([builders[label](0) for label in labels])
    mx.synchronize()
    first = builders[labels[0]]
    copies = calibrate_copies(lambda c: dispatch(first, c))
    samples: dict = {label: [] for label in labels}
    for i in range(rounds):
        if guard is not None:
            guard(f"round {i + 1}/{rounds}")
        start = i % len(labels) if rotate else 0
        for label in labels[start:] + labels[:start]:
            samples[label].append(dispatch(builders[label], copies) / copies)
    return samples


def interleaved_samples(build_a, build_b, rounds: int,
                        guard=None) -> tuple:
    """The two-arm case, in arm order A then B in every round.

    A thin wrapper over ``interleaved_arms`` so there is one sampler rather
    than two: the pack gates, the pricing probe and the knob ladders all get
    the same warm-up, the same single calibrated batch and the same guard
    seam. Rotation is OFF here on purpose: every certificate this function has
    published was measured with B always following A, and turning rotation on
    would change what those gates measure without anyone asking for it.
    """
    samples = interleaved_arms({"a": build_a, "b": build_b}, rounds, guard,
                               rotate=False)
    return samples["a"], samples["b"]
