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
