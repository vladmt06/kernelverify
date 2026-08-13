"""One-off probe: where exactly are the B=8 bottleneck faults detectable?

Reads the cached verdict table and prints, for each fault of interest, the
breakdown of detecting cases by input mode, dtype, and dims. The point is to
see whether the misses live at single features (a mode the core skips) or at
feature *pairs* (mode x shape), which decides between fixing the core's
feature list and moving to pairwise coverage.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.score_oracles import build_verdicts  # noqa: E402

FAULTS = [
    "flash_attention[init_max_zero=True]",
    "l2norm[eps=1e-06]",
    "rmsnorm[eps=0.0]",
    "rmsnorm[eps_outside=True]",
    "softmax[read_past_row=True]",
]


def main() -> int:
    table = build_verdicts()["table"]
    for name in FAULTS:
        verdicts = table[name]
        hits = [c for c, hit in verdicts.items() if hit]
        total = len(verdicts)
        print(f"\n{'=' * 72}\n{name}: {len(hits)}/{total} detecting cases")

        by_mode = Counter(c.distribution for c in hits)
        by_dtype = Counter(c.dtype for c in hits)
        by_pair = Counter((c.distribution, c.dtype) for c in hits)
        print(f"  by mode : {dict(by_mode)}")
        print(f"  by dtype: {dict(by_dtype)}")
        print(f"  by (mode, dtype): {dict(by_pair)}")

        # Per-dimension value histogram of the detecting cases.
        dim_values: dict[str, Counter] = {}
        for c in hits:
            for dim, value in c.dims:
                dim_values.setdefault(dim, Counter())[value] += 1
        for dim, counter in sorted(dim_values.items()):
            print(f"  dim {dim}: {dict(sorted(counter.items()))}")

        # And the full case list when it is small enough to read.
        if len(hits) <= 12:
            for c in sorted(hits, key=lambda c: (c.distribution, c.dtype, c.dims)):
                print(f"    {dict(c.dims)} {c.dtype} {c.distribution} seed={c.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
