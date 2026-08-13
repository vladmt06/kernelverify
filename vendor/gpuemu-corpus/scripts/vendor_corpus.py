#!/usr/bin/env python3
"""Vendor the validated corpus from a gpuemu-paper checkout into this package.

Run this script at release time (and during local dev) to populate
``gpuemu_corpus/data/`` from ``../gpuemu-paper/corpus/``. The canonical research
source remains the gpuemu-paper repo; this package only mirrors the data files.

Usage:
    python3 scripts/vendor_corpus.py [--paper-repo ../gpuemu-paper]

The vendor step is intentionally a script rather than a build hook so the
gpuemu-paper repo is never an implicit dependency of `pip install`. CI runs
this script before `python -m build`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def vendor(paper_repo: Path, out: Path) -> int:
    src = paper_repo / "corpus"
    if not src.is_dir():
        print(f"ERROR: {src} does not exist", flush=True)
        return 1
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(
        src,
        out,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            ".pytest_cache",
            "gpuemu.toml",  # regenerated from meta.json at runtime
            "build_toml.py",  # only used in the paper repo
            "_artifacts",  # large PTX samples; not part of the user-facing corpus
        ),
    )
    # Count what we just vendored.
    metas = sorted(out.glob("*/meta.json"))
    print(f"vendored {len(metas)} ops into {out}", flush=True)
    for m in metas:
        print(f"  - {m.parent.name}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--paper-repo",
        default=str(Path(__file__).resolve().parents[2] / "gpuemu-paper"),
        help="Path to the gpuemu-paper checkout (default: ../gpuemu-paper)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Override the vendor destination (default: gpuemu_corpus/data)",
    )
    args = ap.parse_args()
    out = (
        Path(args.out)
        if args.out is not None
        else Path(__file__).resolve().parents[1] / "gpuemu_corpus" / "data"
    )
    return vendor(Path(args.paper_repo), out)


if __name__ == "__main__":
    raise SystemExit(main())
