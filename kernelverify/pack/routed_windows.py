"""Where the wide-tile qmv is routed: per shape, per bit width, from measurement.

The routing decision used to be one pair of numbers, MIN/MAX_PROFITABLE_M =
5..11, applied to every shape at every bit width. That was a default (ruling
D3.1) standing in for a measurement that had not been made. The measurement
now exists - bench/results/qmv-boundary-pricing-2026-08-15.json, 96 points,
the six Qwen3-4B decode dispatch shapes x M = 1..16 at 3 bits - and it does
not agree with the default anywhere:

- every shape LOSES at M = 10 or 11, so the old upper bound of 11 was
  shipping a regression at two widths on five of six shapes;
- lm_head wins one width further than the rest (M = 10), because at 151936
  rows MLX's second weight pass costs more than our single pass for longer;
- the widths themselves are not the same set at every shape, which no single
  pair of bounds can express.

So the table is keyed (bits, d_out, d_in) and holds the SET of winning tile
widths, not a (lo, hi) pair. A set can represent a hole in the middle of a
window; a pair either crashes on one or quietly lies about it. Nothing here
is written by hand except the exclusions and the pins - the cells are derived
from the recording at import, so the shipped routing and the recorded
evidence cannot drift apart.

A window is a claim about ONE kernel body at ONE launch config, priced from
ONE recording. All three are pinned below as literals and a consumer asserts
them before trusting a cell: the same M at a different R, or against a
different kernel body, was never measured. The pins are literals rather than
values recomputed at import precisely so that moving any of the three fails
loudly (tests/test_pack_routed_windows.py) instead of silently re-deriving a
table that describes something nobody ran.

This module imports nothing from the package, deliberately: wide_qmv consults
it, so a dependency back on wide_qmv would be a cycle. The kernel-source and
launch-config pins are therefore data here, checked against wide_qmv itself
by the test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

RECORDING_PATH = (Path(__file__).resolve().parents[2] / "bench" / "results"
                  / "qmv-boundary-pricing-2026-08-15.json")

# sha256 of the recording these windows were derived from. Checked at import,
# so re-running the probe over the same filename cannot silently re-derive the
# table: the derivation is re-read and re-pinned on purpose or not at all.
RECORDING_SHA256 = "4a7c500fce9c43c86297ea74f6a8c5bc131c119cca101c80bcb8e1f284330ee9"

# sha256 of kernelverify.pack.wide_qmv.WIDE_QMV_MSL, the kernel body the
# recording priced. A window says nothing about an edited kernel; the test
# re-hashes the live source against this.
KERNEL_SOURCE_SHA256 = "43c9533c507aad9c55af91cbc708a923858b91d841b97966ba9bc6b01c2c873e"

# R (rows per simdgroup) at every tile width the recording swept, as data
# rather than as a second copy of wide_qmv.rows_per_simdgroup. Every point of
# the recording is checked against it at import, and wide_qmv's own function
# is checked against it by the test, so the priced launch config and the
# shipped one are the same one.
ROWS_PER_SIMDGROUP = MappingProxyType({
    1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4,
    9: 4, 10: 4, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2,
})

# Ruling D2 (Vlad, 2026-08-15). lm_head M=4 came back WIN (interval
# [1.018, 1.028]) and is still NOT routed. The pre-registration this run was
# read under (bench/price_qmv_boundary.py, ruling D3, written before any
# number existed) permits the routed window to NARROW only: an unevidenced
# cell is dropped, a new cell is not added. Every window below is strictly
# inside the old 5..11 default, so adopting them is pure narrowing and needs
# no new authority; M=4 is outside it, so adopting that cell would be reading
# a widening out of a rule that pre-registered narrowing. It becomes routable
# through the widening rule now pre-registered in the probe's docstring:
# WIN again in a later run, contiguous with this set, and adopted together
# with gate coverage of the cell.
EXCLUDED_CELLS = MappingProxyType({(3, 151936, 2560): frozenset({4})})


def _load() -> dict:
    """The recording, refusing to be read as anything but the pinned bytes."""
    raw = RECORDING_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != RECORDING_SHA256:
        raise ValueError(
            f"{RECORDING_PATH.name} has changed (sha256 {digest}, pinned "
            f"{RECORDING_SHA256}): the routed windows describe the pinned "
            "recording, so re-derive the table and re-pin it deliberately")
    return json.loads(raw)


def _derive(record: dict) -> tuple[dict, dict]:
    """The priced shapes and the routed cells, read out of the recording.

    A cell routes when the probe judged it WIN, which by its own rule
    (bench/price_qmv_boundary.py, classify) means the whole ratio interval
    cleared 1.0 from above - ratio_lo > 1 - on a round whose reference arm
    held steady. Both halves are required here: a REJECTED round can carry a
    ratio_lo above 1 and still describe the machine rather than the kernels.
    LOSS and REFUSED cells route nowhere; REFUSED in particular is undecided,
    not measured-bad.

    A shape that was priced and won nowhere stays in `priced` and drops out of
    `windows`, so "0 of 6 shapes" can still name its denominator.
    """
    bits = record["bits"]

    priced: dict[int, list[tuple[int, int]]] = {}
    windows: dict[tuple[int, int, int], set[int]] = {}
    for point in record["points"]:
        m, r = point["m"], point["r"]
        if ROWS_PER_SIMDGROUP[m] != r:
            raise ValueError(
                f"{point['site']} M={m} was priced at R={r}, but the pinned "
                f"launch config launches R={ROWS_PER_SIMDGROUP[m]}: the "
                "recording does not describe the shipped kernel")
        shape = (point["d_out"], point["d_in"])
        shapes = priced.setdefault(bits, [])
        if shape not in shapes:
            shapes.append(shape)
        window = windows.setdefault((bits, *shape), set())
        if point["verdict"] == "WIN" and point["ratio_lo"] > 1.0:
            window.add(m)
    for key, excluded in EXCLUDED_CELLS.items():
        windows[key] -= excluded
    return ({b: tuple(s) for b, s in priced.items()},
            {key: frozenset(ms) for key, ms in windows.items() if ms})


_RECORD = _load()
RECORDING_DATE = _RECORD["date"]
_PRICED_SHAPES, _WINDOWS = _derive(_RECORD)

# Every (d_out, d_in) the recording swept, per bit width, whether it won
# anywhere or not: the denominator of any "routed at N shapes" statement.
PRICED_SHAPES = MappingProxyType(_PRICED_SHAPES)
ROUTED_WINDOWS = MappingProxyType(_WINDOWS)

PROVENANCE = (f"bench/results/{RECORDING_PATH.name} ({RECORDING_DATE}, "
              f"sha256 {RECORDING_SHA256[:12]})")


def window_for(bits: int, d_out: int, d_in: int) -> frozenset[int]:
    """The tile widths measured profitable at this shape and width; empty for
    an unpriced shape or bit width, which is the honest reading of no
    evidence rather than a default."""
    return ROUTED_WINDOWS.get((bits, d_out, d_in), frozenset())


def dispatch_boundary_prose(bits: int, m: int) -> str:
    """One sentence stating where this (bits, M) specialization is routed.

    Single-sourced here so the certificate emitter cannot describe a boundary
    the pack does not enforce: its prose predated even the old upper bound and
    still claimed an open-ended `M >= 5` after the window had closed at 11.
    """
    head = "should_dispatch(m, bits, d_out, d_in) is per-shape and per-width"
    tail = ("a routing boundary, not a correctness bound - the certificate's "
            "correctness claim holds wherever this specialization is launched")
    shapes = PRICED_SHAPES.get(bits, ())
    if not shapes:
        return (f"{head}: {bits}-bit is unpriced, so it routes nowhere and MLX "
                "serves every shape at this width until its own pricing run "
                f"lands; {tail}")
    routed = [f"{d_out}x{d_in}" for d_out, d_in in shapes
              if m in window_for(bits, d_out, d_in)]
    where = (f"{len(routed)} of the {len(shapes)} priced dispatch shapes "
             f"({', '.join(routed)})" if routed
             else f"none of the {len(shapes)} priced dispatch shapes")
    return (f"{head}: at BITS={bits}, M={m} it routes here at {where} and "
            f"nowhere else, per {PROVENANCE}; {tail}")
