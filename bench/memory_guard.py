"""The machine's memory-guard vocabulary, single-sourced (ruling D1).

Every heavy measurement harness refuses the same three ways - over its own
footprint budget, behind the machine-wide measurement lock, or into a
machine that has no room - and the 03:29 collapse on 2026-08-15 (a 39.5 GB
serving calibration and a 27.9 GB pricing probe Jetsam-killed together on a
36 GB machine) is what happens when each harness owns its own copy of that
vocabulary: per-harness numbering and per-harness locks let two "protected"
runs collapse the machine side by side. So the exit codes, the footprint
reader and the low-memory refusal live HERE, imported by the serving
calibration and the boundary-pricing probe alike, never copied. The
machine-wide lock itself lives in bench/machine_state.py with the other
machine-scope rules; this module carries the process- and memory-scope
vocabulary.

phys_footprint is the number Jetsam kills on. RSS is not it: the 03:29 run
self-reported 2.4 GB RSS while dying at a 39.5 GB footprint, because
compressed pages and IOKit/Metal memory charge the footprint without being
resident.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import subprocess

# Exit codes. 0 = attested, 1 = measured-and-stopped. Everything else is a
# gate with its own number, so the coordinator can tell a protective refusal
# from a measurement verdict without parsing stdout.
#
# 2 is NOT one of them, and used to be: it meant "no usable Metal device"
# while argparse also exits 2 on any bad argument, and the Python interpreter
# exits 2 on a failed import at startup. A child dying either of those ways
# was therefore reported as a missing GPU - the coordinator sent to look at
# the hardware for a typo. The device gate now has EXIT_NO_DEVICE and goes
# through the normal refusal path (checkpoint written, cell marked), leaving
# 2 to mean what the interpreter already makes it mean.
EXIT_BUDGET_REFUSAL = 3   # this process's phys_footprint crossed the budget
EXIT_LOCK_HELD = 4        # another heavy measurement holds the machine lock
EXIT_LOW_MEMORY = 5       # machine-wide available memory too low for a cell
EXIT_CHILD_DEATH = 6      # a measurement child died; its cell is named
EXIT_NO_DEVICE = 7        # no usable Metal device: nothing can be measured


def machine_ram_gb() -> float:
    """Total unified memory in decimal GB, from sysctl hw.memsize."""
    out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                         capture_output=True, text=True)
    return int(out.stdout.strip()) / 1e9


_RUSAGE_INFO_V4 = 4
_RUSAGE_V4_WORDS = 40   # 16-byte uuid + 35 uint64 fields, rounded up
_PHYS_FOOTPRINT_WORD = 9        # ri_phys_footprint: uuid is words 0-1
_LIFETIME_FOOTPRINT_WORD = 30   # ri_lifetime_max_phys_footprint
_libproc = None


def phys_footprint_gb() -> tuple[float, float]:
    """(current, lifetime peak) phys_footprint of THIS process, decimal GB.

    phys_footprint is the number Jetsam kills on (module docstring). The
    budget reads it via proc_pid_rusage (one ctypes call - cheap enough to
    check between every record, no per-check subprocess).

    Word offsets follow rusage_info_v4 in <libproc.h> (ri_uuid occupies
    words 0-1, ri_phys_footprint is word 9, ri_lifetime_max_phys_footprint
    word 30); test_footprint_reader_reports_this_process_truthfully pins
    them against a live allocation.
    """
    global _libproc
    if _libproc is None:
        _libproc = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)
    words = (ctypes.c_uint64 * _RUSAGE_V4_WORDS)()
    ret = _libproc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4,
                                   ctypes.byref(words))
    if ret != 0:
        raise RuntimeError(
            f"proc_pid_rusage failed with {ret} (errno {ctypes.get_errno()}); "
            f"a budget that cannot read the footprint must stop, not guess")
    return (words[_PHYS_FOOTPRINT_WORD] / 1e9,
            words[_LIFETIME_FOOTPRINT_WORD] / 1e9)


class BudgetExceeded(RuntimeError):
    """The footprint crossed the budget: the run refuses with a distinct
    exit code. What a refusal preserves is the harness's own policy (the
    serving calibration keeps its checkpoint; the pricing probe discards the
    run). Never a silently shrunk grid."""

    def __init__(self, cell: str, footprint_gb: float, budget_gb: float):
        super().__init__(
            f"footprint {footprint_gb:.2f} GB over budget "
            f"{budget_gb:.2f} GB at cell {cell}")
        self.cell = cell
        self.footprint_gb = footprint_gb
        self.budget_gb = budget_gb


class BudgetGuard:
    """The footprint cutoff, checked between cells and (per child) between
    records. ``reader`` is injectable so the refusal path is testable without
    allocating tens of GB."""

    def __init__(self, budget_gb: float, reader=None):
        self.budget_gb = budget_gb
        self._reader = phys_footprint_gb if reader is None else reader

    def check(self, cell: str) -> float:
        current, _peak = self._reader()
        if current > self.budget_gb:
            raise BudgetExceeded(cell, current, self.budget_gb)
        return current


class LowMemoryRefusal(RuntimeError):
    """The machine cannot offer the room this harness may still grow into:
    refuse before the cell, never push a doomed allocation into compression
    and Jetsam."""

    def __init__(self, cell: str, available_gb: float, needed_gb: float):
        super().__init__(
            f"machine has {available_gb:.2f} GB available, cell needs room "
            f"for the {needed_gb:.2f} GB budget at cell {cell}")
        self.cell = cell
        self.available_gb = available_gb
        self.needed_gb = needed_gb


def available_memory_gb() -> float:
    """Machine-wide available memory in decimal GB.

    kern.memorystatus_level is the memorystatus (Jetsam) subsystem's own
    percentage of available memory - the same authority that killed the three
    runs - so the refusal reads the exact meter the killer reads. Same
    refusal-gate idiom as bench/machine_state.py: sample, refuse, name why.
    """
    out = subprocess.run(["sysctl", "-n", "kern.memorystatus_level"],
                         capture_output=True, text=True)
    return int(out.stdout.strip()) / 100.0 * machine_ram_gb()


def require_available_memory(needed_gb: float, cell: str,
                             reader=None) -> float:
    read = available_memory_gb if reader is None else reader
    available = read()
    if available < needed_gb:
        raise LowMemoryRefusal(cell, available, needed_gb)
    return available


def positive_float_arg(text: str, what: str, unit: str) -> float:
    """A CLI number a gate depends on: infinities, NaN and non-positives are
    rejected at parse time, never becoming a budget that cannot bite."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{what} {text!r} is not a number")
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            f"{what} must be positive finite {unit}, got {text!r}")
    return value


def budget_gb_arg(text: str) -> float:
    """The `--budget-gb` parser every harness with a footprint budget shares,
    so the budget cannot be validated one way here and another way there."""
    return positive_float_arg(text, "budget", "decimal GB")
