"""Replacing one module-level name in mlx-lm, reversibly and visibly.

metalrunner runs mlx-lm's own trainer rather than a copy of it, so wherever
it needs different behaviour it replaces a name that mlx-lm looks up at call
time. That is the seam. This module is the only place a replacement is
installed, so there is exactly one answer to "what did metalrunner change
about this process", and it is a list rather than a search.

Three properties matter more than convenience here.

It refuses rather than stacking. Before replacing anything it checks that the
name currently holds the object mlx-lm itself defined, by asking the object
which module it was defined in. If something else has already replaced it,
installing on top would produce a run whose behaviour is the composition of
two patches nobody wrote down, and a measurement taken through it would be
measuring that composition. So the install refuses and nothing is changed.
What this check catches is a foreign object left at the seam; what it cannot
catch is a replacement that copied the original's identity metadata, or an
edit to the original's own source. The second of those is what the file
hashes in versions.py are for.

It counts. Every call through a replaced name is counted, so afterwards the
run can state how many times each replacement was actually reached. A seam
that was installed and never called is the failure this exists to make
visible: without a count, a report saying an operation was replaced and a
run in which it never happened look exactly alike.

It restores, and says if it could not. Removal puts every original back in
reverse order. If a name no longer holds what this module put there, some
other code replaced it mid-run; the original is restored anyway so the
process is left clean, and the seam is named in `foreign_on_removal` so the
run can be treated as suspect instead of quietly trusted.

A name is only a seam if the caller looks it up at call time. `run()` calls
`train_model(...)` as a module global, so replacing `mlx_lm.lora.train_model`
reaches it; a name already bound into a default argument, a closure or a
compiled function would not be reached, and no amount of care here changes
that.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


class SeamRefusal(RuntimeError):
    """The seam was not what metalrunner expected, so nothing was changed."""


@dataclass(frozen=True)
class Seam:
    """A module-level name, by the module that defines it."""

    module: str
    attribute: str

    def __str__(self) -> str:
        return f"{self.module}.{self.attribute}"


class Installation:
    """The set of replacements active in this process, and their counts."""

    def __init__(self):
        self._order: list[Seam] = []
        self._originals: dict[Seam, object] = {}
        self._installed: dict[Seam, object] = {}
        self._counts: dict[Seam, int] = {}
        self.foreign_on_removal: list[str] = []

    def install(self, seam: Seam, wrap) -> None:
        """Replace `seam` with `wrap(original)`, counting every call.

        `wrap` receives the object being replaced and returns the object to
        install, so a replacement that delegates to the original does not
        have to look it up again and cannot look up the wrong one.
        """
        if seam in self._originals:
            raise SeamRefusal(f"{seam} is already installed by this run")

        module = importlib.import_module(seam.module)
        try:
            original = getattr(module, seam.attribute)
        except AttributeError:
            raise SeamRefusal(
                f"{seam} does not exist, so metalrunner cannot replace it"
            ) from None

        defined_in = getattr(original, "__module__", None)
        if defined_in != seam.module:
            raise SeamRefusal(
                f"{seam} currently holds an object defined in "
                f"{defined_in!r}, not {seam.module!r}: something else has "
                "already replaced it, and installing on top of it would "
                "produce behaviour nobody wrote down. Nothing was changed.")

        replacement = wrap(original)
        self._counts[seam] = 0
        counted = self._counting(seam, replacement)
        setattr(module, seam.attribute, counted)
        self._originals[seam] = original
        self._installed[seam] = counted
        self._order.append(seam)

    def remove(self) -> None:
        """Put every original back, newest first, and report any surprise."""
        for seam in reversed(self._order):
            module = importlib.import_module(seam.module)
            if getattr(module, seam.attribute, None) is not self._installed[seam]:
                self.foreign_on_removal.append(str(seam))
            setattr(module, seam.attribute, self._originals[seam])
        self._order.clear()
        self._originals.clear()
        self._installed.clear()

    @property
    def counts(self) -> dict[str, int]:
        """How many times each replacement was reached, by seam name."""
        return {str(seam): count for seam, count in self._counts.items()}

    def _counting(self, seam: Seam, replacement):
        def call(*args, **kwargs):
            self._counts[seam] += 1
            return replacement(*args, **kwargs)

        call.__name__ = seam.attribute
        call.__qualname__ = f"metalrunner:{seam}"
        return call

    def __enter__(self) -> "Installation":
        return self

    def __exit__(self, *_exception) -> None:
        # Removal runs on the way out of a failed run too: a process that
        # raised mid-training must not leave a replaced name behind for
        # whatever runs next in the same interpreter.
        self.remove()
