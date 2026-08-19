"""Cases the generator never sees, drawn so it cannot guess them.

A generator that is told what it will be judged on optimises for that, and a
kernel tuned to its own test set is exactly the correctness illusion this
repository exists to refuse. So a frozen incumbent is checked on a draw it
had no way to anticipate, and the pre-registration
(docs/research/2026-08-19-metalrunner-sprint1-prereg.md, section 7) fixes how
the draw is made before any candidate exists.

Three properties, each with a mechanism rather than a promise.

Unpredictable. The seed is sha256 of the candidate's own source hash
concatenated with a secret salt. The salt was generated on 2026-08-19, lives
outside the repository, and never enters a generator's context; only its
sha256 is committed. A generator that could compute its own draw could
special-case it, so the secret is what makes the check mean anything.

Unchangeable. The salt's digest is pinned in this module. Regenerating the
salt mid-sprint to obtain a friendlier draw changes that digest and is
refused here, loudly. The salt is published when the sprint closes, at which
point any reader can recompute every draw that was ever made and confirm it
was not chosen to flatter anything.

Sealed. What comes back is a `Verdict`, which holds one boolean and has
nowhere to put a reason. A failing held-out case that told the generator
which case failed would be feedback, and feedback turns a held-out set into
a training set within a few rounds. The draw is legible to a human reading
the sealed record afterwards, and to nothing in the loop.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

SALT_PATH = Path(__file__).resolve().parents[2] / "bench" / ".heldout_salt"

# Published in the pre-registration before any draw was made. The salt itself
# is published in the closing record.
SALT_SHA256 = "d5a9492c6466660e0d39c90beaae27a3bdaf1402ce00c8697432d129cd94d389"


class SaltError(RuntimeError):
    """The salt is missing, or is not the one the sprint pre-registered."""


@dataclass(frozen=True)
class Verdict:
    """All the loop is allowed to learn from a held-out check.

    One field, deliberately. There is nowhere to attach the failing case, so
    a failure cannot become feedback by someone adding a helpful detail
    string later.
    """

    passed: bool

    def __str__(self) -> str:
        return "PASS" if self.passed else "FAIL"


def load_salt(path: Path | None = None) -> bytes:
    """The secret, checked against the digest committed before any draw."""
    path = SALT_PATH if path is None else path
    try:
        salt = path.read_bytes()
    except FileNotFoundError as error:
        raise SaltError(
            f"no held-out salt at {path}: a draw without it would be one the "
            f"generator could compute for itself, so there is no unsalted "
            f"fallback") from error

    digest = hashlib.sha256(salt).hexdigest()
    if digest != SALT_SHA256:
        raise SaltError(
            f"{path} hashes {digest}, not the pre-registered {SALT_SHA256}: "
            f"this is not the salt the sprint committed to, and a re-rolled "
            f"salt is a re-rolled draw")
    return salt


def seed_for(candidate: str, salt: bytes) -> int:
    """This candidate's draw seed: unpredictable without the salt, and fixed
    for the candidate once the salt is known, so a draw never varies between
    two runs of the same check."""
    return int.from_bytes(
        hashlib.sha256(candidate.encode() + salt).digest()[:8], "big")


def draw(candidate: str, space: dict[str, list], *,
         salt: bytes | None = None) -> dict:
    """Choose one value per axis of `space` for this candidate.

    `space` is the withheld cell set: token-count bands, tile-edge shapes,
    projection family, dtype and input distribution. It is data rather than
    code here because which cells are withheld depends on the operation the
    Day 1 profile selects, while everything above this line does not.
    """
    if not space:
        raise ValueError("an empty held-out space withholds nothing")
    for axis, choices in space.items():
        if not choices:
            raise ValueError(f"held-out axis {axis!r} offers no choices")

    chooser = random.Random(seed_for(candidate, load_salt() if salt is None
                                     else salt))
    return {axis: chooser.choice(choices) for axis, choices in sorted(space.items())}
