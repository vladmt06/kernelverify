"""How much of a composed win belongs to the artifact and how much to the kernel.

One definition, in a module with no imports, because two consumers need it and
they cannot share a module that reaches MLX.

`bench/serve_sub4bit.py` computes it while holding a GPU, and
`bench/spec_decode_rules.py` computes it in a module that must stay importable
where no Metal device exists - importing `mlx.nn` ABORTS the interpreter there,
which is why the spec-decode verdict arithmetic is standard-library-only and
testable on any machine.

It lived in serve_sub4bit and was hand-copied into spec_decode_rules to satisfy
that constraint. The copies were equivalent on the day they were written and
nothing bound them, which is the shape of defect AGENTS.md records for
`bench/external.py`: two spellings of one rule are how a lane reports something
other than what it measured.
"""
from __future__ import annotations


def composed_attribution(comp_pct: float, base_pct: float,
                         noise_pct: float) -> str:
    """Which half of a composed comparison earned the win.

    `comp_pct` is the composed comparison, `base_pct` the same comparison with
    the kernel taken out, and `noise_pct` the floor both are judged against.
    A win the base already had is `artifact-alone`; one that needed both is
    `joint`.
    """
    if abs(comp_pct) <= noise_pct:
        return "inconclusive"
    if comp_pct <= 0:
        return "negative"
    if base_pct > noise_pct:
        return "artifact-alone"
    return "joint"
