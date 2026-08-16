"""ADR 0016's detection price, recomputed from two committed record sets.

What the repair cost
--------------------
Making `factored-groups` round the way a real int-accumulate kernel rounds
widened the ensemble floor on constant rows. That closed ten false positives
against correct device kernels, and it necessarily also widened the tolerance
for one INCORRECT implementation: the fp16-dequant boundary case, which is the
nearest-miss the contract deliberately keeps outside itself. Three of 384
float32 cells flip from caught to not-caught. ADR 0016 named them rather than
counting them, because a detection loss that is not named is not a price, it is
a rounding of the story.

Why this file exists
--------------------
The "after" halves came from `bench/results/quant_device_adequacy.json`. The
"before" halves came from a verbatim pre-repair re-run whose records were never
committed, so for a while the most uncomfortable numbers in the ADR were the
only ones nobody could check. The merge review of 2026-08-16 caught that.
`bench/derive_prerepair_device_records.py` now re-runs the device grid with the
pre-repair member swapped in at runtime and commits its records, and this module
recomputes every published number from the two files.

The comparison is only meaningful because the two runs agree everywhere else:
the generator matches all 768 records and requires every member except
`factored-groups` - including the three device members, genuinely re-measured on
the GPU - to come back bit-identical, so a moved detection price cannot be a
disguised driver or compiler change. That check is re-asserted here on the
committed files rather than trusted from the generator's own header, which is
the mistake its sibling artifact's first draft made.
"""

import json
from pathlib import Path

import pytest

from kernelverify.schemas.quant_contract import ENSEMBLE

ROOT = Path(__file__).resolve().parents[1]
AFTER = ROOT / "bench" / "results" / "quant_device_adequacy.json"
BEFORE = ROOT / "bench" / "results" / "quant_device_adequacy_prerepair.json"

REPAIRED = "factored-groups"
CPU_MEMBERS = tuple(ENSEMBLE)
# ADR 0012's reading, which the boundary table is stated against: K = 3 over the
# CPU floor. Deliberately NOT K_QUANT - this table is a fact about that column
# and must not follow the shipped K when it moves.
K_TABLE = 3.0
BITS = (2, 3, 4, 8)


def _records(path: Path) -> list:
    if not path.exists():
        pytest.fail(f"{path.name} is missing; regenerate with\n"
                    f"  .venv/bin/python -u bench/derive_prerepair_device_records.py")
    return [r for group in json.loads(path.read_text())["records"].values()
            for r in group]


def _identity(record: dict) -> tuple:
    return (record["bits"], record["shape"], record["draw"], record["seed"],
            record["mode"], record["dtype"])


def _tolerance(record: dict) -> float:
    """The shipped shape, at the table's K and floor: max(base, K * CPU floor)."""
    return max(record["base_tol"],
               K_TABLE * max(record["members"][m] for m in CPU_MEMBERS))


def _caught(record: dict) -> bool:
    return record["boundary_fp16_dequant"] > _tolerance(record)


@pytest.fixture(scope="module")
def paired():
    before, after = _records(BEFORE), _records(AFTER)
    assert len(before) == len(after) == 768
    by_id = {_identity(r): r for r in before}
    return [(by_id[_identity(r)], r) for r in after]


def test_only_the_repaired_member_differs_between_the_two_runs(paired):
    """The precondition for reading the two runs as one comparison.

    Re-asserted here on the committed files rather than taken from the
    generator's header: a header saying a check ran is not the check.
    """
    for before, after in paired:
        for name in before["members"]:
            if name == REPAIRED:
                continue
            assert before["members"][name] == after["members"][name], (
                f"member {name} moved at {_identity(after)}: the detection price "
                f"would be measuring two changes at once")
        assert before["boundary_fp16_dequant"] == after["boundary_fp16_dequant"], (
            "the boundary case itself moved; it is the thing being detected")
        assert before["base_tol"] == after["base_tol"]
    moved = sum(b["members"][REPAIRED] != a["members"][REPAIRED] for b, a in paired)
    assert moved, "the repaired member is identical in both runs: one is mislabelled"


def test_the_floor_moves_both_ways_not_only_wider(paired):
    """A qualification ADR 0016's prose does not carry, measured here.

    The ADR describes the trade as the tolerance getting WIDER: the member
    stopped being unrealistically exact, so the floor it feeds grew. That is the
    dominant effect and it is what closed the ten false positives. It is not the
    whole picture. Chaining and pairwise reduction are two legitimate rounding
    orders, and on data that is not constant either can land closer to the fp64
    reference - even on constant rows, because only the ACTIVATION sum is exact
    there while the x*q sum still varies. So the chained member is larger on
    about seven records in ten, not on all of them, and where it was the unique
    maximum and shrank, the CPU floor shrinks with it.

    On this grid the floor tightens on 116 of 768 device records. Nothing was
    harmed by it - no boundary cell becomes newly caught, asserted below - but
    "the tolerance is wider" is a simplification, and a tightening floor is the
    false-positive direction, so it is pinned rather than left implicit.
    """
    def floor(record: dict) -> float:
        return max(record["members"][m] for m in CPU_MEMBERS)

    grew = sum(floor(a) > floor(b) for b, a in paired)
    shrank = sum(floor(a) < floor(b) for b, a in paired)
    unchanged = sum(floor(a) == floor(b) for b, a in paired)

    assert (grew, shrank, unchanged) == (117, 116, 535)
    assert sum(a["members"][REPAIRED] >= b["members"][REPAIRED] for b, a in paired) == 543


def test_the_boundary_counts_move_as_adr_0016_reports(paired):
    """39 / 96 / 96 / 96 -> 38 / 95 / 95 / 96 at bits 2 / 3 / 4 / 8, float32."""
    def counts(index: int, dtype: str) -> list:
        return [sum(1 for pair in paired
                    if pair[index]["dtype"] == dtype and pair[index]["bits"] == bits
                    and _caught(pair[index]))
                for bits in BITS]

    assert counts(0, "float32") == [39, 96, 96, 96]
    assert counts(1, "float32") == [38, 95, 95, 96]
    # fp16 activations catch nothing either side: the floor is inert there, so
    # base_tol carries the tolerance and the boundary sits under it (ADR 0014).
    assert counts(0, "float16") == [0, 0, 0, 0]
    assert counts(1, "float16") == [0, 0, 0, 0]


def test_the_three_lost_cells_are_named_with_their_ratios(paired):
    """The price, named rather than counted - ADR 0016's own standard.

    Every one is at the smallest shape on heavy-tailed constant rows, the same
    regime the ten false positives lived in, which is what makes this one
    trade-off rather than two unrelated effects.
    """
    lost = [(_identity(a), b["boundary_fp16_dequant"] / _tolerance(b),
             a["boundary_fp16_dequant"] / _tolerance(a))
            for b, a in paired if _caught(b) and not _caught(a)]

    assert len(lost) == 3, lost
    assert all(shape == "512x512" and draw == "heavy-tailed"
               and mode == "constant-rows" and dtype == "float32"
               for (_, shape, draw, _, mode, dtype), _, _ in lost)

    table = {ident[0]: (round(before, 3), round(after, 3)) for ident, before, after in lost}
    assert table == {2: (1.248, 0.953), 3: (1.640, 0.835), 4: (1.963, 0.980)}
    assert all(before > 1.0 > after for before, after in table.values())

    # Nothing flips the other way: the repair only ever widens the tolerance.
    gained = [_identity(a) for b, a in paired if _caught(a) and not _caught(b)]
    assert gained == [], gained
