"""The record guards itself.

The claims audit of 2026-08-15 found eight stale facts in AGENTS.md. They
drifted because nothing checked them: a count, a package list and a path list
are all facts about the tree, and a fact about the tree can be tested against
the tree. These are those tests.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENTS = (ROOT / "AGENTS.md").read_text()


def test_adr_numbering_is_contiguous():
    nums = sorted(int(p.name[:4]) for p in (ROOT / "docs/adr").glob("0*.md"))
    assert nums == list(range(1, len(nums) + 1)), nums


def test_agents_md_catalogue_count_matches_the_tree():
    from kernelverify.mutation.catalogue import CATALOGUE
    m = re.search(r"fault catalogue \((\d+) entries\)", AGENTS)
    assert m and int(m.group(1)) == len(CATALOGUE), (m and m.group(1), len(CATALOGUE))


def test_every_path_agents_md_layout_names_exists():
    layout = AGENTS.split("## Layout", 1)[1].split("\n## ", 1)[0]
    for path in re.findall(r"`((?:kernelverify|bench|docs|vendor)/[^`]+?)`", layout):
        if path.endswith("/") or path.endswith(".py") or path.endswith(".md") or path.endswith(".mm") or path.endswith(".sh"):
            assert (ROOT / path.rstrip("/")).exists(), f"AGENTS.md Layout names {path}, which does not exist"


def test_agents_md_empty_packages_list_is_true():
    m = re.search(r"Empty packages: `([^`]+)`(?:, `([^`]+)`)*", AGENTS)
    claimed = set(re.findall(r"`([a-z_]+)`", m.group(0))) if m else set()
    actual = {p.name for p in (ROOT / "kernelverify").iterdir()
              if p.is_dir() and not p.name.startswith("__")
              and (p / "__init__.py").exists()                                   # a package, not a data dir
              and all(f.name == "__init__.py" and f.stat().st_size == 0 for f in p.glob("*.py"))}
    assert claimed == actual, (claimed, actual)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "H2 of the 2026-08-16 audit-amendments plan is blocked: the source of this "
        "record is bench/.cache/score_oracles_rerun_20260815.log, and bench/.cache/ "
        "is gitignored and per-worktree, so the log does not exist in the hygiene "
        "worktree and no lane may fabricate it. The test is written now so the "
        "obligation is in the suite rather than in someone's memory. It is STRICT "
        "on purpose: the moment the record is committed this goes XPASS and turns "
        "the suite red, which is the signal to delete this marker."
    ),
)
def test_the_battery_100_percent_claim_has_a_committed_record():
    rec = ROOT / "bench/results/score_oracles-2026-08-15.txt"
    assert rec.exists()
    assert "B=16: none, every viable fault caught in every run" in rec.read_text()
