"""The record guards itself.

The claims audit of 2026-08-15 found eight stale facts in AGENTS.md. They
drifted because nothing checked them: a count, a package list and a path list
are all facts about the tree, and a fact about the tree can be tested against
the tree. These are those tests.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = (ROOT / "AGENTS.md").read_text()


def test_adr_numbering_is_contiguous():
    nums = sorted(int(p.name[:4]) for p in (ROOT / "docs/adr").glob("0*.md"))
    assert nums == list(range(1, len(nums) + 1)), nums


def test_agents_md_catalogue_count_matches_the_tree():
    from kernelverify.mutation.catalogue import CATALOGUE
    m = re.search(r"fault catalogue \((\d+) entries\)", AGENTS)
    # Two different failures, told apart on purpose: a reworded line reads as a
    # count mismatch of (None, 65) otherwise, and sends the reader to the wrong file.
    assert m, "AGENTS.md no longer says 'fault catalogue (N entries)'; this test parses that exact phrasing"
    assert int(m.group(1)) == len(CATALOGUE), (m.group(1), len(CATALOGUE))


def test_every_path_agents_md_layout_names_exists():
    # Every section that names paths, not just Layout: the Running block and the
    # Mistakes entries cite files too, and a path that has moved is as stale there.
    #
    # Runtime outputs are excluded by directory, not waved through case by case.
    # Everything under these is written BY a harness and gitignored, so it is
    # absent in a fresh clone and its absence says nothing about the record being
    # stale - which is the only thing this test is for. Source paths that moved
    # are the failure it must catch, and those all live outside them.
    outputs = (".cache/", ".models/", ".baselines/", ".corpus/", ".certificates/")
    for path in re.findall(r"`((?:kernelverify|bench|docs|vendor)/[^`]+?)`", AGENTS):
        if any(marker in path for marker in outputs):
            continue
        if path.endswith(("/", ".py", ".md", ".mm", ".sh", ".json", ".txt")):
            assert (ROOT / path.rstrip("/")).exists(), f"AGENTS.md names {path}, which does not exist"


def test_agents_md_empty_packages_list_is_true():
    m = re.search(r"Empty packages: `([^`]+)`(?:, `([^`]+)`)*", AGENTS)
    claimed = set(re.findall(r"`([a-z_]+)`", m.group(0))) if m else set()
    actual = {p.name for p in (ROOT / "kernelverify").iterdir()
              if p.is_dir() and not p.name.startswith("__")
              and (p / "__init__.py").exists()                                   # a package, not a data dir
              and all(f.name == "__init__.py" and f.stat().st_size == 0 for f in p.glob("*.py"))}
    assert claimed == actual, (claimed, actual)


def test_the_battery_100_percent_claim_has_a_committed_record():
    rec = ROOT / "bench/results/score_oracles-2026-08-15.txt"
    assert rec.exists()
    assert "B=16: none, every viable fault caught in every run" in rec.read_text()


def test_the_100_percent_record_scored_todays_catalogue():
    """The half that makes the claim's staleness visible.

    AGENTS.md and ADR 0008 both say the claim goes stale the moment the
    catalogue outgrows what the record scored. Nothing enforced that until
    here: the test above only greps a fixed string, so the population could
    double and the record would still 'back' the claim. This compares the
    record's own synthesised count against the live catalogue, which is the
    comparison both documents promise a reader.
    """
    from kernelverify.mutation.catalogue import CATALOGUE
    text = (ROOT / "bench/results/score_oracles-2026-08-15.txt").read_text()
    m = re.search(r"fault population: (\d+) synthesised, (\d+) viable", text)
    assert m, "the record no longer states its population in the parsed form"
    scored, viable = int(m.group(1)), int(m.group(2))
    assert scored == len(CATALOGUE), (
        f"the committed record scored {scored} faults but the catalogue now holds "
        f"{len(CATALOGUE)}; the 100% claim is stale until score_oracles is re-run")
    assert viable <= scored
