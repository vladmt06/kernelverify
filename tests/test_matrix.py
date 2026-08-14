"""The matrix must refuse before it renders.

Every test here pins a refusal, because the renderer's value is not that it
draws tables - it is that a number nobody may believe never reaches a reader,
and that the reason travels with the refusal. The gates come from
bench/.baselines/SCHEMA.md and each one exists because a real measurement was
once wrong in exactly that way.
"""

import json

import pytest

from kernelverify.report.matrix import (
    Claim,
    classify,
    comparable,
    load_rows,
    render,
    utilisation,
)


def row(**over):
    base = {
        "schema_version": 2,
        "run_id": "20260814T120000Z-abcdef12",
        "row_id": "20260814T120000Z-abcdef12/decode-llamacpp",
        "measured_at": "2026-08-14T12:00:00Z",
        "provenance_tier": "owner-run",
        "binding": True,
        "binding_blockers": [],
        "machine": {"chip": "Apple M3 Pro", "hw_model": "Mac15,6", "os": "macOS",
                    "os_build": "26.5.2"},
        "stack": {"name": "llama.cpp", "version": "a94d563"},
        "model": {"name": "Qwen3-4B", "quant": "Q4_K_M"},
        "measurement": {"kind": "decode", "matmul_width": 1,
                        "width_mechanism": "prompt-width"},
        "result": {"metric": "tok/s", "median": 45.67, "spread_pct": 1.2,
                   "reps": 3, "samples": [45.6, 45.67, 45.8],
                   "min_sample_ms": 21.9, "floor_ms": 1.0,
                   "below_timing_floor": False},
        "sampling": {"interleaved": True, "rotation": "round-robin", "rounds": 5,
                     "group": "qwen3-4b-decode", "group_members": 2},
        "roofline": {"bytes_per_pass": 2491000000, "byte_model": "tensor-table",
                     "denominator_name": "bandwidth_read", "denominator_gbs": 135.5,
                     "achieved_gbs": 114.2, "bandwidth_utilisation_pct": 84.3,
                     "binding_resource": "memory",
                     "roofline_utilisation_pct": None},
    }
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# Publishability
# ---------------------------------------------------------------------------
def test_a_clean_owner_run_row_is_publishable():
    claim = classify(row())
    assert claim.publishable and claim.reasons == ()


def test_non_binding_row_is_refused_with_every_blocker_kept():
    claim = classify(row(binding=False,
                         binding_blockers=["machine busy: Terminal at 30.2% CPU",
                                           "repeats disagree by 93.9%"]))
    assert not claim.publishable
    assert "Terminal at 30.2% CPU" in claim.reasons[0]
    assert len(claim.reasons) == 2, "every blocker must survive to the reader"


def test_non_binding_with_no_blocker_still_refuses():
    """A row claiming failure without saying why is itself the problem."""
    claim = classify(row(binding=False, binding_blockers=[]))
    assert not claim.publishable and claim.reasons


def test_community_row_is_never_publishable_even_when_binding():
    claim = classify(row(provenance_tier="community-unattested", binding=True))
    assert not claim.publishable and claim.unattested


def test_unknown_provenance_tier_is_refused():
    assert not classify(row(provenance_tier="borrowed-laptop")).publishable


# ---------------------------------------------------------------------------
# Comparability: a separate question with separate fields
# ---------------------------------------------------------------------------
def test_binding_does_not_imply_comparable():
    """Two rows can both be publishable and still not be a valid ratio."""
    a = classify(row())
    b = classify(row(row_id="other", sampling={"group": "a-different-group"}))
    assert a.publishable and b.publishable
    ok, why = comparable(a, b)
    assert not ok and "different sampling groups" in why


def test_non_interleaved_rows_are_never_compared():
    a = classify(row())
    b = classify(row(row_id="other", sampling={"interleaved": False}))
    ok, why = comparable(a, b)
    assert not ok and "clock" in why


def test_same_group_interleaved_rows_are_comparable():
    a = classify(row())
    b = classify(row(row_id="other", stack={"name": "mlx-lm", "version": "0.31.3"}))
    ok, _ = comparable(a, b)
    assert ok


# ---------------------------------------------------------------------------
# The utilisation column is chosen by the row, not by the renderer
# ---------------------------------------------------------------------------
def test_compute_bound_row_reports_the_compute_column():
    resource, pct = utilisation(row(roofline={"binding_resource": "compute",
                                              "roofline_utilisation_pct": 60.4,
                                              "bandwidth_utilisation_pct": 1.1}))
    assert (resource, pct) == ("compute", 60.4), \
        "a prefill row read as bandwidth shows 1% where the truth is 60%"


def test_memory_bound_row_reports_the_bandwidth_column():
    assert utilisation(row())[0] == "bandwidth"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_render_publishes_a_clean_row_and_its_mechanism():
    text = render([row()])
    assert "Qwen3-4B" in text and "45.7" in text
    assert "prompt-width" in text, "width without its mechanism overstates comparability"


def test_render_never_publishes_a_refused_number_as_fact():
    blocked = row(binding=False, binding_blockers=["on battery"])
    text = render([blocked])
    assert "## Refused" in text and "on battery" in text
    assert "## Measurements" in text
    measurements = text.split("## Measurements", 1)[1].split("##", 1)[0]
    assert "45.7" not in measurements, "a refused median leaked into the published table"


def test_render_keeps_contributed_rows_in_their_own_section():
    text = render([row(provenance_tier="community-unattested")])
    assert "Contributed claims" in text
    published = text.split("## Measurements", 1)[1].split("##", 1)[0]
    assert "45.7" not in published


def test_render_calls_a_difference_smaller_than_the_spread_no_difference():
    a = row()
    b = row(row_id="b", stack={"name": "mlx-lm"},
            result={"median": 46.1, "spread_pct": 5.0})
    text = render([a, b])
    assert "no difference" in text, \
        "a 1.01x ratio inside a 5% spread is noise and must be labelled"


def test_unattested_row_never_enters_a_comparison():
    """Found by rendering real output: a contributed row sharing a sampling
    group was appearing inside the ratio table, which is exactly the merge the
    schema forbids."""
    a = row()
    contributed = row(row_id="contributed", provenance_tier="community-unattested",
                      stack={"name": "someone-elses-fork"})
    text = render([a, contributed])
    # Excluding it drops the group below two members, so no comparison renders
    # at all - the strongest form of the refusal.
    comparisons = (text.split("## Comparisons", 1)[1].split("\n## ", 1)[0]
                   if "## Comparisons" in text else "")
    assert "someone-elses-fork" not in comparisons
    assert "Contributed claims" in text, "the row is still shown, just never merged"


def test_non_binding_row_may_ratio_inside_its_own_run_but_is_labelled():
    a = row()
    busy = row(row_id="busy", stack={"name": "mlx-lm"}, binding=False,
               binding_blockers=["machine busy"],
               result={"median": 52.1, "spread_pct": 1.0})
    text = render([a, busy])
    comparisons = text.split("## Comparisons", 1)[1].split("\n## ", 1)[0]
    assert "1.14x" in comparisons
    assert "not publishable as an absolute" in comparisons


def test_non_binding_row_from_another_run_gets_no_ratio():
    a = row()
    other = row(row_id="other", run_id="20260814T990000Z-ffffffff",
                stack={"name": "mlx-lm"}, binding=False,
                binding_blockers=["machine busy"])
    text = render([a, other])
    comparisons = text.split("## Comparisons", 1)[1].split("\n## ", 1)[0]
    assert "inside their own run only" in comparisons


def test_render_states_the_honest_empty_case():
    text = render([row(binding=False, binding_blockers=["machine busy"])])
    assert "No row in the record is publishable" in text


def test_ceiling_row_renders_without_a_model():
    ceiling = row(model=None,
                  measurement={"kind": "ceiling", "name": "bandwidth_read",
                               "matmul_width": None},
                  result={"metric": "gbs", "median": 135.5},
                  roofline=None, sampling=None)
    text = render([ceiling])
    assert "bandwidth_read" in text and "135.5" in text


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_load_reads_every_dated_file_in_order(tmp_path):
    (tmp_path / "2026-08-13.jsonl").write_text(json.dumps(row(row_id="older")) + "\n")
    (tmp_path / "2026-08-14.jsonl").write_text(
        json.dumps(row(row_id="newer")) + "\n\n")
    loaded = load_rows(tmp_path)
    assert [r["row_id"] for r in loaded] == ["older", "newer"]


def test_a_damaged_record_fails_loudly(tmp_path):
    """The producer validates before writing, so an unparseable line means the
    record is damaged; skipping it quietly would publish a partial matrix."""
    (tmp_path / "2026-08-14.jsonl").write_text('{"row_id": "truncated"\n')
    with pytest.raises(ValueError, match="not valid JSON"):
        load_rows(tmp_path)
