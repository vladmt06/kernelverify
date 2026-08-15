"""The matrix must refuse before it renders.

Every test here pins a refusal, because the renderer's value is not that it
draws tables - it is that a number nobody may believe never reaches a reader,
and that the reason travels with the refusal. The gates come from
bench/.baselines/SCHEMA.md and each one exists because a real measurement was
once wrong in exactly that way.
"""

import json
import re
from pathlib import Path

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
    comparisons = text.split("## Comparisons", 1)[1].split("\n## ", 1)[0]
    assert "no difference" in comparisons, \
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
# ---------------------------------------------------------------------------
# Audit regressions: each pins a confirmed way the first version misled
# ---------------------------------------------------------------------------
def test_decode_and_prefill_in_one_group_are_not_comparable():
    """The P1 from the audit, reproduced on the real record: the producer sets
    sampling.group to the run id, so one group held decode, prefill and four
    matmul widths, and the render published a decode row at 0.07x of its own
    stack's prefill row - a workload ratio dressed as a stack ratio."""
    a = classify(row())
    b = classify(row(row_id="p", measurement={"kind": "prefill",
                                              "matmul_width": 1024,
                                              "n_prompt": 1024},
                     result={"median": 615.3}))
    ok, why = comparable(a, b)
    assert not ok and "different workloads" in why
    text = render([a.row, b.row])
    assert "14.2" not in text and "0.07x" not in text, \
        "a cross-workload ratio reached the reader"


def test_same_workload_different_stack_still_compares():
    """The gate must not break the comparison the table exists for."""
    a = classify(row())
    b = classify(row(row_id="m", stack={"name": "mlx-lm"},
                     measurement={"width_mechanism": "batch-size"},
                     result={"median": 52.1, "spread_pct": 2.0}))
    ok, _ = comparable(a, b)
    assert ok, "width_mechanism and model differ across stacks by construction"


def test_comparison_rows_carry_their_identity():
    """Six rows all labelled with the same stack string were unattributable;
    the table now carries model and width-with-mechanism per row."""
    a = row()
    b = row(row_id="m", stack={"name": "mlx-lm"},
            measurement={"width_mechanism": "batch-size"},
            result={"median": 52.1, "spread_pct": 2.0})
    text = render([a, b])
    comparisons = text.split("## Comparisons", 1)[1]
    assert "Qwen3-4B" in comparisons
    assert "prompt-width" in comparisons and "batch-size" in comparisons
    assert "baseline" in comparisons, "the baseline row must identify itself"


def test_unknown_binding_resource_is_never_read_as_bandwidth():
    """The producer stamps MLX prefill rows binding_resource='unknown' because
    no parameter count exists to place them; defaulting those to bandwidth
    publishes the 1%-catastrophe the column exists to prevent."""
    resource, pct = utilisation(row(roofline={"binding_resource": "unknown",
                                              "bandwidth_utilisation_pct": 1.2}))
    assert pct is None and "unknown" in resource
    resource, pct = utilisation(row(roofline={"binding_resource": "Compute",
                                              "roofline_utilisation_pct": 60.0,
                                              "bandwidth_utilisation_pct": 1.1}))
    assert pct is None, "a casing variant must not silently become bandwidth"


def test_unknown_provenance_tier_never_enters_a_comparison():
    """The exclusion asks the trust question, not a literal match: a row
    tagged 'Owner-Run' (unrecognised) must be barred exactly like the known
    contributed tier."""
    a = row()
    impostor = row(row_id="i", provenance_tier="Owner-Run",
                   stack={"name": "secret-fork"},
                   result={"median": 91.3, "spread_pct": 1.0})
    text = render([a, impostor])
    comparisons = (text.split("## Comparisons", 1)[1].split("\n## ", 1)[0]
                   if "## Comparisons" in text else "")
    assert "secret-fork" not in comparisons
    assert "secret-fork" in text, "the row is still shown, in the contributed section"


def test_contributed_rows_carry_their_own_gate_results():
    """A contributed row that failed its own dispersion gate rendered as a
    bare number; its bindingness must travel with it."""
    contributed = row(provenance_tier="community-unattested", binding=False,
                      binding_blockers=["repeats disagree by 93.9%"])
    text = render([contributed])
    section = text.split("## Contributed", 1)[1]
    assert "93.9%" in section
    assert "0 refused, 1 contributed" in text


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


# ---------------------------------------------------------------------------
# Second-pass regressions: units, noise symmetry, run identity
# ---------------------------------------------------------------------------
def test_mismatched_metrics_never_form_a_ratio():
    """The kernel lanes report latency in us while the baseline record reports
    tokens/s; the moment both land in one group a unitless division would
    publish a number that is not a ratio of anything."""
    a = classify(row())
    b = classify(row(row_id="other", result={"metric": "us"}))
    ok, why = comparable(a, b)
    assert not ok and "metric" in why


def test_mismatched_metric_row_is_refused_in_the_rendered_table():
    text = render([row(), row(row_id="b", stack={"name": "mlx-lm", "version": "0.31"},
                              result={"metric": "us", "median": 11.0})])
    assert "different metrics" in text


def test_ratio_header_names_the_unit_it_divides():
    """2.6x of a throughput is a win and 2.6x of a latency is a loss; the
    header must say which one the column holds."""
    text = render([row(), row(row_id="b", stack={"name": "mlx-lm", "version": "0.31"},
                              result={"median": 91.3})])
    assert "(tok/s)" in text


def test_a_noisy_baseline_swallows_a_small_difference():
    """The swallow gate was one-sided: a tight row against a noisy baseline
    published a 'difference' inside the baseline's own noise band."""
    base = row(result={"spread_pct": 20.0})
    other = row(row_id="b", stack={"name": "mlx-lm", "version": "0.31"},
                result={"median": 52.0, "spread_pct": 1.0})
    text = render([base, other])
    comparisons = text.split("## Comparisons", 1)[1].split("\n## ", 1)[0]
    assert "no difference" in comparisons


def test_a_measurement_row_names_its_run():
    """Two binding runs of the same workload must not render as
    indistinguishable duplicate absolutes."""
    text = render([row()])
    assert "20260814T120000Z-abcdef12" in text


# ---------------------------------------------------------------------------
# The v3 boundary: logical model identity, and no cross-version ratios
# ---------------------------------------------------------------------------
def row_v3(**over):
    base = row(schema_version=3, model={"logical_name": "qwen3-4b"})
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def test_v3_rows_gate_on_the_logical_model():
    """D12.3: two v3 rows measuring different logical models never form a
    ratio, however identically they were sampled."""
    a = classify(row_v3())
    b = classify(row_v3(row_id="other", stack={"name": "mlx-lm"},
                        model={"logical_name": "llama3-8b"}))
    ok, why = comparable(a, b)
    assert not ok and "logical" in why


def test_v3_rows_with_one_logical_model_still_compare():
    a = classify(row_v3())
    b = classify(row_v3(row_id="other", stack={"name": "mlx-lm"},
                        model={"name": "a-different-artefact-per-stack"}))
    ok, _ = comparable(a, b)
    assert ok, "the artefact name differs per stack by construction; the logical name is the identity"


def test_a_v3_row_missing_its_logical_name_never_compares():
    a = classify(row_v3(model={"logical_name": None}))
    b = classify(row_v3(row_id="other", model={"logical_name": None}))
    ok, why = comparable(a, b)
    assert not ok and "logical" in why


def test_v2_rows_keep_their_original_gates():
    """D12.3: the logical-name gate is a v3 rule. The shipped v2 record has no
    logical_name and must keep comparing exactly as before."""
    a = classify(row())
    b = classify(row(row_id="other", stack={"name": "mlx-lm"}))
    assert "logical_name" not in a.row["model"]
    ok, _ = comparable(a, b)
    assert ok


def test_cross_version_rows_never_compare():
    """D12.4: what a sampling group means changed at the v3 boundary, so a
    cross-version ratio would pair rows measured under different contracts."""
    a = classify(row())
    b = classify(row_v3(row_id="other"))
    ok, why = comparable(a, b)
    assert not ok and "schema version" in why


# ---------------------------------------------------------------------------
# The footer: provenance and the legend (D13)
# ---------------------------------------------------------------------------
def test_footer_states_the_renderer_commit_and_record_files():
    text = render([row()], sources=[Path("bench/.baselines/2026-08-14.jsonl")])
    footer = text.split("## Provenance and legend", 1)[1]
    assert "2026-08-14.jsonl" in footer
    assert re.search(r"at commit `[0-9a-f]{7,}`", footer), \
        "the renderer must name its own commit, or the page cannot be traced"


def test_legend_defines_every_label_numerically():
    footer = render([row()]).split("## Provenance and legend", 1)[1]
    for label in ("no difference", "ratio only", "directional"):
        assert label in footer
    assert "summed spreads" in footer
    assert "131.7 to 93.1" in footer, \
        "directional must be defined by the measured drift, not as a vibe"


# ---------------------------------------------------------------------------
# The real record (D12.3, D12.4): content, not just a page that renders
# ---------------------------------------------------------------------------
RECORD_2026_08_14 = (Path(__file__).resolve().parents[1]
                     / "bench" / ".baselines" / "2026-08-14.jsonl")


def test_the_real_v2_record_keeps_its_comparison_content():
    """The mandated regression: the shipped 2026-08-14 record is v2, and every
    renderer change must leave its comparison CONTENT standing, not merely the
    page rendering. Four workload cells each pair llama.cpp against mlx-lm,
    and these ratios are the numbers that record actually supports."""
    rows = [json.loads(line)
            for line in RECORD_2026_08_14.read_text().splitlines() if line.strip()]
    text = render(rows)
    comparisons = text.split("## Comparisons", 1)[1].split("\n## ", 1)[0]
    for cell in ("### decode at width 1",
                 "### matmul_width at width 1",
                 "### matmul_width at width 8",
                 "### matmul_width at width 16"):
        assert cell in comparisons, f"the {cell!r} cell vanished from the render"
    for ratio in ("1.12x", "1.19x", "2.25x", "0.70x"):
        assert ratio in comparisons, f"the record's {ratio} ratio vanished"
    # Scope the pairing check to the v2 run's own cells: the same dated file
    # now legitimately carries later binding runs, whose cells must not be
    # counted against the v2 record's four.
    v2_cells = [chunk for chunk in comparisons.split("### ")
                if chunk and "20260814T145747Z-6bc2d222" in chunk.splitlines()[0]]
    assert len(v2_cells) == 4, "the v2 record's four cells must survive"
    assert all(chunk.count("| mlx-lm 0.31.3 |") == 1 for chunk in v2_cells), \
        "every v2 cell must still pair both stacks"


def test_the_real_v3_record_keeps_its_comparison_content():
    """D3.7 at the renderer: the v3 cells of the 00:05 binding run must render
    identically after the batch_decode cells were added to the producer. The
    2.46x and 1.13x cells are width-mechanism cells (batch-size on mlx), and
    per D3.8 they are never cited as a serving comparison; here they are
    protected as cells, by identity, whatever their kind label says."""
    rows = [json.loads(line)
            for line in RECORD_2026_08_14.read_text().splitlines() if line.strip()]
    text = render(rows)
    comparisons = text.split("## Comparisons", 1)[1].split("\n## ", 1)[0]
    v3_cells = [chunk for chunk in comparisons.split("### ")
                if chunk and "20260814T230251Z-b50ff2a8" in chunk.splitlines()[0]]
    assert len(v3_cells) == 4, "the v3 record's four two-arm cells must survive"
    for cell in ("### decode at width 1, group `20260814T230251Z-b50ff2a8/decode-w1`",
                 "### matmul_width at width 1, group `20260814T230251Z-b50ff2a8/matmul_width-w1`",
                 "### matmul_width at width 8, group `20260814T230251Z-b50ff2a8/matmul_width-w8`",
                 "### matmul_width at width 16, group `20260814T230251Z-b50ff2a8/matmul_width-w16`"):
        assert cell in comparisons, f"the {cell!r} cell vanished from the render"
    for ratio in ("1.10x", "1.14x", "2.46x", "1.13x"):
        assert ratio in comparisons, f"the v3 record's {ratio} ratio vanished"
    assert all(chunk.count("| mlx-lm 0.31.3 |") == 1 for chunk in v3_cells), \
        "every v3 two-arm cell must still pair both stacks"


# ---------------------------------------------------------------------------
# mlx-only serving cells (D6): the label must reach the reader
# ---------------------------------------------------------------------------
def test_a_batch_decode_row_renders_its_mlx_only_scope():
    """The serving cells exist on one stack only this block, so the rendered
    kind must carry the scope: a bare 'batch_decode' line invites setting the
    aggregate against the other stack's single-stream decode row."""
    r = row(stack={"name": "mlx-lm", "version": "0.31.3"},
            measurement={"kind": "batch_decode", "matmul_width": 8,
                         "width_mechanism": "batch-size", "n_parallel": 8,
                         "stack_scope": "mlx-only"})
    table = render([r]).split("## Measurements", 1)[1].split("\n## ", 1)[0]
    assert "batch_decode (mlx-only)" in table
