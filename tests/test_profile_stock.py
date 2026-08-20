"""The Day 1 profile harness, everywhere it can be pinned without a device.

Almost all of it can be. The harness's job is to gather numbers and refuse
when they cannot mean what they would be read to mean, and both halves of that
are arithmetic over records. What genuinely needs the GPU - that the marks
fire inside a real step, that compilation can be toggled between modes, that
the marked pass computes what the plain pass computes - lives in
tests/test_profile_stock_live.py.

The tests that matter most here are the refusals. A harness that gathered a
plausible wrong number and published it would look exactly like a harness that
worked, so each refusal is asserted together with the fact that nothing was
published.
"""

import json
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import profile_instrument as pi  # noqa: E402
import profile_rules as rules  # noqa: E402
import profile_stock as ps  # noqa: E402
from decode_rules import RunInvalid  # noqa: E402
from harness_runner import PreconditionFailed  # noqa: E402
from memory_guard import EXIT_NO_DEVICE, EXIT_PRECONDITION  # noqa: E402

DEPTH, ADAPTED = 36, 16
CTX = {"cell": "B", "batch": 4, "width": 161, "tokens": 640,
       "model": "qwen3-4b-4bit-g64", "adapted": ADAPTED}


def _decomposed(totals, *, context=None, counts=None, elapsed=1.0):
    """One instrumented pass, in the shape `decompose` returns."""
    covered = sum(totals.values())
    return {
        "regions": {name: {pi.FORWARD: total, pi.BACKWARD: 0.0}
                    for name, total in totals.items()},
        "totals": dict(totals),
        "covered": covered,
        "remainder": elapsed - covered,
        "elapsed": elapsed,
        "counts": counts if counts is not None else {
            name: {pi.FORWARD: 1, pi.BACKWARD: 1} for name in totals},
        "context": dict(context if context is not None else CTX),
    }


def _mode(totals=None, *, samples, peaks=None, rounds=3, context=None):
    record = {"totals_s": list(samples),
              "peak_gb": list(peaks if peaks is not None
                              else [10.0] * len(samples)),
              "passes": []}
    if totals is not None:
        record["passes"] = [_decomposed(totals, context=context,
                                        elapsed=sample)
                            for sample in samples[:rounds]]
    return record


def _cell_record(cell="B", *, modes=None, identity_ok=True,
                 completeness_ok=True, cheap=False):
    """One child's record.

    `cheap` gives the instrument a cost section 3.3's 2% reconciliation limit
    can accept. A real instrument does not: measured on 2026-08-20 it cost
    between 1.08x and 2.86x the plain step, so with the registered limit no
    real recording binds until Amendment 5 replaces it. Tests about anything
    other than that limit use the cheap instrument so they are testing what
    they say they are testing.
    """
    context = dict(CTX, cell=cell)
    scale = (lambda base: [1.00, 1.01, 1.02]) if cheap else (lambda base: base)
    modes = modes if modes is not None else {
        "compiled": _mode(samples=[0.95, 1.00, 1.05]),
        "plain": _mode(samples=[0.98, 1.00, 1.02]),
        "instr-A": _mode({"attn-core": 0.30},
                         samples=scale([1.30, 1.32, 1.34]), context=context),
        "instr-L": _mode({"head-matmul": 0.10, "cross-entropy": 0.05},
                         samples=scale([1.10, 1.12, 1.14]), context=context),
        "instr-Q": _mode({"qmm": 0.50, "head-matmul": 0.10},
                         samples=scale([1.90, 1.92, 1.94]), context=context),
    }
    return {
        "cell": cell,
        "context": context,
        "depth": DEPTH,
        "adapted": ADAPTED,
        "batch": {"rows": 4, "width": 161, "tokens": 640,
                  "supervised_tokens": 300, "mask_prompt": True},
        "modes": modes,
        "identity": {"gradients_compared": 96, "modes": {
            mode: {"loss_equal": identity_ok,
                   "gradients_differing": [] if identity_ok else ["a.b.lora_a"],
                   "foreign_on_removal": []}
            for mode in modes if ps.MODE_REGIONS[mode]}},
        "completeness": {"ok": completeness_ok,
                         "mismatches": [] if completeness_ok else [
                             {"region": "qmm", "direction": pi.BACKWARD,
                              "expected": 109, "observed": None}],
                         "expected": {}, "observed": {}},
        "stock": {"seams": {}, "certified": 0, "forced_to_stock": False},
        "rounds": 3,
    }


def _plan(**over):
    plan = {"schema_version": 1, "data": "bench/.data/dolly", "seed": 7,
            "optimizer": "adam", "optimizer_config": {}, "learning_rate": 1e-5,
            "memory_ceiling_gb": 24.0, "wall_cap_seconds": 900.0,
            "cells": ["B"], "rounds": 3}
    plan.update(over)
    return plan


# ---------------------------------------------------------------------------
# Which modes run where, and in what order
# ---------------------------------------------------------------------------
def test_the_context_carries_the_token_count_the_matmuls_actually_run_at():
    """mlx-lm pads a batch to `width` and `default_loss` trains on all but the
    last column, so every matmul sees one fewer token per row than the padded
    width. A floor sweep that measured at the padded width would agree with
    the profile about everything except the shape it measured."""
    context = ps.cell_context("B", batch=4, width=161, model="m", adapted=16)
    assert context["tokens"] == 4 * 160
    assert context["tokens"] != 4 * 161


def test_every_mode_leads_exactly_once_over_a_full_rotation():
    """Without rotation the first mode of every round always runs on a machine
    that has just been idle, so the mode order would be part of what the
    profile measures."""
    modes = ps.MODES_DECIDING
    leaders = [ps.rotated(modes, index)[0] for index in range(len(modes))]
    assert sorted(leaders) == sorted(modes)
    for index in range(len(modes)):
        rotation = ps.rotated(modes, index)
        assert set(rotation) == set(modes)
        assert rotation[1:] == tuple(
            modes[(index + offset) % len(modes)]
            for offset in range(1, len(modes)))


def test_only_the_cell_that_decides_splits_its_instrumented_passes():
    """One combined pass inflates the heavily marked region's share and
    deflates the others, which is exactly the three-way comparison the rule
    makes. The cells that decide nothing pay the bias and are labelled."""
    assert ps.modes_for(rules.PRIMARY_CELL) == ps.MODES_DECIDING
    for cell in rules.CELLS:
        if cell != rules.PRIMARY_CELL:
            assert ps.modes_for(cell) == ps.MODES_REPORTING


def test_every_cell_can_price_its_own_compile_transfer():
    """Amendment 4 reports the ratio beside EVERY share it publishes, and the
    cells that decide nothing still publish shares - section 4.3 requires them
    reported and requires cell C's ordering compared against cell B's."""
    for cell in rules.CELLS:
        assert "compiled" in ps.modes_for(cell), cell
        assert "plain" in ps.modes_for(cell), cell


def test_an_unregistered_cell_is_refused_rather_than_defaulted():
    with pytest.raises(RunInvalid):
        ps.modes_for("Z")


def test_each_split_mode_scores_one_candidate_and_the_combined_pass_scores_all():
    for candidate in rules.CANDIDATES:
        assert ps.candidates_from(f"instr-{candidate}") == (candidate,)
    assert sorted(ps.candidates_from("instr-all")) == sorted(rules.CANDIDATES)
    assert ps.candidates_from("plain") == ()
    assert ps.candidates_from("compiled") == ()


def test_every_mode_either_cell_runs_has_regions_registered_for_it():
    """A mode with no entry would raise inside the child, on the machine,
    after the lock was taken and the model was loaded."""
    for cell in rules.CELLS:
        for mode in ps.modes_for(cell):
            assert mode in ps.MODE_REGIONS
    for regions in ps.MODE_REGIONS.values():
        assert set(regions) <= set(pi.REGIONS)


# ---------------------------------------------------------------------------
# The counts: what proves a seam was reached
# ---------------------------------------------------------------------------
def test_the_backward_expectation_is_not_the_forward_one():
    """Under LoRA the blocks below the first adapted one have no trainable
    parameter beneath them and no backward at all. A profile that expected a
    backward per layer would accept a log that was missing most of one, and
    every share built on it would be wrong."""
    counts = ps.expected_counts(DEPTH, ADAPTED, ["attn-core", "qmm"])
    assert counts["attn-core"][pi.FORWARD] == DEPTH
    assert counts["attn-core"][pi.BACKWARD] == ADAPTED
    assert counts["qmm"][pi.FORWARD] == ps.PROJECTIONS_PER_BLOCK * DEPTH
    assert counts["qmm"][pi.BACKWARD] == (
        ps.PROJECTIONS_PER_BLOCK * ADAPTED - ps.INPUT_CONSUMING_PROJECTIONS)


def test_the_step_regions_fire_once_per_step_in_each_direction():
    counts = ps.expected_counts(DEPTH, ADAPTED, ["head-matmul",
                                                 "cross-entropy"])
    for region in ("head-matmul", "cross-entropy"):
        assert counts[region] == {pi.FORWARD: 1, pi.BACKWARD: 1}


@pytest.mark.parametrize("depth,adapted", [(0, 1), (36, 0), (36, 37), (-1, 1)])
def test_an_arrangement_mlx_lm_cannot_produce_is_refused(depth, adapted):
    with pytest.raises(RunInvalid):
        ps.expected_counts(depth, adapted, ["attn-core"])


def test_a_region_with_no_registered_count_is_refused_not_skipped():
    with pytest.raises(RunInvalid):
        ps.expected_counts(DEPTH, ADAPTED, ["attn-core", "swiglu"])


def test_a_seam_that_never_fired_is_caught_rather_than_read_as_zero():
    """The failure this check exists for. A region absent from the log has no
    span, so its share reads as zero and its time lands silently in the
    remainder; only a count can tell that apart from an operation that really
    costs nothing."""
    expected = ps.expected_counts(DEPTH, ADAPTED, ["attn-core", "qmm"])
    observed = {"attn-core": {pi.FORWARD: DEPTH, pi.BACKWARD: ADAPTED}}
    report = ps.completeness(observed, expected)
    assert not report["ok"]
    assert [m["region"] for m in report["mismatches"]] == ["qmm", "qmm"]
    assert all(m["observed"] is None for m in report["mismatches"])


def test_a_count_that_is_merely_short_is_caught_too():
    expected = ps.expected_counts(DEPTH, ADAPTED, ["attn-core"])
    observed = {"attn-core": {pi.FORWARD: DEPTH - 1, pi.BACKWARD: ADAPTED}}
    report = ps.completeness(observed, expected)
    assert not report["ok"]
    assert report["mismatches"] == [
        {"region": "attn-core", "direction": pi.FORWARD,
         "expected": DEPTH, "observed": DEPTH - 1}]


def test_counts_that_match_the_arrangement_pass():
    expected = ps.expected_counts(DEPTH, ADAPTED, sorted(pi.REGIONS))
    assert ps.completeness(expected, expected)["ok"]


# ---------------------------------------------------------------------------
# Reducing rounds to one reading
# ---------------------------------------------------------------------------
def test_the_median_is_taken_per_region_not_over_ratios():
    """Section 3.3 takes both sides of a share as medians over the eligible
    rounds, so this is a ratio of medians."""
    passes = [_decomposed({"qmm": value}, elapsed=2.0)
              for value in (0.10, 0.40, 0.30)]
    combined = ps.median_decomposition(passes)
    assert combined["totals"]["qmm"] == statistics.median([0.10, 0.40, 0.30])
    assert combined["passes"] == 3


def test_rounds_that_measured_different_workloads_are_not_averaged():
    passes = [_decomposed({"qmm": 0.1}),
              _decomposed({"qmm": 0.1}, context=dict(CTX, batch=1))]
    with pytest.raises(RunInvalid, match="not repeats"):
        ps.median_decomposition(passes)


def test_a_region_present_in_one_round_and_absent_in_another_is_refused():
    passes = [_decomposed({"qmm": 0.1, "head-matmul": 0.2}),
              _decomposed({"qmm": 0.1})]
    with pytest.raises(RunInvalid, match="cannot be reduced"):
        ps.median_decomposition(passes)


def test_rounds_whose_counts_moved_are_refused():
    """The workload changed between rounds, and a median across them describes
    neither of the two things that ran."""
    passes = [_decomposed({"qmm": 0.1}),
              _decomposed({"qmm": 0.1},
                          counts={"qmm": {pi.FORWARD: 2, pi.BACKWARD: 1}})]
    with pytest.raises(RunInvalid, match="workload changed"):
        ps.median_decomposition(passes)


def test_a_median_needs_at_least_one_pass():
    with pytest.raises(RunInvalid):
        ps.median_decomposition([])


# ---------------------------------------------------------------------------
# The cell reading
# ---------------------------------------------------------------------------
def test_each_share_at_the_deciding_cell_comes_from_its_own_pass():
    """The whole reason cell B splits: a share must carry only the cost of its
    own marks, and its denominator must come from the pass that carries
    none."""
    reading = ps.cell_reading(_cell_record())
    plain_total = statistics.median([0.98, 1.00, 1.02])
    assert reading["plain_total_s"] == plain_total
    assert reading["shares"]["A"]["from_mode"] == "instr-A"
    assert reading["shares"]["A"]["share"] == pytest.approx(0.30 / plain_total)
    assert reading["shares"]["L"]["share"] == pytest.approx(0.15 / plain_total)
    assert reading["shares"]["Q"]["share"] == pytest.approx(0.60 / plain_total)
    assert not any(share["cross_region_bias"]
                   for share in reading["shares"].values())


def test_the_output_head_is_counted_in_both_l_and_q():
    """The tied head IS a quantized matmul, so it belongs to both shares. The
    two therefore overlap and must never be composed, which is what Amendment
    5 records; here it is just asserted to be true of the arithmetic."""
    reading = ps.cell_reading(_cell_record())
    assert "head-matmul" in reading["shares"]["L"]["regions"]
    assert "head-matmul" in reading["shares"]["Q"]["regions"]


def test_a_reporting_cell_labels_its_shares_as_carrying_cross_region_bias():
    modes = {"compiled": _mode(samples=[1.0, 1.0, 1.0]),
             "plain": _mode(samples=[1.0, 1.0, 1.0]),
             "instr-all": _mode({"attn-core": 0.3, "qmm": 0.5,
                                 "head-matmul": 0.1, "cross-entropy": 0.05},
                                samples=[2.0, 2.0, 2.0],
                                context=dict(CTX, cell="C"))}
    reading = ps.cell_reading(_cell_record("C", modes=modes))
    assert set(reading["shares"]) == set(rules.CANDIDATES)
    assert all(share["cross_region_bias"]
               for share in reading["shares"].values())


def test_the_plain_arm_disagreeing_with_itself_makes_the_rounds_ineligible():
    """Interleaving the modes equalises a clock excursion across them but
    cannot detect one, so the reference arm's own spread is the detector."""
    record = _cell_record()
    record["modes"]["plain"] = _mode(samples=[0.5, 1.0, 1.5])
    reading = ps.cell_reading(record)
    assert not reading["rounds_eligible"]
    assert reading["plain_spread_pct"] == pytest.approx(100.0)


def test_a_steady_plain_arm_leaves_the_rounds_eligible():
    assert ps.cell_reading(_cell_record())["rounds_eligible"]


def test_a_share_refuses_to_divide_two_different_workloads():
    """The numerator comes from the instrumented pass and the denominator from
    the plain pass, so nothing but the context check stands between a split
    profile and a ratio of two batches."""
    record = _cell_record()
    record["modes"]["instr-A"] = _mode({"attn-core": 0.3},
                                       samples=[1.3, 1.3, 1.3],
                                       context=dict(CTX, width=999))
    with pytest.raises(RunInvalid, match="two workloads"):
        ps.cell_reading(record)


def test_the_reconciliation_is_measured_against_the_step_without_the_marks():
    """Amendment 4 says the regions and the remainder are measured inside the
    marked step and compared against "that same uncompiled step's own
    end-to-end time, taken without the interior boundaries". Compared against
    the marked pass's own elapsed it could not fail; compared against the
    plain pass, as registered, it is the instrument's cost against a 2%
    limit."""
    reading = ps.cell_reading(_cell_record())
    for mode, report in reading["reconciles"].items():
        assert not report["ok"], mode
        assert report["limit_pct"] == rules.RECONCILE_PCT
        assert report["same_check_as"].startswith("instrument_cost")
    # And the two really are one quantity read at two limits.
    plain = statistics.median([0.98, 1.00, 1.02])
    gap = reading["reconciles"]["instr-Q"]["gap_pct"] / 100.0
    cost = reading["instrument_cost"]["instr-Q"]["ratio"]
    assert gap == pytest.approx(cost - 1.0)


def test_a_cheap_enough_instrument_reconciles():
    reading = ps.cell_reading(_cell_record(cheap=True))
    assert all(report["ok"] for report in reading["reconciles"].values())


def test_a_real_instrument_cost_blocks_on_the_registered_two_percent(
        monkeypatch):
    """The state the profile is actually in. Section 3.3's limit was written
    before any instrument existed, and no instrument that can put a clock
    inside an MLX backward meets it."""
    monkeypatch.setattr(rules, "INSTRUMENT_COST_BAND", (1.0, 3.0))
    blockers = ps.binding_blockers(_record())
    assert any("section 3.3" in one for one in blockers)


def test_a_share_carries_its_own_marks_and_says_so(monkeypatch):
    """Measured 2026-08-20: the split-pass design stops one candidate's marks
    from deflating another's share, and does not stop a candidate's own marks
    from inflating its own. Every share is therefore an upper bound, and the
    record has to say that where the number is."""
    reading = ps.cell_reading(_cell_record())
    plain = statistics.median([0.98, 1.00, 1.02])
    for candidate, share in reading["shares"].items():
        assert "upper bound" in share["caveat"], candidate
        assert share["instrument_excess_s"] > 0.0, candidate
    assert reading["shares"]["Q"]["instrument_excess_s"] == pytest.approx(
        statistics.median([1.90, 1.92, 1.94]) - plain)


def test_a_share_that_is_not_a_fraction_of_a_step_blocks_the_recording(
        monkeypatch):
    """The reading a perturbation this size eventually produces. A share above
    one is not a share, and `gain` would refuse it much later and much less
    legibly."""
    monkeypatch.setattr(rules, "INSTRUMENT_COST_BAND", (1.0, 5.0))
    record = _cell_record()
    record["modes"]["instr-Q"] = _mode({"qmm": 1.20, "head-matmul": 0.30},
                                       samples=[2.9, 2.9, 2.9],
                                       context=dict(CTX, cell="B"))
    blockers = ps.binding_blockers(_record(cells={"B": record}))
    assert any("not a fraction of a step" in one for one in blockers)


def test_a_combined_pass_attributes_no_excess_to_any_one_candidate():
    """It marks every region, so the excess belongs to all of them and
    splitting it would need an apportionment nobody registered."""
    modes = {"compiled": _mode(samples=[1.0, 1.0, 1.0]),
             "plain": _mode(samples=[1.0, 1.0, 1.0]),
             "instr-all": _mode({"attn-core": 0.3, "qmm": 0.5,
                                 "head-matmul": 0.1, "cross-entropy": 0.05},
                                samples=[2.0, 2.0, 2.0],
                                context=dict(CTX, cell="C"))}
    reading = ps.cell_reading(_cell_record("C", modes=modes))
    assert all(share["instrument_excess_s"] is None
               for share in reading["shares"].values())


def test_the_instrument_cost_is_priced_per_mode():
    """Measured 2026-08-20 at 0.6B, the modes cost very differently: marking
    the loss pair cost 1.11x and marking the projections cost 1.93x. One band
    over one number would hide that."""
    reading = ps.cell_reading(_cell_record())
    plain = statistics.median([0.98, 1.00, 1.02])
    assert reading["instrument_cost"]["instr-Q"]["ratio"] == pytest.approx(
        statistics.median([1.90, 1.92, 1.94]) / plain)
    assert reading["instrument_cost"]["instr-L"]["ratio"] < \
        reading["instrument_cost"]["instr-Q"]["ratio"]


# ---------------------------------------------------------------------------
# What stops a recording from binding
# ---------------------------------------------------------------------------
def _record(**over):
    cells = over.pop("cells", {"B": _cell_record()})
    record = {"closing_idle": {"idle": True}, "cells": cells,
              "readings": {name: ps.cell_reading(one)
                           for name, one in cells.items()}}
    record.update(over)
    return record


def test_an_unregistered_instrument_cost_band_blocks_a_binding_recording():
    """Unregistered is not the same as passed. While no band exists the
    profile has no way to say the marked pass described the same step."""
    blockers = ps.binding_blockers(_record())
    assert any("no band is registered" in one for one in blockers)


def test_a_registered_band_the_cost_sits_inside_stops_blocking(monkeypatch):
    monkeypatch.setattr(rules, "INSTRUMENT_COST_BAND", (1.0, 2.5))
    assert ps.binding_blockers(
        _record(cells={"B": _cell_record(cheap=True)})) == []


def test_a_cost_outside_the_registered_band_blocks(monkeypatch):
    monkeypatch.setattr(rules, "INSTRUMENT_COST_BAND", (1.0, 1.5))
    blockers = ps.binding_blockers(_record())
    assert any("instr-Q" in one and "outside" in one for one in blockers)


def test_a_machine_that_went_busy_at_the_closing_gate_blocks(monkeypatch):
    monkeypatch.setattr(rules, "INSTRUMENT_COST_BAND", (1.0, 2.5))
    blockers = ps.binding_blockers(_record(
        cells={"B": _cell_record(cheap=True)}, closing_idle={"idle": False}))
    assert any("closing gate" in one for one in blockers)


def test_a_marked_pass_that_changed_the_computation_blocks(monkeypatch):
    monkeypatch.setattr(rules, "INSTRUMENT_COST_BAND", (1.0, 2.5))
    blockers = ps.binding_blockers(
        _record(cells={"B": _cell_record(identity_ok=False, cheap=True)}))
    assert any("do not compute the same thing" in one for one in blockers)


def test_a_seam_that_never_fired_blocks(monkeypatch):
    monkeypatch.setattr(rules, "INSTRUMENT_COST_BAND", (1.0, 2.5))
    blockers = ps.binding_blockers(
        _record(cells={"B": _cell_record(completeness_ok=False, cheap=True)}))
    assert any("region counts differ" in one for one in blockers)


def test_a_compile_ratio_outside_amendment_four_s_band_blocks(monkeypatch):
    monkeypatch.setattr(rules, "INSTRUMENT_COST_BAND", (1.0, 2.5))
    record = _cell_record(cheap=True)
    record["modes"]["compiled"] = _mode(samples=[0.4, 0.4, 0.4])
    blockers = ps.binding_blockers(_record(cells={"B": record}))
    assert any("compiled-to-uncompiled" in one for one in blockers)


# ---------------------------------------------------------------------------
# The plan, and what a stock profile may not be asked to do
# ---------------------------------------------------------------------------
def test_a_plan_may_not_name_a_kernel_for_a_profile_of_stock():
    """A plan that could name a kept kernel could produce a profile of
    somebody's kernel labelled as stock."""
    for banned in ("kept_candidates", "measurement_module"):
        with pytest.raises(RunInvalid, match="installs nothing"):
            ps.validate_plan(_plan(**{banned: ["x"]}))


def test_the_fixed_plan_says_in_writing_that_nothing_was_installed():
    fixed = ps.validate_plan(_plan())
    assert fixed["kept_candidates"] == []
    assert fixed["measurement_module"] == ps.STOCK_MEASUREMENT_MODULE


@pytest.mark.parametrize("over", [
    {"cells": []}, {"cells": ["Z"]}, {"schema_version": 2}, {"rounds": 0},
])
def test_a_plan_that_cannot_produce_a_reading_is_refused(over):
    with pytest.raises(RunInvalid):
        ps.validate_plan(_plan(**over))


def test_every_required_key_is_required():
    for key in ("data", "seed", "optimizer", "memory_ceiling_gb", "cells"):
        plan = _plan()
        plan.pop(key)
        with pytest.raises(RunInvalid, match="missing"):
            ps.validate_plan(plan)


# ---------------------------------------------------------------------------
# The recording
# ---------------------------------------------------------------------------
def test_a_refused_run_says_so_in_the_file_name():
    day = __import__("datetime").date(2026, 8, 20)
    assert ps.recording_path("r", day, kind="binding",
                             closing_idle=False).name.endswith(".REFUSED.json")
    assert ps.recording_path("r", day, kind="calibration",
                             closing_idle=True).name == (
        "profile-stock-calibration-2026-08-20.json")


def test_a_recording_is_never_overwritten(tmp_path):
    """A run that could overwrite its own recording could be re-run until it
    produced a number somebody liked, and nothing in the file would show it."""
    runtime = ps.SystemRuntime()
    path = tmp_path / "rec.json"
    runtime.write_record(path, {"a": 1})
    with pytest.raises(PreconditionFailed, match="never overwritten"):
        runtime.write_record(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


# ---------------------------------------------------------------------------
# The parent loop
# ---------------------------------------------------------------------------
class _Runtime:
    def __init__(self, cells=("B",)):
        self.events = []
        self.lock_result = (True, "acquired")
        self.idle_results = [{"idle": True, "blockers": []},
                             {"idle": True, "blockers": []}]
        self.cells = cells
        self.writes = []

    def acquire_lock(self):
        self.events.append("lock")
        return self.lock_result

    def release_lock(self):
        self.events.append("release")

    def fingerprint(self):
        return {"cores": 12, "chip": "Apple M3 Pro"}

    def idle_check(self, cores):
        return self.idle_results.pop(0)

    def require_memory(self, budget, cell):
        self.events.append(f"memory:{cell}")

    def preflight(self, plan):
        self.events.append("preflight")
        return {"plan_sha256": "plan-sha"}

    def run_cell(self, plan, provenance, cell):
        self.events.append(f"run:{cell}")
        return _cell_record(cell, cheap=True)

    def write_record(self, path, record):
        self.writes.append((Path(path), record))

    def today(self):
        return __import__("datetime").date(2026, 8, 20)


def test_a_binding_run_refuses_before_the_lock_while_no_band_is_registered():
    """The refusal that keeps the calibration run honest: a cost accepted
    after it was seen is not a limit, so the binding run cannot start until
    the band is written down."""
    runtime = _Runtime()
    assert ps.run_profile(_plan(), runtime) == EXIT_PRECONDITION
    assert runtime.events == []
    assert runtime.writes == []


def test_the_calibration_run_needs_no_band_and_binds_nothing():
    runtime = _Runtime()
    assert ps.run_profile(_plan(cells=["A", "B"]), runtime,
                          calibrate=True) == 0
    (path, record), = runtime.writes
    assert "calibration" in path.name
    assert record["binding"] is False
    # Only the primary cell: the calibration exists to price the instrument
    # where the instrument will be split, and nowhere else.
    assert list(record["cells"]) == [rules.PRIMARY_CELL]


def test_a_binding_run_with_a_registered_band_binds(monkeypatch):
    monkeypatch.setattr(rules, "INSTRUMENT_COST_BAND", (1.0, 2.5))
    runtime = _Runtime()
    assert ps.run_profile(_plan(), runtime) == 0
    (path, record), = runtime.writes
    assert record["binding"] is True
    assert record["binding_blockers"] == []
    assert path.name == "profile-stock-binding-2026-08-20.json"


def test_a_run_that_went_busy_at_the_closing_gate_records_and_does_not_bind(
        monkeypatch):
    monkeypatch.setattr(rules, "INSTRUMENT_COST_BAND", (1.0, 2.5))
    runtime = _Runtime()
    runtime.idle_results = [{"idle": True, "blockers": []},
                            {"idle": False, "blockers": ["load"]}]
    assert ps.run_profile(_plan(), runtime) == 1
    (path, record), = runtime.writes
    assert record["binding"] is False
    assert path.name.endswith(".REFUSED.json")


def test_a_busy_machine_at_the_opening_gate_measures_nothing(monkeypatch):
    monkeypatch.setattr(rules, "INSTRUMENT_COST_BAND", (1.0, 2.5))
    runtime = _Runtime()
    runtime.idle_results = [{"idle": False, "blockers": ["load"]}]
    assert ps.run_profile(_plan(), runtime) != 0
    assert runtime.writes == []
    assert "release" in runtime.events


def test_the_lock_is_released_even_when_a_cell_raises(monkeypatch):
    monkeypatch.setattr(rules, "INSTRUMENT_COST_BAND", (1.0, 2.5))
    runtime = _Runtime()
    runtime.run_cell = lambda *a, **k: (_ for _ in ()).throw(
        PreconditionFailed("no"))
    assert ps.run_profile(_plan(), runtime) == EXIT_PRECONDITION
    assert runtime.events[-1] == "release"


# ---------------------------------------------------------------------------
# The ruling
# ---------------------------------------------------------------------------
def _sweep(*, context=None, killed=False, deltas=None):
    ratios = {name: (1.05 if killed else 1.60) for name in rules.SHAPES}
    deltas = deltas or {"L": None, "A": None, "Q": None}
    return {
        "schema_version": 1, "binding": True, "binding_blockers": [],
        "cells": {"B": {
            "context": dict(context if context is not None
                            else dict(CTX, cell="B")),
            "ceiling_ratios": ratios,
            "candidates": {
                "L": {"numerator": [0.15, 0.16], "denominator": [0.05, 0.06],
                      "footprint_delta_bytes": deltas["L"],
                      "assumption": "a streamed kernel reaches stock "
                                    "throughput on the reduced problem"},
                "A": {"numerator": [0.30, 0.31], "denominator": [0.20, 0.21],
                      "footprint_delta_bytes": deltas["A"],
                      "assumption": "the fused backward would cost twice its "
                                    "forward"},
                "Q": {"per_shape": {
                          name: {"forward": {"count": 36, "numerator": [0.01],
                                             "denominator": [0.008]},
                                 "backward": {"count": 16, "numerator": [0.02],
                                              "denominator": [0.016]}}
                          for name in rules.SHAPES},
                      "footprint_delta_bytes": deltas["Q"],
                      "assumption": "dense fp16 is a ceiling because it skips "
                                    "dequantisation"},
            }}}}


def _profile(monkeypatch):
    monkeypatch.setattr(rules, "INSTRUMENT_COST_BAND", (1.0, 2.5))
    reporting = {"compiled": _mode(samples=[1.0, 1.0, 1.0]),
                 "plain": _mode(samples=[1.0, 1.0, 1.0]),
                 "instr-all": _mode({"attn-core": 0.3, "qmm": 0.5,
                                     "head-matmul": 0.1,
                                     "cross-entropy": 0.05},
                                    samples=[1.0, 1.01, 1.02],
                                    context=dict(CTX, cell="C"))}
    cells = {"B": _cell_record("B", cheap=True),
             "C": _cell_record("C", modes=reporting)}
    record = _record(cells=cells)
    record["binding_blockers"] = ps.binding_blockers(record)
    record["binding"] = not record["binding_blockers"]
    return record


def test_the_ruling_names_the_recordings_and_the_arithmetic_it_rested_on(
        monkeypatch):
    """A ruling that could not be traced to the exact numbers and the exact
    rule code that produced it is an opinion with a timestamp."""
    artifact = ps.decide(_profile(monkeypatch), _sweep())
    rested = artifact["rested_on"]
    assert set(rested) == {"profile_sha256", "sweep_sha256",
                           "profile_rules_sha256"}
    assert all(len(value) == 64 for value in rested.values())


def test_the_ruling_selects_and_carries_what_it_selected_on(monkeypatch):
    artifact = ps.decide(_profile(monkeypatch), _sweep())
    assert artifact["ruling"]["verdict"] in {
        "SELECTED", "NO SINGLE OPERATION REACHES THE FLOOR"}
    assert set(artifact["shares"]) == set(rules.CANDIDATES)
    assert artifact["cell"] == rules.PRIMARY_CELL


def test_a_killed_q_never_reaches_the_selection(monkeypatch):
    artifact = ps.decide(_profile(monkeypatch), _sweep(killed=True))
    assert artifact["kill_q"]["killed"]
    assert "Q" not in {row["candidate"] for row in artifact["ruling"]["ranked"]}


def test_a_live_q_does_reach_the_selection(monkeypatch):
    artifact = ps.decide(_profile(monkeypatch), _sweep())
    assert not artifact["kill_q"]["killed"]
    assert "Q" in {row["candidate"] for row in artifact["ruling"]["ranked"]}


def test_the_ruling_refuses_two_recordings_that_measured_different_workloads(
        monkeypatch):
    """The cross-harness hazard. The share comes from one run and the ratio
    from another, and nothing else in the pipeline would notice if the two had
    measured different batches."""
    sweep = _sweep(context=dict(CTX, cell="B", width=2049))
    with pytest.raises(RunInvalid, match="two workloads"):
        ps.decide(_profile(monkeypatch), sweep)


def test_the_ruling_refuses_a_profile_missing_a_candidate(monkeypatch):
    """A candidate absent from the table is not a candidate that scored zero,
    and a rule that ranked the rest would report a comparison of two where
    three were registered."""
    profile = _profile(monkeypatch)
    profile["readings"]["B"]["shares"].pop("A")
    with pytest.raises(RunInvalid, match="no share for"):
        ps.decide(profile, _sweep())


def test_the_ruling_refuses_a_sweep_missing_a_floor(monkeypatch):
    sweep = _sweep()
    sweep["cells"]["B"]["candidates"].pop("Q")
    with pytest.raises(RunInvalid, match="no credited ratio"):
        ps.decide(_profile(monkeypatch), sweep)


def test_the_ruling_refuses_a_recording_that_does_not_bind(monkeypatch):
    profile = _profile(monkeypatch)
    profile["binding"] = False
    profile["binding_blockers"] = ["the machine was not idle"]
    with pytest.raises(RunInvalid, match="does not bind"):
        ps.decide(profile, _sweep())


def test_the_footprint_tie_break_is_labelled_as_a_fact_about_the_floor(
        monkeypatch):
    """Amendment 5's substitution: the tie-break asks for the peak-footprint
    delta of a kernel that does not exist yet, so what stands in is the delta
    across the floor arms."""
    artifact = ps.decide(_profile(monkeypatch), _sweep())
    assert "about the floor" in artifact["footprint_delta_source"]
    assert "upper bound" in artifact["gain_is_an_upper_bound"]


def test_an_unmeasured_footprint_sends_a_tie_to_table_order(monkeypatch):
    artifact = ps.decide(_profile(monkeypatch), _sweep())
    assert artifact["ruling"].get("footprint_measured") in (False, None)


def test_the_other_cells_are_reported_and_decide_nothing(monkeypatch):
    artifact = ps.decide(_profile(monkeypatch), _sweep())
    assert set(artifact["reported_only"]) == {"C"}


# ---------------------------------------------------------------------------
# The child's refusals, which are all a caller ever sees of it
# ---------------------------------------------------------------------------
def test_the_child_refuses_when_the_device_is_switched_off(tmp_path,
                                                           monkeypatch):
    monkeypatch.setenv("KV_FORCE_NO_METAL", "1")
    assert ps.child_main(tmp_path / "task.json",
                         tmp_path / "out.json") == EXIT_NO_DEVICE


def test_the_child_refuses_a_kind_it_does_not_implement(tmp_path, monkeypatch):
    monkeypatch.delenv("KV_FORCE_NO_METAL", raising=False)
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"kind": "train", "budget_gb": 8.0,
                                "parent_pid": 1}))
    out = tmp_path / "out.json"
    monkeypatch.setattr(ps, "_ChildGuard", lambda *a, **k: type(
        "G", (), {"check": lambda self, cell: 0.0})())
    monkeypatch.setattr(ps, "stack_record", lambda: {})
    monkeypatch.setattr(ps, "_check_child_inputs", lambda task, stack: None)
    assert ps.child_main(task, out) == EXIT_PRECONDITION
    assert not out.exists()
