"""CPU-only tests for the batched-decode decision arithmetic."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from batch_decode_rules import (
    B_GRID,
    IN_ZONE,
    LOOP_KERNEL_RATIO,
    PERPLEXITY_PAIR,
    arm_summary,
    ceiling_for,
    decide_ob1,
    decide_ob2,
    decide_ob3,
    decide_zone,
    decode_passes,
    excluded_round_cap,
    expected_routed_calls,
    per_stream_tps,
    stream_identity_labels,
    validate_arm_record,
)
from decode_rules import (
    RoundSample,
    RunInvalid,
    classify_passes,
    clears,
    is_decider,
    result_line,
)


def _round(
    *,
    a1: float = 110.0,
    a2: float = 100.0,
    a3: float = 100.0,
    a4: float = 100.0,
    identity: dict[str, object] | None = None,
    valid: bool = True,
) -> RoundSample:
    return RoundSample(
        generation_tps={1: a1, 2: a2, 3: a3, 4: a4},
        identity={} if identity is None else identity,
        valid=valid,
    )


def _arm_record(
    *,
    generation_tokens: int = 12,
    tokens: list[list[int]] | None = None,
    finish_reasons: list[str] | None = None,
) -> dict:
    return {
        "generation_tokens": generation_tokens,
        "tokens": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        if tokens is None
        else tokens,
        "finish_reasons": ["length"] * 4
        if finish_reasons is None
        else finish_reasons,
    }


def _zone_cells() -> dict[int, list[RoundSample]]:
    return {b: [_round()] for b in IN_ZONE}


def _zone_counts() -> tuple[dict[int, int], dict[int, int]]:
    return ({b: 1 for b in IN_ZONE}, {b: 1 for b in IN_ZONE})


# Treating the one registered prefill as a decode pass changes both tuples.
def test_classify_passes_accepts_matching_width_and_one_prefill():
    passes, prefill = classify_passes(
        [(5, 1), (1, 2555)],
        is_pass=lambda width: width == 5,
        prefill_width=2555,
    )
    assert passes == ((5, 1),)
    assert prefill == ((1, 2555),)


# Allowing two candidate prefills makes the counted seam ambiguous.
def test_classify_passes_refuses_a_second_prefill():
    with pytest.raises(RunInvalid, match="two"):
        classify_passes(
            [(1, 5), (1, 6)],
            is_pass=lambda width: width == 4,
            prefill_width=20,
        )


# Accepting an unregistered width would hide a stray width-8 call at B=16.
def test_classify_passes_refuses_an_unknown_width():
    with pytest.raises(RunInvalid, match="8.*8176"):
        classify_passes(
            [(1, 8)],
            is_pass=lambda width: width == 16,
            prefill_width=16 * 511,
        )


# Flattening a rank-3 activation shape would admit a call from the wrong seam.
def test_classify_passes_refuses_a_non_rank_two_shape():
    with pytest.raises(RunInvalid, match="rank 3"):
        classify_passes(
            [(1, 5, 2560)],
            is_pass=lambda width: width == 5,
            prefill_width=2555,
        )


# Counting the routed prefill would make expected and observed agree by construction.
def test_expected_routed_calls_sums_decode_passes_only():
    observed = [(5, 1), (5, 511)]
    passes = decode_passes(observed, b=5, prompt_t=512)
    assert expected_routed_calls(passes, {5: 3, 2555: 999}) == 3


# Returning aggregate throughput would overstate one stream by the batch size.
def test_per_stream_tps_divides_the_aggregate_by_batch_size():
    assert per_stream_tps(120.0, 6) == 20.0


# Extending a measured ceiling outside the routed zone would create false deciders.
def test_ceiling_for_is_registered_in_zone_and_zero_outside():
    assert {b: ceiling_for(b) for b in IN_ZONE} == {
        5: 5.70,
        6: 14.69,
        7: 16.44,
        8: 15.65,
        9: 15.65,
    }
    assert {b: ceiling_for(b) for b in set(B_GRID) - IN_ZONE} == {
        1: 0.0,
        4: 0.0,
        11: 0.0,
        12: 0.0,
        16: 0.0,
    }


# Comparing streams globally instead of per stream would invent a mismatch label.
def test_stream_identity_labels_are_empty_when_every_stream_matches():
    streams = [[1, 2], [3, 4]]
    assert stream_identity_labels(a1=streams, a2=streams, a4=streams) == {}


# Dropping stream coordinates would make a divergence impossible to locate.
def test_stream_identity_labels_name_stream_and_position():
    assert stream_identity_labels(
        a1=[[1, 9], [3, 8]],
        a2=[[1, 2], [3, 4]],
        a4=[[1, 2], [3, 4]],
    ) == {"kernel-diverged": {0: 1, 1: 1}}


# Zipping equal prefixes without a length check misses the first absent token.
def test_stream_identity_labels_name_the_first_missing_position():
    assert stream_identity_labels(
        a1=[[1], [3, 4]],
        a2=[[1, 2], [3, 4]],
        a4=[[1, 2], [3, 4]],
    ) == {"kernel-diverged": {0: 1}}


# Letting arm 4 differ from arm 2 destroys the forced-stock control.
def test_stream_identity_labels_refuse_a_different_control():
    with pytest.raises(RunInvalid, match="stream 1.*position 1"):
        stream_identity_labels(
            a1=[[1, 2], [3, 4]],
            a2=[[1, 2], [3, 4]],
            a4=[[1, 2], [3, 9]],
        )


# Accepting the wrong aggregate token count changes the clock denominator.
def test_validate_arm_record_refuses_the_wrong_generation_total():
    with pytest.raises(RunInvalid, match="generation_tokens"):
        validate_arm_record(_arm_record(generation_tokens=11), b=4, gen_tokens=3)


# Accepting a short stream would make aggregate divided by B inexact.
def test_validate_arm_record_refuses_a_short_stream():
    tokens = [[1, 2, 3], [4, 5], [7, 8, 9], [10, 11, 12]]
    with pytest.raises(RunInvalid, match="stream 1"):
        validate_arm_record(_arm_record(tokens=tokens), b=4, gen_tokens=3)


# Accepting a stop finish would permit a decode width smaller than B.
def test_validate_arm_record_refuses_a_finish_other_than_length():
    with pytest.raises(RunInvalid, match="finish reason.*stream 2"):
        validate_arm_record(
            _arm_record(finish_reasons=["length", "length", "stop", "length"]),
            b=4,
            gen_tokens=3,
        )


# Marking a cell unstable without any excluded round would discard valid evidence.
def test_excluded_round_cap_allows_zero_divergences():
    assert excluded_round_cap([_round()]) is None


# Tightening the cap to zero would discard the one registered rare divergence.
def test_excluded_round_cap_allows_one_divergence():
    rounds = [_round(identity={"kernel-diverged": {0: 1}}), _round()]
    assert excluded_round_cap(rounds) is None


# Allowing a second divergent round would let exclusions hide a pattern.
def test_excluded_round_cap_marks_two_divergences_unstable():
    rounds = [
        _round(identity={"kernel-diverged": {0: 1}}),
        _round(identity={"kernel-diverged": {1: 2}}),
        _round(),
    ]
    assert excluded_round_cap(rounds) == "identity-unstable"


# Reversing the arm-1 over arm-2 comparison changes this clear win.
def test_ob1_in_zone_win_clears_the_floor_upward():
    result = decide_ob1(6, [_round()], routed_calls=1, routed_sites_at_width=1)
    assert result["delta_pct"] == pytest.approx(10.0)
    assert result["noise_floor_pct"] == 0.0
    assert result["verdict"] == "WIN"


# Removing the downward branch changes a clear kernel loss to NULL.
def test_ob1_in_zone_loss_clears_the_floor_downward():
    result = decide_ob1(
        6,
        [_round(a1=90.0)],
        routed_calls=1,
        routed_sites_at_width=1,
    )
    assert result["verdict"] == "LOSS"


# Making the strict floor boundary inclusive changes this NULL to WIN.
def test_ob1_in_zone_delta_equal_to_floor_is_null():
    rounds = [
        _round(a1=105.46875),
        _round(a1=112.5),
        _round(a1=119.53125),
    ]
    result = decide_ob1(6, rounds, routed_calls=1, routed_sites_at_width=1)
    assert result["delta_pct"] == result["noise_floor_pct"] == 12.5
    assert result["verdict"] == "NULL"


# Reading a powered-down cell's large point estimate creates a false verdict.
def test_ob1_in_zone_non_decider_is_never_null():
    rounds = [
        _round(a1=200.0, a2=90.0),
        _round(a1=200.0, a2=100.0),
        _round(a1=200.0, a2=110.0),
    ]
    result = decide_ob1(5, rounds, routed_calls=1, routed_sites_at_width=1)
    assert result["decider"] is False
    assert result["verdict"] == "not-a-decider"


# Comparing a null cell against arm 2 instead of arm 4 rejects a clean control.
def test_ob1_null_cell_reads_null_when_table_calls_and_control_agree():
    result = decide_ob1(
        1,
        [_round(a1=100.0, a2=90.0, a4=100.0)],
        routed_calls=0,
        routed_sites_at_width=0,
    )
    assert result["delta_pct"] == pytest.approx((100 / 90 - 1) * 100)
    assert result["control_delta_pct"] == 0.0
    assert result["verdict"] == "NULL"


# Ignoring either routing control would let an intercepted null cell look clean.
@pytest.mark.parametrize(
    "routed_calls,routed_sites",
    [(1, 0), (0, 1)],
)
def test_ob1_null_cell_with_routing_is_uncontrolled(routed_calls, routed_sites):
    result = decide_ob1(
        16,
        [_round(a1=100.0, a4=100.0)],
        routed_calls=routed_calls,
        routed_sites_at_width=routed_sites,
    )
    assert result["verdict"] == "NULL-uncontrolled"


# Letting a control delta clear F14 would attribute host drift to a null kernel.
def test_ob1_null_cell_outside_the_control_floor_is_uncontrolled():
    result = decide_ob1(
        16,
        [_round(a1=110.0, a4=100.0)],
        routed_calls=0,
        routed_sites_at_width=0,
    )
    assert result["control_delta_pct"] == pytest.approx(10.0)
    assert result["verdict"] == "NULL-uncontrolled"


# Reusing F12 for the capability comparison changes both delta and floor here.
def test_ob2_reports_its_own_delta_floor_and_perplexity_pair():
    rounds = [
        _round(a1=105.46875, a2=50.0, a3=100.0),
        _round(a1=112.5, a2=50.0, a3=100.0),
        _round(a1=119.53125, a2=50.0, a3=100.0),
    ]
    result = decide_ob2(6, rounds)
    assert result["delta_pct"] == result["noise_floor_pct"] == 12.5
    assert result["perplexity_pair"] == PERPLEXITY_PAIR


# Changing composed attribution loses at least one registered OB2 reading.
@pytest.mark.parametrize(
    "round_sample,want",
    [
        (_round(a1=100.0, a2=100.0, a3=100.0), "inconclusive"),
        (_round(a1=90.0, a2=100.0, a3=100.0), "negative"),
        (_round(a1=120.0, a2=110.0, a3=100.0), "artifact-alone"),
        (_round(a1=110.0, a2=90.0, a3=100.0), "joint"),
    ],
)
def test_ob2_uses_every_registered_attribution_branch(round_sample, want):
    assert decide_ob2(6, [round_sample])["attribution"] == want


# Treating equality with the loop as an upward finding changes this reading.
def test_ob3_at_or_below_loop_includes_equal_ratio():
    loop_ratio = LOOP_KERNEL_RATIO[5]
    result = decide_ob3(
        5,
        [_round(a1=100 * loop_ratio, a2=100.0, a3=100.0)],
    )
    assert result["ratio_delta_pct"] == pytest.approx(0.0)
    assert result["verdict"] == "AT-OR-BELOW-LOOP"


# Making the threshold inclusive reports a boundary value as above-loop.
def test_ob3_engine_above_loop_is_strictly_beyond_the_threshold():
    loop_ratio = LOOP_KERNEL_RATIO[5]
    boundary = decide_ob3(
        5,
        [_round(a1=100 * loop_ratio * 1.0005, a2=100.0, a3=100.0)],
    )
    above = decide_ob3(
        5,
        [_round(a1=100 * loop_ratio * 1.000501, a2=100.0, a3=100.0)],
    )
    assert boundary["threshold_pct"] == 0.05
    assert boundary["verdict"] == "AT-OR-BELOW-LOOP"
    assert above["verdict"] == "ENGINE-ABOVE-LOOP"


# Applying an absolute threshold would mislabel a large downward move as above-loop.
def test_ob3_above_loop_finding_is_upward_only():
    result = decide_ob3(5, [_round(a1=50.0, a2=100.0, a3=100.0)])
    assert result["ratio_delta_pct"] < 0
    assert result["verdict"] == "AT-OR-BELOW-LOOP"


# Inventing a loop baseline at an unmeasured width changes this registered label.
@pytest.mark.parametrize("b", [7, 9])
def test_ob3_unmeasured_width_has_no_loop_number(b):
    result = decide_ob3(b, [_round()])
    assert result["verdict"] == "no-loop-number"


# Requiring anything less than all five passing cells weakens the zone claim.
def test_zone_is_go_only_when_every_in_zone_cell_passes():
    counts, sites = _zone_counts()
    result = decide_zone(
        _zone_cells(),
        routed_calls_by_b=counts,
        routed_sites_by_b=sites,
    )
    assert result["verdict"] == "GO"


# Ignoring one OB1 loss would let a failed width hide inside the range.
def test_zone_is_no_go_for_one_ob1_loss():
    cells = _zone_cells()
    cells[7] = [_round(a1=90.0)]
    counts, sites = _zone_counts()
    assert decide_zone(
        cells, routed_calls_by_b=counts, routed_sites_by_b=sites
    )["verdict"] == "NO-GO"


# Ignoring one OB1 null would turn an unresolved width into a product claim.
def test_zone_is_no_go_for_one_ob1_null():
    cells = _zone_cells()
    cells[7] = [_round(a1=100.0)]
    counts, sites = _zone_counts()
    assert decide_zone(
        cells, routed_calls_by_b=counts, routed_sites_by_b=sites
    )["verdict"] == "NO-GO"


# Ignoring one non-decider would claim evidence the design could not resolve.
def test_zone_is_no_go_for_one_non_decider():
    cells = _zone_cells()
    cells[5] = [
        _round(a1=200.0, a2=90.0),
        _round(a1=200.0, a2=100.0),
        _round(a1=200.0, a2=110.0),
    ]
    counts, sites = _zone_counts()
    assert decide_zone(
        cells, routed_calls_by_b=counts, routed_sites_by_b=sites
    )["verdict"] == "NO-GO"


# Reading through a second divergence would let exclusions hide instability.
def test_zone_is_no_go_for_one_identity_unstable_cell():
    cells = _zone_cells()
    cells[6] = [
        _round(identity={"kernel-diverged": {0: 1}}),
        _round(identity={"kernel-diverged": {1: 2}}),
        _round(),
    ]
    counts, sites = _zone_counts()
    result = decide_zone(
        cells, routed_calls_by_b=counts, routed_sites_by_b=sites
    )
    assert result["cells"][6] == "identity-unstable"
    assert result["verdict"] == "NO-GO"


# Treating a capability delta inside F13 as sufficient weakens the zone gate.
def test_zone_is_no_go_when_one_ob2_delta_is_inside_its_floor():
    cells = _zone_cells()
    cells[8] = [_round(a1=110.0, a2=100.0, a3=110.0)]
    counts, sites = _zone_counts()
    assert decide_zone(
        cells, routed_calls_by_b=counts, routed_sites_by_b=sites
    )["verdict"] == "NO-GO"


# Silently skipping a registered width would make a partial grid look complete.
def test_zone_refuses_a_missing_in_zone_cell():
    cells = _zone_cells()
    del cells[9]
    counts, sites = _zone_counts()
    with pytest.raises(RunInvalid, match="B=9"):
        decide_zone(cells, routed_calls_by_b=counts, routed_sites_by_b=sites)


# Keeping only one endpoint cell would hide exact ties in the quoted range.
def test_zone_ranges_keep_every_tied_endpoint():
    counts, sites = _zone_counts()
    result = decide_zone(
        _zone_cells(),
        routed_calls_by_b=counts,
        routed_sites_by_b=sites,
    )
    assert result["ob1_delta_range_pct"] == {
        "min": pytest.approx(10.0),
        "min_bs": (5, 6, 7, 8, 9),
        "max": pytest.approx(10.0),
        "max_bs": (5, 6, 7, 8, 9),
    }
    assert result["ob2_delta_range_pct"]["min_bs"] == (5, 6, 7, 8, 9)
    assert result["ob2_delta_range_pct"]["max_bs"] == (5, 6, 7, 8, 9)


# Omitting fixed fields or serializing floats as strings makes the line unusable.
def test_result_line_merges_fields_outcome_and_plain_float_results():
    payload = json.loads(
        result_line({"batch": 5}, "OB1", {"delta_pct": 5.7})[8:]
    )
    assert payload == {"batch": 5, "outcome": "OB1", "delta_pct": 5.7}
    assert isinstance(payload["delta_pct"], float)


def _import_roots(module_name: str) -> set[str]:
    source = (Path(__file__).parents[1] / "bench" / f"{module_name}.py").read_text()
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


# Adding a GPU or third-party dependency to the shared core makes this red.
def test_decode_rules_imports_only_stdlib_plus_machine_state():
    assert _import_roots("decode_rules") <= sys.stdlib_module_names | {
        "machine_state"
    }


# Adding a GPU or unrelated sibling dependency to batch rules makes this red.
def test_batch_rules_import_only_the_registered_cpu_only_modules():
    assert _import_roots("batch_decode_rules") <= sys.stdlib_module_names | {
        "attribution",
        "decode_rules",
        "machine_state",
    }


# The three boundary tests below were added at the merge gate after mutation
# testing showed the suite accepted a loosened comparison at each of them.
# Section 5 registers "a delta equal to its floor does not clear it" and
# section 6 registers OB2 as clearing "upward", so each of these is a rule the
# document fixed before the run rather than a preference about strictness.


# Reading OB2 with abs() instead of a signed comparison makes this red, which
# is the mutation that matters most: a cell where our kernel is far SLOWER
# than stock 4-bit would otherwise count as clearing its floor and could carry
# the zone to GO.
def test_ob2_a_large_loss_does_not_clear_the_floor():
    rounds = [
        _round(a1=80.0, a2=100.0, a3=100.0, a4=100.0),
        _round(a1=80.0, a2=100.0, a3=100.0, a4=100.0),
    ]
    result = decide_ob2(6, rounds)
    assert result["delta_pct"] == pytest.approx(-20.0)
    assert result["clears_floor_upward"] is False


# Every registered floor comparison goes through this one rule, so loosening
# any of them to >= makes this red. Section 5 registers that a delta equal to
# its floor does not clear it. The equality is asserted here rather than
# through decide_ob1, decide_ob2 or decide_ob3 because the registered floors
# and ceilings are decimal values with no exact binary form, so no sample set
# drives a delta onto one of them exactly.
def test_a_delta_equal_to_its_floor_does_not_clear_it():
    assert clears(0.10, 0.10) is False
    assert clears(0.10000000000000002, 0.10) is True
    assert clears(0.09, 0.10) is False
    # The downward direction is the same rule on the negated delta, so a LOSS
    # exactly at the floor is a NULL rather than a LOSS.
    assert clears(-0.10, 0.10) is False


# Reading OB3 against the wrong threshold, or comparing the composed ratio
# instead of the kernel ratio, makes this red. The threshold at B = 6 is the
# larger of this run's kernel floor and the published loop floor of 0.10.
def test_ob3_compares_the_kernel_ratio_against_the_larger_threshold():
    loop_ratio = LOOP_KERNEL_RATIO[6]
    engine_ratio = loop_ratio * 1.01                # a full point above the loop
    rounds = [
        _round(a1=100.0 * engine_ratio, a2=100.0),
        _round(a1=100.0 * engine_ratio, a2=100.0),
    ]
    result = decide_ob3(6, rounds)
    assert result["threshold_pct"] == pytest.approx(0.10)
    assert result["ratio_delta_pct"] == pytest.approx(1.0)
    assert result["verdict"] == "ENGINE-ABOVE-LOOP"

    below = [
        _round(a1=100.0 * loop_ratio * 0.999, a2=100.0),
        _round(a1=100.0 * loop_ratio * 0.999, a2=100.0),
    ]
    assert decide_ob3(6, below)["verdict"] == "AT-OR-BELOW-LOOP"


# Loosening the decider rule to >= makes the first assertion red. Section 5
# registers a decider as ceiling STRICTLY greater than the floor, and a cell
# whose ceiling exactly equals its floor could not have decided anything. The
# equality is asserted on the rule itself because the registered ceilings have
# no exact binary form, so no sample set drives a floor to one of them exactly;
# the second assertion pins that decide_ob1 feeds the rule this cell's own
# ceiling and floor, in that order.
def test_a_ceiling_at_or_below_the_floor_is_not_a_decider():
    assert is_decider(ceiling=5.70, floor=5.70) is False
    assert is_decider(ceiling=5.70, floor=5.69) is True

    # B = 5 has the grid's smallest ceiling, 5.70 percent; arm-1 samples
    # spanning 8 percent of their median put F12 above it.
    rounds = [
        _round(a1=96.0, a2=100.0, a3=100.0, a4=100.0),
        _round(a1=104.0, a2=100.0, a3=100.0, a4=100.0),
    ]
    result = decide_ob1(5, rounds, routed_calls=1, routed_sites_at_width=1)
    assert result["noise_floor_pct"] > result["ceiling_pct"]
    assert result["decider"] is False
    assert result["verdict"] == "not-a-decider"


# Dropping the host-cost pair leaves section 5's "arm 4 against arm 2,
# reported" to be divided by hand, with no floor to read it against.
def test_ob1_reports_the_interception_host_cost_against_stock():
    result = decide_ob1(
        6,
        [_round(a1=110.0, a2=100.0, a4=98.0)],
        routed_calls=1,
        routed_sites_at_width=1,
    )
    assert result["host_cost_delta_pct"] == pytest.approx((98 / 100 - 1) * 100)
    assert result["host_cost_floor_pct"] == 0.0


# A host cost read from an excluded round would carry the divergence it exists
# to detect.
def test_the_host_cost_reads_the_eligible_rounds_only():
    rounds = [
        _round(a1=110.0, a2=100.0, a4=98.0),
        _round(a1=110.0, a2=100.0, a4=50.0, identity={"kernel-diverged": {0: 1}}),
    ]
    result = decide_ob1(6, rounds, routed_calls=1, routed_sites_at_width=1)
    assert result["host_cost_delta_pct"] == pytest.approx((98 / 100 - 1) * 100)


# Section 5 asks for a median and a spread per arm per cell; without them a
# reader cannot see which arm was noisy, only the larger of the two floors.
def test_arm_summary_gives_the_median_and_spread_over_eligible_rounds():
    rounds = [
        _round(a1=100.0),
        _round(a1=110.0),
        _round(a1=120.0),
    ]
    summary = arm_summary(6, rounds, 1)
    assert summary["median_tps"] == pytest.approx(110.0)
    assert summary["spread_pct"] == pytest.approx(100 * (120 - 100) / 110)
    assert summary["eligible_rounds"] == 3


# Summarising an excluded round would describe an arm by tokens the identity
# check already rejected.
def test_arm_summary_leaves_out_an_excluded_round():
    rounds = [
        _round(a1=100.0),
        _round(a1=110.0),
        _round(a1=900.0, identity={"kernel-diverged": {0: 2}}),
        _round(a1=100.0, valid=False),
    ]
    summary = arm_summary(6, rounds, 1)
    assert summary["median_tps"] == pytest.approx(105.0)
    assert summary["eligible_rounds"] == 2
