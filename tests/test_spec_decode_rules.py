"""CPU-only tests for the speculative-decode decision arithmetic."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from spec_decode_rules import (
    K_GRID,
    ExactCountMismatch,
    NoVerifyPasses,
    RoundSample,
    RunInvalid,
    accepted_per_pass,
    assert_exact_count,
    ceiling_for,
    decide_o1,
    decide_o2,
    decide_o3,
    decide_o4,
    delta_pct,
    eligible_rounds,
    expected_routed_calls,
    identity_labels,
    in_window,
    is_decider,
    noise_floor,
    spread_pct,
    verify_share,
)


def _round(
    *,
    a0: float = 100.0,
    a1: float = 100.0,
    a2: float = 100.0,
    a3: float = 100.0,
    identity: dict[str, int] | None = None,
    valid: bool = True,
) -> RoundSample:
    return RoundSample(
        generation_tps={0: a0, 1: a1, 2: a2, 3: a3},
        identity={} if identity is None else identity,
        valid=valid,
    )


def _grid(**overrides: list[RoundSample]) -> dict[int, list[RoundSample]]:
    cells = {k: [_round()] for k in K_GRID}
    cells.update({int(k): rounds for k, rounds in overrides.items()})
    return cells


# Adding or removing a registered K in spec_decode_rules.K_GRID makes this red.
def test_k_grid_is_the_registered_grid():
    assert K_GRID == (2, 4, 6, 8, 10)


# Treating K=2 or K=10 as routed in spec_decode_rules.in_window makes this red.
def test_null_cells_are_the_out_of_window_ks():
    assert [k for k in K_GRID if not in_window(k)] == [2, 10]


# Replacing the per-width lookup with one fixed wrapped-site count makes this red.
def test_expected_routed_calls_sums_per_site_and_only_routed_passes():
    shapes = [
        (1, 64, 2560),
        (1, 7, 2560),
        (1, 7, 2560),
        (1, 3, 2560),
    ]
    assert expected_routed_calls(
        shapes, {64: 0, 7: 252, 3: 0}
    ) == 504
    assert expected_routed_calls(
        shapes, {64: 0, 7: 216, 3: 0}
    ) == 432


# Replacing prod(shape[:-1]) with shape[-2] makes this red for M=6.
def test_expected_routed_calls_flattens_every_leading_dimension():
    assert expected_routed_calls([(2, 3, 2560)], {6: 17}) == 17


# Replacing true division with floor division makes the 3.2 result red.
def test_accepted_per_pass_is_tokens_over_verify_passes():
    assert accepted_per_pass(generation_tokens=128, verify_passes=40) == 3.2


# Deleting the zero-pass guard makes this raise ZeroDivisionError instead.
def test_zero_verify_passes_is_a_typed_refusal():
    with pytest.raises(NoVerifyPasses):
        accepted_per_pass(generation_tokens=128, verify_passes=0)


# Making NoVerifyPasses a bare arithmetic error breaks the common refusal type.
def test_no_verify_passes_is_run_invalid():
    assert issubclass(NoVerifyPasses, RunInvalid)


# Reversing verify_time and generation_time makes this return 4.0 instead.
def test_verify_share_uses_the_cell_level_time_totals():
    assert verify_share(verify_time=1.0, generation_time=4.0) == 0.25


# Averaging per-round shares instead of dividing the two sums makes this red.
def test_verify_share_sums_eligible_round_times_before_dividing():
    assert verify_share(
        verify_time=[0.1, 0.9], generation_time=[1.0, 3.0]
    ) == 0.25


# Substituting the unused M=6 gain or confirmation-run gains makes this red.
def test_ceiling_uses_every_registered_binding_input():
    assert {k: ceiling_for(k=k, share=0.5) for k in K_GRID} == pytest.approx(
        {2: 0.0, 4: 2.85, 6: 8.22, 8: 7.825, 10: 0.0}
    )


# Changing the strict ceiling > floor rule to >= makes the equality case red.
def test_decider_requires_the_ceiling_to_clear_the_floor_strictly():
    assert is_decider(ceiling=9.9, floor=0.3)
    assert not is_decider(ceiling=0.3, floor=0.3)
    assert not is_decider(ceiling=0.2, floor=0.3)


# Dividing by the mean instead of the median changes this from 60 percent.
def test_spread_pct_uses_the_median():
    assert spread_pct([8.0, 10.0, 14.0]) == 60.0


# Letting an empty sample reach statistics.median changes the typed error.
def test_empty_spread_is_run_invalid():
    with pytest.raises(RunInvalid):
        spread_pct([])


# Reversing numerator and denominator changes both signed deltas.
def test_delta_pct_preserves_comparison_direction():
    assert delta_pct(125.0, 100.0) == 25.0
    assert delta_pct(80.0, 100.0) == pytest.approx(-20.0)


# Returning either one arm's spread instead of their maximum makes one assertion red.
def test_noise_floor_is_the_larger_compared_arm_spread():
    quiet = [99.0, 100.0, 101.0]
    noisy = [80.0, 100.0, 120.0]
    assert noise_floor(quiet, noisy) == 40.0
    assert noise_floor(noisy, quiet) == 40.0


def _o1_grid(selected: list[RoundSample]) -> dict[int, list[RoundSample]]:
    return _grid(
        **{
            "2": [_round(a2=0.1)],
            "4": [_round(a2=0.2)],
            "6": selected,
            "8": [_round(a2=0.3)],
            "10": [_round(a2=0.4)],
        }
    )


# Swapping O1's positive branch to >= or comparing arm 0 over arm 2 makes this red.
def test_o1_win_requires_a_positive_delta_beyond_the_floor():
    result = decide_o1(_o1_grid([_round(a2=80.0, a0=64.0)]))
    assert result["primary_k"] == 6
    assert result["delta_pct"] == 25.0
    assert result["verdict"] == "WIN"


# Removing O1's negative-beyond-floor branch makes this verdict red.
def test_o1_deceleration_requires_a_negative_delta_beyond_the_floor():
    result = decide_o1(_o1_grid([_round(a2=60.0, a0=64.0)]))
    assert result["verdict"] == "DECELERATES"


# Changing O1's +floor comparison from strict to inclusive makes this WIN.
def test_o1_positive_floor_boundary_is_inconclusive():
    rounds = [
        _round(a2=1.0, a0=1.0),
        _round(a2=2.0, a0=1.0),
        _round(a2=3.0, a0=1.0),
    ]
    result = decide_o1(_o1_grid(rounds))
    assert result["delta_pct"] == result["noise_floor_pct"] == 100.0
    assert result["verdict"] == "INCONCLUSIVE"


# Changing O1's -floor comparison from strict to inclusive makes this DECELERATES.
def test_o1_negative_floor_boundary_is_inconclusive():
    rounds = [
        _round(a2=0.75, a0=2.0),
        _round(a2=1.0, a0=2.0),
        _round(a2=1.25, a0=2.0),
    ]
    result = decide_o1(_o1_grid(rounds))
    assert result["delta_pct"] == -result["noise_floor_pct"] == -50.0
    assert result["verdict"] == "INCONCLUSIVE"


# Breaking an exact best-median tie in favour of the larger K makes this red.
def test_o1_exact_best_k_tie_selects_the_smaller_k():
    cells = _grid(
        **{
            "2": [_round(a2=50.0)],
            "4": [_round(a2=80.0)],
            "6": [_round(a2=80.0)],
            "8": [_round(a2=70.0)],
            "10": [_round(a2=60.0)],
        }
    )
    assert decide_o1(cells)["primary_k"] == 4


# Keeping mlx-m-dependent rounds in O1 makes K=6 win instead of K=4.
def test_o1_selects_best_k_after_its_identity_exclusions():
    cells = _o1_grid(
        [
            _round(a2=70.0),
            _round(a2=1000.0, identity={"mlx-m-dependent": 2}),
        ]
    )
    cells[4] = [_round(a2=80.0)]
    assert decide_o1(cells)["primary_k"] == 4


# Allowing a registered null cell with equal arms to return anything but NULL makes this red.
@pytest.mark.parametrize("k", [2, 10])
def test_o2_registered_null_cells_read_null(k):
    result = decide_o2(
        k, [_round(a1=100.0, a2=100.0)], ceiling=0.0, routed_calls=0
    )
    assert result["decider"] is False
    assert result["verdict"] == "NULL"


# Ignoring routed_calls lets this invalid null control be reported as NULL.
def test_o2_null_cell_with_a_routed_call_invalidates_the_run():
    with pytest.raises(RunInvalid, match="K=2"):
        decide_o2(
            2, [_round(a1=100.0, a2=100.0)], ceiling=0.0, routed_calls=1
        )


# Validating only positive null-cell deltas lets the negative case escape.
@pytest.mark.parametrize("a1", [110.0, 90.0])
def test_o2_null_cell_outside_spread_in_either_direction_is_invalid(a1):
    with pytest.raises(RunInvalid, match="K=10"):
        decide_o2(
            10, [_round(a1=a1, a2=100.0)], ceiling=0.0, routed_calls=0
        )


# Reading the observed delta before ceiling <= floor makes this false WIN a verdict.
def test_o2_non_decider_is_not_interpreted_even_with_a_large_delta():
    rounds = [
        _round(a1=200.0, a2=95.0),
        _round(a1=200.0, a2=100.0),
        _round(a1=200.0, a2=105.0),
    ]
    result = decide_o2(4, rounds, ceiling=10.0, routed_calls=0)
    assert result["noise_floor_pct"] == 10.0
    assert result["decider"] is False
    assert result["verdict"] == "not-a-decider"


# Reversing O2's positive branch or making it inclusive changes this verdict.
def test_o2_decider_win_clears_the_floor():
    result = decide_o2(
        6, [_round(a1=110.0, a2=100.0)], ceiling=1.0, routed_calls=1
    )
    assert result["verdict"] == "WIN"


# Removing or reversing O2's negative branch changes this verdict.
def test_o2_decider_loss_clears_the_floor():
    result = decide_o2(
        6, [_round(a1=90.0, a2=100.0)], ceiling=1.0, routed_calls=1
    )
    assert result["verdict"] == "LOSS"


# Changing either exact spread boundary to WIN or LOSS makes one case red.
@pytest.mark.parametrize(
    "rounds",
    [
        [
            _round(a1=1.0, a2=1.0),
            _round(a1=2.0, a2=1.0),
            _round(a1=3.0, a2=1.0),
        ],
        [
            _round(a1=0.75, a2=2.0),
            _round(a1=1.0, a2=2.0),
            _round(a1=1.25, a2=2.0),
        ],
    ],
)
def test_o2_decider_spread_boundaries_are_null(rounds):
    result = decide_o2(6, rounds, ceiling=101.0, routed_calls=1)
    assert abs(result["delta_pct"]) == result["noise_floor_pct"]
    assert result["verdict"] == "NULL"


# Promoting the exploratory maximum to primary or changing its label makes this red.
def test_o3_keeps_k6_primary_and_labels_the_grid_max_exploratory():
    cells = _grid(
        **{
            "6": [_round(a0=100.0, a1=110.0, a2=105.0)],
            "8": [_round(a0=100.0, a1=150.0, a2=100.0)],
        }
    )
    result = decide_o3(cells)
    assert result["primary_k"] == 6
    assert result["composed_pct"] == pytest.approx(10.0)
    assert result["speculation_pct"] == pytest.approx(5.0)
    assert result["kernel_pct"] == pytest.approx(100 * (110 / 105 - 1))
    assert result["exploratory"] == {
        "delta_pct": 50.0,
        "ks": (8,),
        "label": "exploratory, selection-biased",
    }


# Selecting one unregistered winner instead of retaining tied exploratory Ks makes this red.
def test_o3_exploratory_ties_keep_every_tied_k():
    cells = _grid(
        **{
            "4": [_round(a0=100.0, a1=120.0)],
            "6": [_round(a0=100.0, a1=110.0)],
            "8": [_round(a0=100.0, a1=120.0)],
        }
    )
    assert decide_o3(cells)["exploratory"]["ks"] == (4, 8)


# Changing the copied A/B threshold table makes at least one attribution red.
@pytest.mark.parametrize(
    "round_sample,want",
    [
        (_round(a0=100.0, a1=100.0, a2=100.0), "inconclusive"),
        (_round(a0=100.0, a1=90.0, a2=100.0), "negative"),
        (_round(a0=100.0, a1=120.0, a2=110.0), "artifact-alone"),
        (_round(a0=100.0, a1=110.0, a2=90.0), "joint"),
    ],
)
def test_o3_uses_every_registered_attribution_branch(round_sample, want):
    assert decide_o3(_grid(**{"6": [round_sample]}))["attribution"] == want


# Making comp_pct == either signed floor conclusive makes one case red.
@pytest.mark.parametrize(
    "rounds",
    [
        [
            _round(a0=1.0, a1=1.0),
            _round(a0=1.0, a1=2.0),
            _round(a0=1.0, a1=3.0),
        ],
        [
            _round(a0=2.0, a1=0.75),
            _round(a0=2.0, a1=1.0),
            _round(a0=2.0, a1=1.25),
        ],
    ],
)
def test_o3_composed_spread_boundaries_are_inconclusive(rounds):
    assert decide_o3(_grid(**{"6": rounds}))["attribution"] == "inconclusive"


# Changing base_pct > floor to >= changes this exact-boundary label.
def test_o3_base_at_the_floor_is_joint():
    rounds = [
        _round(a0=1.0, a1=1.5, a2=1.5),
        _round(a0=1.0, a1=2.0, a2=1.5),
        _round(a0=1.0, a1=2.5, a2=1.5),
    ]
    result = decide_o3(_grid(**{"6": rounds}))
    assert result["noise_floor_pct"] == result["speculation_pct"] == 50.0
    assert result["attribution"] == "joint"


# Wiring O4 to arm 0 instead of arm 3 changes the ratio and attribution.
def test_o4_is_ours_vs_stock_4bit_at_k6():
    cells = _grid(
        **{"6": [_round(a0=1000.0, a1=120.0, a2=110.0, a3=100.0)]}
    )
    result = decide_o4(cells)
    assert result["primary_k"] == 6
    assert result["ratio_composed_vs_4bit"] == 1.2
    assert result["composed_pct"] == pytest.approx(20.0)
    assert result["base_pct"] == pytest.approx(10.0)
    assert result["attribution"] == "artifact-alone"


# Bypassing the shared attribution helper in decide_o4 makes one branch red.
@pytest.mark.parametrize(
    "round_sample,want",
    [
        (_round(a0=1000.0, a1=100.0, a2=100.0, a3=100.0), "inconclusive"),
        (_round(a0=1000.0, a1=90.0, a2=100.0, a3=100.0), "negative"),
        (_round(a0=1000.0, a1=120.0, a2=110.0, a3=100.0), "artifact-alone"),
        (_round(a0=1000.0, a1=110.0, a2=90.0, a3=100.0), "joint"),
    ],
)
def test_o4_uses_every_registered_attribution_branch(round_sample, want):
    assert decide_o4(_grid(**{"6": [round_sample]}))["attribution"] == want


# Comparing only equal-length prefixes makes the three equal sequences look divergent.
def test_identity_labels_are_empty_when_required_sequences_match():
    assert identity_labels(
        a1=[1, 2, 3], a2=[1, 2, 3], a0=[1, 2, 3], a4=[1, 2, 3]
    ) == {}


# Renaming or merging the kernel-diverged comparison makes this red.
def test_identity_labels_name_the_kernel_mismatch_position():
    assert identity_labels(
        a1=[1, 2, 3], a2=[1, 2, 4], a0=[1, 2, 4], a4=[1, 2, 4]
    ) == {"kernel-diverged": 2}


# Renaming or merging the mlx-m-dependent comparison makes this red.
def test_identity_labels_name_the_mlx_mismatch_position():
    assert identity_labels(
        a1=[1, 2, 4], a2=[1, 2, 4], a0=[1, 2, 3], a4=[1, 2, 4]
    ) == {"mlx-m-dependent": 2}


# Changing the second identity comparison to elif drops one simultaneous label.
def test_identity_labels_can_report_both_named_mismatches():
    assert identity_labels(
        a1=[1, 9, 4], a2=[1, 2, 4], a0=[1, 2, 3], a4=[1, 2, 4]
    ) == {"kernel-diverged": 1, "mlx-m-dependent": 2}


# Using zip without a sequence-length check misses this strict-prefix mismatch.
def test_identity_labels_report_the_first_missing_token_position():
    assert identity_labels(
        a1=[1, 2], a2=[1, 2, 3], a0=[1, 2, 3], a4=[1, 2, 3]
    ) == {"kernel-diverged": 2}


# Removing the arm-4 equality check lets a control that is not stock survive.
def test_arm4_differing_from_arm2_invalidates_the_run():
    with pytest.raises(RunInvalid, match="arm 4"):
        identity_labels(
            a1=[1, 2, 4], a2=[1, 2, 4], a0=[1, 2, 4], a4=[9, 9, 9]
        )


# Applying one global divergence exclusion changes at least one outcome below.
def test_each_outcome_applies_only_its_registered_identity_exclusions():
    kernel = {"kernel-diverged": 0}
    mlx = {"mlx-m-dependent": 0}
    both = {**kernel, **mlx}
    cells = _grid(
        **{
            "4": [_round(a0=100.0, a1=100.0, a2=80.0)],
            "6": [
                _round(a0=100.0, a1=110.0, a2=100.0, a3=100.0),
                _round(
                    a0=1.0,
                    a1=1000.0,
                    a2=1000.0,
                    a3=1.0,
                    identity=both,
                ),
                _round(
                    a0=100.0,
                    a1=110.0,
                    a2=100.0,
                    a3=100.0,
                    identity=mlx,
                ),
                _round(
                    a0=1.0,
                    a1=1000.0,
                    a2=1000.0,
                    a3=1.0,
                    identity=kernel,
                ),
            ],
        }
    )
    assert decide_o1(cells)["primary_k"] == 6
    assert decide_o2(6, cells[6], ceiling=1000.0, routed_calls=1)[
        "eligible_rounds"
    ] == 2
    assert decide_o3(cells)["eligible_rounds"] == 2
    # O4 excludes kernel-diverged too, ruled 2026-08-18 before any measurement:
    # the two rounds carrying that label go, the mlx-only round stays.
    assert decide_o4(cells)["eligible_rounds"] == 2


# Letting an invalid hard-fallback round enter any outcome changes these medians.
def test_hard_fallback_invalid_rounds_are_ineligible_for_every_outcome():
    cells = _grid(
        **{
            "6": [
                _round(a0=1.0, a1=1000.0, a2=1000.0, a3=1.0, valid=False),
                _round(a0=100.0, a1=110.0, a2=100.0, a3=100.0),
            ]
        }
    )
    assert decide_o1(cells)["arm2_tps"] == 100.0
    assert decide_o2(6, cells[6], ceiling=1.0, routed_calls=1)[
        "arm1_tps"
    ] == 110.0
    assert decide_o3(cells)["arm1_tps"] == 110.0
    assert decide_o4(cells)["arm1_tps"] == 110.0


# The harness sums arm 2's verify and generation times over the O2-eligible
# rounds (doc, section 5) and must not own a second copy of the exclusion rule.
# Re-deriving eligibility inside serve_spec_decode.py rather than calling this
# is what these three tests exist to make impossible.
def test_public_eligibility_applies_both_filters_for_the_named_outcome():
    kept = _round(a2=100.0)
    rounds = [
        _round(a2=1.0, valid=False),
        _round(a2=2.0, identity={"kernel-diverged": 3}),
        kept,
    ]
    assert eligible_rounds(rounds, "O2") == (kept,)
    # O1 excludes the MLX label, not the kernel one, so the same input keeps a
    # different pair: an outcome-blind filter would return the same tuple twice.
    assert len(eligible_rounds(rounds, "O1")) == 2


# Letting the public selector and the verdict's own selection diverge makes this red.
def test_the_public_selection_is_the_one_every_verdict_reads():
    rounds = [
        _round(a1=1000.0, a2=1000.0, identity={"kernel-diverged": 1}),
        _round(a1=110.0, a2=100.0),
        _round(a1=112.0, a2=100.0),
    ]
    decision = decide_o2(6, rounds, ceiling=1.0, routed_calls=1)
    assert len(eligible_rounds(rounds, "O2")) == decision["eligible_rounds"]


# Returning an empty tuple instead of refusing would let the harness divide by zero.
def test_public_eligibility_refuses_when_nothing_survives():
    with pytest.raises(RunInvalid):
        eligible_rounds([_round(valid=False)], "O2")


# Passing empty paired sets to statistics.median instead of RunInvalid makes these red.
@pytest.mark.parametrize(
    "call",
    [
        lambda: decide_o1(
            {k: [_round(identity={"mlx-m-dependent": 0})] for k in K_GRID}
        ),
        lambda: decide_o2(
            6,
            [_round(identity={"kernel-diverged": 0})],
            ceiling=1.0,
            routed_calls=1,
        ),
        lambda: decide_o3(
            {k: [_round(identity={"kernel-diverged": 0})] for k in K_GRID}
        ),
        lambda: decide_o4(_grid(**{"6": [_round(valid=False)]})),
    ],
)
def test_no_eligible_paired_round_invalidates_every_outcome(call):
    with pytest.raises(RunInvalid):
        call()


# Raising on equal counts instead of only observed != expected makes this red.
def test_exact_count_accepts_an_exact_match():
    assert assert_exact_count(
        observed=252, expected=252, cell="qwen3-0.6b K=6 arm 1"
    ) is None


# Returning instead of raising, or omitting the cell text, makes this red.
def test_exact_count_mismatch_is_typed_and_names_the_cell():
    with pytest.raises(ExactCountMismatch, match="qwen3-0.6b K=6 arm 1"):
        assert_exact_count(
            observed=251, expected=252, cell="qwen3-0.6b K=6 arm 1"
        )


# Making ExactCountMismatch unrelated to RunInvalid breaks the harness catch path.
def test_exact_count_mismatch_is_run_invalid():
    assert issubclass(ExactCountMismatch, RunInvalid)


def _import_roots(module_name: str) -> set[str]:
    source = (Path(__file__).parents[1] / "bench" / f"{module_name}.py").read_text()
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


# Adding `import numpy` or `import serve_sub4bit` to either module makes this red,
# and so does adding a non-stdlib import to the shared sibling, which is the point:
# the guarantee is that importing this module cannot reach mlx.nn, and it has to
# hold through what it imports rather than only at its own top line. A machine
# with no Metal device ABORTS the interpreter on `import mlx.nn` rather than
# raising, so an offending import one level down would take the whole collection
# with it and this test would never get to fail.
def test_the_rules_module_cannot_reach_outside_the_standard_library():
    allowed_siblings = {"attribution"}
    roots = _import_roots("spec_decode_rules")
    assert roots <= sys.stdlib_module_names | allowed_siblings
    for sibling in roots & allowed_siblings:
        assert _import_roots(sibling) <= sys.stdlib_module_names, (
            f"bench/{sibling}.py is imported by the rules module, so its own "
            "imports have to be standard library too")
