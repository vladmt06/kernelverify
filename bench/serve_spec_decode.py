"""End-to-end speculative decode across the pre-registered five-arm grid.

The method and outcomes are fixed in
``docs/research/2026-08-18-spec-decode-e2e.md`` sections 1 through 8.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402
from mlx_lm.generate import stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from machine_state import MeasurementLock  # noqa: E402
from memory_guard import (  # noqa: E402
    EXIT_BUDGET_REFUSAL,
    EXIT_LOCK_HELD,
    EXIT_LOW_MEMORY,
    EXIT_NOT_IDLE,
    EXIT_PRECONDITION,
    BudgetExceeded,
    LowMemoryRefusal,
    budget_gb_arg,
)
from model_calls import _CallCounter, counting_calls  # noqa: E402
from serve_sub4bit import (  # noqa: E402
    MODEL_3BIT,
    MODEL_4BIT,
    MODELS_ROOT,
    NotIdle,
    PreconditionFailed,
    ServeGuard,
    check_idle_after,
    install_patch,
    load_model,
    make_prompts,
    provenance,
    require_idle,
    require_pinned_zone,
    routed_sites_at,
    routed_sites_for,
    verify_pins,
)
from spec_decode_rules import (  # noqa: E402
    K_GRID,
    PRIMARY_K,
    RoundSample,
    RunInvalid,
    accepted_per_pass,
    assert_exact_count,
    ceiling_for,
    decide_o1,
    decide_o2,
    decide_o3,
    decide_o4,
    eligible_rounds,
    expected_routed_calls,
    identity_labels,
    verification_passes,
    verify_share,
)


LOCK_NAME = "serve_spec_decode"
BUDGET_GB = 24.0
PROMPT_T = 64
GEN_TOKENS = 128
ROUNDS = 5
DRAFT_MODELS = (
    "qwen3-0.6b-4bit-g64",
    "qwen3-1.7b-4bit-g64",
)
ARM_ORDER = (1, 2, 3, 4, 0)


def _single_prompt(prompts):
    shape = getattr(prompts, "shape", None)
    if shape is not None and len(shape) == 2:
        return prompts[0]
    return prompts


def _generate(target, tokenizer, prompt, *, draft_model, k, sync, cell):
    tokens = []
    last_response = None
    with counting_calls(target, sync=sync) as counter:
        sampler = make_sampler(temp=0.0)
        if draft_model is None:
            responses = stream_generate(
                target,
                tokenizer,
                prompt,
                max_tokens=GEN_TOKENS,
                sampler=sampler,
            )
        else:
            responses = stream_generate(
                target,
                tokenizer,
                prompt,
                max_tokens=GEN_TOKENS,
                draft_model=draft_model,
                num_draft_tokens=k,
                sampler=sampler,
            )
        for response in responses:
            tokens.append(response.token)
            last_response = response

    if last_response is None:
        raise RunInvalid(f"{cell}: stream_generate returned no response")
    return {
        "tokens": tokens,
        "generation_tokens": last_response.generation_tokens,
        "generation_tps": last_response.generation_tps,
        "generation_time": (
            last_response.generation_tokens / last_response.generation_tps
        ),
        "finish_reason": last_response.finish_reason,
        "target_call_shapes": list(counter.shapes),
        "counter_seconds": list(counter.seconds),
    }


def _run_arm(
    arm: int,
    model3,
    model4,
    tokenizer,
    prompt,
    draft_model,
    k: int,
    *,
    sync: bool = False,
    cell: str,
):
    target = model4 if arm == 3 else model3
    patch = None
    routed_calls = 0
    site_cells = ()
    hard_fallbacks = {}
    try:
        if arm in (1, 4):
            patch = install_patch(target)
            patch.mode = "fused" if arm == 1 else "stock"
            patch.reset()

        run = _generate(
            target,
            tokenizer,
            prompt,
            draft_model=None if arm == 0 else draft_model,
            k=None if arm == 0 else k,
            sync=sync,
            cell=cell,
        )
        if patch is not None:
            routed_calls = patch.calls
            site_cells = tuple(patch.site_cells)
            hard_fallbacks = dict(patch.hard_fallbacks())
    finally:
        if patch is not None:
            patch.uninstall()

    # The classifier is the rules module's, so the rule lives in one place;
    # the cell's K applies to arm 0 as well, whose width-1 decode steps all
    # sit under K+1 and whose one prefill is the same width-63 call.
    try:
        verification_shapes = verification_passes(
            run["target_call_shapes"], k=k, prompt_t=PROMPT_T
        )
    except RunInvalid as error:
        raise RunInvalid(f"{cell}: {error}") from error
    run["verification_shapes"] = verification_shapes
    run["verify_passes"] = len(verification_shapes)
    if arm != 0:
        run["accepted_per_pass"] = accepted_per_pass(
            run["generation_tokens"], run["verify_passes"]
        )

    run["routed_calls"] = routed_calls
    run["hard_fallbacks"] = hard_fallbacks
    run["site_cells"] = site_cells
    if arm == 1:
        # Section 4: the sum runs over the observed VERIFICATION passes. It
        # would agree numerically over every call today only because the
        # table declines the prefill width, and that is not the rule.
        routed_sites = routed_sites_for(verification_shapes, site_cells)
        run["routed_sites_at"] = routed_sites
        run["expected_routed_calls"] = expected_routed_calls(
            verification_shapes, routed_sites
        )
    elif arm == 4:
        run["expected_routed_calls"] = 0
    return run


def _public_sample(run, round_number: int, round_sample: RoundSample):
    sample = {
        "round": round_number,
        "tokens": run["tokens"],
        "generation_tokens": run["generation_tokens"],
        "generation_tps": run["generation_tps"],
        "generation_time": run["generation_time"],
        "finish_reason": run["finish_reason"],
        "target_call_shapes": run["target_call_shapes"],
        "counter_seconds": run["counter_seconds"],
        "identity": dict(round_sample.identity),
        "valid": round_sample.valid,
    }
    if "accepted_per_pass" in run:
        sample["verify_passes"] = run["verify_passes"]
        sample["accepted_per_pass"] = run["accepted_per_pass"]
    if "expected_routed_calls" in run:
        sample["routed_calls"] = run["routed_calls"]
        sample["expected_routed_calls"] = run["expected_routed_calls"]
        sample["hard_fallbacks"] = run["hard_fallbacks"]
        if "routed_sites_at" in run:
            sample["routed_sites_at"] = run["routed_sites_at"]
    return sample


def _result_line(draft: str, outcome: str, result: dict) -> str:
    return "RESULT: " + json.dumps(
        {"draft": draft, "outcome": outcome, **result}
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--budget-gb",
        type=budget_gb_arg,
        default=BUDGET_GB,
        help="phys_footprint budget in decimal GB",
    )
    # A registered input, not a knob: the parent pre-registration fixed K=6
    # and the K=4 follow-up names 4 in writing before its run (docs/research/
    # 2026-08-18-spec-decode-k4-followup.md, section 3). Reading any other
    # cell as primary needs its own pre-registration first.
    parser.add_argument(
        "--primary-k",
        type=int,
        default=PRIMARY_K,
        choices=K_GRID,
        help="the pre-registered primary cell O3 and O4 read "
             f"(default {PRIMARY_K})",
    )
    args = parser.parse_args(argv)

    try:
        manifest = verify_pins(
            MODELS_ROOT,
            [MODEL_3BIT, MODEL_4BIT, *DRAFT_MODELS],
        )
    except RuntimeError as error:
        print(f"REFUSED (exit {EXIT_PRECONDITION}): {error}")
        return EXIT_PRECONDITION

    lock = MeasurementLock(LOCK_NAME)
    acquired, detail = lock.acquire()
    if not acquired:
        print(
            f"REFUSAL (exit {EXIT_LOCK_HELD}): machine measurement lock "
            f"{detail}; one heavy measurement at a time"
        )
        return EXIT_LOCK_HELD

    exit_code = 0
    current_cell = "startup"
    try:
        require_idle("before")
        require_pinned_zone()
        guard = ServeGuard(args.budget_gb)
        print(json.dumps(provenance(manifest)))

        model3, tokenizer = load_model(MODEL_3BIT)
        model4, _ = load_model(MODEL_4BIT)
        drafts = []
        for draft_name in DRAFT_MODELS:
            draft_model, _ = load_model(draft_name)
            drafts.append((draft_name, draft_model))
        prompt = _single_prompt(make_prompts(tokenizer, PROMPT_T, 1))

        cells_by_draft = {}
        metadata_by_draft = {}
        for draft_name, draft_model in drafts:
            cells = {}
            cell_metadata = {}
            for k in K_GRID:
                cell = f"{draft_name} K={k}"
                current_cell = cell
                mx.clear_cache()
                guard(cell)

                for arm in ARM_ORDER:
                    _run_arm(
                        arm,
                        model3,
                        model4,
                        tokenizer,
                        prompt,
                        draft_model,
                        k,
                        cell=f"{cell} warm arm{arm}",
                    )
                    guard(f"{cell} warm arm{arm}")

                probe = _run_arm(
                    2,
                    model3,
                    model4,
                    tokenizer,
                    prompt,
                    draft_model,
                    k,
                    sync=True,
                    cell=f"{cell} probe arm2",
                )
                guard(f"{cell} probe arm2")
                probe_width_seconds = [
                    seconds
                    for shape, seconds in zip(
                        probe["target_call_shapes"], probe["counter_seconds"]
                    )
                    if math.prod(shape) == k + 1
                ]
                if not probe_width_seconds:
                    raise RunInvalid(
                        f"{cell}: arm 2 probe observed no width "
                        f"{k + 1} verification pass"
                    )
                per_pass_seconds = statistics.median(probe_width_seconds)

                runs_by_arm = {arm: [] for arm in ARM_ORDER}
                rounds = []
                arm2_by_round = {}
                for round_index in range(ROUNDS):
                    round_cell = f"{cell} round {round_index + 1}"
                    current_cell = round_cell
                    round_runs = {}
                    for arm in ARM_ORDER:
                        run = _run_arm(
                            arm,
                            model3,
                            model4,
                            tokenizer,
                            prompt,
                            draft_model,
                            k,
                            cell=f"{round_cell} arm{arm}",
                        )
                        round_runs[arm] = run
                        runs_by_arm[arm].append(run)
                        guard(f"{round_cell} arm{arm}")

                    identity = identity_labels(
                        a1=round_runs[1]["tokens"],
                        a2=round_runs[2]["tokens"],
                        a0=round_runs[0]["tokens"],
                        a4=round_runs[4]["tokens"],
                    )
                    valid = not (
                        round_runs[1]["hard_fallbacks"]
                        or round_runs[4]["hard_fallbacks"]
                    )
                    round_sample = RoundSample(
                        generation_tps={
                            arm: round_runs[arm]["generation_tps"]
                            for arm in (0, 1, 2, 3)
                        },
                        identity=identity,
                        valid=valid,
                    )
                    rounds.append(round_sample)
                    arm2_by_round[id(round_sample)] = round_runs[2]

                current_cell = cell
                # A hard fallback invalidates only its round. Counting it here
                # would turn that registered exclusion into a run-wide stop.
                valid_indices = [
                    index for index, round_sample in enumerate(rounds)
                    if round_sample.valid
                ]
                for index in valid_indices:
                    run = runs_by_arm[1][index]
                    assert_exact_count(
                        run["routed_calls"],
                        run["expected_routed_calls"],
                        f"{cell} round {index + 1} arm 1",
                    )
                for index, run in enumerate(runs_by_arm[4]):
                    assert_exact_count(
                        run["routed_calls"],
                        0,
                        f"{cell} round {index + 1} arm 4",
                    )
                routed_calls = sum(
                    run["routed_calls"] for run in runs_by_arm[1]
                )
                expected_calls = sum(
                    run["expected_routed_calls"] for run in runs_by_arm[1]
                )
                control_calls = sum(
                    run["routed_calls"] for run in runs_by_arm[4]
                )

                eligible = eligible_rounds(rounds, "O2")
                eligible_arm2 = [
                    arm2_by_round[id(round_sample)]
                    for round_sample in eligible
                ]
                share = verify_share(
                    [
                        per_pass_seconds * run["verify_passes"]
                        for run in eligible_arm2
                    ],
                    [run["generation_time"] for run in eligible_arm2],
                )
                ceiling = ceiling_for(k, share)
                site_cells = runs_by_arm[1][0]["site_cells"]
                routed_sites_at_width = routed_sites_at(k + 1, site_cells)

                for arm in ARM_ORDER:
                    row = {
                        "draft": draft_name,
                        "k": k,
                        "arm": arm,
                        "gen": GEN_TOKENS,
                        "rounds": ROUNDS,
                        "samples": [
                            _public_sample(run, index + 1, rounds[index])
                            for index, run in enumerate(runs_by_arm[arm])
                        ],
                    }
                    if arm == 1:
                        row.update(
                            {
                                "routed_calls": routed_calls,
                                "expected_routed_calls": expected_calls,
                                "routed_sites_at_width": (
                                    routed_sites_at_width
                                ),
                            }
                        )
                    elif arm == 2:
                        row.update(
                            {
                                "probe_target_call_shapes": probe[
                                    "target_call_shapes"
                                ],
                                "probe_seconds": probe["counter_seconds"],
                                "probe_k_plus_1_seconds": (
                                    probe_width_seconds
                                ),
                                "median_probe_pass_seconds": (
                                    per_pass_seconds
                                ),
                                "verify_share": share,
                                "ceiling_pct": ceiling,
                            }
                        )
                    elif arm == 4:
                        row.update(
                            {
                                "routed_calls": control_calls,
                                "expected_routed_calls": 0,
                            }
                        )
                    print(json.dumps(row))

                cells[k] = rounds
                cell_metadata[k] = {
                    "ceiling": ceiling,
                    "routed_calls": routed_calls,
                    "routed_sites_at_width": routed_sites_at_width,
                }
            cells_by_draft[draft_name] = cells
            metadata_by_draft[draft_name] = cell_metadata

        result_lines = []
        for draft_name, _ in drafts:
            cells = cells_by_draft[draft_name]
            metadata = metadata_by_draft[draft_name]
            current_cell = f"{draft_name} O1"
            result_lines.append(
                _result_line(draft_name, "O1", decide_o1(cells))
            )
            for k in K_GRID:
                current_cell = f"{draft_name} K={k} O2"
                result_lines.append(
                    _result_line(
                        draft_name,
                        "O2",
                        decide_o2(
                            k,
                            cells[k],
                            ceiling=metadata[k]["ceiling"],
                            routed_calls=metadata[k]["routed_calls"],
                            routed_sites_at_width=metadata[k][
                                "routed_sites_at_width"
                            ],
                        ),
                    )
                )
            current_cell = f"{draft_name} O3"
            result_lines.append(
                _result_line(
                    draft_name, "O3",
                    decide_o3(cells, primary_k=args.primary_k),
                )
            )
            current_cell = f"{draft_name} O4"
            result_lines.append(
                _result_line(
                    draft_name, "O4",
                    decide_o4(cells, primary_k=args.primary_k),
                )
            )
        for line in result_lines:
            print(line)
    except NotIdle as error:
        print(f"REFUSAL (exit {EXIT_NOT_IDLE}): {error}")
        return EXIT_NOT_IDLE
    except PreconditionFailed as error:
        print(f"REFUSAL (exit {EXIT_PRECONDITION}): {error}")
        return EXIT_PRECONDITION
    except BudgetExceeded as error:
        print(f"REFUSAL (exit {EXIT_BUDGET_REFUSAL}): {error}")
        return EXIT_BUDGET_REFUSAL
    except LowMemoryRefusal as error:
        print(f"REFUSAL (exit {EXIT_LOW_MEMORY}): {error}")
        return EXIT_LOW_MEMORY
    except RunInvalid as error:
        print(f"STOP: {current_cell}: {error}")
        exit_code = 1
    finally:
        lock.release()
    return check_idle_after(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
