"""Measure batched decode through mlx-lm's engine under docs/research/2026-08-18-batch-decode-e2e.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402
from mlx_lm.generate import BatchGenerator  # noqa: E402

from batch_decode_rules import (  # noqa: E402
    B_GRID,
    IN_ZONE,
    decide_ob1,
    decide_ob2,
    decide_ob3,
    decide_zone,
    decode_passes,
    expected_routed_calls,
    stream_identity_labels,
    validate_arm_record,
)
from decode_rules import (  # noqa: E402
    RoundSample,
    RunInvalid,
    assert_exact_count,
    result_line,
)
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
from model_calls import counting_calls  # noqa: E402
from serve_sub4bit import (  # noqa: E402
    MODEL_3BIT,
    MODEL_4BIT,
    MODELS_ROOT,
    PROMPT_SEED,
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


LOCK_NAME = "serve_batch_decode"
BUDGET_GB = 24.0
PROMPT_T = 512
GEN_TOKENS = 128
ROUNDS = 5
PROMPT_STRIDE = 64
ARMS = (1, 2, 3, 4)
ENGINE = {
    "completion_batch_size": 32,
    "prefill_batch_size": 16,
    "prefill_step_size": 2048,
}


def _prompt_digest(token_lists) -> str:
    return hashlib.sha256(
        json.dumps(token_lists, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stream_prompts(tokenizer, t: int, b: int, *, stride: int):
    ids = tokenizer.encode(PROMPT_SEED * 40)
    required = (b - 1) * stride + t
    if len(ids) < required:
        raise RuntimeError(
            f"prompt stream has {len(ids)} tokens, but B={b}, T={t}, and "
            f"stride={stride} require {required}"
        )
    token_lists = [
        ids[stream * stride:stream * stride + t]
        for stream in range(b)
    ]
    return mx.array(token_lists, dtype=mx.int32), token_lists


def _run_arm(arm, model3, model4, tokenizer, prompts, *, cell):
    target = model4 if arm == 3 else model3
    prompt_lists = prompts.tolist()
    b = len(prompt_lists)
    patch = None
    routed_calls = 0
    site_cells = ()
    hard_fallbacks = {}
    try:
        if arm in (1, 4):
            patch = install_patch(target)
            patch.mode = "fused" if arm == 1 else "stock"
            patch.reset()

        gen = BatchGenerator(
            target,
            max_tokens=GEN_TOKENS,
            stop_tokens=None,
            **ENGINE,
        )
        uids = gen.insert(prompt_lists, [GEN_TOKENS] * len(prompt_lists))
        tokens = {uid: [] for uid in uids}
        finish = {}
        try:
            with counting_calls(target) as counter, gen.stats() as stats:
                while responses := gen.next_generated():
                    for response in responses:
                        tokens[response.uid].append(response.token)
                        if response.finish_reason is not None:
                            finish[response.uid] = response.finish_reason
        finally:
            gen.close()

        stats_record = dict(vars(stats))
        run = {
            "tokens": [tokens[uid] for uid in uids],
            "finish_reasons": [finish.get(uid) for uid in uids],
            **stats_record,
            "target_call_shapes": list(counter.shapes),
        }
        if patch is not None:
            routed_calls = patch.calls
            site_cells = tuple(patch.site_cells)
            hard_fallbacks = dict(patch.hard_fallbacks())
    finally:
        if patch is not None:
            patch.uninstall()

    try:
        decode_shapes = decode_passes(
            run["target_call_shapes"],
            b=b,
            prompt_t=PROMPT_T,
        )
        validate_arm_record(run, b=b, gen_tokens=GEN_TOKENS)
    except RunInvalid as error:
        raise RunInvalid(f"{cell}: {error}") from error

    run["decode_shapes"] = decode_shapes
    run["decode_passes"] = len(decode_shapes)
    run["routed_calls"] = routed_calls
    run["site_cells"] = site_cells
    run["hard_fallbacks"] = hard_fallbacks
    if arm == 1:
        routed_sites = routed_sites_for(decode_shapes, site_cells)
        run["routed_sites_at"] = routed_sites
        run["expected_routed_calls"] = expected_routed_calls(
            decode_shapes,
            routed_sites,
        )
    elif arm == 4:
        run["expected_routed_calls"] = 0
    return run


def _public_sample(run, round_number: int, round_sample: RoundSample):
    return {
        "round": round_number,
        **run,
        "identity": dict(round_sample.identity),
        "valid": round_sample.valid,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--budget-gb",
        type=budget_gb_arg,
        default=BUDGET_GB,
        help="phys_footprint budget in decimal GB",
    )
    args = parser.parse_args(argv)

    try:
        manifest = verify_pins(MODELS_ROOT, [MODEL_3BIT, MODEL_4BIT])
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
        prompts_by_b = {}
        prompt_digests = {}
        for b in B_GRID:
            prompts, token_lists = _stream_prompts(
                tokenizer,
                PROMPT_T,
                b,
                stride=PROMPT_STRIDE,
            )
            prompts_by_b[b] = prompts
            prompt_digests[b] = _prompt_digest(token_lists)
        print(f"PROMPTS: {json.dumps(prompt_digests)}")

        cells = {}
        routed_calls_by_b = {}
        routed_sites_by_b = {}
        for b in B_GRID:
            cell = f"B={b}"
            current_cell = cell
            mx.clear_cache()
            guard(cell)
            prompts = prompts_by_b[b]

            for arm in ARMS:
                _run_arm(
                    arm,
                    model3,
                    model4,
                    tokenizer,
                    prompts,
                    cell=f"{cell} warm arm{arm}",
                )
                guard(f"{cell} warm arm{arm}")

            runs_by_arm = {arm: [] for arm in ARMS}
            rounds = []
            for round_index in range(ROUNDS):
                round_number = round_index + 1
                round_cell = f"{cell} round {round_number}"
                current_cell = round_cell
                offset = round_index % len(ARMS)
                round_arms = ARMS[offset:] + ARMS[:offset]
                round_runs = {}
                for arm in round_arms:
                    run = _run_arm(
                        arm,
                        model3,
                        model4,
                        tokenizer,
                        prompts,
                        cell=f"{round_cell} arm{arm}",
                    )
                    round_runs[arm] = run
                    runs_by_arm[arm].append(run)
                    guard(f"{round_cell} arm{arm}")

                pass_counts = {
                    arm: round_runs[arm]["decode_passes"] for arm in ARMS
                }
                if len(set(pass_counts.values())) != 1:
                    raise RunInvalid(
                        f"{round_cell}: decode passes differ across arms: "
                        f"{pass_counts}"
                    )
                identity = stream_identity_labels(
                    a1=round_runs[1]["tokens"],
                    a2=round_runs[2]["tokens"],
                    a4=round_runs[4]["tokens"],
                )
                valid = not (
                    round_runs[1]["hard_fallbacks"]
                    or round_runs[4]["hard_fallbacks"]
                )
                rounds.append(
                    RoundSample(
                        generation_tps={
                            arm: round_runs[arm]["generation_tps"]
                            for arm in ARMS
                        },
                        identity=identity,
                        valid=valid,
                    )
                )

            current_cell = cell
            for index, run in enumerate(runs_by_arm[1]):
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

            routed_calls = sum(run["routed_calls"] for run in runs_by_arm[1])
            expected_calls = sum(
                run["expected_routed_calls"] for run in runs_by_arm[1]
            )
            control_calls = sum(
                run["routed_calls"] for run in runs_by_arm[4]
            )
            site_cells = runs_by_arm[1][0]["site_cells"]
            routed_sites_at_width = routed_sites_at(b, site_cells)

            for arm in ARMS:
                row = {
                    "b": b,
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
                            "routed_sites_at_width": routed_sites_at_width,
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

            cells[b] = rounds
            routed_calls_by_b[b] = routed_calls
            routed_sites_by_b[b] = routed_sites_at_width
            print(
                result_line(
                    {},
                    "OB1",
                    decide_ob1(
                        b,
                        rounds,
                        routed_calls=routed_calls,
                        routed_sites_at_width=routed_sites_at_width,
                    ),
                )
            )
            if b in IN_ZONE:
                print(result_line({}, "OB2", decide_ob2(b, rounds)))
                print(result_line({}, "OB3", decide_ob3(b, rounds)))

        current_cell = "ZONE"
        print(
            result_line(
                {},
                "ZONE",
                decide_zone(
                    cells,
                    routed_calls_by_b=routed_calls_by_b,
                    routed_sites_by_b=routed_sites_by_b,
                ),
            )
        )
    except NotIdle as error:
        print(f"REFUSAL (exit {EXIT_NOT_IDLE}): {error}")
        return EXIT_NOT_IDLE
    except BudgetExceeded as error:
        print(f"REFUSAL (exit {EXIT_BUDGET_REFUSAL}): {error}")
        return EXIT_BUDGET_REFUSAL
    except LowMemoryRefusal as error:
        print(f"REFUSAL (exit {EXIT_LOW_MEMORY}): {error}")
        return EXIT_LOW_MEMORY
    except PreconditionFailed as error:
        print(f"REFUSAL (exit {EXIT_PRECONDITION}): {error}")
        return EXIT_PRECONDITION
    except RunInvalid as error:
        print(f"STOP: {current_cell}: {error}")
        exit_code = 1
    finally:
        lock.release()
    return check_idle_after(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
