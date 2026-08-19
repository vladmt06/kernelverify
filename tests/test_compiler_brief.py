"""What the generator is told, and the leak tests that matter.

Feedback is the most dangerous surface in the loop. Two leaks would turn a
verified loop into one that only looks verified, and each gets a test that
tries to find it rather than a comment saying it cannot happen.

The sealed one is first. A brief is built over a store full of held-out
failures and read for any trace of what those cases were. If the words of a
held-out cell ever reach a brief, the held-out set becomes a training set
within a few rounds and every later verdict is worth less than it appears.

The second is drift. The rules the generator is handed are derived from the
lint's own token list, so a brief cannot come to describe a rule the lint
does not enforce.

Pure Python, no MLX and no GPU.
"""

import pytest

from kernelverify.compiler.brief import Operation, attempts, build, writing_rules
from kernelverify.compiler.funnel import Funnel
from kernelverify.compiler.heldout import Verdict
from kernelverify.compiler.lint import NARROW_TYPES, lint
from kernelverify.compiler.stages import heldout_stage
from kernelverify.compiler.store import CandidateStore

OP = Operation(
    name="quantized_matmul",
    goal="Multiply a float activation by a 4-bit group-64 weight.",
    signature="out[M, D_OUT] = x[M, D_IN] @ dequant(w, scales, biases)",
    constraints=("group size is 64", "bits is 4"),
)

SECRET_SPACE = {
    "tokens": [4093, 4099],
    "distribution": ["opposed-signs-8f3a", "constant-rows-91bc"],
    "shape": [(9728, 2560)],
}


@pytest.fixture()
def store(tmp_path):
    return CandidateStore(tmp_path)


# ---------------------------------------------------------------------------
# The sealed draw must not reach the brief
# ---------------------------------------------------------------------------
def test_a_brief_over_held_out_failures_names_no_held_out_cell(store):
    """The leak test. Every candidate below fails its held-out check, and the
    brief must say so without saying what it failed on."""
    funnel = Funnel([heldout_stage(SECRET_SPACE, lambda source, cells: False)])
    for i in range(5):
        candidate = store.propose(f"kernel void k{i}() {{ }}", origin="generator")
        funnel.screen(store, candidate)

    text = build(OP, store)

    assert "heldout:failed" in text, "the verdict itself is allowed through"
    for axis, choices in SECRET_SPACE.items():
        for choice in choices:
            assert str(choice) not in text, (
                f"the held-out {axis} value {choice!r} reached the brief; a "
                f"failing case that names itself is feedback, and feedback "
                f"turns a held-out set into a training set")


def test_a_held_out_pass_leaks_nothing_either(store):
    funnel = Funnel([heldout_stage(SECRET_SPACE, lambda source, cells: True)])
    candidate = store.propose("kernel void k() { }", origin="generator")
    funnel.screen(store, candidate)

    text = build(OP, store)
    assert "heldout:passed" in text
    assert "4093" not in text and "4099" not in text


def test_the_verdict_type_is_all_the_stage_can_report(store):
    """Where the sealing actually lives: the stage writes an empty detail,
    so there is nothing for a brief to redact."""
    funnel = Funnel([heldout_stage(SECRET_SPACE, lambda source, cells: False)])
    candidate = store.propose("kernel void k() { }", origin="generator")
    funnel.screen(store, candidate)

    [event] = store.get(candidate).events
    assert event["detail"] == ""
    assert str(Verdict(False)) == "FAIL"


# ---------------------------------------------------------------------------
# The rules cannot drift from the lint
# ---------------------------------------------------------------------------
def test_every_type_the_lint_refuses_is_named_in_the_rules():
    rules = writing_rules()
    for token in NARROW_TYPES:
        assert token in rules, (
            f"the lint refuses {token} and the brief never mentions it, so the "
            f"session would kill candidates for a rule it did not give")


def test_the_load_idiom_the_brief_recommends_actually_passes_the_lint():
    """The strongest form of the anti-drift check: take the brief's own
    accepted example and its own refused example, and run the real lint."""
    assert "float4 v = float4(*(device const half4*)p);" in writing_rules()
    assert lint("float4 v = float4(*(device const half4*)p);").ok
    assert not lint("half4 v = *(device const half4*)p;").ok
    assert lint("out[i] = (half)acc;").ok


# ---------------------------------------------------------------------------
# What the brief says about the operation and the history
# ---------------------------------------------------------------------------
def test_the_first_round_says_there_is_no_history(store):
    assert "this is the first round" in build(OP, store)


def test_the_operation_reaches_the_brief_whole(store):
    text = build(OP, store)
    assert OP.name in text and OP.goal in text and OP.signature in text
    for constraint in OP.constraints:
        assert constraint in text


def test_a_failure_and_its_reason_reach_the_generator(store):
    """The lint's reason is exactly the feedback that stops a generator
    repeating itself, so unlike the held-out detail it must come through."""
    candidate = store.propose("half acc = 0;", origin="generator")
    store.record(candidate, stage="lint", passed=False,
                 detail="line 1: half names a value")

    text = attempts(store)
    assert "lint:failed" in text and "half names a value" in text


def test_the_census_counts_every_attempt_even_the_ones_not_shown(store):
    for i in range(25):
        candidate = store.propose(f"kernel void k{i}() {{ }}", origin="generator")
        store.record(candidate, stage="lint", passed=False, detail="bf16 tile")

    text = attempts(store, limit=5)
    assert "5 earlier attempts not shown" not in text
    assert "20 earlier attempts not shown" in text
    assert "lint:failed: 25" in text, (
        "a truncated list must never make the census read smaller than the run")
