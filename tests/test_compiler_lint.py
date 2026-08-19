"""The C1 attestation: does a candidate compute in binary32 or narrower?

The calibration test is the first one. A lint that refuses everything is
useless, so it is pointed at the two Metal kernels this repository has
already written, verified and gated, at both type bindings they run under;
if it flags those, it is wrong about the contract rather than right about
the candidate.

Everything after that is the refusal side: an accumulator, a signature, a
simdgroup tile, and the same accumulator hidden behind a template parameter.

Pure text, no MLX and no GPU.
"""

import pytest

from kernelverify.compiler.lint import lint
from kernelverify.pack import kv_attention, wide_qmv


# ---------------------------------------------------------------------------
# Calibration against code that is already known good
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kernel", [wide_qmv, kv_attention],
                         ids=["wide_qmv", "kv_attention"])
@pytest.mark.parametrize("bound", ["half", "float"])
def test_the_shipped_kernels_pass_at_both_bindings(kernel, bound):
    report = lint(kernel.mlx_door_source(), types={"T": bound})
    assert report.ok, report.reason


# ---------------------------------------------------------------------------
# What C1 allows: narrow types that name storage, and the one rounding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("line", [
    "device const half* x = nullptr;",
    "device half4* out = nullptr;",
    "threadgroup half tile[64];",
    "threadgroup bfloat staging[32];",
    "out[i] = (half)acc;",
    "out[i] = half(acc);",
    "float4 v = float4(*(device const half4*)p);",
])
def test_storage_and_casts_are_admissible(line):
    assert lint(line).ok, f"{line!r} should be allowed"


# ---------------------------------------------------------------------------
# What C1 refuses: narrow types that name values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("line,token", [
    ("half acc = 0.0h;", "half"),
    ("half4 partial = 0;", "half4"),
    ("bfloat sum = 0;", "bfloat"),
    ("packed_half4 staged = load(p);", "packed_half4"),
    ("simdgroup_matrix<half, 8, 8> acc;", "half"),
    ("void helper(half x) { }", "half"),
])
def test_a_narrow_value_is_refused_and_the_token_is_named(line, token):
    report = lint(line)
    assert not report.ok
    [violation] = report.violations
    assert violation.token == token
    assert token in report.reason and "C1" in report.reason


def test_the_violation_carries_its_line_number_and_the_line_itself():
    source = "float a = 0;\nfloat b = 1;\nhalf acc = 0.0h;\nfloat c = 2;\n"
    [violation] = lint(source).violations
    assert violation.line == 3
    assert violation.text.strip() == "half acc = 0.0h;"


def test_every_violation_is_reported_not_just_the_first():
    source = "half a = 0;\nfloat b = 1;\nbfloat c = 2;\n"
    assert [v.line for v in lint(source).violations] == [1, 3]


# ---------------------------------------------------------------------------
# The template binding, which is where the same violation hides
# ---------------------------------------------------------------------------
def test_an_accumulator_typed_by_a_template_parameter_is_resolved_first():
    """`T acc` is not a violation until you know what T is. The source that
    gets compiled is the bound one, so that is the source that is judged."""
    source = "T acc = 0;"
    assert lint(source, types={"T": "float"}).ok
    refused = lint(source, types={"T": "half"})
    assert not refused.ok and refused.violations[0].token == "half"


def test_a_bfloat_binding_is_caught_too_because_training_is_bf16():
    assert not lint("T acc = 0;", types={"T": "bfloat"}).ok


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------
def test_a_narrow_type_inside_a_comment_is_not_a_violation():
    assert lint("// half acc would be wrong here\nfloat acc = 0;").ok
    assert lint("/* half acc\n   also fine */\nfloat acc = 0;").ok


def test_stripping_a_block_comment_leaves_line_numbers_true():
    source = "/* one\n   two */\nhalf acc = 0;\n"
    assert lint(source).violations[0].line == 3
