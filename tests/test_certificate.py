"""A certificate must never read as stronger than the verification behind it.

Each test pins one way a certificate could overclaim. The audit is the gate
that stops such a certificate being written at all, so most of these assert a
refusal rather than a rendering.
"""

import json

import pytest

from kernelverify.report.certificate import (
    EXTRACTED_UNVALIDATED,
    EXTRACTED_VALIDATED,
    NUMERIC,
    STRUCTURAL,
    WRAPPER_ASSEMBLED,
    Certificate,
    ClauseAttestation,
    PerformanceClaim,
    audit,
    contract_clauses,
    emit,
    render_markdown,
)


def cert(**over):
    base = dict(
        kernel_name="wide_qmv",
        operator="quantized_matmul",
        translation_unit="[[kernel]] void wide_qmv(...) { }",
        extraction_provenance=EXTRACTED_VALIDATED,
        compile_options={"math_mode": "safe"},
        clauses=contract_clauses(fp16_activations=False),
        catalogue_fingerprint="catalogue-65-faults-abc123",
        harness_commit="e2bb208",
        contract_version="2",
        tolerance_model={"K_quant": 3.0, "ensemble_members": ["dequant-pairwise",
                                                              "lut-gather"]},
        policy="boundary pairs + random",
        budget=16,
        cases_run=16,
        cases_passed=16,
        seed_protocol="seeds 0..7 at the benchmark shape, sweep seed 1234",
        chip_generation="Apple M3",
        toolchain={"mlx": "0.32.0", "macos": "26.5.2"},
    )
    base.update(over)
    return Certificate(**base)


# ---------------------------------------------------------------------------
# The two blocks stay structurally separate
# ---------------------------------------------------------------------------
def test_byte_bound_block_disclaims_being_a_claim_about_outputs():
    block = cert().byte_bound()
    assert "assert" in block["_meaning"] and "nothing" in block["_meaning"]
    assert len(block["translation_unit_sha256"]) == 64


def test_protocol_bound_block_never_claims_byte_exactness():
    assertion = cert().protocol_bound()["_assertion"]
    assert "NOT asserted to be" in assertion and "identical" in assertion


def test_the_two_blocks_do_not_overlap():
    """A hash must not appear among the assertions, or a reader will read it
    as one."""
    doc = cert().to_json()
    assert "translation_unit_sha256" in doc["byte_bound"]
    assert "translation_unit_sha256" not in doc["protocol_bound"]


def test_what_a_source_hash_cannot_cover_is_stated():
    listed = " ".join(cert().byte_bound()["not_attested_by_these_hashes"])
    assert "compile options" in listed
    assert "contiguity" in listed


# ---------------------------------------------------------------------------
# Clause attestation: the mechanism travels with the clause
# ---------------------------------------------------------------------------
def test_fp16_activations_move_c1_to_source_attestation():
    """The measured result: tree-reduced half accumulation is separable on 0 of
    280 fp16-activation cases, so claiming a numeric check there would be a
    check we never ran."""
    clauses = contract_clauses(fp16_activations=True)
    c1 = next(c for c in clauses if c.clause == "C1")
    assert c1.mechanism == STRUCTURAL
    assert "0/280" in c1.evidence and c1.scope


def test_fp32_activations_keep_c1_numeric():
    c1 = next(c for c in contract_clauses(fp16_activations=False) if c.clause == "C1")
    assert c1.mechanism == NUMERIC


def test_a_clause_without_evidence_is_rejected():
    with pytest.raises(ValueError, match="no evidence"):
        ClauseAttestation("C1", "intermediates wide", NUMERIC, evidence="")


def test_unknown_mechanism_is_rejected():
    with pytest.raises(ValueError, match="unknown attestation mechanism"):
        ClauseAttestation("C1", "intermediates wide", "vibes", evidence="looks fine")


def test_missing_c1_voids_the_certificate():
    partial = tuple(c for c in contract_clauses(fp16_activations=False)
                    if c.clause != "C1")
    problems = audit(cert(clauses=partial))
    assert any("C1" in p for p in problems)


def test_a_certificate_with_no_clauses_is_rejected_outright():
    with pytest.raises(ValueError, match="asserts nothing"):
        cert(clauses=())


# ---------------------------------------------------------------------------
# The audit refuses overclaiming certificates
# ---------------------------------------------------------------------------
def test_clean_certificate_has_no_complaints():
    assert audit(cert()) == []


def test_a_failed_case_means_no_certificate():
    problems = audit(cert(cases_run=16, cases_passed=15))
    assert any("did not pass" in p for p in problems)


def test_fewer_cases_than_budget_is_flagged():
    problems = audit(cert(cases_run=4, cases_passed=4, budget=16))
    assert any("budget" in p for p in problems)


def test_wrapper_assembled_must_be_declared_plainly():
    problems = audit(cert(extraction_provenance=WRAPPER_ASSEMBLED))
    assert any("our own wrapper" in p for p in problems)


def test_unvalidated_extraction_cannot_claim_the_top_tier():
    problems = audit(cert(extraction_provenance=EXTRACTED_UNVALIDATED))
    assert any("reproduction check" in p for p in problems)


def test_cases_passed_cannot_exceed_cases_run():
    with pytest.raises(ValueError, match="more cases passed"):
        cert(cases_run=4, cases_passed=5)


# ---------------------------------------------------------------------------
# Performance claims are scoped or they are not made
# ---------------------------------------------------------------------------
def test_unscoped_speed_claim_is_refused():
    claim = PerformanceClaim(ratio=1.4, against="mlx", scope="", binding=False)
    assert any("no scope" in p for p in audit(cert(performance=(claim,))))


def test_binding_speed_claim_needs_a_sampling_group():
    claim = PerformanceClaim(ratio=1.4, against="mlx", scope="batch 5-11",
                             binding=True)
    assert any("sampling group" in p for p in audit(cert(performance=(claim,))))


def test_directional_claim_renders_as_directional():
    claim = PerformanceClaim(ratio=1.4, against="mlx quantized_matmul",
                             scope="batch 5-11", binding=False)
    text = render_markdown(cert(performance=(claim,)))
    assert "directional" in text and "1.40x" in text


# ---------------------------------------------------------------------------
# Emission is gated by the audit
# ---------------------------------------------------------------------------
def test_emit_refuses_a_certificate_the_audit_complains_about(tmp_path):
    with pytest.raises(ValueError, match="refusing to emit"):
        emit([cert(cases_run=16, cases_passed=10)], tmp_path)
    assert not list(tmp_path.glob("*.json")), "nothing may be written on refusal"


def test_emit_writes_valid_json(tmp_path):
    [path] = emit([cert()], tmp_path)
    doc = json.loads(path.read_text())
    assert doc["kernel"] == "wide_qmv"
    assert doc["validity"]["binding_on"]["chip_generation"] == "Apple M3"
    assert "advisory" in doc["validity"]["elsewhere"]


# ---------------------------------------------------------------------------
# The human summary leads with limits
# ---------------------------------------------------------------------------
def test_markdown_states_the_limits_before_any_speed_claim():
    text = render_markdown(cert(performance=(
        PerformanceClaim(1.4, "mlx", "batch 5-11", False),)))
    assert text.index("does not assert") < text.index("## Speed")


def test_markdown_names_the_mechanism_per_clause():
    text = render_markdown(cert(clauses=contract_clauses(fp16_activations=True)))
    assert "reading the kernel source" in text and "battery measurement" in text
