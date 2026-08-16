"""What a pack certificate SAYS about the contract and tolerance it applied.

`tests/test_pack_certificates.py` drives the whole emitter and is skipped
wholesale without a Metal device, because emitting a certificate means running
a gate. The family facts below are pure functions of a family name, and they
are the sentences a reader trusts a certificate for, so they are pinned here on
any machine rather than only on one with a GPU.
"""

import hashlib
from pathlib import Path

import emit_pack_certificates as emitter

from kernelverify.pack import kv_attention, moe_dispatch, wide_qmv
from kernelverify.schemas.native_ops import K_NATIVE, K_QUANT
from kernelverify.schemas.quant_contract import QUANT_ENSEMBLE_VERSION

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SOURCE = ROOT / "kernelverify" / "schemas" / "quant_contract.py"


def test_the_kv_certificate_says_its_k_is_borrowed_and_uncalibrated():
    """A certificate that names a contract version is claiming the tolerance
    was calibrated for that contract. kv_attention's was not: it reuses
    quantized_matmul's K over an ensemble no harness has scored."""
    prose = emitter.contract_version_for(kv_attention.KERNEL_NAME)
    assert "K borrowed from quantized_matmul, uncalibrated" in prose


def test_the_wide_qmv_tolerance_model_identifies_the_ensemble_by_version_and_hash():
    """The 26 committed certificates name the quant ensemble by member NAME
    only, and the factored-groups repair moved a member's arithmetic while
    keeping its name. A certificate that survives that change unaltered is
    vouching for a floor it never saw, so it carries the version and the
    source hash the version is supposed to track."""
    model = emitter.tolerance_model_for(wide_qmv.KERNEL_NAME)
    assert model["ensemble_version"] == QUANT_ENSEMBLE_VERSION
    assert model["ensemble_source_sha256"] == hashlib.sha256(
        CONTRACT_SOURCE.read_bytes()).hexdigest()
    assert model["K_quant"] == K_QUANT


def test_the_moe_tolerance_model_states_the_k_moe_actually_multiplies():
    """native_ops.moe_tolerance multiplies K_NATIVE = 1.5 over an unquantized
    ensemble. The certificate said K_quant = 4.0, which is a different number
    for a tolerance nobody applied - a false statement in the artifact whose
    whole job is to be true."""
    model = emitter.native_tolerance_model("moe_dispatch")
    assert model["K_native"] == K_NATIVE
    assert "K_quant" not in model
    assert "K_NATIVE" in model["form"]


def test_the_kv_tolerance_model_keeps_the_k_it_borrowed():
    model = emitter.native_tolerance_model("kv_attention")
    assert model["K_quant"] == K_QUANT
    assert "K_native" not in model


def test_the_two_native_families_route_to_their_own_tolerance_model():
    assert (emitter.tolerance_model_for(kv_attention.KERNEL_NAME)
            == emitter.native_tolerance_model("kv_attention"))
    assert emitter.tolerance_model_for(moe_dispatch.ROUTING_NAME) == {
        "criterion": "top-2 indices bit-exact against the contract's "
                     "tie-to-lower-index rule; no numeric tolerance",
        "gate_weights": "renormalized top-2 probabilities",
    }, "the routing kernel is index-exact and carries no K at all"
