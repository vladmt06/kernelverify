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
