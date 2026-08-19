"""metalrunner actually trains: the one test that runs the real thing.

Every other test in `test_metalrunner_lora.py` asserts what does not happen,
with the trainer replaced by something that raises. That is the right shape
for refusal paths and it proves nothing about the happy path, so this file
runs a real fine-tune: the smallest pinned model, a handful of examples, two
iterations, through the actual entry point with no fakes anywhere.

It is deliberately tiny. The point is not a measurement and no timing is
taken; the point is that the product runs, produces an adapter, and writes a
receipt that describes it. A minute of GPU is a development smoke test, not
a binding measurement, and it never touches the machine-wide measurement
lock.

The model is a gitignored local artefact, so the test skips when it is
absent rather than failing on a fresh clone.
"""

import json
from pathlib import Path

import pytest

from conftest import requires_metal

MODEL = Path(__file__).resolve().parents[1] / "bench" / ".models" / "qwen3-0.6b-4bit-g64"

EXAMPLES = [
    {"text": "Q: What colour is the sky? A: Blue."},
    {"text": "Q: What colour is grass? A: Green."},
    {"text": "Q: What colour is snow? A: White."},
    {"text": "Q: What colour is coal? A: Black."},
]


@pytest.fixture()
def dataset(tmp_path):
    for split in ("train", "valid"):
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in EXAMPLES) + "\n")
    return tmp_path


@pytest.mark.slow
@requires_metal
def test_a_real_two_iteration_fine_tune_runs_and_accounts_for_itself(dataset,
                                                                     tmp_path):
    if not MODEL.is_dir():
        pytest.skip(f"the pinned smoke model is not present at {MODEL}")

    from metalrunner.lora import main

    adapters = tmp_path / "adapters"
    code = main(["--model", str(MODEL), "--train", "--data", str(dataset),
                 "--iters", "2", "--batch-size", "1", "--num-layers", "2",
                 "--max-seq-length", "64", "--steps-per-eval", "2",
                 "--steps-per-report", "1", "--adapter-path", str(adapters)])

    assert code == 0, "the entry point must complete a real fine-tune"
    assert (adapters / "adapters.safetensors").exists(), "no adapter was written"

    receipt = json.loads((adapters / "metalrunner-receipt.json").read_text())

    # The receipt describes THIS run, not a template.
    assert receipt["base_model"]["kind"] == "local"
    assert len(receipt["base_model"]["config_sha256"]) == 64
    assert receipt["training"]["iters"] == 2
    assert receipt["training"]["fine_tune_type"] == "lora"
    assert receipt["adapter"]["sha256"] is not None, (
        "the receipt must fingerprint the adapter it accounts for")
    assert receipt["peak_memory_bytes"] > 0

    # And it declines to claim what it did not check.
    assert any("speed" in line for line in receipt["not_attested"])

    # Nothing was routed, so the record must say so rather than being empty.
    assert receipt["routing"] and not any(r["routed"] for r in receipt["routing"])
    assert all(r["reason"] for r in receipt["routing"])
