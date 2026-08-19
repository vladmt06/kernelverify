"""The user-facing entry point: what it refuses, and what it never touches.

metalrunner runs someone else's training job. The failure that matters is
not a crash, it is a run that quietly differs from what the user believed
they were running, so most of these tests are about what does NOT happen: a
stack that was never verified must not be patched, an unsupported mode must
not reach the trainer, and an unreadable quantization must not be guessed.

The load-bearing test is the last kind. mlx-lm's `run` loads a model and
trains, so a test asserting a refusal returned the right exit code proves
little on its own; each refusal test also asserts the trainer was never
reached, with a fake that raises if it is.

No model is loaded and no training runs anywhere in this file.
"""

import json
import sys
import types

import pytest

from metalrunner import receipt, routing
from metalrunner.lora import (
    EXIT_UNSUPPORTED_MODE,
    EXIT_UNVERIFIED_STACK,
    main,
    parse,
    unsupported_mode,
)
from metalrunner.versions import (
    MLX_LM_VERSION,
    MLX_VERSION,
    SEAM_HASHES,
    Stack,
    UnverifiedStack,
    differences,
    observed,
    require_verified_stack,
)


@pytest.fixture()
def never_trains(monkeypatch):
    """mlx-lm's trainer, replaced by something that fails loudly if reached."""
    reached = []

    def explode(_args):
        reached.append(True)
        raise AssertionError("the trainer was reached after a refusal")

    import mlx_lm.lora

    monkeypatch.setattr(mlx_lm.lora, "run", explode)
    return reached


# ---------------------------------------------------------------------------
# The stack pin
# ---------------------------------------------------------------------------
def test_this_machine_runs_the_verified_stack():
    assert differences(observed()) == []
    require_verified_stack()


@pytest.mark.parametrize("field,value", [("mlx", "9.9.9"), ("mlx_lm", "9.9.9")])
def test_a_different_version_is_a_difference_named_in_words(field, value):
    stack = observed()
    bumped = Stack(mlx=value if field == "mlx" else stack.mlx,
                   mlx_lm=value if field == "mlx_lm" else stack.mlx_lm,
                   seams=stack.seams)
    [problem] = differences(bumped)
    assert value in problem
    assert (MLX_VERSION if field == "mlx" else MLX_LM_VERSION) in problem


def test_a_changed_seam_is_caught_even_when_the_version_did_not_move():
    """A patch release can move a function without moving the version a user
    sees, which is why the files are hashed and not just counted."""
    stack = observed()
    tampered = Stack(mlx=stack.mlx, mlx_lm=stack.mlx_lm,
                     seams={**stack.seams, "tuner/trainer.py": "0" * 64})
    [problem] = differences(tampered)
    assert "tuner/trainer.py has changed" in problem


def test_a_missing_seam_file_is_a_difference_not_a_crash(tmp_path):
    stack = observed(mlx_lm_root=tmp_path)
    assert len(differences(stack)) >= len(SEAM_HASHES)


def test_an_unverified_stack_refuses_and_never_reaches_the_trainer(
        monkeypatch, never_trains, capsys):
    """The direction that matters: refuse, do not degrade. A silent fallback
    would let someone believe they ran verified kernels when they ran none."""
    import metalrunner.lora as entry

    monkeypatch.setattr(entry, "require_verified_stack",
                        lambda: (_ for _ in ()).throw(
                            UnverifiedStack("mlx-lm is 9.9.9")))

    assert main(["--model", "some/model", "--train"]) == EXIT_UNVERIFIED_STACK
    assert not never_trains, "nothing may run on an unverified stack"
    assert "9.9.9" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Mode refusals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["dora", "full"])
def test_an_unsupported_mode_is_refused_in_one_line(mode):
    message = unsupported_mode(types.SimpleNamespace(fine_tune_type=mode))
    assert message is not None
    assert "mlx_lm.lora" in message, "a refusal must name what to run instead"
    assert message.count("\n") == 0


def test_lora_is_not_refused():
    assert unsupported_mode(types.SimpleNamespace(fine_tune_type="lora")) is None


@pytest.mark.parametrize("mode", ["dora", "full"])
def test_an_unsupported_mode_never_reaches_the_trainer(mode, never_trains,
                                                       capsys):
    code = main(["--model", "some/model", "--train", "--fine-tune-type", mode])
    assert code == EXIT_UNSUPPORTED_MODE
    assert not never_trains
    assert "mlx_lm.lora" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Argument compatibility: every mlx-lm flag, unchanged
# ---------------------------------------------------------------------------
def test_the_parser_is_mlx_lms_own_so_every_flag_is_accepted():
    from mlx_lm.lora import build_parser

    ours = parse(["--model", "m", "--train", "--iters", "7"])
    theirs = build_parser().parse_args(["--model", "m", "--train",
                                        "--iters", "7"])
    for flag in vars(theirs):
        assert hasattr(ours, flag), f"metalrunner dropped the flag {flag}"
    assert ours.iters == 7 and ours.model == "m" and ours.train


def test_unspecified_arguments_take_mlx_lms_defaults():
    from mlx_lm.lora import CONFIG_DEFAULTS

    args = parse(["--model", "m", "--train"])
    for key, value in CONFIG_DEFAULTS.items():
        if getattr(args, key, None) is not None:
            continue
        assert value is None or getattr(args, key) == value


def test_a_config_file_is_read_and_the_command_line_still_wins(tmp_path):
    config = tmp_path / "cfg.yaml"
    config.write_text("iters: 11\nbatch_size: 3\n")
    args = parse(["--config", str(config), "--iters", "5"])
    assert args.iters == 5, "the command line must beat the config file"
    assert args.batch_size == 3, "and the config file must still be read"


# ---------------------------------------------------------------------------
# Routing, and the honesty of the report
# ---------------------------------------------------------------------------
def test_nothing_routes_today_and_every_decline_carries_a_reason():
    decisions = routing.decide("lora", 4, 64, on_chip="Apple M3 Pro")
    assert len(decisions) == len(routing.KNOWN_OPERATIONS)
    assert not any(d.routed for d in decisions)
    assert all(d.reason for d in decisions)
    assert "nothing certified to route" in decisions[0].reason


def test_an_unreadable_quantization_declines_rather_than_assuming():
    [first, *_] = routing.decide("lora", None, None, on_chip="Apple M3 Pro")
    assert "could not be read" in first.reason
    assert "must not be routed onto another" in first.reason


def test_an_unverified_format_is_named_in_the_decline():
    [first, *_] = routing.decide("lora", 8, 32, on_chip="Apple M3 Pro")
    assert "8-bit group-32" in first.reason and "4-bit group-64" in first.reason


def test_the_report_states_that_it_changed_nothing_when_it_routed_nothing():
    """The report is read by someone deciding whether to trust a number. With
    nothing routed it must say so plainly rather than looking like a product
    doing work."""
    decisions = routing.decide("lora", 4, 64, on_chip="Apple M3 Pro")
    text = routing.render(decisions, observed(), model="m",
                          fine_tune_type="lora", bits=4, group_size=64)
    assert "not routed (3)" in text
    assert "This run is stock mlx-lm" in text
    assert "makes no speed claim" in text
    assert MLX_LM_VERSION in text


# ---------------------------------------------------------------------------
# The receipt
# ---------------------------------------------------------------------------
def test_the_quantization_is_read_from_the_models_own_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(
        {"quantization": {"bits": 4, "group_size": 64}}))
    assert receipt.read_quantization(str(tmp_path)) == (4, 64)


def test_an_absent_or_unquantized_config_reads_as_unknown(tmp_path):
    assert receipt.read_quantization(str(tmp_path)) == (None, None)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert receipt.read_quantization(str(tmp_path)) == (None, None)


def test_the_receipt_says_what_it_does_not_attest(tmp_path):
    record = receipt.build(
        args=types.SimpleNamespace(model="m", fine_tune_type="lora", seed=1),
        stack=observed(),
        decisions=routing.decide("lora", 4, 64, on_chip="Apple M3 Pro"),
        chip="Apple M3 Pro", adapter_path=str(tmp_path), peak_bytes=1234,
        started="a", finished="b")

    assert record["not_attested"], "a receipt that overstates is worse than none"
    assert any("speed" in line for line in record["not_attested"])
    assert any("shadow" in line for line in record["not_attested"])
    assert record["stack"]["mlx_lm"] == MLX_LM_VERSION
    assert len(record["routing"]) == len(routing.KNOWN_OPERATIONS)


def test_the_receipt_lands_beside_the_adapter_and_reloads(tmp_path):
    record = receipt.build(
        args=types.SimpleNamespace(model="m"), stack=observed(), decisions=[],
        chip="c", adapter_path=str(tmp_path), peak_bytes=None,
        started="a", finished="b")
    path = receipt.write(record, str(tmp_path))
    assert path == tmp_path / "metalrunner-receipt.json"
    assert json.loads(path.read_text())["metalrunner_receipt"] == 1


def test_a_run_with_nowhere_to_put_a_receipt_says_so_rather_than_inventing():
    assert receipt.write({"a": 1}, None) is None


def test_a_local_model_is_fingerprinted_and_a_hub_id_is_not_pretended_to_be(
        tmp_path):
    (tmp_path / "config.json").write_text("{}")
    local = receipt.fingerprint_model(str(tmp_path))
    assert local["kind"] == "local" and len(local["config_sha256"]) == 64

    remote = receipt.fingerprint_model("mlx-community/Qwen3-4B-4bit")
    assert remote["kind"] == "hub-id" and remote["config_sha256"] is None


# ---------------------------------------------------------------------------
# The control arm: the wrapper installed, routing forced off
# ---------------------------------------------------------------------------
def test_force_stock_is_off_unless_the_environment_says_otherwise(monkeypatch):
    monkeypatch.delenv(routing.FORCE_STOCK_ENV, raising=False)
    assert not routing.forced_to_stock()
    monkeypatch.setenv(routing.FORCE_STOCK_ENV, "0")
    assert not routing.forced_to_stock(), "an explicit 0 must not arm it"
    monkeypatch.setenv(routing.FORCE_STOCK_ENV, "1")
    assert routing.forced_to_stock()


def test_a_forced_run_declines_for_that_reason_and_no_other():
    """The decline must read as the control it is. An ordinary lack of
    coverage and a deliberate control arm are different facts, and a
    measurement that confused them would be comparing the wrong things."""
    decisions = routing.decide("lora", 4, 64, on_chip="Apple M3 Pro",
                               force_stock=True)
    assert not any(d.routed for d in decisions)
    assert all("control run" in d.reason for d in decisions)
    assert all(routing.FORCE_STOCK_ENV in d.reason for d in decisions)


def test_the_report_announces_a_control_run(monkeypatch):
    monkeypatch.setenv(routing.FORCE_STOCK_ENV, "1")
    text = routing.render(routing.decide("lora", 4, 64, on_chip="c"),
                          observed(), model="m", fine_tune_type="lora",
                          bits=4, group_size=64)
    assert "CONTROL RUN" in text
    assert "measures the wrapper's own cost" in text


def test_a_control_receipt_is_unmistakable_afterwards(tmp_path):
    """Arm 3 of the end-to-end measurement produces receipts too, and one of
    them must never be quoted as a real run."""
    record = receipt.build(
        args=types.SimpleNamespace(model="m"), stack=observed(),
        decisions=routing.decide("lora", 4, 64, on_chip="c", force_stock=True),
        chip="c", adapter_path=str(tmp_path), peak_bytes=1,
        started="a", finished="b", forced_to_stock=True)

    assert record["forced_to_stock"] is True
    assert all("control run" in r["reason"] for r in record["routing"])


def test_an_ordinary_receipt_is_not_marked_as_a_control(tmp_path):
    record = receipt.build(
        args=types.SimpleNamespace(model="m"), stack=observed(), decisions=[],
        chip="c", adapter_path=str(tmp_path), peak_bytes=1,
        started="a", finished="b")
    assert record["forced_to_stock"] is False
