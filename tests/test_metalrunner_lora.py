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
    EXIT_SEAM_REFUSED,
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


# ---------------------------------------------------------------------------
# The seam: what metalrunner replaces inside mlx-lm, and what it records
#
# `run()` takes a `training_callback` argument and overwrites it on its own
# next line with whatever `--report-to` built, so that parameter is dead in
# mlx-lm 0.31.3. The callback is injected at `train_model` instead, which is
# where it actually arrives. These tests drive main() with a stubbed trainer
# so the wiring is exercised without loading a model.
# ---------------------------------------------------------------------------
@pytest.fixture()
def stub_trainer(monkeypatch):
    """mlx-lm's `train_model` and `run`, reduced to the shape main() needs.

    The stub declares itself as defined in mlx_lm.lora because that is what
    it stands in for, and the installer refuses a name holding a foreign
    object. `run` looks the trainer up on the module at call time, exactly as
    the real one does, which is the property that makes the name a seam.
    """
    import mlx_lm.lora

    seen = {}

    def train_model(args, model, train_set, valid_set,
                    training_callback=None):
        seen["callback"] = training_callback
        training_callback.on_train_loss_report(
            {"iteration": 1, "train_loss": 2.5, "trained_tokens": 128})

    train_model.__module__ = "mlx_lm.lora"
    monkeypatch.setattr(mlx_lm.lora, "train_model", train_model)

    def run(args, training_callback=None):
        import mlx_lm.lora as module

        module.train_model(args, None, None, None, None)

    monkeypatch.setattr(mlx_lm.lora, "run", run)
    return seen


def _train(tmp_path, extra=()):
    return main(["--model", "some/model", "--train",
                 "--adapter-path", str(tmp_path)] + list(extra))


def test_the_receipt_carries_what_the_trainer_reported(stub_trainer, tmp_path):
    """The loss curve reaches the receipt through mlx-lm's own callback, so
    a run's own numbers survive without anyone parsing its printed output."""
    assert _train(tmp_path) == 0

    record = json.loads((tmp_path / "metalrunner-receipt.json").read_text())
    assert record["progress"]["train_reports"] == 1
    assert record["progress"]["last_train_loss"] == 2.5
    # Supervised tokens, the quantity the end-to-end fairness rule compares.
    assert record["progress"]["trained_tokens"] == 128


def test_the_receipt_says_how_often_the_seam_was_reached(stub_trainer,
                                                         tmp_path):
    """An installed seam that was never called and one that carried the
    whole run look identical without this count."""
    assert _train(tmp_path) == 0

    record = json.loads((tmp_path / "metalrunner-receipt.json").read_text())
    assert record["seams"]["calls"] == {"mlx_lm.lora.train_model": 1}
    assert record["seams"]["foreign_on_removal"] == []


def test_the_recorded_loss_is_never_called_attested(stub_trainer, tmp_path):
    """Recording a number is not checking it, and the receipt has to say so
    or it converts an absent check into an apparent one."""
    assert _train(tmp_path) == 0

    record = json.loads((tmp_path / "metalrunner-receipt.json").read_text())
    assert any("loss values" in line for line in record["not_attested"])
    assert "Not recomputed" in record["progress"]["_meaning"]


def test_the_seam_is_gone_once_the_run_is_over(stub_trainer, tmp_path):
    """metalrunner must not outlive its own run inside the interpreter."""
    import mlx_lm.lora

    before = mlx_lm.lora.train_model
    assert _train(tmp_path) == 0
    assert mlx_lm.lora.train_model is before


def test_a_foreign_patch_on_the_seam_refuses_before_training(
        monkeypatch, never_trains, capsys):
    """The disk hashes can pass while the running process disagrees with
    them. That is the same condition the stack check exists to catch, found
    a moment later, so it gets the same answer: refuse, change nothing."""
    import mlx_lm.lora

    def someone_elses(*_args, **_kwargs):
        raise AssertionError("the foreign trainer was reached")

    someone_elses.__module__ = "some_other_package"
    monkeypatch.setattr(mlx_lm.lora, "train_model", someone_elses)

    assert main(["--model", "some/model", "--train"]) == EXIT_SEAM_REFUSED
    assert not never_trains
    assert "already replaced it" in capsys.readouterr().err
    assert mlx_lm.lora.train_model is someone_elses


def test_a_user_who_asked_for_reporting_still_gets_it(stub_trainer, tmp_path):
    """Recording sits in front of whatever `--report-to` built rather than
    replacing it, so instrumenting a run costs the user nothing."""
    import metalrunner.lora as entry

    downstream = []

    class Downstream:
        def on_train_loss_report(self, info):
            downstream.append(info)

        def on_val_loss_report(self, info):
            pass

    recorder = entry.progress.LossRecorder()
    wrapped = entry._recording(recorder)(lambda *a, **k: None)
    wrapped(None, None, None, None, Downstream())
    assert recorder.wrapped is not None

    recorder.on_train_loss_report({"train_loss": 1.0, "trained_tokens": 4})
    assert len(downstream) == 1


# ---------------------------------------------------------------------------
# The user path routes through the SAME installer the end-to-end measurement
# drives. Two code paths would mean the measurement measures a twin of the
# product rather than the product.
# ---------------------------------------------------------------------------
def test_nothing_certified_routes_nothing_and_the_receipt_says_so(
        stub_trainer, tmp_path):
    """Today's honest state, in numbers rather than prose: the report says
    nothing was routed and the receipt now counts it."""
    assert _train(tmp_path) == 0

    record = json.loads((tmp_path / "metalrunner-receipt.json").read_text())
    evidence = record["measurement"]
    assert evidence["routed_candidates"] == []
    assert evidence["routed_calls"] == evidence["routing_decisions"] == 0
    assert evidence["wrapper_installed"] is False, "uninstalled before writing"
    assert len(evidence["wrapper_sha256"]) == 64


def test_the_kernel_installer_is_removed_even_when_training_raises(
        monkeypatch, tmp_path):
    """A run that dies mid-step must not leave a routed kernel behind for
    whatever runs next in this interpreter."""
    import mlx_lm.lora

    from metalrunner import measurement

    uninstalled = []
    real_install = measurement.install

    def watched(**kwargs):
        patch = real_install(**kwargs)
        real_uninstall = patch.uninstall

        def note():
            uninstalled.append(True)
            real_uninstall()

        patch.uninstall = note
        return patch

    monkeypatch.setattr(measurement, "install", watched)

    def explode(_args, training_callback=None):
        raise RuntimeError("training blew up")

    monkeypatch.setattr(mlx_lm.lora, "run", explode)

    with pytest.raises(RuntimeError, match="training blew up"):
        _train(tmp_path)
    assert uninstalled == [True]


def test_a_refusing_installer_stops_the_run_and_takes_the_seam_back_out(
        monkeypatch, never_trains, capsys):
    """The installer refusing means this process is not what it claims, so
    it does not train, and the recorder seam does not outlive the attempt."""
    import mlx_lm.lora

    from metalrunner import measurement

    before = mlx_lm.lora.train_model
    monkeypatch.setattr(
        measurement, "install",
        lambda **_k: (_ for _ in ()).throw(
            measurement.MeasurementRefusal("candidate ab has no entry")))

    code = main(["--model", "some/model", "--train"])
    assert code == EXIT_SEAM_REFUSED
    assert not never_trains, "a refusing installer must not reach the trainer"
    assert "no entry" in capsys.readouterr().err
    assert mlx_lm.lora.train_model is before


def test_a_certified_operation_is_reported_routed(monkeypatch):
    """The report and the installer read one table, so a certified kernel
    cannot be routed while the report calls it declined, or the reverse."""
    from metalrunner import routing

    entry = {"candidate_sha256": "a" * 64,
             "operation": routing.KNOWN_OPERATIONS[0][0],
             "chip": "Apple M3 Pro", "bits": 4, "group_size": 64,
             "pricing_recording_sha256": "c" * 64}

    decisions = routing.decide("lora", 4, 64, on_chip="Apple M3 Pro",
                               force_stock=False, certified=(entry,))
    routed = [d for d in decisions if d.routed]
    assert [d.operation for d in routed] == [entry["operation"]]
    assert routed[0].reason == "certified and priced here"
    # The others decline for the honest reason about THIS table.
    assert all("no kernel certified for this operation" in d.reason
               for d in decisions if not d.routed)


def test_forced_stock_declines_even_a_certified_operation(monkeypatch):
    from metalrunner import routing

    entry = {"candidate_sha256": "a" * 64,
             "operation": routing.KNOWN_OPERATIONS[0][0],
             "chip": "Apple M3 Pro", "bits": 4, "group_size": 64,
             "pricing_recording_sha256": "c" * 64}

    decisions = routing.decide("lora", 4, 64, on_chip="Apple M3 Pro",
                               force_stock=True, certified=(entry,))
    assert not any(d.routed for d in decisions)
    assert all(routing.FORCE_STOCK_ENV in d.reason for d in decisions)


@pytest.mark.parametrize(("bits", "group_size", "chip"), [
    (8, 64, "Apple M3 Pro"),        # unverified width
    (4, 32, "Apple M3 Pro"),        # unverified group size
    (None, None, "Apple M3 Pro"),   # unreadable quantization
    (4, 64, "Apple M1"),            # certified somewhere else
])
def test_eligible_declines_rather_than_assuming(bits, group_size, chip):
    from metalrunner import routing

    entry = {"candidate_sha256": "a" * 64,
             "operation": routing.KNOWN_OPERATIONS[0][0],
             "chip": "Apple M3 Pro", "bits": 4, "group_size": 64,
             "pricing_recording_sha256": "c" * 64}
    assert routing.eligible("lora", bits, group_size, on_chip=chip,
                            certified=(entry,)) == ()


def test_eligible_returns_the_entry_when_everything_matches():
    from metalrunner import routing

    entry = {"candidate_sha256": "a" * 64,
             "operation": routing.KNOWN_OPERATIONS[0][0],
             "chip": "Apple M3 Pro", "bits": 4, "group_size": 64,
             "pricing_recording_sha256": "c" * 64}
    assert routing.eligible("lora", 4, 64, on_chip="Apple M3 Pro",
                            certified=(entry,)) == (entry,)
