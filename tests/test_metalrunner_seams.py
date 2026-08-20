"""What metalrunner changes about a running process, and what it refuses to.

Two mechanisms are pinned here. The seam installer, which is the only place
this package replaces a name inside mlx-lm, and the progress recorder, which
is what it puts at the one seam it installs today.

The tests that matter most are the refusals and the restoration. A patch that
silently stacks on top of another patch, or one that survives the run that
installed it, both produce a process whose behaviour nobody wrote down; a
measurement taken through such a process is measuring something else. So the
installer is exercised against a foreign object already at the seam, against
a name that does not exist, against a second install, and against a body that
raises.

No mlx-lm module is patched in this file. The seams are taken on a module
built for the purpose, so a failing test here cannot leave the trainer
modified for whatever runs next.
"""

import sys
import types

import pytest

from metalrunner import progress, seams


@pytest.fixture()
def target(monkeypatch):
    """A throwaway module with one function in it, to be replaced."""
    module = types.ModuleType("metalrunner_seam_target")

    def greet(name):
        return f"stock {name}"

    greet.__module__ = module.__name__
    module.greet = greet
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.fixture()
def seam(target):
    return seams.Seam(target.__name__, "greet")


# ---------------------------------------------------------------------------
# Installing and restoring
# ---------------------------------------------------------------------------
def test_the_replacement_is_what_callers_reach(target, seam):
    installation = seams.Installation()
    installation.install(seam, lambda _original: lambda name: f"ours {name}")
    assert target.greet("x") == "ours x"


def test_the_wrapper_is_handed_the_object_it_replaced(target, seam):
    """A replacement that delegates must not have to look the original up
    again, because by then the name it would look up is its own."""
    installation = seams.Installation()

    def wrap(original):
        return lambda name: original(name).upper()

    installation.install(seam, wrap)
    assert target.greet("x") == "STOCK X"


def test_removal_puts_the_original_back(target, seam):
    original = target.greet
    installation = seams.Installation()
    installation.install(seam, lambda _o: lambda name: "ours")
    installation.remove()
    assert target.greet is original
    assert installation.foreign_on_removal == []


def test_the_context_manager_restores_after_a_raising_body(target, seam):
    """A training run that dies mid-step must not leave a replaced name
    behind for the next thing in this interpreter."""
    original = target.greet
    installation = seams.Installation()
    installation.install(seam, lambda _o: lambda name: "ours")
    with pytest.raises(RuntimeError):
        with installation:
            raise RuntimeError("training blew up")
    assert target.greet is original


# ---------------------------------------------------------------------------
# Counting: a seam that was never reached must be visible as such
# ---------------------------------------------------------------------------
def test_every_call_through_the_seam_is_counted(target, seam):
    installation = seams.Installation()
    installation.install(seam, lambda _o: lambda name: "ours")
    target.greet("a")
    target.greet("b")
    assert installation.counts == {str(seam): 2}


def test_a_seam_that_was_never_called_reports_zero_not_absence(target, seam):
    """Zero and missing are different claims. An installed seam with no
    calls says the replacement never ran; a missing key would say nothing
    was installed, which is the confusion this count exists to prevent."""
    installation = seams.Installation()
    installation.install(seam, lambda _o: lambda name: "ours")
    assert installation.counts == {str(seam): 0}


def test_the_count_survives_removal(target, seam):
    """The receipt is written after the seams come out, so the counts have
    to outlive them."""
    installation = seams.Installation()
    installation.install(seam, lambda _o: lambda name: "ours")
    target.greet("a")
    installation.remove()
    assert installation.counts == {str(seam): 1}


# ---------------------------------------------------------------------------
# Refusals: nothing is changed when the seam is not what it should be
# ---------------------------------------------------------------------------
def test_a_foreign_object_at_the_seam_refuses_and_changes_nothing(target, seam):
    """Installing on top of someone else's patch would make the run behave
    like the composition of two patches, which is not a thing anyone wrote."""
    def someone_elses(name):
        return "theirs"

    someone_elses.__module__ = "some_other_package"
    target.greet = someone_elses

    installation = seams.Installation()
    with pytest.raises(seams.SeamRefusal, match="already replaced it"):
        installation.install(seam, lambda _o: lambda name: "ours")
    assert target.greet is someone_elses
    assert installation.counts == {}


def test_a_missing_name_refuses(target):
    installation = seams.Installation()
    absent = seams.Seam(target.__name__, "no_such_function")
    with pytest.raises(seams.SeamRefusal, match="does not exist"):
        installation.install(absent, lambda _o: None)


def test_installing_the_same_seam_twice_refuses(target, seam):
    installation = seams.Installation()
    installation.install(seam, lambda _o: lambda name: "ours")
    with pytest.raises(seams.SeamRefusal, match="already installed"):
        installation.install(seam, lambda _o: lambda name: "ours again")


def test_a_name_replaced_mid_run_is_named_and_the_original_still_returns(
        target, seam):
    """Something patched the seam while the run was in flight. The process
    is left clean either way, and the run is marked suspect rather than
    quietly trusted."""
    original = target.greet
    installation = seams.Installation()
    installation.install(seam, lambda _o: lambda name: "ours")
    target.greet = lambda name: "an interloper"

    installation.remove()
    assert target.greet is original
    assert installation.foreign_on_removal == [str(seam)]


def test_seams_come_out_newest_first(monkeypatch):
    """Removal order is the reverse of installation order.

    It matters when one replacement wraps another: taking the outer one out
    last would leave the inner original buried under it. Order is observed
    through a module that records what is written to it, because setting the
    attribute back IS the removal.
    """
    class Recording(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            object.__setattr__(self, "restored", [])

        def __setattr__(self, name, value):
            self.restored.append(name)
            object.__setattr__(self, name, value)

    module = Recording("metalrunner_seam_order")

    def one():
        return 1

    def two():
        return 2

    one.__module__ = two.__module__ = module.__name__
    module.one, module.two = one, two
    monkeypatch.setitem(sys.modules, module.__name__, module)

    installation = seams.Installation()
    for name in ("one", "two"):
        installation.install(seams.Seam(module.__name__, name),
                             lambda original: original)
    module.restored.clear()
    installation.remove()

    assert module.restored == ["two", "one"]
    assert module.one is one and module.two is two


# ---------------------------------------------------------------------------
# The progress recorder
# ---------------------------------------------------------------------------
def test_it_keeps_every_report_the_trainer_makes():
    recorder = progress.LossRecorder()
    recorder.on_train_loss_report({"iteration": 1, "train_loss": 2.5,
                                   "trained_tokens": 100})
    recorder.on_train_loss_report({"iteration": 2, "train_loss": 2.0,
                                   "trained_tokens": 210})
    recorder.on_val_loss_report({"iteration": 2, "val_loss": 2.2})

    summary = recorder.summary()
    assert summary["train_reports"] == 2 and summary["val_reports"] == 1
    assert summary["first_train_loss"] == 2.5
    assert summary["last_train_loss"] == 2.0
    assert summary["min_train_loss"] == 2.0
    assert summary["last_val_loss"] == 2.2
    assert summary["trained_tokens"] == 210


def test_the_recorded_report_is_a_copy():
    """The trainer reuses nothing today, but a record that can be edited
    from outside after the fact is not a record."""
    recorder = progress.LossRecorder()
    reported = {"iteration": 1, "train_loss": 2.5, "trained_tokens": 100}
    recorder.on_train_loss_report(reported)
    reported["train_loss"] = 99.0
    assert recorder.summary()["last_train_loss"] == 2.5


def test_a_run_that_reported_nothing_summarises_without_arithmetic():
    """A run with fewer steps than one reporting interval, or one that died
    early, must produce a summary rather than an exception."""
    summary = progress.LossRecorder().summary()
    assert summary["train_reports"] == 0
    assert summary["last_train_loss"] is None
    assert summary["trained_tokens"] is None


def test_a_user_who_asked_for_wandb_still_gets_wandb():
    """Recording must not cost the user the reporting they asked for."""
    class Downstream:
        def __init__(self):
            self.train, self.val = [], []

        def on_train_loss_report(self, info):
            self.train.append(info)

        def on_val_loss_report(self, info):
            self.val.append(info)

    downstream = Downstream()
    recorder = progress.LossRecorder().chained(downstream)
    recorder.on_train_loss_report({"train_loss": 1.0, "trained_tokens": 5})
    recorder.on_val_loss_report({"val_loss": 1.1})

    assert len(downstream.train) == 1 and len(downstream.val) == 1
    assert recorder.summary()["train_reports"] == 1


def test_no_downstream_callback_is_not_an_error():
    """`--report-to` unset means mlx-lm builds no callback at all, which is
    the ordinary case."""
    recorder = progress.LossRecorder().chained(None)
    recorder.on_train_loss_report({"train_loss": 1.0, "trained_tokens": 5})
    assert recorder.summary()["train_reports"] == 1


# ---------------------------------------------------------------------------
# Re-exports: bound in one module, defined in another
#
# mlx-lm binds `train` and `get_reporting_callbacks` into mlx_lm.lora but
# defines them under mlx_lm.tuner. A seam on either is a legitimate re-export,
# and a check that demanded the binding and defining modules agree would refuse
# exactly the names a measurement needs to reach.
# ---------------------------------------------------------------------------
def test_a_declared_re_export_installs(monkeypatch):
    home = types.ModuleType("metalrunner_seam_home")
    away = types.ModuleType("metalrunner_seam_away")

    def borrowed():
        return "stock"

    borrowed.__module__ = away.__name__
    away.borrowed = borrowed
    home.borrowed = borrowed
    for module in (home, away):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    seam = seams.Seam(home.__name__, "borrowed", defined_in=away.__name__)
    installation = seams.Installation()
    installation.install(seam, lambda _o: lambda: "ours")
    assert home.borrowed() == "ours"
    installation.remove()
    assert home.borrowed is borrowed


def test_an_undeclared_re_export_still_refuses(monkeypatch):
    """Silence about the re-export is not permission for one: a seam that did
    not say where its object comes from is refused, so the surprise surfaces
    at the seam rather than being absorbed."""
    home = types.ModuleType("metalrunner_seam_home2")
    away = types.ModuleType("metalrunner_seam_away2")

    def borrowed():
        return "stock"

    borrowed.__module__ = away.__name__
    home.borrowed = borrowed
    for module in (home, away):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    installation = seams.Installation()
    with pytest.raises(seams.SeamRefusal, match="already replaced it"):
        installation.install(seams.Seam(home.__name__, "borrowed"),
                             lambda _o: lambda: "ours")


def test_a_foreign_object_at_a_declared_re_export_refuses(monkeypatch):
    """Declaring the origin narrows the check, it does not remove it."""
    home = types.ModuleType("metalrunner_seam_home3")
    away = types.ModuleType("metalrunner_seam_away3")

    def interloper():
        return "theirs"

    interloper.__module__ = "some_other_package"
    home.borrowed = interloper
    for module in (home, away):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    installation = seams.Installation()
    with pytest.raises(seams.SeamRefusal, match="already replaced it"):
        installation.install(
            seams.Seam(home.__name__, "borrowed", defined_in=away.__name__),
            lambda _o: lambda: "ours")


def test_the_real_mlx_lm_re_exports_are_reachable():
    """The acceptance case, against the live install rather than a fake: the
    two names an end-to-end measurement reaches are both re-exports, and both
    must be installable."""
    import mlx_lm.lora

    for attribute, origin in (("train", "mlx_lm.tuner.trainer"),
                              ("get_reporting_callbacks",
                               "mlx_lm.tuner.callbacks")):
        assert getattr(mlx_lm.lora, attribute).__module__ == origin
        installation = seams.Installation()
        installation.install(
            seams.Seam("mlx_lm.lora", attribute, defined_in=origin),
            lambda original: original)
        installation.remove()
        assert getattr(mlx_lm.lora, attribute).__module__ == origin


# ---------------------------------------------------------------------------
# Names that live on a class, not directly on the module
# ---------------------------------------------------------------------------
# The Day 1 profile has to reach `QuantizedLinear.__call__` and
# `QuantizedEmbedding.as_linear`, which are methods rather than module
# globals. A method is looked up on its class at call time exactly the way a
# module global is looked up on its module, so it is a seam by the same
# argument, and the seam names it with a dotted path.
# ---------------------------------------------------------------------------
@pytest.fixture()
def classy(monkeypatch):
    """A throwaway module holding a class with two methods on it."""
    module = types.ModuleType("metalrunner_seam_class")

    class Widget:
        def __call__(self, value):
            return f"stock call {value}"

        def sideways(self, value):
            return f"stock sideways {value}"

    Widget.__module__ = module.__name__
    Widget.__call__.__module__ = module.__name__
    Widget.sideways.__module__ = module.__name__
    module.Widget = Widget
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_a_method_seam_is_reached_by_calling_the_instance(classy):
    """The mechanism the profile depends on: replacing the method on the
    class changes what `instance(...)` does, for every instance."""
    seam = seams.Seam(classy.__name__, "Widget.__call__")
    installation = seams.Installation()
    installation.install(seam, lambda _o: lambda self, value: f"ours {value}")
    assert classy.Widget()("x") == "ours x"


def test_a_method_seam_still_receives_the_instance(classy):
    """`self` arrives as the first argument, so a replacement that delegates
    can still reach the instance's own state."""
    seen = []
    seam = seams.Seam(classy.__name__, "Widget.sideways")
    installation = seams.Installation()

    def wrap(original):
        def call(self, value):
            seen.append(self)
            return original(self, value)
        return call

    installation.install(seam, wrap)
    widget = classy.Widget()
    assert widget.sideways("y") == "stock sideways y"
    assert seen == [widget]


def test_removing_a_method_seam_puts_the_method_back(classy):
    original = classy.Widget.__call__
    seam = seams.Seam(classy.__name__, "Widget.__call__")
    with seams.Installation() as installation:
        installation.install(seam, lambda _o: lambda self, value: "ours")
    assert classy.Widget.__call__ is original
    assert classy.Widget()("x") == "stock call x"


def test_calls_through_a_method_seam_are_counted(classy):
    """The count is what proves a region was actually reached. The profile
    reads it to check attention fired once per layer per step rather than
    trusting that it did."""
    seam = seams.Seam(classy.__name__, "Widget.__call__")
    installation = seams.Installation()
    installation.install(seam, lambda original: original)
    widget = classy.Widget()
    for _ in range(3):
        widget("x")
    assert installation.counts[str(seam)] == 3


def test_a_missing_step_on_the_way_refuses_and_names_it(classy):
    """A class renamed upstream must not read as a seam holding something
    unexpected: the two have different repairs, so they get different
    refusals."""
    installation = seams.Installation()
    with pytest.raises(seams.SeamRefusal, match="has no 'Gadget'"):
        installation.install(seams.Seam(classy.__name__, "Gadget.__call__"),
                             lambda _o: lambda self: None)


def test_a_missing_method_on_a_real_class_refuses(classy):
    installation = seams.Installation()
    with pytest.raises(seams.SeamRefusal, match="does not exist"):
        installation.install(seams.Seam(classy.__name__, "Widget.absent"),
                             lambda _o: lambda self: None)


def test_a_foreign_method_at_a_class_seam_refuses(classy):
    """The origin check does not weaken on the way through a class."""
    def interloper(self, value):
        return "theirs"

    interloper.__module__ = "some_other_package"
    classy.Widget.__call__ = interloper

    installation = seams.Installation()
    with pytest.raises(seams.SeamRefusal, match="already replaced it"):
        installation.install(seams.Seam(classy.__name__, "Widget.__call__"),
                             lambda _o: lambda self, value: "ours")


def test_replacing_call_on_an_instance_reaches_nothing(classy):
    """Why the seam goes on the class and not the object.

    Python resolves `instance(...)` through the type, so a `__call__` written
    onto the instance is never consulted. An instrument built that way
    installs nothing, raises nothing, and reports every region as absent,
    which is the failure this test exists to keep out of the profile.
    """
    widget = classy.Widget()
    widget.__call__ = lambda value: f"ours {value}"
    assert widget("x") == "stock call x"


def test_the_real_profile_patch_points_are_reachable():
    """The acceptance case against the live install: all four names the Day 1
    profile marks must be installable, and two of them are methods."""
    import mlx.nn  # noqa: F401
    import mlx_lm.models.qwen3  # noqa: F401

    points = (
        ("mlx_lm.models.qwen3", "scaled_dot_product_attention",
         "mlx_lm.models.base"),
        ("mlx.nn", "QuantizedLinear.__call__", "mlx.nn.layers.quantized"),
        ("mlx.nn", "QuantizedEmbedding.as_linear", "mlx.nn.layers.quantized"),
        ("mlx.nn", "losses.cross_entropy", "mlx.nn.losses"),
    )
    for module, attribute, origin in points:
        seam = seams.Seam(module, attribute, defined_in=origin)
        owner, name = seam.resolve()
        assert getattr(owner, name).__module__ == origin
        installation = seams.Installation()
        installation.install(seam, lambda original: original)
        installation.remove()
        assert getattr(*seam.resolve()).__module__ == origin
        assert installation.foreign_on_removal == []
