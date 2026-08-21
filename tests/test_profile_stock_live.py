"""The profile harness against a real model, for the claims only a GPU proves.

What lives here is what arithmetic over records cannot pin:

  the process is stock          - every seam still holds the object mlx-lm
                                  defines, checked the way the installer
                                  checks it;
  the batch is mlx-lm's         - the width a step runs at is a property of
                                  the data under mlx-lm's own padding rule,
                                  not of the configuration, and the harness
                                  must record the width that rule produces
                                  rather than the one the plan names;
  the settings are the model's  - a dial asked for 0.75 places a whole number
                                  of quantization groups or of rows, and the
                                  fraction it lands on is a property of the
                                  model's dimensions rather than of the
                                  ladder;
  the manifest really runs      - every arm of one registered width built,
                                  traced under its own seams, and timed with
                                  no installer active at all.

The arm machinery itself - one trace per arm, two arms not sharing a graph, no
seam installed during timing, the full setting bit-identical to stock, every
patched class restored - is proved in tests/test_profile_knobs_live.py, which
owns it. This file does not restate those claims.

Qwen3-0.6B stands in for the pinned 4B. The structure the harness reads - a
projection per layer, attention per layer, one output head, a backward that
reaches only the adapted blocks - is the same in both, and this one loads in
seconds. Its rows are chosen so mlx-lm's own padding rule lands the batch on
the registered SHORT band's width, which is what lets the real context be
built rather than stubbed.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import pin_dolly  # noqa: E402
import profile_instrument as pi  # noqa: E402
import profile_knobs as pk  # noqa: E402
import profile_rules as rules  # noqa: E402
import profile_stock as ps  # noqa: E402
from conftest import requires_metal  # noqa: E402

MODEL = (Path(__file__).resolve().parents[1] / "bench" / ".models"
         / "qwen3-0.6b-4bit-g64")
BATCH = 2
WIDTH = "short"

# Rows of deliberately different lengths, all inside the 33-to-64 token band,
# so the batch pads to the registered short band's own width of 65 and the
# padding rule is visible rather than incidental.
ROWS = [
    {"prompt": "Name three primary colours.",
     "completion": "Red, blue and yellow are the three primary colours used."},
    {"prompt": "Summarise the water cycle briefly.",
     "completion": "Water evaporates, condenses into clouds, and falls as "
                   "rain."},
    {"prompt": "What is 12 times 12?",
     "completion": "It is one hundred and forty four."},
]

PLAN = {"seed": 7, "optimizer": "adam", "optimizer_config": {},
        "learning_rate": 1e-5}


class _Guard:
    """The child's memory guard, reduced to the one method a build calls."""

    def __init__(self):
        self.checks = []

    def check(self, label):
        self.checks.append(label)
        return 0.0


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("corpus")
    (directory / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ROWS))
    return directory


@pytest.fixture(scope="module")
def target():
    """The model, the adapters and a settled optimizer, built once.

    Loading the model costs far more than the steps being measured, and every
    test here reads the same arrangement rather than provoking a new one.
    """
    provenance = {"base_model": {"directory": str(MODEL)}}
    return ps._load_model(PLAN, provenance)


@pytest.fixture(scope="module")
def batch(target, corpus):
    return ps._fixed_batch(PLAN, target.tokenizer, str(corpus),
                           batch=BATCH, mask_prompt=True)


@requires_metal
def test_the_process_is_the_stock_one_the_profile_claims_to_measure():
    """A profile of stock taken through somebody's patch is a profile of the
    patch, and nothing in the numbers would say so."""
    report = ps.stock_process()
    assert set(report["seams"]) == {str(region.seam)
                                    for region in pi.REGIONS.values()}
    assert report["certified"] == 0
    for _seam_name, module in report["seams"].items():
        assert module and not module.startswith("metalrunner")


@requires_metal
def test_a_replaced_seam_makes_the_process_refuse_to_be_profiled():
    """The check has to be able to fail, or it is decoration. Installing
    through the real installer and asking again is the only honest way to show
    that it does."""
    recorder = pi.Recorder()
    installation = pi.install(["attn-core"], recorder)
    try:
        with pytest.raises(Exception, match="not stock"):
            ps.stock_process()
    finally:
        installation.remove()
    assert ps.stock_process()["certified"] == 0


@requires_metal
def test_the_batch_width_is_the_one_mlx_lm_s_own_padding_rule_produces(
        corpus, target, batch):
    """`max_seq_length` is a cap, not a target. mlx-lm pads each batch to one
    plus the next multiple of 32 above its own longest row, so the width is a
    property of the data, and a harness that recorded the registered number
    instead would name a shape the step never ran."""
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset

    train, _, _ = load_dataset(
        ps._dataset_args({}, str(corpus), True), target.tokenizer)
    cached = CacheDataset(train)
    longest = max(len(cached[index][0]) for index in range(len(cached)))

    assert batch.width == pin_dolly.batch_width(longest,
                                                max_seq_length=rules.SEQ_LEN)
    assert batch.width == rules.WIDTHS[WIDTH]["batch_width"]
    assert batch.rows == BATCH


@requires_metal
def test_the_context_the_batch_produces_is_the_registered_one(target, batch):
    context = ps.cell_context(
        "B", WIDTH, batch=batch.rows, batch_width=batch.width,
        model=MODEL.name, adapted=target.adapted,
        supervised=batch.supervised, supervised_of=batch.supervised_of,
        batch_sha256=batch.digest)
    assert context["operation_width"] == batch.width - 1
    assert context["tokens"] == batch.rows * (batch.width - 1)
    assert 0.0 < context["supervised_fraction"] < 1.0


@requires_metal
def test_the_prompt_mask_survives_the_real_loader(batch):
    """Candidate L's whole premise is that some tokens are not supervised. If
    the loader lost the prompt boundary every token would be supervised and
    the profile would be measuring the all-token case while labelled masked."""
    import mlx.core as mx

    _tokens, lengths = batch.batch
    assert bool(mx.all(lengths[:, 0] > 0))
    assert batch.supervised < batch.supervised_of


@requires_metal
def test_the_supervised_count_is_mlx_lm_s_own_mask_and_not_an_approximation(
        target, batch):
    """Derived from the batch rather than read off a step, because every arm's
    loss is meaningless at a dialled setting and a count taken from one would
    be a count of whatever that arm happened to compute."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.trainer import default_loss

    del tree_flatten, nn
    mx.disable_compile()
    try:
        _loss, toks = default_loss(target.model, *batch.batch)
        mx.eval(toks)
    finally:
        mx.enable_compile()
    assert int(toks.item()) == batch.supervised


@requires_metal
def test_the_structural_pass_counts_exactly_what_the_arrangement_says(
        target, batch):
    """The check that catches a seam installed and never reached. Asserted
    against the harness's own expectation, so the expectation and the machine
    have to agree rather than the test restating one of them."""
    report = ps._structural_pass(target.model, batch.batch)
    expected = ps.expected_counts(target.depth, target.adapted,
                                  sorted(pi.REGIONS))
    checked = ps.completeness(report["counts"], expected)
    assert checked["ok"], checked["mismatches"]
    assert report["foreign_on_removal"] == []


@requires_metal
def test_the_marked_pass_computes_what_the_plain_pass_computes(target, batch):
    """Over every gradient array rather than a summary: a summary can agree
    while the arrays beneath it do not."""
    report = ps._structural_pass(target.model, batch.batch)
    assert report["gradients_compared"] > 0
    assert report["loss_equal"]
    assert report["gradients_differing"] == []


@requires_metal
def test_the_structural_pass_leaves_compilation_enabled(target, batch):
    """Enabled is the state a process starts in. A harness that left it off
    would silently change what runs next in the same interpreter, including
    the rest of this suite."""
    import mlx.core as mx

    ps._structural_pass(target.model, batch.batch)
    traces = []

    @mx.compile
    def step(value):
        traces.append(1)
        return value + 1

    step(mx.array(1.0))
    step(mx.array(2.0))
    assert len(traces) == 1


@requires_metal
@pytest.mark.parametrize("candidate", ["L", "Q", "P3"])
def test_every_dial_places_the_fraction_the_model_can_realise(target, batch,
                                                              candidate):
    """A fit run on the fraction that was asked for rather than the one that
    was placed is a fit on numbers nothing measured."""
    knob = ps._knob_for(candidate)
    prepared = knob.prepare(target.model, batch.width)
    settings = ps._actual_settings(candidate, prepared)
    assert sorted(settings) == sorted(pk.PHIS)
    placed = [settings[phi]["actual_phi"] for phi in pk.PHIS]
    assert len(set(placed)) == len(pk.PHIS)
    assert placed == sorted(placed, reverse=True)
    for phi in pk.PHIS:
        for site in settings[phi]["sites"].values():
            assert 0 < site["kept"] <= site["full"]
            assert site["fraction"] == pytest.approx(settings[phi]["actual_phi"])


@requires_metal
def test_the_attention_dial_the_amendment_names_is_the_one_that_is_built(
        target, batch):
    """Amendment 7 clause 35 names candidate A's dial by rule, so no run
    reselects it and a recording cannot depend on which dial happened to price
    best that night."""
    knob = ps._knob_for("A")
    assert knob is pk.ATTENTION_KNOBS[ps.ATTENTION_DIAL]
    assert ps.ATTENTION_DIAL == "kv-length"
    prepared = knob.prepare(target.model, batch.width)
    settings = ps._actual_settings("A", prepared)
    full = pk.attention_width(batch.width)
    assert settings[1.0]["sites"][f"attention:{full}"]["kept"] == full


@requires_metal
def test_a_candidate_with_no_registered_dial_is_refused_not_guessed():
    with pytest.raises(Exception, match="no dial is registered"):
        ps._knob_for("P2")


@pytest.fixture(scope="module")
def built(target, batch):
    """Every arm of the short width, built, traced under its seams and warmed.

    This is the harness's inner loop run for real. It is a module fixture
    because building 26 compiled steps costs far more than timing them.
    """
    return ps._build_width(target, batch, WIDTH, _Guard())


@requires_metal
def test_the_whole_registered_manifest_builds(built):
    labels = [arm.label for arm in built["compiled"]]
    assert labels == [arm.label for arm in ps.arm_manifest(WIDTH)]
    assert len(labels) == 26
    assert set(built["roles"]) == set(labels)


@requires_metal
def test_every_arm_carries_the_fraction_it_actually_placed(built):
    for label, role in built["roles"].items():
        if role["nominal_phi"] is None:
            assert role["actual_phi"] is None, label
        else:
            assert role["actual_phi"] is not None, label
            assert role["sites"], label


@requires_metal
def test_the_stock_and_ablated_arms_carry_no_setting_at_all(built):
    """Their `phi` is absent rather than a fictitious zero or one, because a
    fit that read either would be fitting a point no dial placed."""
    for label, role in built["roles"].items():
        if role["role"] in (pk.STOCK, pk.ABLATION):
            assert role["nominal_phi"] is None and role["actual_phi"] is None


@requires_metal
def test_every_arm_of_the_manifest_is_timed_in_every_round(target, batch,
                                                           built):
    """All the arms at one width go through ONE `timed_rounds` call. Running
    one round group per candidate would let the machine drift between two
    candidates the rule then compares."""
    samples = pk.timed_rounds(built["compiled"], batch.batch, rounds=2)
    assert set(samples) == set(built["roles"])
    for label, values in samples.items():
        assert len(values) == 2, label
        assert all(value > 0.0 for value in values), label


@requires_metal
def test_the_reducer_reads_a_real_width_end_to_end(target, batch, built):
    """The arithmetic and the machine meeting: real arms, real samples, read
    by the function the harness will use.

    It asserts the SHAPE and not the numbers. Nothing here says which gates a
    0.6B proxy clears at a resolution floor nothing has measured, and a test
    that did would be pinning this machine's noise.
    """
    samples = pk.timed_rounds(built["compiled"], batch.batch, rounds=2)
    context = ps.cell_context(
        "B", WIDTH, batch=batch.rows, batch_width=batch.width,
        model=MODEL.name, adapted=target.adapted,
        supervised=batch.supervised, supervised_of=batch.supervised_of,
        batch_sha256=batch.digest)
    measured = {
        "width": WIDTH, "context": context, "peak_gb": 1.0,
        "arms": {label: dict(built["roles"][label], context=context,
                             samples_ms=[value * 1000.0
                                         for value in samples[label]])
                 for label in samples},
    }
    reading = ps.width_reading(measured, resolution_floor_ms=0.155)
    assert set(reading["entries"]) == set(ps.CANDIDATES_AT_WIDTH[WIDTH])
    for candidate, entry in reading["entries"].items():
        assert entry["type"] in ("reading", "missing_share"), candidate
        evidence = entry if entry["type"] == "reading" else entry["evidence"]
        assert evidence["stock_median_ms"] > 0.0
        assert len(evidence["per_round"]) == 2
    # P2 is measured by candidate A's dial and candidate A is at the long
    # width alone, so the short width can never PASS this test.
    assert reading["partition"]["outcome"] in ("NOT RUN", "REJECT")
