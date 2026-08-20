"""The profile harness against a real model, for the claims only a GPU proves.

Three things cannot be pinned by arithmetic over records, and all three are
load-bearing:

  the process is stock          - every seam still holds the object mlx-lm
                                  defines, checked the way the installer
                                  checks it;
  the modes really alternate    - Amendment 4 needs a compiled total and an
                                  uncompiled total from one child on one
                                  batch, and MLX refuses an eval inside a
                                  compiled step, so the whole harness rests on
                                  the global compile switch surviving repeated
                                  toggling in one process;
  the batch is mlx-lm's         - the width a step runs at is a property of
                                  the data under mlx-lm's own padding rule,
                                  not of the configuration, and the harness
                                  must record the width that rule produces
                                  rather than the one the plan names.

Qwen3-0.6B stands in for the pinned 4B. The structure the harness reads - a
projection per layer, attention per layer, one output head, a backward that
reaches only the adapted blocks - is the same in both, and this one loads in
seconds.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import pin_dolly  # noqa: E402
import profile_instrument as pi  # noqa: E402
import profile_rules as rules  # noqa: E402
import profile_stock as ps  # noqa: E402
from conftest import requires_metal  # noqa: E402

MODEL = (Path(__file__).resolve().parents[1] / "bench" / ".models"
         / "qwen3-0.6b-4bit-g64")
BATCH = 2

# Short rows of deliberately different lengths, so the batch pads to the
# longest of them and the padding rule is visible rather than incidental.
ROWS = [
    {"prompt": "Name three primary colours.",
     "completion": "Red, blue and yellow."},
    {"prompt": "Summarise the water cycle in one paragraph, with detail "
               "about evaporation, condensation and precipitation.",
     "completion": "Water evaporates from oceans and lakes, rises, cools and "
                   "condenses into clouds, and returns as rain or snow, "
                   "which flows back to the sea."},
    {"prompt": "What is 12 times 12?", "completion": "144."},
]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("corpus")
    (directory / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ROWS))
    return directory


@pytest.fixture(scope="module")
def target(corpus):
    """The model, the optimizer and one fixed batch, through the real loader.

    Built once: loading the model costs far more than the steps being measured
    and every test here reads the same arrangement rather than provoking a new
    one.
    """
    plan = {"seed": 7, "optimizer": "adam", "optimizer_config": {},
            "learning_rate": 1e-5}
    provenance = {"base_model": {"directory": str(MODEL)},
                  "data": {"directory": str(corpus)}}
    return ps._load_target(plan, provenance, batch=BATCH, mask_prompt=True)


@requires_metal
def test_the_process_is_the_stock_one_the_profile_claims_to_measure(target):
    """A profile of stock taken through somebody's patch is a profile of the
    patch, and nothing in the numbers would say so."""
    report = ps.stock_process()
    assert set(report["seams"]) == {str(region.seam)
                                    for region in pi.REGIONS.values()}
    assert report["certified"] == 0
    for seam_name, module in report["seams"].items():
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
        corpus, target):
    """`max_seq_length` is a cap, not a target. mlx-lm pads each batch to one
    plus the next multiple of 32 above its own longest row, so the width is a
    property of the data, and a harness that recorded the registered number
    instead would name a shape the step never ran."""
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset

    tokenizer = load_tokenizer(MODEL)
    train, _, _ = load_dataset(
        ps._dataset_args({}, str(corpus), True), tokenizer)
    cached = CacheDataset(train)
    longest = max(len(cached[index][0]) for index in range(len(cached)))

    assert target.width == pin_dolly.batch_width(longest,
                                                 max_seq_length=rules.SEQ_LEN)
    assert target.width < rules.SEQ_LEN
    assert target.rows == BATCH


@requires_metal
def test_the_prompt_mask_survives_the_real_loader(target):
    """Candidate L's whole premise is that some tokens are not supervised. If
    the loader lost the prompt boundary every token would be supervised and
    the profile would be measuring the all-token case while labelled masked."""
    import mlx.core as mx

    _tokens, lengths = target.batch
    offsets = lengths[:, 0]
    assert bool(mx.all(offsets > 0))


@requires_metal
def test_the_marked_passes_compute_what_the_plain_pass_computes(target):
    """Over every gradient array rather than a summary: a summary can agree
    while the arrays beneath it do not."""
    import mlx.core as mx

    mx.disable_compile()
    try:
        report = ps._identity(target.model, target.batch, ps.MODES_DECIDING)
    finally:
        mx.enable_compile()
    assert report["gradients_compared"] > 0
    assert set(report["modes"]) == {"instr-A", "instr-L", "instr-Q"}
    for mode, verdict in report["modes"].items():
        assert verdict["loss_equal"], mode
        assert verdict["gradients_differing"] == [], mode
        assert verdict["foreign_on_removal"] == [], mode


@pytest.fixture(scope="module")
def measured(target):
    """Every deciding mode, twice around, on one built step.

    This is the harness's inner loop run for real: the same step object driven
    compiled and uncompiled with the marks installed and removed between, which
    is the arrangement Amendment 4 asks for and the one MLX makes awkward.
    """
    step, state = ps._build_step(target.model, target.optimizer)
    context = ps.cell_context("B", batch=target.rows, width=target.width,
                              model=MODEL.name, adapted=target.adapted)
    for mode in ps.MODES_DECIDING:
        ps.timed_mode(step, state, target.batch, mode, context=context)
    return {mode: [ps.timed_mode(step, state, target.batch, mode,
                                 context=context) for _ in range(2)]
            for mode in ps.MODES_DECIDING}


@requires_metal
def test_every_mode_produces_a_timed_step(measured):
    for mode, runs in measured.items():
        assert all(run["elapsed_s"] > 0.0 for run in runs), mode
        assert all(run["foreign_on_removal"] == [] for run in runs), mode


@requires_metal
def test_the_counts_are_exactly_what_the_arrangement_says(measured, target):
    """The check that catches a seam installed and never reached. Asserted
    against the harness's own expectation, so the expectation and the machine
    have to agree rather than the test restating one of them."""
    observed = {}
    for runs in measured.values():
        for run in runs:
            if run["decomposed"] is not None:
                observed.update(run["decomposed"]["counts"])
    marked = sorted({region for mode in ps.MODES_DECIDING
                     for region in ps.MODE_REGIONS[mode]})
    expected = ps.expected_counts(target.depth, target.adapted, marked)
    report = ps.completeness(observed, expected)
    assert report["ok"], report["mismatches"]


@requires_metal
def test_the_counts_do_not_move_between_repeats(measured):
    """Two rounds of one mode that fired different numbers of times are not
    repeats, and the median across them describes neither."""
    for mode, runs in measured.items():
        passes = [run["decomposed"] for run in runs
                  if run["decomposed"] is not None]
        if passes:
            assert ps.median_decomposition(passes)["passes"] == len(passes)


@requires_metal
def test_compilation_is_left_enabled_however_a_mode_ends(target):
    """Enabled is the state a process starts in. A harness that left it off
    would silently change what runs next in the same interpreter, including
    the rest of this suite."""
    import mlx.core as mx

    step, state = ps._build_step(target.model, target.optimizer)
    context = ps.cell_context("B", batch=target.rows, width=target.width,
                              model=MODEL.name, adapted=target.adapted)
    for mode in ps.MODES_DECIDING:
        ps.timed_mode(step, state, target.batch, mode, context=context)
        assert not mx.is_compile_disabled() if hasattr(
            mx, "is_compile_disabled") else True

    def explode(*_args, **_kwargs):
        raise RuntimeError("the step failed mid-measurement")

    with pytest.raises(RuntimeError):
        ps.timed_mode(explode, state, target.batch, "instr-Q", context=context)
    # The marks were removed and compilation restored on the way out, so a
    # plain step still runs and still computes.
    after = ps.timed_mode(step, state, target.batch, "compiled",
                          context=context)
    assert after["elapsed_s"] > 0.0


@requires_metal
def test_the_shares_a_real_cell_produces_are_fractions_of_a_real_step(
        measured, target):
    """The arithmetic and the machine meeting: real logs, read by the function
    the harness will use, over a denominator from the pass that carries no
    marks."""
    record = {
        "cell": "B",
        "context": ps.cell_context("B", batch=target.rows, width=target.width,
                                   model=MODEL.name, adapted=target.adapted),
        "modes": {mode: {
            "totals_s": [run["elapsed_s"] for run in runs],
            "peak_gb": [1.0 for _ in runs],
            "passes": [run["decomposed"] for run in runs
                       if run["decomposed"] is not None],
        } for mode, runs in measured.items()},
    }
    reading = ps.cell_reading(record)
    assert set(reading["shares"]) == set(rules.CANDIDATES)
    for mode, cost in reading["instrument_cost"].items():
        assert cost["ratio"] > 1.0, mode

    # BLOCKER, measured here: the registered share carries its own marks.
    #
    # Section 3.3 defines f as the region's wall time over the step's, and the
    # split-pass design was supposed to make that safe by taking the
    # denominator from the unmarked pass. It does stop one candidate's marks
    # from deflating another's share, and it does NOT stop a candidate's own
    # marks from inflating its own: the marks sit inside the spans they
    # bracket and outside the plain step they divide. The inflation is
    # proportional to how many marks a region carries, and that is exactly the
    # quantity that differs most between the three candidates - 197 marked
    # calls for Q, 32 for A, 2 for L - so it biases the comparison the rule
    # makes. At 0.6B it is severe enough that Q's share comes out above 1,
    # which is not a fraction of anything.
    #
    # The correction is measurable rather than estimated: in a split pass
    # nothing but this candidate's regions is marked, so the whole difference
    # between the instrumented total and the plain total was spent inside
    # those spans. It is computed and reported and deliberately not used,
    # because a share redefined after seeing a number is not a share.
    assert reading["shares"]["Q"]["share"] > 1.0
    assert all(share["instrument_excess_s"] > 0.0
               for share in reading["shares"].values())
    # And the excess is larger than the spans it would have to be subtracted
    # from, for the two heavily marked candidates. That is why no correction
    # is applied: the marks' cost is not all inside the spans, and how much of
    # it is cannot be recovered from these two numbers.
    for candidate in ("A", "Q"):
        share = reading["shares"][candidate]
        assert share["instrument_excess_s"] > share["marked_total_s"], candidate
    blockers = ps.binding_blockers({
        "closing_idle": {"idle": True}, "cells": {}, "readings": {"B": reading}})
    assert any("not a fraction of a step" in one for one in blockers)
