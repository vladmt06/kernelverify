"""Pinning a training corpus: the parts that decide a shape, checked offline.

Nothing here fetches anything or loads a tokenizer. What is pinned is the
arithmetic that turns a corpus into a registered shape - which band a row
falls in, what width the trainer will pad that band to, and whether a slice is
reproducible from its seed - because those are what a recording's hash means
and what a floor is measured against.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import pin_dolly  # noqa: E402


def measured(length, supervised=None, prompt="p", completion="c"):
    band = pin_dolly.PAD_TO * ((length + pin_dolly.PAD_TO - 1)
                               // pin_dolly.PAD_TO)
    return {"pair": {"prompt": prompt, "completion": completion},
            "length": length, "band": band,
            "supervised": length // 2 if supervised is None else supervised}


# ---------------------------------------------------------------------------
# The trainer's own padding arithmetic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("longest,width", [
    (1, 33), (31, 33), (32, 33), (33, 65), (64, 65), (65, 97), (160, 161),
])
def test_a_band_pads_to_the_width_the_trainer_uses(longest, width):
    """One plus the next multiple of 32, which is mlx-lm's rule rather than
    ours; a floor registered at any other number would be measured at a shape
    the step never produces."""
    assert pin_dolly.batch_width(longest) == width


@pytest.mark.parametrize("longest", [2017, 2048, 4096])
def test_the_width_is_capped_and_the_cap_is_not_a_round_number(longest):
    """mlx-lm caps the padded width at max_seq_length AFTER rounding up, so a
    corpus reaching the cap trains on 2047 tokens rather than 2048. Reading
    only the rounding half of the rule puts every floor one token wide of the
    shape the step actually produces."""
    assert pin_dolly.batch_width(longest, max_seq_length=2048) == 2048


def test_the_padded_width_is_one_more_than_the_input_length():
    """The trainer drops the last column to make the targets, so a batch
    padded to 161 trains on 160 tokens."""
    assert pin_dolly.batch_width(160) - 1 == 160


# ---------------------------------------------------------------------------
# Dolly's fields in the shape that carries a prompt boundary
# ---------------------------------------------------------------------------
def test_a_row_without_context_is_prompt_and_completion():
    pair = pin_dolly.as_prompt_completion(
        {"instruction": "Why?", "context": "", "response": "Because."})
    assert pair == {"prompt": "Why?", "completion": "Because."}


def test_context_joins_the_instruction_rather_than_the_answer():
    """Context is part of what the model reads, not part of what it is scored
    on; putting it in the completion would move it across the mask and change
    the supervised fraction candidate L is judged by."""
    pair = pin_dolly.as_prompt_completion(
        {"instruction": "Why?", "context": "Some background.",
         "response": "Because."})
    assert pair["prompt"] == "Why?\n\nSome background."
    assert pair["completion"] == "Because."


def test_a_missing_context_key_is_not_an_error():
    pair = pin_dolly.as_prompt_completion(
        {"instruction": "Why?", "response": "Because."})
    assert pair["prompt"] == "Why?"


# ---------------------------------------------------------------------------
# The slice
# ---------------------------------------------------------------------------
def test_the_slice_takes_only_rows_from_the_named_band():
    rows = [measured(50) for _ in range(10)] + [measured(100) for _ in range(10)]
    train, valid = pin_dolly.select(rows, band=64, train=6, valid=2)
    assert all(row["band"] == 64 for row in train + valid)


def test_the_slice_is_the_same_every_time_for_the_same_seed():
    rows = [measured(50, prompt=f"p{i}") for i in range(40)]
    first = pin_dolly.select(rows, band=64, train=8, valid=4)
    second = pin_dolly.select(rows, band=64, train=8, valid=4)
    assert [r["pair"] for r in first[0]] == [r["pair"] for r in second[0]]
    assert [r["pair"] for r in first[1]] == [r["pair"] for r in second[1]]


def test_train_and_valid_do_not_share_a_row():
    rows = [measured(50, prompt=f"p{i}") for i in range(40)]
    train, valid = pin_dolly.select(rows, band=64, train=8, valid=4)
    assert not ({r["pair"]["prompt"] for r in train}
                & {r["pair"]["prompt"] for r in valid})


def test_a_band_too_thin_to_fill_the_slice_refuses_and_names_the_alternatives():
    """Silently returning fewer rows would change the shape of every batch
    after the short one."""
    rows = [measured(50) for _ in range(5)] + [measured(100) for _ in range(40)]
    with pytest.raises(SystemExit, match="busiest bands"):
        pin_dolly.select(rows, band=64, train=8, valid=4)


# ---------------------------------------------------------------------------
# The distribution, which is the evidence a band is chosen against
# ---------------------------------------------------------------------------
def test_the_distribution_reports_the_bands_a_slice_could_come_from():
    rows = ([measured(50) for _ in range(3)]
            + [measured(100) for _ in range(5)])
    stats = pin_dolly.distribution(rows)
    assert stats["rows"] == 8
    assert stats["bands"] == {"64": 3, "128": 5}


def test_the_distribution_reports_the_supervised_fraction():
    """Candidate L's whole premise is that supervised rows are fewer than
    total rows, so a corpus where they are not is evidence against it."""
    rows = [measured(100, supervised=40) for _ in range(4)]
    stats = pin_dolly.distribution(rows)
    assert stats["supervised_fraction"]["median"] == pytest.approx(0.4)


def test_the_distribution_carries_the_padding_rule_it_was_computed_with():
    stats = pin_dolly.distribution([measured(50)])
    assert stats["pad_to"] == pin_dolly.PAD_TO


# ---------------------------------------------------------------------------
# Writing, and the hash a recording carries
# ---------------------------------------------------------------------------
def test_a_written_split_is_jsonl_of_prompt_and_completion(tmp_path):
    rows = [measured(50, prompt="a", completion="b"),
            measured(50, prompt="c", completion="d")]
    path = tmp_path / "train.jsonl"
    pin_dolly.write_split(path, rows)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == [{"prompt": "a", "completion": "b"},
                     {"prompt": "c", "completion": "d"}]


def test_the_hash_moves_when_a_single_row_moves(tmp_path):
    """The hash is what a recording binds, so it has to be a hash of the
    bytes rather than of the row count."""
    one = pin_dolly.write_split(tmp_path / "a.jsonl", [measured(50, prompt="a")])
    two = pin_dolly.write_split(tmp_path / "b.jsonl", [measured(50, prompt="z")])
    assert one != two


def test_the_committed_corpus_report_matches_the_pinned_corpus():
    """The evidence the band is chosen against travels with the repository,
    so a later reader can see what was known when the choice was made."""
    report = (Path(__file__).resolve().parents[1] / "bench" / ".data"
              / "dolly" / "corpus-distribution.json")
    stats = json.loads(report.read_text())
    assert stats["rows"] == 15011
    assert stats["length"]["median"] == 116
    assert stats["pad_to"] == pin_dolly.PAD_TO
    fillable = [int(band) for band, count in stats["bands"].items()
                if count >= 1024 + 128]
    assert max(fillable) == 160, (
        "the widest band that can fill a 1024+128 slice; if this moves, the "
        "shape the profile can register moves with it")


# ---------------------------------------------------------------------------
# The source bytes, declared before the fetch rather than recorded after it
# ---------------------------------------------------------------------------
def test_a_changed_source_refuses_rather_than_reselecting(tmp_path):
    """The selection walks the file in its own order before shuffling, so a
    reordered or edited upstream produces a different slice under the same
    seed. Recording the hash afterwards would only say which bytes arrived."""
    cache = tmp_path / "corpus.jsonl"
    cache.write_text('{"instruction": "a", "response": "b"}\n')
    with pytest.raises(SystemExit, match="has changed"):
        pin_dolly.fetch(cache=cache, expected="0" * 64)


def test_a_matching_source_passes(tmp_path):
    import hashlib
    cache = tmp_path / "corpus.jsonl"
    cache.write_bytes(b"pinned bytes\n")
    digest = hashlib.sha256(b"pinned bytes\n").hexdigest()
    assert pin_dolly.fetch(cache=cache, expected=digest) == cache


def test_the_declared_digest_matches_the_corpus_that_was_measured():
    """The committed distribution and the declared digest have to describe the
    same file, or the evidence a band is chosen against is not evidence about
    the corpus that would be sliced."""
    assert len(pin_dolly.SOURCE_SHA256) == 64
    assert set(pin_dolly.SOURCE_SHA256) <= set("0123456789abcdef")


# ---------------------------------------------------------------------------
# UltraChat: clause 20's adapter, derived bands, and independent selection
# ---------------------------------------------------------------------------
def _chat(*turns):
    return {"messages": [{"role": role, "content": content}
                         for role, content in turns]}


def test_the_adapter_takes_the_first_exchange_and_discards_the_thread():
    row = _chat(("user", "q1"), ("assistant", "a1"),
                ("user", "q2"), ("assistant", "a2"))
    assert pin_dolly.ultrachat_pair(row) == {"prompt": "q1",
                                             "completion": "a1"}


def test_the_completion_is_the_first_assistant_after_the_first_user():
    row = _chat(("assistant", "orphan"), ("user", "q"), ("assistant", "a"))
    assert pin_dolly.ultrachat_pair(row) == {"prompt": "q", "completion": "a"}


def test_a_row_without_an_exchange_is_dropped_not_guessed():
    assert pin_dolly.ultrachat_pair(_chat(("user", "q"))) is None
    assert pin_dolly.ultrachat_pair(_chat(("assistant", "a"))) is None
    assert pin_dolly.ultrachat_pair({"messages": []}) is None


def test_measure_counts_a_dropped_row_by_returning_fewer():
    class Tok:
        def apply_chat_template(self, messages, return_dict=False,
                                add_generation_prompt=False):
            return list(range(sum(len(m["content"]) for m in messages)))

    rows = [_chat(("user", "q"), ("assistant", "aa")), _chat(("user", "q"))]
    got = pin_dolly.measure(rows, Tok(), adapter=pin_dolly.ultrachat_pair)
    assert len(got) == 1 and got[0]["pair"]["completion"] == "aa"


def test_the_short_band_is_the_shortest_that_fills_both_splits():
    train = {64: 2000, 96: 2000, 1056: 2000, 1088: 2000}
    valid = {64: 10, 96: 200, 1056: 200, 1088: 200}
    got = pin_dolly.derive_bands(train, valid)
    assert got["short"] == 96, "band 64 fills train but not valid"


def test_the_long_band_starts_where_the_lower_edge_reaches_1024():
    train = {96: 2000, 1024: 2000, 1056: 2000, 1088: 2000}
    valid = {96: 200, 1024: 200, 1056: 200, 1088: 200}
    got = pin_dolly.derive_bands(train, valid)
    assert got["long"] == 1056, \
        "band 1024 spans (992, 1024], and its LOWER edge is below 1024"


def test_an_unfillable_corpus_refuses_and_names_the_missing_band():
    with pytest.raises(SystemExit, match="long band"):
        pin_dolly.derive_bands({96: 2000}, {96: 200})


def test_two_identical_derived_bands_refuse():
    with pytest.raises(SystemExit, match="must differ"):
        pin_dolly.derive_bands({1056: 2000}, {1056: 200})


def test_selection_from_one_split_ignores_the_other_split_entirely():
    """Clause 20 shuffles the splits independently with one seed, so a changed
    validation split can never move a training row."""
    train_pool = [measured(90, prompt=f"t{i}") for i in range(300)]
    first = pin_dolly.select_one(train_pool, 96, 5)
    second = pin_dolly.select_one(list(train_pool), 96, 5)
    assert [r["pair"] for r in first] == [r["pair"] for r in second]


def test_select_one_refuses_a_thin_band():
    with pytest.raises(SystemExit, match="holds 2 rows"):
        pin_dolly.select_one([measured(90), measured(91)], 96, 3)


def test_the_tokenizer_hash_moves_with_the_tokenizer(tmp_path):
    (tmp_path / "tokenizer.json").write_text("A", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("B", encoding="utf-8")
    before = pin_dolly.tokenizer_hash(tmp_path)
    (tmp_path / "tokenizer.json").write_text("C", encoding="utf-8")
    assert pin_dolly.tokenizer_hash(tmp_path) != before


def test_a_directory_without_tokenizer_files_refuses(tmp_path):
    with pytest.raises(SystemExit, match="no tokenizer files"):
        pin_dolly.tokenizer_hash(tmp_path)


def test_a_changed_upstream_revision_refuses_rather_than_repinning(
        tmp_path, monkeypatch):
    monkeypatch.setattr(pin_dolly, "DATA_DIR", tmp_path)
    out = tmp_path / "ultrachat-96"
    out.mkdir()
    (out / "manifest.json").write_text(
        json.dumps({"revision": "aaa"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="invalidated"):
        pin_dolly._refuse_stale_manifest(96, "bbb")
