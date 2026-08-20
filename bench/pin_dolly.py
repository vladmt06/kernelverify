#!/usr/bin/env python3
"""Freeze a fixed slice of databricks-dolly-15k as the pinned training corpus.

Why a pinned slice rather than the dataset
------------------------------------------
Every measurement this repository binds names the exact bytes it measured. A
dataset fetched at run time is not those bytes: the upstream file can change,
the selection can shuffle, and two runs a week apart become incomparable
without anything looking wrong. So the slice is chosen once, written here, and
hashed into every recording that reads it.

Why a length band, and what it costs
------------------------------------
mlx-lm does not train at a fixed sequence length. Its iterator sorts examples
by length and pads each batch only to one plus the next multiple of 32 above
the longest row in that batch, so the width of a step is a property of the
data rather than of the configuration, and `max_seq_length` is a cap that a
normal instruction corpus never reaches. Drawing the slice from a single
32-token band makes every batch the same width, which is what lets a shape be
registered before the run rather than discovered after it.

The cost is stated rather than hidden. Dolly's rows are short - the corpus
median is 116 tokens - so a band wide enough to fill the slice sits far below
the 2048 the pre-registration names. `--report` writes the whole corpus
distribution beside the slice so the band is a visible choice against measured
evidence, and the recording carries both.

Usage
-----
    python bench/pin_dolly.py --report            # distribution only, no write
    python bench/pin_dolly.py --band 64 --train 1024 --valid 128
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "bench" / ".models" / "qwen3-4b-4bit-g64"
OUT_DIR = ROOT / "bench" / ".data" / "dolly"
SOURCE_URL = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k"
              "/resolve/main/databricks-dolly-15k.jsonl")
CACHE = ROOT / "bench" / ".cache" / "databricks-dolly-15k.jsonl"

# The exact upstream bytes this slice was drawn from, declared before the
# fetch rather than recorded after it. Both the URL and a local cache are
# mutable, and the selection walks the file in its own order, so a reordered
# or edited upstream would silently produce a different slice under the same
# seed. Verified 2026-08-20.
SOURCE_SHA256 = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"

# The iterator's own padding arithmetic, mlx_lm/tuner/trainer.py: a batch is
# padded to 1 + PAD_TO * ceil(longest / PAD_TO). Carried here so the band and
# the width it produces are computed by the same rule the trainer applies.
PAD_TO = 32

# mlx-lm caps a padded batch here, so it is half of the width rule rather
# than a separate setting. `bench/train_lora_e2e.py` registers 2048.
MAX_SEQ_LENGTH = 2048
SEED = 20260820


def fetch(url: str = SOURCE_URL, cache: Path = CACHE,
          expected: str | None = SOURCE_SHA256) -> Path:
    """The upstream file, downloaded once, reused, and checked against a
    digest declared before the fetch.

    Checking after the fact would only record which bytes arrived. The point
    of declaring it first is that a changed upstream, or a cache someone
    edited, stops the run instead of quietly producing a different slice from
    the same seed.
    """
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            cache.write_bytes(response.read())
    if expected is not None:
        digest = hashlib.sha256(cache.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(
                f"{cache} hashes {digest}, not the declared {expected}: the "
                f"upstream corpus or the local cache has changed, and the "
                f"same seed would now select different rows. Re-declare "
                f"SOURCE_SHA256 deliberately, or restore the pinned file.")
    return cache


def as_prompt_completion(row: dict) -> dict:
    """Dolly's own fields in the shape mlx-lm masks prompts for.

    `CompletionsDataset` is the format that carries a prompt boundary, and the
    boundary is what candidate L's whole premise rests on: without it every
    token is supervised and there is no eliminated work to eliminate.
    """
    context = row.get("context") or ""
    instruction = row["instruction"]
    prompt = f"{instruction}\n\n{context}" if context else instruction
    return {"prompt": prompt, "completion": row["response"]}


def batch_width(longest: int, max_seq_length: int = MAX_SEQ_LENGTH) -> int:
    """What the trainer pads a batch of this longest row to.

    Both halves of mlx-lm's rule, not just the first: the batch is padded to
    one plus the next multiple of 32, and then capped at `max_seq_length`. The
    cap is why a corpus whose rows reach the cap produces a width one token
    BELOW the round number, and a floor registered at the round number would
    be measured at a shape the step never produces.
    """
    return min(1 + PAD_TO * ((longest + PAD_TO - 1) // PAD_TO), max_seq_length)


def measure(rows, tokenizer, adapter=None) -> list[dict]:
    """Token length and supervised fraction for every row, once.

    `adapter` maps a raw corpus row to a prompt/completion pair and defaults
    to Dolly's; rows the adapter returns None for are dropped and counted by
    the caller. One measurement rule for every corpus, per the plan's
    one-band-arithmetic requirement.
    """
    adapter = as_prompt_completion if adapter is None else adapter
    measured = []
    for row in rows:
        pair = adapter(row)
        if pair is None:
            continue
        messages = [{"role": "user", "content": pair["prompt"]},
                    {"role": "assistant", "content": pair["completion"]}]
        tokens = tokenizer.apply_chat_template(messages, return_dict=False)
        prompt_tokens = tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True, return_dict=False)
        length = len(tokens)
        measured.append({
            "pair": pair,
            "length": length,
            "band": PAD_TO * ((length + PAD_TO - 1) // PAD_TO),
            "supervised": length - len(prompt_tokens),
        })
    return measured


def distribution(measured) -> dict:
    """The corpus as a whole, which is the evidence a band is chosen against.

    This is also the padding-fraction measurement the profile owes: it reads
    off the dataset, needs no device, and says how much of a step at any
    chosen width would be padding rather than tokens.
    """
    lengths = sorted(row["length"] for row in measured)
    fractions = sorted(row["supervised"] / row["length"] for row in measured)

    def at(values, quantile):
        return values[int(quantile * (len(values) - 1))]

    bands: dict[int, int] = {}
    for row in measured:
        bands[row["band"]] = bands.get(row["band"], 0) + 1
    return {
        "rows": len(measured),
        "length": {"min": lengths[0], "p25": at(lengths, 0.25),
                   "median": at(lengths, 0.5), "p75": at(lengths, 0.75),
                   "p95": at(lengths, 0.95), "max": lengths[-1]},
        "supervised_fraction": {"p25": at(fractions, 0.25),
                                "median": at(fractions, 0.5),
                                "p75": at(fractions, 0.75)},
        "bands": {str(band): count for band, count in sorted(bands.items())},
        "pad_to": PAD_TO,
    }


def select(measured, band: int, train: int, valid: int, seed: int = SEED):
    """A deterministic slice of one band, or a refusal naming the shortfall."""
    pool = [row for row in measured if row["band"] == band]
    wanted = train + valid
    if len(pool) < wanted:
        raise SystemExit(
            f"band {band} holds {len(pool)} rows, and {wanted} were asked "
            f"for; choose a band with more rows or a smaller slice. The "
            f"busiest bands are {_busiest(measured)}")
    random.Random(seed).shuffle(pool)
    return pool[:train], pool[train:wanted]


def _busiest(measured, top: int = 5) -> str:
    bands: dict[int, int] = {}
    for row in measured:
        bands[row["band"]] = bands.get(row["band"], 0) + 1
    ranked = sorted(bands.items(), key=lambda item: -item[1])[:top]
    return ", ".join(f"{band}:{count}" for band, count in ranked)


def write_split(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row["pair"], ensure_ascii=False) + "\n"
                   for row in rows)
    path.write_text(body, encoding="utf-8")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# UltraChat 200k, the corpus of Amendment 5 clause 20. The module name is
# historical: this file is the repository's one corpus pinner, and the band
# arithmetic above is shared by every corpus it pins.
# ---------------------------------------------------------------------------
ULTRACHAT_REPO = "HuggingFaceH4/ultrachat_200k"
ULTRACHAT_SPLITS = {"train": "train_sft", "valid": "test_sft"}
TRAIN_ROWS = 1024
VALID_ROWS = 128
LONG_EDGE = 1024
HF_CACHE = ROOT / "bench" / ".cache" / "hf"
DATA_DIR = ROOT / "bench" / ".data"


def ultrachat_pair(row: dict) -> dict | None:
    """Clause 20's chat adapter: the first user turn becomes the prompt, the
    first assistant turn after it becomes the completion, and the rest of the
    thread is discarded. A row without that shape is dropped, and the caller
    counts the drops so the distribution report can name them.
    """
    messages = row.get("messages") or []
    prompt = None
    for message in messages:
        if prompt is None:
            if message.get("role") == "user" and message.get("content"):
                prompt = message["content"]
        elif message.get("role") == "assistant" and message.get("content"):
            return {"prompt": prompt, "completion": message["content"]}
    return None


def resolve_revision(repo: str = ULTRACHAT_REPO) -> str:
    """The upstream commit hash, resolved once at pin time (clause 20).

    The hash is committed in the manifest beside the slices; a consumer that
    finds a different upstream refuses rather than re-pinning silently.
    """
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(repo).sha
    if not sha:
        raise SystemExit(f"{repo} resolved to no commit hash; nothing to pin")
    return sha


def fetch_ultrachat(revision: str, repo: str = ULTRACHAT_REPO,
                    cache_dir: Path = HF_CACHE) -> dict[str, list[Path]]:
    """Every parquet shard of both registered splits, at the pinned revision."""
    from huggingface_hub import HfApi, hf_hub_download

    names = HfApi().list_repo_files(repo, repo_type="dataset",
                                    revision=revision)
    shards: dict[str, list[Path]] = {}
    for split, upstream in ULTRACHAT_SPLITS.items():
        matching = sorted(n for n in names
                          if n.startswith(f"data/{upstream}-")
                          and n.endswith(".parquet"))
        if not matching:
            raise SystemExit(f"{repo}@{revision} holds no parquet shards for "
                             f"split {upstream}")
        shards[split] = [Path(hf_hub_download(
            repo, name, repo_type="dataset", revision=revision,
            cache_dir=str(cache_dir))) for name in matching]
    return shards


def read_parquet_rows(paths) -> list[dict]:
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for path in paths:
        table = pq.read_table(path, columns=["messages"])
        rows.extend(table.to_pylist())
    return rows


def band_counts(measured) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in measured:
        counts[row["band"]] = counts.get(row["band"], 0) + 1
    return counts


def derive_bands(train_counts: dict[int, int], valid_counts: dict[int, int],
                 *, train: int = TRAIN_ROWS, valid: int = VALID_ROWS,
                 long_edge: int = LONG_EDGE) -> dict[str, int]:
    """Clause 20's two bands, DERIVED rather than chosen.

    The short band is the shortest 32-token band holding `train` training
    rows AND `valid` validation rows; the long band is the shortest such band
    whose LOWER edge, the upper edge minus 32, is at or above `long_edge`.
    Registering the rule registers the answer, so nothing here is a choice.
    """
    def fills(band: int) -> bool:
        return (train_counts.get(band, 0) >= train
                and valid_counts.get(band, 0) >= valid)

    candidates = sorted(set(train_counts) | set(valid_counts))
    short = next((band for band in candidates if fills(band)), None)
    long = next((band for band in candidates
                 if band - PAD_TO >= long_edge and fills(band)), None)
    if short is None or long is None:
        which = [name for name, band in (("short", short), ("long", long))
                 if band is None]
        raise SystemExit(
            f"no {' or '.join(which)} band can fill {train} training and "
            f"{valid} validation rows; the profile does not run on a relaxed "
            f"rule. Busiest training bands: "
            f"{_busiest(_counts_as_measured(train_counts))}")
    if short == long:
        raise SystemExit(
            f"both derived bands are {short}: the two registered widths must "
            f"differ, and this corpus cannot supply two")
    return {"short": short, "long": long}


def _counts_as_measured(counts: dict[int, int]) -> list[dict]:
    return [{"band": band} for band, n in counts.items() for _ in range(n)]


def select_one(measured, band: int, count: int, seed: int = SEED):
    """A deterministic slice of one band from ONE split.

    Clause 20 shuffles the two splits independently, each with the same seed,
    because they are separate upstream populations and a pooled shuffle would
    not be reproducible from either alone.
    """
    pool = [row for row in measured if row["band"] == band]
    if len(pool) < count:
        raise SystemExit(f"band {band} holds {len(pool)} rows of this split, "
                         f"and {count} were asked for")
    random.Random(seed).shuffle(pool)
    return pool[:count]


def tokenizer_hash(model_dir: Path = MODEL_DIR) -> str:
    """One digest over the tokenizer files, committed with the band edges,
    because the bands are token counts and a different tokenizer makes them
    different numbers."""
    digest = hashlib.sha256()
    names = sorted(name for name in
                   ("tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json")
                   if (model_dir / name).exists())
    if not names:
        raise SystemExit(f"{model_dir} holds no tokenizer files to hash")
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update((model_dir / name).read_bytes())
    return digest.hexdigest()


def _supervised_median(rows) -> float:
    fractions = sorted(r["supervised"] / r["length"] for r in rows)
    return fractions[len(fractions) // 2]


def pin_ultrachat_band(name: str, band: int, measured_train, measured_valid,
                       revision: str, tok_hash: str, dropped: dict,
                       out_root: Path = DATA_DIR) -> dict:
    train = select_one(measured_train, band, TRAIN_ROWS)
    valid = select_one(measured_valid, band, VALID_ROWS)
    out = out_root / f"ultrachat-{band}"
    manifest = {
        "dataset": ULTRACHAT_REPO,
        "config": "default",
        "splits": ULTRACHAT_SPLITS,
        "revision": revision,
        "chat_adapter": ("the first user turn is the prompt, the first "
                         "assistant turn after it is the completion, and the "
                         "rest of the thread is discarded"),
        "seed": SEED,
        "cell": name,
        "band": band,
        "band_lower_edge": band - PAD_TO,
        "batch_width": batch_width(band),
        "tokenizer_sha256": tok_hash,
        "dropped_rows": dropped,
        "supervised_fraction": {
            "train_median": _supervised_median(train),
            "valid_median": _supervised_median(valid),
        },
        "train": {"rows": len(train),
                  "sha256": write_split(out / "train.jsonl", train)},
        "valid": {"rows": len(valid),
                  "sha256": write_split(out / "valid.jsonl", valid)},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _measured_cached(split: str, revision: str, tok_hash: str,
                     shards, tokenizer) -> tuple[list[dict], int]:
    """Measure one UltraChat split once and cache the result, because the two
    splits hold 231k rows and the tokenizer is the slow part. The cache is
    keyed by revision and tokenizer hash, so a changed upstream or tokenizer
    measures fresh rather than serving stale numbers.
    """
    cache = (ROOT / "bench" / ".cache" /
             f"ultrachat-{split}-{revision[:12]}-{tok_hash[:12]}.jsonl")
    if cache.exists():
        lines = cache.read_text(encoding="utf-8").splitlines()
        dropped = json.loads(lines[0])["dropped"]
        return [json.loads(line) for line in lines[1:]], dropped
    rows = read_parquet_rows(shards)
    measured = measure(rows, tokenizer, adapter=ultrachat_pair)
    dropped = len(rows) - len(measured)
    cache.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"dropped": dropped}) + "\n" + "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in measured)
    cache.write_text(body, encoding="utf-8")
    return measured, dropped


def _refuse_stale_manifest(band: int, revision: str) -> None:
    existing = DATA_DIR / f"ultrachat-{band}" / "manifest.json"
    if existing.exists():
        pinned = json.loads(existing.read_text(encoding="utf-8"))["revision"]
        if pinned != revision:
            raise SystemExit(
                f"{existing} pins revision {pinned} and upstream now "
                f"resolves {revision}: the slice is invalidated, not "
                f"re-pinned silently. Delete the slice deliberately to "
                f"re-pin.")


def pin_ultrachat(report_only: bool = False) -> int:
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    tokenizer = load_tokenizer(MODEL_DIR)
    tok_hash = tokenizer_hash()
    revision = resolve_revision()
    shards = fetch_ultrachat(revision)
    measured, dropped = {}, {}
    for split in ULTRACHAT_SPLITS:
        measured[split], dropped[split] = _measured_cached(
            split, revision, tok_hash, shards[split], tokenizer)

    bands = derive_bands(band_counts(measured["train"]),
                         band_counts(measured["valid"]))
    report = {
        "dataset": ULTRACHAT_REPO,
        "revision": revision,
        "tokenizer_sha256": tok_hash,
        "derived_bands": bands,
        "dropped_rows": dropped,
        "train": distribution(measured["train"]),
        "valid": distribution(measured["valid"]),
    }
    if report_only:
        print(json.dumps(report, indent=2))
        return 0

    for name, band in bands.items():
        _refuse_stale_manifest(band, revision)
    manifests = [pin_ultrachat_band(name, band, measured["train"],
                                    measured["valid"], revision, tok_hash,
                                    dropped)
                 for name, band in bands.items()]
    (DATA_DIR / "ultrachat-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifests, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", choices=("dolly", "ultrachat"),
                        default="dolly",
                        help="ultrachat DERIVES both registered bands per "
                             "Amendment 5 clause 20 and pins both; dolly "
                             "keeps the original --band interface")
    parser.add_argument("--band", type=int, default=None,
                        help="upper edge of the 32-token band to draw from "
                             "(dolly only; ultrachat derives its bands)")
    parser.add_argument("--train", type=int, default=1024)
    parser.add_argument("--valid", type=int, default=128)
    parser.add_argument("--report", action="store_true",
                        help="print the corpus distribution and write nothing")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    if args.corpus == "ultrachat":
        if args.band is not None:
            raise SystemExit("ultrachat's bands are derived by clause 20's "
                             "rule, not chosen; --band applies to dolly only")
        return pin_ultrachat(report_only=args.report)

    from mlx_lm.tokenizer_utils import load as load_tokenizer

    tokenizer = load_tokenizer(MODEL_DIR)
    rows = [json.loads(line) for line in
            fetch().read_text(encoding="utf-8").splitlines() if line.strip()]
    measured = measure(rows, tokenizer)
    stats = distribution(measured)

    if args.report or args.band is None:
        print(json.dumps(stats, indent=2))
        if args.band is None and not args.report:
            print("\nno --band given, so nothing was written; the busiest "
                  f"bands are {_busiest(measured)}", file=sys.stderr)
            return 2
        return 0

    train, valid = select(measured, args.band, args.train, args.valid)
    manifest = {
        "source": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "seed": SEED,
        "band": args.band,
        "batch_width": batch_width(args.band),
        "supervised_tokens_per_row": {
            "train_median": sorted(r["supervised"] for r in train)[len(train) // 2],
        },
        "train": {"rows": len(train),
                  "sha256": write_split(args.out / "train.jsonl", train)},
        "valid": {"rows": len(valid),
                  "sha256": write_split(args.out / "valid.jsonl", valid)},
        "corpus": stats,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "corpus"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
