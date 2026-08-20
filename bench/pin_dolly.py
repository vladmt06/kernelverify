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

# The iterator's own padding arithmetic, mlx_lm/tuner/trainer.py: a batch is
# padded to 1 + PAD_TO * ceil(longest / PAD_TO). Carried here so the band and
# the width it produces are computed by the same rule the trainer applies.
PAD_TO = 32
SEED = 20260820


def fetch(url: str = SOURCE_URL, cache: Path = CACHE) -> Path:
    """The upstream file, downloaded once and reused."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            cache.write_bytes(response.read())
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


def batch_width(longest: int) -> int:
    """What the trainer pads a batch of this longest row to."""
    return 1 + PAD_TO * ((longest + PAD_TO - 1) // PAD_TO)


def measure(rows, tokenizer) -> list[dict]:
    """Token length and supervised fraction for every row, once."""
    measured = []
    for row in rows:
        pair = as_prompt_completion(row)
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--band", type=int, default=None,
                        help="upper edge of the 32-token band to draw from")
    parser.add_argument("--train", type=int, default=1024)
    parser.add_argument("--valid", type=int, default=128)
    parser.add_argument("--report", action="store_true",
                        help="print the corpus distribution and write nothing")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

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
        "source_sha256": hashlib.sha256(fetch().read_bytes()).hexdigest(),
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
