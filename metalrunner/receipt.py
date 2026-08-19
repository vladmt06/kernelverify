"""What a training run can prove about itself afterwards.

mlx-lm saves adapter weights and nothing else, so a fine-tuned adapter
arrives with no record of what produced it: which base model, which data,
which seeds, which stack, whether anything unusual was swapped into the
step. Six months later that adapter is an artefact nobody can account for.

The receipt is that account. It records what was observed, and it is
explicit about what it does not attest, because a receipt that overstates
is worse than none: it converts an absent check into an apparent one.

What it attests: the stack it verified, the chip, the routing decisions and
their reasons, the arguments the run was given, fingerprints of the base
model config and the adapter that came out, and the peak memory the process
reached.

It also carries what the trainer reported about its own progress, which
reaches this package through mlx-lm's own callback rather than through its
printed output, and a count of how many times each replaced name was
actually called.

What it does NOT attest, and says so in its own text: per-step numerical
containment, which would need shadow computation the run did not do; the
loss values, which are recorded exactly as the trainer reported them and
are not recomputed here; and anything at all about speed.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

NOT_ATTESTED = (
    "per-step numerical containment: no shadow computation was run",
    "the loss values: they are mlx-lm's own reported numbers, recorded "
    "verbatim, and nothing here recomputed or checked them",
    "throughput or speed: this receipt records what ran, never how fast",
)


def _digest_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def fingerprint_model(model: str) -> dict:
    """What identifies the base model, when it is a local path.

    A hub id is recorded as given, because the bytes behind it are not this
    package's to vouch for; a local directory gets its config hashed, which
    is what pins the architecture and the quantization.
    """
    path = Path(model)
    if not path.is_dir():
        return {"model": model, "kind": "hub-id", "config_sha256": None}
    return {"model": str(path), "kind": "local",
            "config_sha256": _digest_file(path / "config.json")}


def read_quantization(model: str) -> tuple[int | None, int | None]:
    """Bits and group size from the model's own config, or (None, None).

    None means unreadable, and unreadable must stay unreadable: guessing a
    quantization is how a kernel verified for one format gets routed onto
    another.
    """
    path = Path(model) / "config.json"
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None
    quant = config.get("quantization")
    if not isinstance(quant, dict):
        return None, None
    bits, group = quant.get("bits"), quant.get("group_size")
    return (bits if isinstance(bits, int) else None,
            group if isinstance(group, int) else None)


def build(*, args, stack, decisions, chip: str, adapter_path: str | None,
          peak_bytes: int | None, started: str, finished: str,
          forced_to_stock: bool = False, progress: dict | None = None,
          seams: dict | None = None, measurement: dict | None = None) -> dict:
    """The record itself, as a plain dictionary."""
    adapters = Path(adapter_path) if adapter_path else None
    adapter_file = adapters / "adapters.safetensors" if adapters else None
    return {
        "metalrunner_receipt": 1,
        "_meaning": "what this run did, recorded by metalrunner. Read "
                    "not_attested before quoting anything from it.",
        "not_attested": list(NOT_ATTESTED),
        "started_utc": started,
        "finished_utc": finished,
        "machine": {"chip": chip, "platform": platform.platform(),
                    "python": platform.python_version()},
        "stack": {"mlx": stack.mlx, "mlx_lm": stack.mlx_lm,
                  "seams_verified": stack.seams},
        "base_model": fingerprint_model(getattr(args, "model", "")),
        "training": {
            "fine_tune_type": getattr(args, "fine_tune_type", None),
            "iters": getattr(args, "iters", None),
            "batch_size": getattr(args, "batch_size", None),
            "max_seq_length": getattr(args, "max_seq_length", None),
            "num_layers": getattr(args, "num_layers", None),
            "learning_rate": getattr(args, "learning_rate", None),
            "seed": getattr(args, "seed", None),
            "data": getattr(args, "data", None),
        },
        "routing": [{"operation": d.operation, "routed": d.routed,
                     "reason": d.reason} for d in decisions],
        # A control arm's receipt must be unmistakable afterwards: a forced
        # run measures the wrapper, not a kernel, and quoting it as a real
        # run would be quoting the wrong thing.
        "forced_to_stock": forced_to_stock,
        # What the trainer said about itself, and how often each replaced
        # name was reached. A routing report claiming an operation was
        # replaced and a run in which it never happened are the same
        # document without this count.
        "progress": progress,
        "seams": seams,
        # What routing actually did, from the same installer the end-to-end
        # measurement reads. A run that routed nothing says so here in
        # numbers, which is a different claim from the report's prose.
        "measurement": measurement,
        "adapter": {
            "path": str(adapters) if adapters else None,
            "sha256": _digest_file(adapter_file) if adapter_file else None,
        },
        "peak_memory_bytes": peak_bytes,
    }


def write(record: dict, adapter_path: str | None) -> Path | None:
    """Beside the adapter, because that is the artefact it accounts for.

    A run with nowhere to put it says so and returns None rather than
    inventing a location.
    """
    if not adapter_path:
        return None
    directory = Path(adapter_path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "metalrunner-receipt.json"
        path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        return path
    except OSError:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
