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

What it does NOT attest, and says so in its own text: per-step numerical
containment, which would need shadow computation the run did not do; and
the loss curve, which lives in mlx-lm's own reporting callbacks and is not
reachable from outside its trainer without instrumenting it.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

NOT_ATTESTED = (
    "per-step numerical containment: no shadow computation was run",
    "the loss curve: mlx-lm's trainer owns its reporting and this run did "
    "not instrument it",
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
          peak_bytes: int | None, started: str, finished: str) -> dict:
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
