"""`metalrunner.lora`: mlx-lm's own trainer, with the stack checked first.

Run it exactly as you would run `mlx_lm.lora`. Every flag, every config
file, every default is mlx-lm's, because they are literally mlx-lm's: this
module builds mlx-lm's parser, merges the configuration mlx-lm's way, and
hands the result to mlx-lm's own `run`.

Why a separate entry point rather than a flag on the original. mlx-lm
0.31.3's `main()` parses with a closed parser and calls `run()` directly,
with no hook a foreign flag could reach; that was reproduced before this was
designed. Upstreaming a backend hook is worth proposing and it does not gate
this, so the user types a different module name and nothing else changes.

The order below is the whole safety argument, and it is deliberate:

1. Verify the stack. A version or seam that is not the verified one refuses
   the run outright, before anything is loaded, patched or written. Falling
   back to stock silently would be friendlier and would let someone believe
   they ran verified kernels when they ran none.
2. Parse the user's arguments, mlx-lm's way.
3. Refuse the modes this package does not cover, in one line each.
4. Decide routing and PRINT it, before training, so what is about to happen
   to the step is known in advance rather than reported afterwards.
5. Run mlx-lm's trainer.
6. Write the receipt.

Today step 4 always decides to route nothing, because no training kernel has
been kept yet. The run is then stock mlx-lm with a stack check and a
receipt, the report says exactly that, and no speed is claimed anywhere.
"""

from __future__ import annotations

import os
import sys
import types

from metalrunner import receipt, routing
from metalrunner.versions import UnverifiedStack, require_verified_stack

EXIT_UNVERIFIED_STACK = 3
EXIT_UNSUPPORTED_MODE = 4

UNSUPPORTED_MODES = {
    "dora": "metalrunner covers LoRA only; run mlx_lm.lora for DoRA.",
    "full": "metalrunner covers LoRA only; run mlx_lm.lora for full "
            "fine-tuning.",
}


def parse(argv=None) -> types.SimpleNamespace:
    """mlx-lm's own parser and its own configuration merge.

    This mirrors `mlx_lm.lora.main()`'s argument handling rather than
    reimplementing it: the parser, the defaults and the precedence rule
    (command line beats config file) all come from mlx-lm. It is copied
    because that function offers no seam, and it is safe to copy only
    because the stack check pins the version it was copied from.
    """
    import yaml
    from mlx_lm.lora import CONFIG_DEFAULTS, build_parser, yaml_loader

    parsed = build_parser().parse_args(argv)
    config = parsed.config
    args = vars(parsed)
    if config:
        print("Loading configuration file", config)
        with open(config, "r") as handle:
            loaded = yaml.load(handle, yaml_loader)
        for key, value in loaded.items():
            if args.get(key, None) is None:
                args[key] = value
    for key, value in CONFIG_DEFAULTS.items():
        if args.get(key, None) is None:
            args[key] = value
    return types.SimpleNamespace(**args)


def unsupported_mode(args) -> str | None:
    return UNSUPPORTED_MODES.get(getattr(args, "fine_tune_type", "lora"))


def main(argv=None) -> int:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    try:
        stack = require_verified_stack()
    except UnverifiedStack as refusal:
        print(refusal, file=sys.stderr)
        return EXIT_UNVERIFIED_STACK

    args = parse(argv)

    refusal = unsupported_mode(args)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return EXIT_UNSUPPORTED_MODE

    bits, group_size = receipt.read_quantization(args.model)
    chip = routing.chip()
    decisions = routing.decide(args.fine_tune_type, bits, group_size,
                               on_chip=chip)
    print(routing.render(decisions, stack, model=args.model,
                         fine_tune_type=args.fine_tune_type,
                         bits=bits, group_size=group_size))

    started = receipt.utc_now()
    from mlx_lm.lora import run

    run(args)

    print(_finish(args, stack, decisions, chip, started))
    return 0


def _finish(args, stack, decisions, chip, started) -> str:
    import mlx.core as mx

    try:
        peak = int(mx.get_peak_memory())
    except Exception:
        peak = None
    record = receipt.build(args=args, stack=stack, decisions=decisions,
                           chip=chip, adapter_path=getattr(args, "adapter_path",
                                                           None),
                           peak_bytes=peak, started=started,
                           finished=receipt.utc_now(),
                           forced_to_stock=routing.forced_to_stock())
    written = receipt.write(record, getattr(args, "adapter_path", None))
    if written is None:
        return "metalrunner: no adapter path, so no receipt was written."
    return f"metalrunner receipt: {written}"


if __name__ == "__main__":
    raise SystemExit(main())
