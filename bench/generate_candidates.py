"""Run a bounded LLM generation session against the real funnel.

Thin wiring over library code: the round-trip itself lives in
kernelverify/compiler/llm.py where the fake-model tests pin it, and this
script only chooses the backend, the operation and the bounds, then prints
what the session's own records say happened.

The funnel here is compile -> lint -> unwritten and nothing else. Pricing is
not disabled by a flag; it is absent from the stage list, and the assertion
below makes a timing stage unwireable by accident, which is rule V1 enforced
a third time at the one place a hurried edit would add it.

The model id has no default. A defaulted model id is a plausible wrong
number: the journal would pin an id nobody chose.

Run from the repository root; the codex backend refuses untrusted
directories, and a session started elsewhere surfaces that as counted
`exit` refusals rather than working by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from kernelverify.compiler.brief import Operation  # noqa: E402
from kernelverify.compiler.funnel import Funnel  # noqa: E402
from kernelverify.compiler.llm import (  # noqa: E402
    KERNEL_SCHEMA,
    claude_model,
    codex_model,
    generate_session,
)
from kernelverify.compiler.stages import (  # noqa: E402
    compile_stage,
    lint_stage,
    unwritten_stage,
)
from kernelverify.compiler.store import CandidateStore  # noqa: E402
from kernelverify.runners import MetalRunner  # noqa: E402
from kernelverify.runners.spec import (  # noqa: E402
    Binding,
    BindingKind,
    LaunchSpec,
    RunCase,
)

# The demo operation: a scaled elementwise map, the smallest thing with a
# real correctness surface. The sprint's chosen operation (L, A or Q) lands
# here once the Day 1 profile rule picks it; until then the loop is exercised
# on something whose gates are already trusted.
N = 4096


def demo_operation() -> tuple[Operation, dict]:
    operation = Operation(
        name="scale2",
        goal=("Multiply every element of a float32 vector by 2. The kernel "
              "must write every element of the output exactly once."),
        signature=("[[kernel]] void scale2(device const float* x [[buffer(0)]], "
                   "device float* out [[buffer(1)]], "
                   "constant uint& n [[buffer(2)]], "
                   "uint gid [[thread_position_in_grid]]); "
                   "launched with one thread per element, n threads total"),
        constraints=(
            "the entry point must be named scale2, with exactly the three "
            "buffers above in that order",
            f"n is {N} in the screening case and any positive value in general",
            "guard against gid >= n",
        ),
    )
    case = RunCase(inputs={"x": np.arange(1, N + 1, dtype=np.float32)},
                   params={"n": N}, output_shapes=[((N,), "float32")],
                   label="scale2 screen")
    context = {
        "entry_point": "scale2",
        "bindings": (Binding(BindingKind.INPUT, "x"), Binding(BindingKind.OUTPUT),
                     Binding(BindingKind.SCALAR, "n", "uint32")),
        "launch": LaunchSpec(grid=("n", 1, 1), threadgroup=(64, 1, 1)),
        "probe_case": case,
        "gate_cases": [case],
    }
    return operation, context


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path,
                        help="candidate store directory (created if absent)")
    parser.add_argument("--backend", choices=("claude", "codex"),
                        default="claude")
    parser.add_argument("--model", required=True,
                        help="exact model id; no default on purpose")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--per-round", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on NEW candidates per round")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="seconds per model call")
    args = parser.parse_args(argv)

    store = CandidateStore(args.store)
    schema_path = args.store / "schema.json"
    schema_path.write_text(json.dumps(KERNEL_SCHEMA, indent=1) + "\n")

    if args.backend == "claude":
        model = claude_model(args.model, timeout=args.timeout)
    else:
        model = codex_model(args.model, schema_path, timeout=args.timeout)

    funnel = Funnel([compile_stage(), lint_stage(), unwritten_stage()])
    assert all(stage.verifies for stage in funnel.stages), (
        "this session must not contain a timing stage (rule V1)")

    operation, context = demo_operation()
    context["runner"] = MetalRunner()

    print(f"model {model.model_id} via {args.backend}; store {args.store}")
    rounds = generate_session(operation, model, funnel, store,
                              context=context, rounds=args.rounds,
                              per_round=args.per_round, limit=args.limit)

    for number, entry in enumerate(rounds, 1):
        print(f"\nround {number}")
        print(entry.summary())
    print("\nfinal census")
    for outcome, count in sorted(store.census().items()):
        print(f"  {outcome}: {count}")
    survivors = [c for r in rounds if r.session for c in r.session.survived]
    print(f"\nsurvivors: {len(survivors)}")
    for candidate in survivors:
        print(f"  {candidate[:12]} ({store.get(candidate).origin})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
