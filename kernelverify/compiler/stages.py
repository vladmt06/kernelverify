"""The funnel's real stages, wired to the machinery that already exists.

`funnel.py` decides what may run after what and knows nothing about kernels.
This is where its stages actually do something: the compile stage drives the
real `MetalRunner`, the lint stage drives the real C1 attestation, and the
held-out stage drives the sealed draw. Each is a thin adapter on purpose,
because the judgement lives in the module being adapted and duplicating any
of it here would give the loop two places to disagree with itself.

One decision is made here and nowhere else: which runner statuses are a
verdict about the candidate and which are a problem with the harness. A
candidate that fails to compile, fails to launch, hangs, or takes the worker
process down with it has told us something about itself, and that is a
refusal. A spec the case does not match, or a machine with no GPU, has told
us something about our own setup, and recording that as a bad kernel would
put our bugs in the candidate census. Those raise instead, which the funnel
records as errored rather than failed.
"""

from __future__ import annotations

from kernelverify.compiler.funnel import Stage, StageOutcome
from kernelverify.compiler.heldout import Verdict, draw
from kernelverify.compiler.lint import lint
from kernelverify.compiler.support import screen as support_screen
from kernelverify.compiler.unwritten import screen
from kernelverify.runners.result import RunStatus
from kernelverify.runners.spec import KernelSpec

# The candidate did something wrong: a verdict about it.
CANDIDATE_FAULTS = {
    RunStatus.COMPILE_ERROR: "did not compile",
    RunStatus.LAUNCH_ERROR: "could not be launched as written",
    RunStatus.TIMEOUT: "did not finish",
    RunStatus.CRASH: "took the worker process down",
}

# We did something wrong: not a verdict about anything.
HARNESS_FAULTS = {
    RunStatus.INVALID_SPEC: "the probe case does not match the spec",
    RunStatus.UNSUPPORTED: "no Metal device on this machine",
}


class HarnessError(RuntimeError):
    """The stage could not reach a verdict, for reasons that are ours."""


def compile_stage(name: str = "compile") -> Stage:
    """Does this source compile and launch at all?

    The cheapest thing that touches the GPU, and the one that kills most of
    what a generator writes. It runs one probe case rather than compiling
    alone, because a kernel that compiles and then fails to launch is just as
    dead and finding that out costs nothing extra.
    """
    def run(source, *, runner, entry_point, bindings, launch, probe_case,
            **_context) -> StageOutcome:
        spec = KernelSpec(source=source, entry_point=entry_point,
                          bindings=bindings, launch=launch)
        result = runner.run_one(spec, probe_case)
        if result.status in HARNESS_FAULTS:
            raise HarnessError(
                f"{HARNESS_FAULTS[result.status]}: {result.detail}")
        if result.status in CANDIDATE_FAULTS:
            return StageOutcome(False, f"{CANDIDATE_FAULTS[result.status]}: "
                                       f"{result.detail}".strip())
        return StageOutcome(True)

    return Stage(name=name, run=run, verifies=True)


def lint_stage(name: str = "lint") -> Stage:
    """Is every intermediate binary32 or wider (contract clause C1)?

    Text only, so it costs nothing and could run first; it runs after compile
    because a source that does not compile is not worth reading.
    """
    def run(source, *, types=None, **_context) -> StageOutcome:
        report = lint(source, types=types)
        return StageOutcome(report.ok, report.reason)

    return Stage(name=name, run=run, verifies=True)


def heldout_stage(space: dict[str, list], check, name: str = "heldout") -> Stage:
    """Does the frozen incumbent survive cases it never saw?

    The detail is empty on failure, always, and that is the whole point. The
    funnel writes a stage's detail into the store, the store is what a brief
    is built from, and a brief that named the failing cell would hand the
    generator the held-out set one candidate at a time. What survives is PASS
    or FAIL, which is what the pre-registration sealed.
    """
    def run(source, *, candidate, **_context) -> StageOutcome:
        verdict = Verdict(bool(check(source, draw(candidate, space))))
        return StageOutcome(verdict.passed, "")

    return Stage(name=name, run=run, verifies=True)


def unwritten_stage(name: str = "unwritten") -> Stage:
    """Was every declared output cell reached by a store?

    Like the compile stage, this builds its spec from the source the funnel
    handed it, never from a spec passed through the context. The two would
    almost always agree, and on the run where they did not the gate would be
    screening text the store never recorded, which is the one thing a
    candidate journal must never allow.
    """
    def run(source, *, runner, entry_point, bindings, launch, gate_cases,
            **_context) -> StageOutcome:
        spec = KernelSpec(source=source, entry_point=entry_point,
                          bindings=bindings, launch=launch)
        report = screen(runner, spec, gate_cases)
        return StageOutcome(report.ok, report.reason)

    return Stage(name=name, run=run, verifies=True)


def support_stage(name: str = "support") -> Stage:
    """Does the output depend on the weight bytes it is specified to depend on?

    Only meaningful for candidates of the quantized-matmul shape, so the
    context supplies the raw weights and the bit width; operations without
    weights simply do not wire this stage. Like every gate stage, the spec is
    built from the source the funnel handed over, so the text screened is the
    text the journal recorded.
    """
    def run(source, *, runner, entry_point, bindings, launch, support_weights,
            support_bits, **_context) -> StageOutcome:
        def spec_for(_d_out, _d_in):
            return KernelSpec(source=source, entry_point=entry_point,
                              bindings=bindings, launch=launch)

        report = support_screen(runner, spec_for, support_weights, support_bits)
        return StageOutcome(report.ok, report.reason)

    return Stage(name=name, run=run, verifies=True)
