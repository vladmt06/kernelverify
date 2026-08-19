"""Does the output actually depend on the bytes it is specified to depend on?

The blind spot this gate exists for was measured, not argued: for a quantized
matmul the packed integer codes carry essentially all of the weight data, no
out-of-range value exists in an integer carrier for a NaN dye to use, and a
real support fault on the codes was shown to pass the shipped tolerance gate
on this repository's own near-zero input mode. A kernel can ignore a quarter
of its weights and leave every other gate green.

The mechanism: perturb exactly one weight element, run the kernel on both
versions, and require the output cells that PROVABLY must differ to differ in
their bits. "Provably" is an inequality, not a threshold: a cell is a witness
only when the fp64 reference moves by more than the sum of both runs' error
envelopes plus two output ulps, because then the two admissible-error
intervals are disjoint and no correct kernel can return the same bits. The
comparison is bit equality on witnesses only, so no tolerance enters.

Judgement is one-directional, deliberately. A witness that failed to change
is a refusal; a non-witness that changed is nothing, because an online
rescale or a blocked formulation legitimately perturbs cells whose exact
value the contract does not pin. That one-directionality is what keeps
rounding coupling and formulation freedom (contract C2, C3) from ever
producing a false positive.

Two independent adversarial reviews broke the first design, and the build
below is the revised one:

- The gate derives BOTH device input sets itself, from the raw weights,
  through the same contract-anchored quantizer and packer (both verified
  bit-identical to MLX by existing probes). The first design accepted
  builder-supplied device buffers and checked a byte range, and the reviewer
  measured that a builder whose packing disagreed with the oracle's by one
  code position refuses a CORRECT kernel with every check green. There is no
  parameter through which pre-packed device bytes can arrive.
- The activation is nonzero on every code whose bits share a word with the
  perturbed code, not only on the perturbed code. A sub-byte code in a
  32-bit word is otherwise mostly invisible to a sparse case by
  construction; the measured margin cost of the denser case is a factor of
  seven, leaving the witness inequality clear by more than an order of
  magnitude at the shipped shapes.
- Probe positions are derived from the BINDING's address arithmetic, not
  from the operator's index space. The reviewer planted two one-line support
  faults in the shipped kernel (a dead guard on the two-word read, 28% of
  codes never read; a wrong lane stride, 3%) and both passed a design that
  probed at "structural" operator indices, because a multiple of 8 or 64 is
  never a word-straddling code. The target set below covers every alignment
  residue a code can have inside a word, completely, and a band of block
  indices derived from the launch's lane stride; the POLICY states which of
  those two is a guarantee and which is a sample.
- The scales and biases bindings are NOT screened, and the policy says so.
  The oracle derives them from the raw weights, so no independent
  perturbation of them is expressible without also moving the codes; the one
  clean edit (a power-of-two group rescale) moves two bindings at once and
  is future work recorded in the sprint record, not a silent gap.

Where the gate would need a rule about what a kernel may do, it abstains,
per amendment 1 of the sprint pre-registration. Every abstention is counted
and named, and `screened` excludes them, so a session can never present what
the gate declined to judge as something it checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kernelverify.pack.wide_qmv import pack_codes
from kernelverify.runners.result import RunStatus
from kernelverify.runners.spec import RunCase
from kernelverify.schemas.quant_contract import (
    QuantContract,
    canonical_quantize,
    r_contract,
)

POLICY = (
    "input support for the packed weight codes of a quantized matmul: one "
    "code is re-valued through the contract quantizer, both input sets are "
    "derived by the gate itself through the same anchored quantize-and-pack "
    "path, and every output cell whose fp64 reference moves by more than both "
    "error envelopes plus two output ulps must change its bits. Alignment "
    "residues inside a word are covered COMPLETELY, so any fault keyed to "
    "where a code sits in a word is caught; block positions are covered for "
    "one lane-stride band and SAMPLED beyond it. The x binding is exercised "
    "by the same witnesses; the scales and biases bindings are NOT screened "
    "here, because the oracle derives them from the raw weights and no "
    "single-binding perturbation of them exists. A wrong VALUE computed from "
    "the right bytes is the tolerance gate's to catch, not this gate's."
)

# Verdicts. SUPPORTED/UNSUPPORTED are judgements; everything else abstains.
SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
NO_WITNESS = "no-witness"
NONDETERMINISTIC = "nondeterministic"
UNCLEAN_EDIT = "unclean-edit"
NO_RUN = "no-run"

_JUDGED = (SUPPORTED, UNSUPPORTED)

# fp32 unit roundoff: the working-precision floor contract clause C1 imposes.
_U32 = 2.0 ** -24


@dataclass(frozen=True)
class Target:
    """One probe: this code of this weight row must reach the output."""

    row: int
    code: int
    label: str


@dataclass(frozen=True)
class TargetVerdict:
    target: Target
    verdict: str
    detail: str = ""
    margin: float = 0.0  # witness strength: reference move over the envelope

    @property
    def screened(self) -> bool:
        return self.verdict in _JUDGED


@dataclass(frozen=True)
class Report:
    policy: str
    verdicts: tuple[TargetVerdict, ...]

    @property
    def ok(self) -> bool:
        return not any(v.verdict == UNSUPPORTED for v in self.verdicts)

    @property
    def screened(self) -> int:
        return sum(1 for v in self.verdicts if v.screened)

    @property
    def reason(self) -> str:
        refused = [v for v in self.verdicts if v.verdict == UNSUPPORTED]
        if refused:
            return "; ".join(f"{v.target.label}: {v.detail}" for v in refused)
        skipped = [v for v in self.verdicts if not v.screened]
        if skipped:
            return "not screened: " + "; ".join(
                f"{v.target.label} ({v.verdict})" for v in skipped)
        return ""


# ---------------------------------------------------------------------------
# Target derivation: the binding's address arithmetic, not the operator's
# ---------------------------------------------------------------------------
def alignment_cycle(bits: int) -> int:
    """How many codes until the bit-alignment pattern inside 32-bit words
    repeats: 32/gcd(bits, 32). Every distinct way a code can sit in a word,
    including straddling two, is one residue of this cycle."""
    return 32 // np.gcd(bits, 32)


def derive_targets(d_out: int, d_in: int, bits: int, *,
                   lane_stride: int = 32, block_codes: int = 8) -> list[Target]:
    """Every alignment residue once (complete), plus a lane-stride band of
    block indices (a wrong stride s in (lane_stride, 2*lane_stride] first
    skips a block in this band; larger errors are sampled by the spread).

    Rows walk the output so a row-mapping fault cannot hide at row 0.
    """
    cycle = alignment_cycle(bits)
    n_cycles = max(d_in // cycle, 1)
    targets = []
    for residue in range(cycle):
        # spread across the row: residue i sits in a different cycle each time
        at_cycle = (residue * max(n_cycles - 1, 1)) // max(cycle - 1, 1)
        code = min(residue + cycle * at_cycle, d_in - 1)
        targets.append(Target(row=residue % d_out, code=code,
                              label=f"align r{residue} k{code}"))
    blocks = d_in // block_codes
    for offset in range(lane_stride):
        block = lane_stride + offset
        if block >= blocks:
            break
        code = block * block_codes + (offset % block_codes)
        targets.append(Target(row=(cycle + offset) % d_out, code=code,
                              label=f"block b{block} k{code}"))
    return targets


# ---------------------------------------------------------------------------
# The envelope: the worst error any admissible implementation can commit
# ---------------------------------------------------------------------------
def envelope_qmm(x: np.ndarray, artefact) -> np.ndarray:
    """gamma(d_in + 3) * sum_k |x_k| * (|s|q + |b|), per output cell, fp64.

    The standard fp32 dot-product rounding bound over the factored
    formulation's term magnitudes, which dominate the dequantized ones since
    |s*q + b| <= |s|q + |b|; the +3 covers the dequantize chain. Everything
    in it is read off the case: no fitted constant.
    """
    g = artefact.contract.group_size
    rows, cols = artefact.q.shape
    q = artefact.q.reshape(rows, cols // g, g).astype(np.float64)
    magnitude = (np.abs(artefact.scales.astype(np.float64))[:, :, None] * q
                 + np.abs(artefact.biases.astype(np.float64))[:, :, None])
    terms = np.abs(x.astype(np.float64)) @ magnitude.reshape(rows, cols).T
    n = cols + 3
    gamma = (n * _U32) / (1.0 - n * _U32)
    return gamma * terms


def _ulp(values: np.ndarray, dtype) -> np.ndarray:
    finfo = np.finfo(dtype)
    return np.abs(values).astype(np.float64) * finfo.eps


# ---------------------------------------------------------------------------
# The perturbation: one code, re-valued through the contract quantizer
# ---------------------------------------------------------------------------
def straddle_mask(code: int, bits: int) -> int:
    """The bits of this code that live in the SECOND 32-bit word, as a mask
    over the code's value; zero when the code sits inside one word."""
    offset = (code * bits) % 32
    if offset + bits <= 32:
        return 0
    in_first = 32 - offset
    return ((1 << bits) - 1) & ~((1 << in_first) - 1)


def perturb_weight(w: np.ndarray, row: int, code: int, bits: int):
    """Raw weights with exactly one code moved, and moved where it hurts.

    For a code contained in one word, the new code is the farthest legal
    value, which maximises the witness margin. For a code that STRADDLES two
    words, the new code flips exactly the bits living in the second word and
    leaves the first word's bits identical: a kernel whose two-word read is
    broken then sees NO change at all, so the fault is caught for every
    original value rather than only when the low bits happen to differ. Both
    choices are read off the packing arithmetic; nothing is fitted.

    The edit happens in RAW weight space and is verified after the fact: the
    perturbed weights are re-quantized and the artefact must show scales and
    biases bit-identical with exactly this one code changed, else the edit
    disturbed its group's anchoring and the target must be dropped rather
    than judged (a perturbation that moved a scale is not attributable to
    one element).

    Returns (w_perturbed, clean).
    """
    contract = QuantContract(bits=bits, group_size=64)
    before = canonical_quantize(w.astype(np.float32), contract)
    group = code // contract.group_size
    q_old = int(before.q[row, code])
    mask = straddle_mask(code, bits)
    if mask:
        q_new = q_old ^ mask
    else:
        q_new = 0 if q_old > (1 << bits) // 2 else (1 << bits) - 1
    scale = np.float32(before.scales[row, group])
    bias = np.float32(before.biases[row, group])
    replaced = np.float32(scale * np.float32(q_new) + bias)

    perturbed = w.astype(np.float32).copy()
    perturbed[row, code] = replaced

    after = canonical_quantize(perturbed, contract)
    clean = (np.array_equal(after.scales, before.scales)
             and np.array_equal(after.biases, before.biases)
             and int(np.sum(after.q != before.q)) == 1
             and after.q[row, code] != q_old)
    return perturbed, clean


def _activation(d_in: int, code: int, bits: int) -> np.ndarray:
    """Nonzero on every code sharing a word with the perturbed one, plus an
    even spread across the row. All values are powers of two, so the
    activation itself adds no rounding of its own."""
    x = np.zeros((1, d_in), dtype=np.float32)
    x[0, :: max(d_in // 16, 1)] = 1.0
    first_bit = code * bits
    last_bit = (code + 1) * bits - 1
    low = (first_bit // 32) * 32 // bits
    high = min((last_bit // 32 + 1) * 32 // bits + 1, d_in)
    x[0, low:high] = 1.0
    return x


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def screen(runner, spec_for, w: np.ndarray, bits: int,
           targets: list[Target] | None = None, *,
           launch_params=None) -> Report:
    """Screen one candidate kernel's weight-code support.

    `spec_for` maps (d_out, d_in) to the candidate's KernelSpec: the gate
    owns the inputs, the caller owns only the kernel. Both input sets for
    every target are derived here, from `w`, through the anchored
    quantize-and-pack path; there is deliberately no way to hand the gate
    pre-packed device bytes.
    """
    d_out, d_in = w.shape
    contract = QuantContract(bits=bits, group_size=64)
    baseline = canonical_quantize(w.astype(np.float32), contract)
    if targets is None:
        targets = derive_targets(d_out, d_in, bits)

    verdicts = []
    for target in targets:
        verdicts.append(_judge(runner, spec_for, w, bits, baseline, contract,
                               target, launch_params or {}))
    return Report(policy=POLICY, verdicts=tuple(verdicts))


def _judge(runner, spec_for, w, bits, baseline, contract, target,
           launch_params) -> TargetVerdict:
    d_out, d_in = w.shape
    perturbed_w, clean = perturb_weight(w, target.row, target.code, bits)
    if not clean:
        return TargetVerdict(target, UNCLEAN_EDIT, detail=(
            "re-quantizing moved a scale, a bias or a second code, so the "
            "edit is not attributable to one element"))

    after = canonical_quantize(perturbed_w, contract)
    x = _activation(d_in, target.code, bits)

    # The oracle side: witnesses from the shipped fp64 reference, differenced
    # rather than assumed, with the envelope bounding every admissible order.
    y0 = r_contract(x, baseline)
    y1 = r_contract(x, after)
    separation = (envelope_qmm(x, baseline) + envelope_qmm(x, after)
                  + 2.0 * _ulp(np.maximum(np.abs(y0), np.abs(y1)), np.float32))
    finite = np.isfinite(y0) & np.isfinite(y1)
    normal = (np.abs(y0) >= np.finfo(np.float32).tiny) & \
             (np.abs(y1) >= np.finfo(np.float32).tiny)
    witnesses = finite & normal & (np.abs(y1 - y0) > separation)
    if not witnesses.any():
        best = float(np.max(np.abs(y1 - y0) / np.maximum(separation, 1e-300)))
        return TargetVerdict(target, NO_WITNESS, margin=best, detail=(
            f"no output cell clears the separation inequality; best margin "
            f"{best:.3f} of the 1.0 required"))
    margin = float(np.max((np.abs(y1 - y0) / separation)[witnesses]))

    # The device side, all three runs derived here from the raw weights.
    def case(artefact, label):
        inputs = {"x": x, "w_q": pack_codes(artefact.q, bits),
                  "scales": artefact.scales, "biases": artefact.biases}
        return RunCase(inputs=inputs, params={"d_out": d_out, "d_in": d_in,
                                              **launch_params},
                       output_shapes=[((1, d_out), "float32")], label=label)

    spec = spec_for(d_out, d_in)
    results = runner.run(spec, [case(baseline, f"{target.label}|control"),
                                case(baseline, f"{target.label}|again"),
                                case(after, f"{target.label}|perturbed")],
                         warmup=0, repeats=0)
    if any(r.status is not RunStatus.OK for r in results):
        return TargetVerdict(target, NO_RUN, detail=(
            "a run did not complete; that verdict belongs to the compile stage"))

    control = results[0].outputs[0].view(np.uint32)
    again = results[1].outputs[0].view(np.uint32)
    perturbed = results[2].outputs[0].view(np.uint32)

    if not np.array_equal(control, again):
        return TargetVerdict(target, NONDETERMINISTIC, detail=(
            "the same case produced different bits twice; dependence cannot "
            "be separated from run-to-run variation, and requiring "
            "determinism would be a new rule (amendment 1), so nothing is "
            "claimed"))

    unchanged = witnesses[0] & (control == perturbed)[0]
    if unchanged.any():
        cell = int(np.argmax(unchanged))
        return TargetVerdict(target, UNSUPPORTED, margin=margin, detail=(
            f"weight code {target.code} of row {target.row} was re-valued, "
            f"the reference moved output cell {cell} by "
            f"{abs(y1[0, cell] - y0[0, cell]):.3e} against an envelope of "
            f"{separation[0, cell]:.3e}, and the kernel's bits did not "
            f"change: the output does not depend on that code"))
    return TargetVerdict(target, SUPPORTED, margin=margin)
