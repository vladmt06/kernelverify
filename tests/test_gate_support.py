"""The input-support gate, and the two measured faults that shaped it.

An adversarial review planted two one-line support faults in the shipped
kernel and showed the first design passing both at 49x to 443x margin: a
dead guard on the two-word code read (28% of weight codes never reach the
output) and a wrong lane stride (blocks silently skipped). Both survive
here as planted faults in a representative candidate that MUST be refused,
because a support gate that cannot catch the faults that motivated it is
decoration.

The candidate is a hand-written 3-bit packed GEMV with the same address
arithmetic classes as the shipped kernel: two-word code reads and a
lane-strided block loop. It is a KernelSpec because generated candidates,
which are what this gate screens, arrive as KernelSpecs.

The review's false positive is closed structurally rather than tested
around: the gate derives every device input itself from the raw weights, so
the packer disagreement that refused a correct kernel cannot be expressed
through its API at all, and a test pins that absence.
"""

import inspect

import numpy as np
import pytest

from conftest import requires_metal
from kernelverify.compiler.support import (
    NO_WITNESS,
    NONDETERMINISTIC,
    POLICY,
    SUPPORTED,
    UNCLEAN_EDIT,
    UNSUPPORTED,
    Target,
    alignment_cycle,
    derive_targets,
    envelope_qmm,
    perturb_weight,
    screen,
    straddle_mask,
)
from kernelverify.runners import MetalRunner
from kernelverify.runners.spec import (
    Binding,
    BindingKind,
    KernelSpec,
    LaunchSpec,
)
from kernelverify.schemas.quant_contract import (
    QuantContract,
    canonical_quantize,
    r_contract,
)

BITS = 3
D_OUT, D_IN = 16, 640  # 10 groups, 80 blocks: the lane-stride band exists

QMV = """
[[kernel]] void qmv(device const float* x [[buffer(0)]],
                    device const uint* w_q [[buffer(1)]],
                    device const float* scales [[buffer(2)]],
                    device const float* biases [[buffer(3)]],
                    device float* out [[buffer(4)]],
                    constant uint& d_out [[buffer(5)]],
                    constant uint& d_in [[buffer(6)]],
                    uint2 tid [[thread_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint BITS = 3;
    uint row = tid.y;
    if (row >= d_out) { return; }
    uint blocks = d_in / 8;
    uint row_words = (d_in * BITS + 31) / 32;
    uint groups = d_in / 64;
    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += 32) {
        for (uint i = 0; i < 8; ++i) {
            uint k = b * 8 + i;
            uint bit = k * BITS;
            uint word = bit / 32;
            uint off = bit % 32;
            uint lo = w_q[row * row_words + word];
            uint q;
            if (off + BITS > 32) {
                uint hi = w_q[row * row_words + word + 1];
                q = ((lo >> off) | (hi << (32 - off))) & 7u;
            } else {
                q = (lo >> off) & 7u;
            }
            uint g = k / 64;
            float wv = scales[row * groups + g] * (float)q
                     + biases[row * groups + g];
            acc = metal::fma(x[k], wv, acc);
        }
    }
    acc = metal::simd_sum(acc);
    if (lane == 0) { out[row] = acc; }
}
"""

# The review's fault A: the two-word read's guard is dead, so every code
# that straddles a word boundary loses the bits in its second word.
DEAD_GUARD = QMV.replace(
    "uint hi = w_q[row * row_words + word + 1];", "uint hi = 0u;")

# The review's fault C: a wrong lane stride, so some blocks are never read.
WRONG_STRIDE = QMV.replace("b += 32", "b += 33")

BINDINGS = (Binding(BindingKind.INPUT, "x"), Binding(BindingKind.INPUT, "w_q"),
            Binding(BindingKind.INPUT, "scales"),
            Binding(BindingKind.INPUT, "biases"), Binding(BindingKind.OUTPUT),
            Binding(BindingKind.SCALAR, "d_out", "uint32"),
            Binding(BindingKind.SCALAR, "d_in", "uint32"))
LAUNCH = LaunchSpec(grid=(32, "d_out", 1), threadgroup=(32, 1, 1))


def spec_for_source(source):
    def spec_for(d_out, d_in):
        return KernelSpec(source=source, entry_point="qmv", bindings=BINDINGS,
                          launch=LAUNCH)
    return spec_for


@pytest.fixture(scope="module")
def runner():
    return MetalRunner()


@pytest.fixture(scope="module")
def weights():
    rng = np.random.default_rng(7)
    return (rng.standard_normal((D_OUT, D_IN)) * 0.05).astype(np.float32)


# ---------------------------------------------------------------------------
# Target derivation: the binding's arithmetic, not the operator's indices
# ---------------------------------------------------------------------------
def test_every_alignment_residue_is_covered_including_the_straddlers():
    """The review's structural point: a probe set built from operator-level
    'special' indices provably never reaches a word-straddling code, because
    a multiple of 8 or 64 is never congruent to 10 or 21 mod 32."""
    targets = derive_targets(D_OUT, D_IN, BITS)
    residues = {t.code % alignment_cycle(BITS) for t in targets
                if t.label.startswith("align")}
    assert residues == set(range(32))
    assert {10, 21} <= residues


def test_the_lane_stride_band_is_covered():
    blocks = {t.code // 8 for t in derive_targets(D_OUT, D_IN, BITS)
              if t.label.startswith("block")}
    assert blocks == set(range(32, 64)), (
        "a wrong stride s in (32, 64] first skips a block in this band")


def test_straddlers_exist_at_3_bits_and_not_at_widths_that_divide_32():
    assert [r for r in range(32) if straddle_mask(r, 3)] == [10, 21]
    assert not any(straddle_mask(r, 4) for r in range(8))
    assert not any(straddle_mask(r, 2) for r in range(16))


# ---------------------------------------------------------------------------
# The perturbation: clean, attributable, and shaped by the packing
# ---------------------------------------------------------------------------
def test_a_perturbation_moves_exactly_one_code_and_no_scale(weights):
    perturbed, clean = perturb_weight(weights, row=3, code=100, bits=BITS)
    assert clean
    contract = QuantContract(bits=BITS, group_size=64)
    before = canonical_quantize(weights, contract)
    after = canonical_quantize(perturbed, contract)
    assert int(np.sum(after.q != before.q)) == 1
    assert np.array_equal(after.scales, before.scales)
    assert np.array_equal(after.biases, before.biases)


def test_a_straddling_code_flips_only_its_second_word_bits(weights):
    """The dead-guard fault is caught for EVERY original value only if the
    first word's bits stay identical, so a broken two-word read sees no
    change at all."""
    code = 10  # residue 10: bits 30, 31 | 32
    mask = straddle_mask(code, BITS)
    assert mask == 0b100
    contract = QuantContract(bits=BITS, group_size=64)
    before = canonical_quantize(weights, contract)
    perturbed, clean = perturb_weight(weights, row=0, code=code, bits=BITS)
    assert clean
    after = canonical_quantize(perturbed, contract)
    q_old, q_new = int(before.q[0, code]), int(after.q[0, code])
    assert q_new == q_old ^ mask
    assert (q_old & 0b011) == (q_new & 0b011), "first-word bits identical"


def test_an_edit_that_would_move_the_scale_is_dropped_not_judged():
    """A group built so the target element IS the group's edge: re-valuing it
    moves the group maximum, the scale moves, and the edit is no longer
    attributable to one element."""
    w = np.full((1, 64), 0.01, dtype=np.float32)
    w[0, 5] = 0.05  # the lone extreme; q_old is the top code, q_new is 0
    perturbed, clean = perturb_weight(w, row=0, code=5, bits=BITS)
    assert not clean


# ---------------------------------------------------------------------------
# The witness inequality, on the host
# ---------------------------------------------------------------------------
def test_the_envelope_is_positive_and_scales_with_the_terms(weights):
    x = np.zeros((1, D_IN), dtype=np.float32)
    x[0, :64] = 1.0
    artefact = canonical_quantize(weights, QuantContract(bits=BITS, group_size=64))
    env = envelope_qmm(x, artefact)
    assert env.shape == (1, D_OUT) and (env > 0).all()
    x2 = x * 2
    assert np.allclose(envelope_qmm(x2, artefact), env * 2)


def test_a_perturbed_code_clears_the_inequality_at_these_shapes(weights):
    """The gate is only worth building if witnesses exist: the reference must
    move by MORE than both envelopes at the gate's own case."""
    from kernelverify.compiler.support import _activation

    contract = QuantContract(bits=BITS, group_size=64)
    before = canonical_quantize(weights, contract)
    perturbed, clean = perturb_weight(weights, row=2, code=100, bits=BITS)
    assert clean
    after = canonical_quantize(perturbed, contract)
    x = _activation(D_IN, 100, BITS)
    moved = np.abs(r_contract(x, after) - r_contract(x, before))
    envelope = envelope_qmm(x, before) + envelope_qmm(x, after)
    assert (moved[0, 2] > 10 * envelope[0, 2]), (
        "the witness should clear the envelope by an order of magnitude")


# ---------------------------------------------------------------------------
# The gate against real kernels
# ---------------------------------------------------------------------------
@requires_metal
def test_the_correct_kernel_is_supported_at_every_screened_target(runner, weights):
    report = screen(runner, spec_for_source(QMV), weights, BITS)
    assert report.ok, report.reason
    assert not any(v.verdict == UNSUPPORTED for v in report.verdicts)
    screened = [v for v in report.verdicts if v.screened]
    assert len(screened) >= 48, (
        f"only {len(screened)} of {len(report.verdicts)} targets screened: "
        f"{report.reason}")
    assert all(v.verdict == SUPPORTED for v in screened)


@requires_metal
def test_the_dead_two_word_guard_is_refused_at_a_straddling_target(runner,
                                                                   weights):
    """The review's fault A: 28% of codes never fully read, passed the first
    design. The straddle-flip perturbation makes it visible: the first word's
    bits are identical, so the broken read sees no change at all."""
    report = screen(runner, spec_for_source(DEAD_GUARD), weights, BITS)
    assert not report.ok
    refused = [v for v in report.verdicts if v.verdict == UNSUPPORTED]
    straddle_residues = {10, 21}
    assert any(v.target.code % 32 in straddle_residues for v in refused), (
        f"the refusals must include a straddler: {[v.target.label for v in refused]}")
    assert "does not depend on that code" in refused[0].detail


@requires_metal
def test_the_wrong_lane_stride_is_refused_at_a_band_target(runner, weights):
    """The review's fault C: stride 33 skips blocks 32 and 65 here, and the
    band exists precisely so the first skipped block of any stride in
    (32, 64] holds a target."""
    report = screen(runner, spec_for_source(WRONG_STRIDE), weights, BITS)
    assert not report.ok
    refused = [v for v in report.verdicts if v.verdict == UNSUPPORTED]
    skipped = {b for b in range(D_IN // 8) if b % 33 == 32}
    assert any(v.target.code // 8 in skipped for v in refused), (
        f"refusals: {[v.target.label for v in refused]}, skipped blocks: {skipped}")


def test_a_nondeterministic_kernel_is_abstained_not_refused(weights):
    """Requiring determinism would be a new rule about what a kernel may do,
    which amendment 1 blocks; its absence costs coverage, never a refusal.
    A planted GPU race fires unreliably, which would make this test flaky, so
    the path is driven by a scripted runner whose two control runs disagree
    by construction."""
    from kernelverify.runners.result import RunResult, RunStatus

    class DisagreeingRunner:
        def run(self, spec, cases, **_kwargs):
            outputs = []
            for index, case in enumerate(cases):
                out = np.full((1, D_OUT), 1.0, dtype=np.float32)
                out[0, 0] += index * 1e-6  # control and again differ
                outputs.append(RunResult(status=RunStatus.OK, outputs=[out],
                                         label=case.label))
            return outputs

    targets = [Target(row=0, code=100, label="probe")]
    report = screen(DisagreeingRunner(), spec_for_source(QMV), weights, BITS,
                    targets=targets)

    [verdict] = report.verdicts
    assert verdict.verdict == NONDETERMINISTIC
    assert report.ok, "an abstention must not refuse the candidate"
    assert report.screened == 0, "and must not count as coverage"
    assert "not screened" in report.reason


# ---------------------------------------------------------------------------
# What the gate refuses to pretend
# ---------------------------------------------------------------------------
def test_the_policy_names_the_unscreened_bindings_and_the_sampled_band():
    assert "scales and biases bindings are NOT screened" in POLICY
    assert "COMPLETELY" in POLICY and "SAMPLED" in POLICY
    assert "tolerance gate's to catch" in POLICY


def test_the_gate_cannot_be_handed_prepacked_device_bytes():
    """The review's false positive lived in the gap between builder-packed
    device buffers and the oracle's raw weights. The gate closes it by
    construction: its signature accepts raw weights and derives everything,
    and there is no parameter for a pre-built RunCase."""
    parameters = inspect.signature(screen).parameters
    assert "w" in parameters and "targets" in parameters
    assert not any(name in parameters for name in ("cases", "control",
                                                   "perturbed", "inputs"))


# ---------------------------------------------------------------------------
# As a funnel stage
# ---------------------------------------------------------------------------
@requires_metal
def test_the_support_stage_kills_the_dead_guard_through_the_funnel(runner,
                                                                   weights,
                                                                   tmp_path):
    from kernelverify.compiler.funnel import Funnel
    from kernelverify.compiler.stages import support_stage
    from kernelverify.compiler.store import CandidateStore

    store = CandidateStore(tmp_path)
    funnel = Funnel([support_stage()])
    candidate = store.propose(DEAD_GUARD, origin="mutant:dead guard")
    result = funnel.screen(store, candidate, runner=runner, entry_point="qmv",
                           bindings=BINDINGS, launch=LAUNCH,
                           support_weights=weights, support_bits=BITS)

    assert not result.passed and result.reached == "support"
    assert "does not depend on that code" in result.detail
    assert store.get(candidate).outcome == "support:failed"
