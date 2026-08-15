"""3-bit device adequacy at the Qwen3-4B E2E serving shapes (lane Q, ADR 0013).

PRE-REGISTRATION
================
This docstring is committed before the calibration runs. The rule was gated
through the plan review (builder + reviewer, two rounds) and traces to the
2026-08-15 pivot design rulings D1 (3-bit serving path), D3.4 (adequacy
precedes certificates; honest serial measuring order) and D4 (2-bit is CUT
from the block and is not run here). If the numbers force a deviation, the
run stops and the rule is renegotiated with the coordinator before anything
further is looked at, never after.

What this discharges
--------------------
ADR 0012 shipped K = 4 over the three-class nine-name floor, measured on the
calibration shapes. The serving shapes Qwen3-4B actually dispatches under
mlx-lm are new input territory for the same contract. The question, exactly:
does the standing tolerance ``max(base, 4 * floor)`` cover every correct
implementation at the serving shapes, or does some cell demand more? Per the
standing repair order (ADR 0005, 0009, 0012), a demand above 4 is answered by
MEMBERSHIP first - further device variants join the floor - and K moves only
if membership cannot close it; either branch is a fleet decision taken with
the coordinator, not inside this harness.

The serving shapes and the artifact condition
---------------------------------------------
The shapes come from the pinned 3-bit artifact
``/Users/vlad/kernelverify/bench/.models/qwen3-4b-3bit-g64`` (ABSOLUTE path:
``bench/.models`` is gitignored, so the artifact lives in the main worktree
only). While the artifact is absent - its conversion is owned by the
coordinator - the shapes are derived from the Qwen3-4B config, recorded in
``QWEN3_4B_DERIVED_CONFIG`` from the HF checkpoint config verified locally,
and the ATTESTATION IS CONDITIONAL: it binds to the serving artifact only if
that artifact's config.json shows uniform bits=3, group_size=64 quantization
on every projection including the tied-embedding lm_head path. The harness
re-derives provenance from the artifact config when present and stops loudly
on any mismatch; ADR 0013 carries the conditional wording until the
coordinator confirms the artifact and this harness has verified it.

Derived (d_out, d_in), deduplicated: q_proj (4096, 2560), kv_proj
(1024, 2560), o_proj (2560, 4096), gate_up_proj (9728, 2560), down_proj
(2560, 9728), lm_head (151936, 2560) via tie_word_embeddings.

The rule
--------
Bits = 3 only. Weight draws, calibration seeds 0/1, independent draw 100/101,
four input modes and both activation dtypes exactly as the ADR 0009/0012
grid. Batch is a grid axis: base set {1, 2, 8, 16} (decode; the ADR 0012
grid's batch; the interior of the measured wide_qmv win zone; the serving
edge), extended by step 2's probe. The steps, in order:

0. CONTINUITY ANCHOR. The STANDING ``calibrate_quant_device.measure`` runs
   unchanged at (1024, 4096), bits=3 - verbatim rng consumption by
   construction - and every member, held-out, fault and boundary error must
   be identical to the 64 cached ADR 0012 records in
   ``bench/.cache/quant_device_adequacy.json``. The pipeline does not
   measure new territory until it reproduces the old. Any difference stops
   the run (toolchain drift is a finding, not a footnote).

1. G0 per serving shape: ``canonical_quantize`` bit-exact against
   ``mx.quantize`` on every (shape, draw, seed). At bits=3 the pack stream
   straddles word boundaries, and the (151936, 2560) row count has never
   been quantized by this harness; any inexact check stops the run.

2. BATCH-REGIME PROBE, because batch-invariance must be measured, not
   assumed. At every serving shape, for each activation dtype: draw x16 (16
   rows, unit mode, normal-0.02 weights, seed 0, the stream order pinned in
   ``probe_batch_regimes``), evaluate every implementation - six CPU
   members, three device members, both held-outs, the fp16-dequant boundary
   - on x16[:B] for every B in 1..16. Batches B and B' MATCH when their
   shared-prefix rows (the first min(B, B') rows) are bit-identical across
   every implementation and both dtypes, jointly; the full pairwise match
   matrix is recorded in the JSON. Coverage is DIRECT-MATCH: a batch B is
   covered iff the matrix shows a direct match between B and some grid
   batch; while uncovered batches exist, the smallest uncovered B joins the
   grid and coverage is recomputed. The connected-component reading of the
   same matrix is reported beside it (both-readings practice), with
   direct-match governing and any divergence between the two readings
   itself reported - the match relation is not transitive, so a component
   could otherwise merge batches that directly mismatch. When components
   are cliques the two readings coincide: one sampled representative per
   measured arithmetic regime, no more. One visibility line beyond the
   gated rule, reported and never gating: any batch covered ONLY through a
   smaller grid batch (a strict-prefix match) is listed as weak-prefix
   covered, so the ADR can say which batches are attested by direct
   sampling and which by measured row-invariance alone. The attestation
   then covers B = 1..16 through measured row-invariance plus the sampled
   set, not through assumption.

3. The main grid: shapes x batches x draws x seeds x modes x dtypes, every
   record carrying the same quantities as the standing harness (members,
   held-outs, faults through the dequant-pairwise carrier unchanged,
   fp16-dequant boundary). Per (shape, draw, seed) block the rng stream is
   consumed in the pinned order: ``make_w`` first, then ``make_x`` per
   (batch, mode, dtype), batch outermost.

4. K DEMAND at full nine-name membership (the standing ``k_demand``,
   leave-one-member-out plus held-outs against the full floor, only cases
   above base tolerance), computed per (shape, batch) cell and pooled, on
   the calibration draw and the independent draw SEPARATELY. Attestation
   requires every cell's demand, on both draws, at or under the shipped
   K = 4. A cell above 4 fires the pre-registered branch: STOP
   interpretation, report the cell, renegotiate membership with the
   coordinator (membership before K, always).

5. Gates at K = 4 per cell and pooled, G1/G2/G3 as the standing ``gates``,
   with one pinned refinement: fault EQUIVALENCE is classified at width
   level (caught nowhere across ALL serving records), then per-cell G2
   gates caught == of and margin >= 10x over that width-level scored set,
   so a fault caught elsewhere but absent in one cell cannot silently drop
   out of that cell's gate.

6. The class-necessity diagnostic (never a gate) and the fp16-dequant
   boundary under both floors, per activation dtype, pooled and per cell,
   reported in the standing shape.

ADR 0012's step-2 trigger (device members against the retired CPU-calibrated
K = 3 tolerance) is deliberately not re-run: the shipped tolerance is now
K = 4 over the joined floor, so the live questions are step 4 and step 5.
The records carry every member and held-out error, so the retired-trigger
quantity stays computable post hoc from the JSON, and ADR 0013 says so.

Exact-value engineering, all of it tested for bit-identity
----------------------------------------------------------
The lm_head shape does not fit the standing evaluation verbatim in 36 GB:
the two serial CPU members materialise (batch, d_out, d_in) twice. The
harness therefore evaluates them in row chunks (elementwise product and
``np.add.accumulate`` along d_in never cross a row boundary and touch no
BLAS), and hoists per-(draw, seed) pure functions of the artefact: the fp64
reference dequant, the fp32 member dequant, the lut-gathered weights, the
int-domain code tensor, the six fault artefacts' fp32 dequants (evaluated
one at a time to bound residency), the fp16-dequant boundary weights and
``mx.quantize`` for the held-out. Every rewritten path carries a unit test
proving bit-identity against the untouched reference implementation; a
helper that is merely close is wrong. Device members and MLX's own
``quantized_matmul`` run per record, unhoisted, as on the standing grid.

Machine discipline
------------------
One measuring lane at a time: the full run executes only in the
coordinator-granted GPU slot, after the pack lane's boundary pricing. The
pre-slot allowance is the probe validated at (1024, 2560) plus unit-test
smoke dispatches. Reruns must reproduce ADR 0013's tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BITS = 3
BASE_BATCHES = (1, 2, 8, 16)
PROBE_BATCH_MAX = 16
K_SHIP = 4.0  # ADR 0012: the shipped K over the three-class nine-name floor

ARTIFACT_DIR = Path("/Users/vlad/kernelverify/bench/.models/qwen3-4b-3bit-g64")
OUT_PATH = Path(__file__).with_name(".cache") / "quant_serving_adequacy.json"
CONTINUITY_CACHE = Path(__file__).with_name(".cache") / "quant_device_adequacy.json"
CONTINUITY_SHAPE = (1024, 4096)

# The Qwen3-4B geometry, recorded from the HF checkpoint config
# (Qwen/Qwen3-4B, verified locally on 2026-08-15) for the artifact-absent
# fallback. The quantization block states the ASSUMED artifact recipe; the
# pinned artifact's own config.json replaces all of this when present.
QWEN3_4B_DERIVED_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 2560,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9728,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
    "quantization": {"group_size": 64, "bits": 3},
}


class ArtifactMismatch(RuntimeError):
    """The artifact (or config) does not match what the attestation covers."""


def verify_artifact_quantization(quantization) -> None:
    """Uniform bits=3, group_size=64, no per-layer overrides: anything else
    means the attestation would cover a different artifact than the one
    serving, so the run stops and the coordinator renegotiates."""
    if not isinstance(quantization, dict):
        raise ArtifactMismatch(
            f"no quantization block ({quantization!r}); expected uniform "
            f"bits={BITS}, group_size=64")
    extras = sorted(set(quantization) - {"group_size", "bits", "mode"})
    if extras:
        raise ArtifactMismatch(
            f"per-layer quantization overrides {extras}: the artifact is not "
            f"uniformly quantized, so the serving shapes do not all carry the "
            f"attested contract; renegotiate with the coordinator")
    if quantization.get("mode", "affine") != "affine":
        raise ArtifactMismatch(
            f"quantization mode {quantization['mode']!r} is not the mlx-affine "
            f"scheme this contract states")
    if quantization.get("bits") != BITS or quantization.get("group_size") != 64:
        raise ArtifactMismatch(
            f"artifact quantization {quantization} is not bits={BITS}, "
            f"group_size=64; this harness attests only that contract")


def derive_serving_shapes(config: dict) -> tuple[tuple[str, tuple[int, int]], ...]:
    """The deduplicated (d_out, d_in) set mlx-lm dispatches for one decode
    step: attention and MLP projections per layer, plus the tied-embedding
    lm_head. k/v and gate/up share shapes and appear once."""
    if not config.get("tie_word_embeddings"):
        raise ArtifactMismatch(
            "tie_word_embeddings is false: the lm_head path would be a "
            "separate head this derivation does not cover")
    hidden = config["hidden_size"]
    head_dim = config["head_dim"]
    q_dim = config["num_attention_heads"] * head_dim
    kv_dim = config["num_key_value_heads"] * head_dim
    inter = config["intermediate_size"]
    shapes = (
        ("q_proj", (q_dim, hidden)),
        ("kv_proj", (kv_dim, hidden)),
        ("o_proj", (hidden, q_dim)),
        ("gate_up_proj", (inter, hidden)),
        ("down_proj", (hidden, inter)),
        ("lm_head", (config["vocab_size"], hidden)),
    )
    group = config.get("quantization", {}).get("group_size", 64)
    for name, (d_out, d_in) in shapes:
        assert d_in % group == 0, (name, d_in, group)
    return shapes


def resolve_serving_source(artifact_dir: Path) -> tuple[dict, str]:
    """The artifact's own config when it exists (verified), the recorded
    derivation otherwise (assumption stated, attestation conditional)."""
    config_path = artifact_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        verify_artifact_quantization(config.get("quantization"))
        return config, f"pinned-artifact:{artifact_dir}"
    return QWEN3_4B_DERIVED_CONFIG, (
        f"derived-from-config: pinned artifact absent at {artifact_dir}; "
        f"shapes derived from the Qwen/Qwen3-4B HF config on the recorded "
        f"assumption of uniform bits={BITS} g64 quantization (attestation "
        f"conditional until the artifact is verified)")


# ---------------------------------------------------------------------------
# Exact-value engineering: hoisted pure functions of the artefact and the
# chunked serial members. Every function here is bit-identical to the
# corresponding path in quant_contract / phase0_contract_k / calibrate_quant_bits,
# proven by tests/test_serving_adequacy.py; none of them is allowed to be
# merely close.
# ---------------------------------------------------------------------------
from kernelverify.schemas.quant_contract import (  # noqa: E402
    ENSEMBLE,
    FAULTS,
    QuantArtefact,
    QuantContract,
    dequantize,
)


def _f32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)


class ArtefactHoists:
    """Per-(draw, seed) precomputation, so the lm_head shape pays for each
    pure function of the artefact once instead of once per record.

    Fault dequants are deliberately NOT stored here: six extra fp32 weight
    matrices would add ~9 GB at lm_head, so ``fault_dequants`` yields them
    one at a time instead.
    """

    def __init__(self, w: np.ndarray, artefact: QuantArtefact):
        g = artefact.contract.group_size
        rows, cols = artefact.q.shape
        levels = 1 << artefact.contract.bits
        self.w64 = dequantize(artefact, np.float64)
        self.w32 = dequantize(artefact, np.float32)
        # lut-gather's weights, exactly member_lut_gather's construction
        table = (artefact.scales.astype(np.float32)[:, :, None]
                 * np.arange(levels, dtype=np.float32)[None, None, :]
                 + artefact.biases.astype(np.float32)[:, :, None])
        q = artefact.q.reshape(rows, cols // g, g)
        self.w_lut32 = np.take_along_axis(table, q, axis=2).reshape(rows, cols)
        # the int-domain members' operands
        self.qg32 = artefact.q.reshape(rows, cols // g, g).astype(np.float32)
        self.scales32 = artefact.scales.astype(np.float32)
        self.biases32 = artefact.biases.astype(np.float32)
        # the fp16-dequant boundary's weights
        self.w16_32 = self.w32.astype(np.float16).astype(np.float32)


def fault_dequants(artefact: QuantArtefact):
    """(name, fp32 dequant of the faulted artefact), one at a time, in
    catalogue order; the consumer frees each before the next is built."""
    for name, fault in FAULTS.items():
        yield name, dequantize(fault(artefact), np.float32)


def eval_ref(x: np.ndarray, w64: np.ndarray) -> np.ndarray:
    """r_contract with the fp64 dequant hoisted."""
    return x.astype(np.float64) @ w64.T


def eval_pairwise(x: np.ndarray, w32: np.ndarray) -> np.ndarray:
    """member_dequant_pairwise; also the fault carrier on a faulted w32."""
    return (_f32(x) @ w32.T).astype(x.dtype)


def eval_lut(x: np.ndarray, w_lut32: np.ndarray) -> np.ndarray:
    """member_lut_gather with the gathered weights hoisted."""
    return (_f32(x) @ w_lut32.T).astype(x.dtype)


def eval_serial_chunked(x: np.ndarray, w32: np.ndarray, chunk_rows: int,
                        reverse: bool = False) -> np.ndarray:
    """member_dequant_serial / member_dequant_reversed in output-row chunks.

    The elementwise product and the accumulation run along d_in and never
    cross a row boundary, and no BLAS is involved, so the chunked evaluation
    is bit-identical to the whole-matrix member at every chunk size; chunking
    exists purely to keep (batch, chunk, d_in) inside RAM at lm_head.
    """
    xf = _f32(x)
    out = np.empty((x.shape[0], w32.shape[0]), dtype=np.float32)
    for start in range(0, w32.shape[0], chunk_rows):
        block = w32[start:start + chunk_rows]
        prod = xf[:, None, :] * block[None, :, :]
        if reverse:
            prod = prod[:, :, ::-1]
        out[:, start:start + chunk_rows] = np.add.accumulate(prod, axis=2)[:, :, -1]
    return out.astype(x.dtype)


def eval_factored_groups(x: np.ndarray, qg32: np.ndarray, scales32: np.ndarray,
                         biases32: np.ndarray) -> np.ndarray:
    """member_factored_groups with the code tensor and parameters hoisted."""
    batch, cols = x.shape
    groups, g = qg32.shape[1], qg32.shape[2]
    xg = _f32(x).reshape(batch, groups, g)
    xq = np.einsum("bgk,rgk->brg", xg, qg32, optimize=True)
    xs = xg.sum(axis=2)
    out = (xq * scales32[None, :, :]).sum(axis=2)
    out += xs @ biases32.T
    return out.astype(x.dtype)


def eval_factored_serial(x: np.ndarray, qg32: np.ndarray, scales32: np.ndarray,
                         biases32: np.ndarray) -> np.ndarray:
    """member_factored_serial with the code tensor and parameters hoisted."""
    batch, cols = x.shape
    groups, g = qg32.shape[1], qg32.shape[2]
    xg = _f32(x).reshape(batch, groups, g)
    xq = np.einsum("bgk,rgk->brg", xg, qg32, optimize=True)
    per_group = (xq * scales32[None, :, :]
                 + xg.sum(axis=2)[:, None, :] * biases32[None, :, :])
    return np.add.accumulate(per_group, axis=2)[:, :, -1].astype(x.dtype)


def eval_block_tiled(x: np.ndarray, w32: np.ndarray) -> np.ndarray:
    """heldout_block_tiled with the fp32 dequant hoisted."""
    xf = _f32(x)
    out = np.zeros((x.shape[0], w32.shape[0]), dtype=np.float32)
    for start in range(0, w32.shape[1], 128):
        out = out + xf[:, start:start + 128] @ w32[:, start:start + 128].T
    return out.astype(x.dtype)


def eval_boundary(x: np.ndarray, w16_32: np.ndarray) -> np.ndarray:
    """boundary_fp16_dequant with the fp16-rounded weights hoisted."""
    return (_f32(x) @ w16_32.T).astype(x.dtype)


# ---------------------------------------------------------------------------
# Step 2 machinery: the batch-regime match matrix and its two readings
# ---------------------------------------------------------------------------
def match_matrix(impl_outputs: list) -> np.ndarray:
    """The pre-registered pairwise relation over B = 1..PROBE_BATCH_MAX.

    ``impl_outputs`` is one dict per (implementation, dtype), mapping B to
    that implementation's output on x16[:B]. Batches B and B' match iff the
    first min(B, B') rows are bit-identical in EVERY dict (the joint AND).
    """
    n = PROBE_BATCH_MAX
    matrix = np.ones((n, n), dtype=bool)
    for outputs in impl_outputs:
        for b in range(1, n + 1):
            for b2 in range(b + 1, n + 1):
                shared = min(b, b2)
                same = np.array_equal(outputs[b][:shared], outputs[b2][:shared])
                matrix[b - 1][b2 - 1] &= same
                matrix[b2 - 1][b - 1] &= same
    return matrix


def match_components(matrix: np.ndarray) -> list:
    """The connected-component reading, reported beside the governing one."""
    n = matrix.shape[0]
    seen, components = set(), []
    for start in range(1, n + 1):
        if start in seen:
            continue
        stack, component = [start], set()
        while stack:
            b = stack.pop()
            if b in component:
                continue
            component.add(b)
            stack.extend(b2 for b2 in range(1, n + 1)
                         if matrix[b - 1][b2 - 1] and b2 not in component)
        seen |= component
        components.append(component)
    return sorted(components, key=min)


def coverage_joins(matrix: np.ndarray, grid_batches) -> list:
    """Direct-match coverage, the governing reading: while some batch has no
    direct match with any grid batch, the smallest such batch joins."""
    grid = set(grid_batches)
    joins = []
    while True:
        uncovered = [b for b in range(1, matrix.shape[0] + 1)
                     if not any(matrix[b - 1][g - 1] for g in grid)]
        if not uncovered:
            return joins
        joins.append(uncovered[0])
        grid.add(uncovered[0])
    return joins


def weak_prefix_cover(matrix: np.ndarray, grid_batches) -> list:
    """Batches covered ONLY through a smaller grid batch (a strict-prefix
    match): visibility for the ADR, never a gate."""
    grid = set(grid_batches)
    return [b for b in range(1, matrix.shape[0] + 1)
            if b not in grid
            and any(matrix[b - 1][g - 1] for g in grid)
            and not any(matrix[b - 1][g - 1] for g in grid if g >= b)]


# ---------------------------------------------------------------------------
# Step 5 machinery: per-cell gates under width-level fault equivalence
# ---------------------------------------------------------------------------
from calibrate_quant_bits import MARGIN_GATE  # noqa: E402
from calibrate_quant_device import ALL_MEMBERS, gates as standing_gates  # noqa: E402
from calibrate_quant_device import floor_of  # noqa: E402


import mlx.core as mx  # noqa: E402

from calibrate_quant_bits import (  # noqa: E402
    CALIBRATION_SEEDS,
    GROUP_SIZE,
    HELDOUT_SEEDS,
    MODES,
    WEIGHT_DRAWS,
    base_tol,
    verify_against_mlx,
)
from calibrate_quant_device import k_demand  # noqa: E402
from calibrate_quant_device import measure as standing_measure  # noqa: E402
from kernelverify.schemas.quant_device import (  # noqa: E402
    DEVICE_MEMBERS,
    DeviceMemberSession,
    make_run_case,
    result_outputs,
)
from phase0_contract_k import err, make_w, make_x  # noqa: E402

PROBE_SEED = 0
SERIAL_CHUNK_TARGET = 1 << 27  # fp32 elements per serial-member chunk (~512 MB)


def serial_chunk_rows(batch: int, d_in: int) -> int:
    return max(1, SERIAL_CHUNK_TARGET // max(1, batch * d_in))


def mlx_qmm_heldout(x: np.ndarray, w_q, scales, biases) -> np.ndarray:
    """mlx_on_device with the (deterministic) mx.quantize hoisted out."""
    out = mx.quantized_matmul(mx.array(x), w_q, scales, biases, transpose=True,
                              group_size=GROUP_SIZE, bits=BITS)
    mx.eval(out)
    return np.array(out)


def mlx_quantize_hoist(w: np.ndarray):
    w_q, scales, biases = mx.quantize(mx.array(w), group_size=GROUP_SIZE,
                                      bits=BITS)
    mx.eval(w_q, scales, biases)
    return w_q, scales, biases


def _np_dtype(name: str):
    return np.float32 if name == "float32" else np.float16


def _cpu_members(x: np.ndarray, hoists: ArtefactHoists, chunk: int) -> dict:
    """All six CPU members through the tested exact-value paths."""
    return {
        "dequant-pairwise": eval_pairwise(x, hoists.w32),
        "dequant-serial": eval_serial_chunked(x, hoists.w32, chunk),
        "lut-gather": eval_lut(x, hoists.w_lut32),
        "factored-groups": eval_factored_groups(x, hoists.qg32, hoists.scales32,
                                                hoists.biases32),
        "factored-serial": eval_factored_serial(x, hoists.qg32, hoists.scales32,
                                                hoists.biases32),
        "dequant-reversed": eval_serial_chunked(x, hoists.w32, chunk, reverse=True),
    }


def _device_member_output(session: DeviceMemberSession, member: str,
                          x: np.ndarray, carrier: dict, d_out: int, d_in: int,
                          label: str) -> np.ndarray:
    dtype_name = str(x.dtype)
    case = make_run_case(
        inputs={"x": np.ascontiguousarray(x), **carrier},
        params={"D_IN": d_in, "D_OUT": d_out, "N_GROUPS": d_in // GROUP_SIZE,
                "GROUP": GROUP_SIZE, "BATCH": int(x.shape[0])},
        output_shapes=[((int(x.shape[0]), d_out), dtype_name)],
        label=label,
    )
    result = session.compiled(member, dtype_name).run(case, warmup=0, repeats=0)
    if not result.ok:
        raise RuntimeError(f"trusted member {member} failed on {label}: "
                           f"{result.status.value} - {result.detail}")
    return result_outputs(result)[0]


def _carrier(artefact) -> dict:
    return {
        "q": np.ascontiguousarray(artefact.q.astype(np.float16)),
        "scales": np.ascontiguousarray(artefact.scales.astype(np.float32)),
        "biases": np.ascontiguousarray(artefact.biases.astype(np.float32)),
    }


# ---------------------------------------------------------------------------
# Step 2: the probe driver
# ---------------------------------------------------------------------------
def probe_batch_regimes(shapes, session: DeviceMemberSession) -> dict:
    """The pre-registered batch-regime probe at every serving shape.

    Evaluates one (implementation, dtype) at a time across B = 1..16 and ANDs
    its pairwise matrix into the joint one, bounding residency at lm_head.
    """
    contract = QuantContract(scheme="mlx-affine", bits=BITS, group_size=GROUP_SIZE)
    report = {}
    for name, (d_out, d_in) in shapes:
        rng = np.random.default_rng(10_000 + PROBE_SEED)
        w = make_w(d_out, d_in, "normal-0.02", rng)
        artefact, exact = verify_against_mlx(w, contract)
        if not exact:
            report[name] = {"g0_exact": False}
            continue
        hoists = ArtefactHoists(w, artefact)
        carrier = _carrier(artefact)
        w_q, scales, biases = mlx_quantize_hoist(w)
        # rng order pinned: float32 then float16
        x16 = {d: make_x(PROBE_BATCH_MAX, d_in, _np_dtype(d), "unit", rng)
               for d in ("float32", "float16")}
        matrix = np.ones((PROBE_BATCH_MAX, PROBE_BATCH_MAX), dtype=bool)
        for dtype_name, x_full in x16.items():
            chunk = serial_chunk_rows(PROBE_BATCH_MAX, d_in)
            evaluators = {
                "dequant-pairwise": lambda xb: eval_pairwise(xb, hoists.w32),
                "dequant-serial": lambda xb: eval_serial_chunked(
                    xb, hoists.w32, chunk),
                "lut-gather": lambda xb: eval_lut(xb, hoists.w_lut32),
                "factored-groups": lambda xb: eval_factored_groups(
                    xb, hoists.qg32, hoists.scales32, hoists.biases32),
                "factored-serial": lambda xb: eval_factored_serial(
                    xb, hoists.qg32, hoists.scales32, hoists.biases32),
                "dequant-reversed": lambda xb: eval_serial_chunked(
                    xb, hoists.w32, chunk, reverse=True),
                **{m: (lambda xb, m=m: _device_member_output(
                       session, m, xb, carrier, d_out, d_in,
                       f"probe {name} {m}")) for m in DEVICE_MEMBERS},
                "block-tiled": lambda xb: eval_block_tiled(xb, hoists.w32),
                "mlx-on-device": lambda xb: mlx_qmm_heldout(xb, w_q, scales, biases),
                "boundary-fp16-dequant": lambda xb: eval_boundary(xb, hoists.w16_32),
            }
            for impl_name, fn in evaluators.items():
                outs = {B: fn(x_full[:B]) for B in range(1, PROBE_BATCH_MAX + 1)}
                matrix &= match_matrix([outs])
                del outs
        report[name] = {
            "g0_exact": True,
            "matrix": matrix.tolist(),
            "components": [sorted(c) for c in match_components(matrix)],
            "joins": coverage_joins(matrix, BASE_BATCHES),
            "weak_prefix_covered": weak_prefix_cover(matrix, BASE_BATCHES),
        }
        del hoists, carrier, w_q, scales, biases
    return report


# ---------------------------------------------------------------------------
# Step 3: the main grid
# ---------------------------------------------------------------------------
def measure_serving(shapes, batches, session: DeviceMemberSession,
                    verbose: bool) -> dict:
    """Every record for the serving grid: the standing quantities, plus the
    batch axis, through the tested exact-value paths. Per (shape, draw, seed)
    block the rng stream is consumed in the pinned order: make_w first, then
    make_x per (batch, mode, dtype), batch outermost."""
    contract = QuantContract(scheme="mlx-affine", bits=BITS, group_size=GROUP_SIZE)
    records, exactness = [], []
    for name, (d_out, d_in) in shapes:
        for draw in WEIGHT_DRAWS:
            for seed in CALIBRATION_SEEDS + HELDOUT_SEEDS:
                rng = np.random.default_rng(10_000 + seed)
                w = make_w(d_out, d_in, draw, rng)
                artefact, exact = verify_against_mlx(w, contract)
                exactness.append(exact)
                xs = {}
                for batch in batches:
                    for mode in MODES:
                        for dtype_name in ("float32", "float16"):
                            xs[(batch, mode, dtype_name)] = make_x(
                                batch, d_in, _np_dtype(dtype_name), mode, rng)
                if not exact:
                    continue  # G0 verdict is taken over exactness at the end
                hoists = ArtefactHoists(w, artefact)
                carrier = _carrier(artefact)
                w_q, scales, biases = mlx_quantize_hoist(w)
                block = []
                for (batch, mode, dtype_name), x in xs.items():
                    ref = eval_ref(x, hoists.w64)
                    scale = float(np.abs(ref).max())
                    chunk = serial_chunk_rows(batch, d_in)
                    members = {n: err(out, ref)
                               for n, out in _cpu_members(x, hoists, chunk).items()}
                    label = (f"b{BITS} {name} {d_out}x{d_in} B{batch} {draw} "
                             f"s{seed} {mode} {dtype_name}")
                    for member in DEVICE_MEMBERS:
                        out = _device_member_output(session, member, x, carrier,
                                                    d_out, d_in, label)
                        members[member] = err(out, ref)
                    record = {
                        "bits": BITS, "proj": name, "shape": f"{d_out}x{d_in}",
                        "batch": batch, "draw": draw, "seed": seed,
                        "heldout_draw": seed in HELDOUT_SEEDS,
                        "mode": mode, "dtype": dtype_name,
                        "ref_scale": scale,
                        "base_tol": base_tol(dtype_name, scale),
                        "members": members,
                        "heldout": {
                            "block-tiled": err(eval_block_tiled(x, hoists.w32), ref),
                            "mlx-on-device": err(
                                mlx_qmm_heldout(x, w_q, scales, biases), ref),
                        },
                        "faults": {},
                        "boundary_fp16_dequant": err(
                            eval_boundary(x, hoists.w16_32), ref),
                    }
                    block.append((record, x, ref))
                for fault_name, w_fault32 in fault_dequants(artefact):
                    for record, x, ref in block:
                        record["faults"][fault_name] = err(
                            eval_pairwise(x, w_fault32), ref)
                    del w_fault32
                records.extend(record for record, _, _ in block)
                del hoists, carrier, block, xs, w_q, scales, biases
            if verbose:
                print(f"    {name} {d_out}x{d_in} {draw} done "
                      f"({len(records)} records)")
    return {"records": records, "bit_exact": all(exactness),
            "exact_checks": len(exactness)}


# ---------------------------------------------------------------------------
# Step 0: the continuity anchor
# ---------------------------------------------------------------------------
def continuity_anchor(session: DeviceMemberSession) -> tuple[bool, str]:
    """The standing measure() at (1024, 4096) bits=3 must reproduce the 64
    cached ADR 0012 records exactly before anything new is measured."""
    cached = json.loads(CONTINUITY_CACHE.read_text())
    shape_key = f"{CONTINUITY_SHAPE[0]}x{CONTINUITY_SHAPE[1]}"
    cached_records = [r for r in cached["records"][str(BITS)]
                      if r["shape"] == shape_key]
    fresh = standing_measure(BITS, [CONTINUITY_SHAPE], session, verbose=False)
    if not fresh["bit_exact"]:
        return False, "G0 failed inside the continuity rerun"
    fresh_records = fresh["records"]
    if len(fresh_records) != len(cached_records):
        return False, (f"{len(fresh_records)} fresh records vs "
                       f"{len(cached_records)} cached")
    for i, (ours, theirs) in enumerate(zip(fresh_records, cached_records)):
        checks = (
            [("base_tol", ours["base_tol"], theirs["base_tol"]),
             ("ref_scale", ours["ref_scale"], theirs["ref_scale"]),
             ("boundary", ours["boundary_fp16_dequant"],
              theirs["boundary_fp16_dequant"])]
            + [(f"member {n}", ours["members"][n], theirs["members"][n])
               for n in theirs["members"]]
            + [(f"heldout {n}", ours["heldout"][n], theirs["heldout"][n])
               for n in theirs["heldout"]]
            + [(f"fault {n}", ours["faults"][n], theirs["faults"][n])
               for n in theirs["faults"]]
        )
        for what, a, b in checks:
            if float(a) != float(b):
                return False, (f"record {i} ({theirs['draw']} s{theirs['seed']} "
                               f"{theirs['mode']} {theirs['dtype']}): {what} "
                               f"{a!r} != cached {b!r}")
    return True, f"{len(cached_records)} records reproduced exactly"


def width_equivalents(records: list, k: float) -> set:
    """Faults caught nowhere across ALL serving records at this width."""
    equivalents = set(FAULTS)
    for r in records:
        tol = max(r["base_tol"], k * floor_of(r, ALL_MEMBERS))
        equivalents -= {n for n in FAULTS if r["faults"][n] > tol}
        if not equivalents:
            break
    return equivalents


def cell_reports(records: list, k: float) -> dict:
    """The standing gates per (shape, batch) cell, with fault equivalence
    pinned at width level so a fault caught elsewhere cannot silently drop
    out of one cell's G2."""
    equivalents = width_equivalents(records, k)
    cells = {}
    for r in records:
        cells.setdefault((r["shape"], r["batch"]), []).append(r)
    reports = {}
    for cell, cell_records in cells.items():
        report = standing_gates(cell_records, k)
        report["equivalent_faults"] = sorted(equivalents)
        scored = {n: f for n, f in report["faults"].items() if n not in equivalents}
        worst = min((f["worst_margin"] for f in scored.values()), default=0.0)
        report["worst_margin"] = worst
        report["gates"]["G2_faults_caught"] = (
            all(f["caught"] == f["of"] for f in scored.values())
            and worst >= MARGIN_GATE)
        report["verdict"] = ("ADEQUATE" if all(report["gates"].values())
                             else "INADEQUATE")
        reports[cell] = report
    return reports


# ---------------------------------------------------------------------------
# Step 4: per-cell and pooled K demand against the shipped K = 4
# ---------------------------------------------------------------------------
def demand_table(records: list) -> dict:
    """k_demand per (shape, batch) cell and pooled, calibration and
    independent draws SEPARATELY, each tested against K_SHIP."""
    cells = {}
    for r in records:
        cells.setdefault((r["shape"], r["batch"]), []).append(r)
    table = {}
    for cell, cell_records in sorted(cells.items()):
        cal = [r for r in cell_records if not r["heldout_draw"]]
        indep = [r for r in cell_records if r["heldout_draw"]]
        needed_cal, binding_cal = k_demand(cal)
        needed_indep, binding_indep = k_demand(indep)
        table[cell] = {
            "k_needed_calibration": needed_cal, "k_binding": binding_cal,
            "k_needed_independent": needed_indep,
            "k_binding_independent": binding_indep,
            "covered": max(needed_cal, needed_indep) <= K_SHIP,
        }
    pooled_cal, pooled_cal_at = k_demand([r for r in records
                                          if not r["heldout_draw"]])
    pooled_indep, pooled_indep_at = k_demand([r for r in records
                                              if r["heldout_draw"]])
    pooled = {
        "k_needed_calibration": pooled_cal, "k_binding": pooled_cal_at,
        "k_needed_independent": pooled_indep,
        "k_binding_independent": pooled_indep_at,
        "covered": max(pooled_cal, pooled_indep) <= K_SHIP,
    }
    return {"cells": table, "pooled": pooled}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="3-bit device adequacy at the Qwen3-4B serving shapes")
    parser.add_argument("--probe-validate", action="store_true",
                        help="pre-slot smoke: the probe at kv_proj (1024, 2560) "
                             "only, no continuity run, no main grid")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config, provenance = resolve_serving_source(ARTIFACT_DIR)
    shapes = derive_serving_shapes(config)
    hashes_path = ARTIFACT_DIR.parent / "PINNED-HASHES.txt"
    hashes = hashes_path.read_text() if hashes_path.exists() else ""

    print(f"contract: mlx-affine, bits={BITS}, group_size={GROUP_SIZE}; "
          f"shipped K={K_SHIP} (ADR 0012, three-class nine-name floor)")
    print(f"source: {provenance}")
    print("serving shapes: "
          + ", ".join(f"{n} {d_out}x{d_in}" for n, (d_out, d_in) in shapes))

    try:
        session = DeviceMemberSession()
    except Exception as error:  # no PyObjC Metal, no GPU: nothing to calibrate
        print(f"no usable Metal device for the device members: {error}")
        return 2

    if args.probe_validate:
        probe = probe_batch_regimes([("kv_proj", (1024, 2560))], session)
        _print_probe(probe)
        return 0

    # -- step 0: the continuity anchor --------------------------------------
    ok, detail = continuity_anchor(session)
    print(f"\nSTEP 0 continuity anchor at {CONTINUITY_SHAPE}, bits={BITS}: "
          f"{'REPRODUCED - ' + detail if ok else 'FAILED - ' + detail}")
    if not ok:
        print("STOP: the pipeline does not reproduce ADR 0012's records; "
              "toolchain drift is a finding to raise with the coordinator, "
              "and nothing new is measured on top of it")
        return 1

    # -- steps 1+2: G0 on the probe artefacts, then the batch-regime probe --
    print("\nSTEP 2 batch-regime probe (G0 checked on each probe artefact):")
    probe = probe_batch_regimes(shapes, session)
    _print_probe(probe)
    if not all(p.get("g0_exact") for p in probe.values()):
        print("STOP: G0 failed on a probe artefact; nothing here describes MLX")
        return 1
    joins = sorted({b for p in probe.values() for b in p["joins"]})
    batches = tuple(sorted(set(BASE_BATCHES) | set(joins)))
    print(f"  grid batches: base {BASE_BATCHES}"
          + (f" + joined {joins} -> {batches}" if joins else " (no joins)"))

    # -- step 3: the main grid ----------------------------------------------
    print("\nSTEP 3 main grid:")
    measured = measure_serving(shapes, batches, session, not args.quiet)
    records = measured["records"]
    print(f"  G0: {'bit-exact on all' if measured['bit_exact'] else 'FAILED'} "
          f"({measured['exact_checks']} checks)")
    if not measured["bit_exact"]:
        print("STOP: canonical_quantize is not bit-exact against mx.quantize "
              "at a serving shape; nothing here describes MLX")
        return 1

    # -- step 4: K demand ---------------------------------------------------
    demands = demand_table(records)
    print(f"\nSTEP 4 K demand vs shipped K = {K_SHIP} "
          f"(per cell; calibration / independent):")
    misses = []
    for (shape, batch), d in demands["cells"].items():
        flag = "" if d["covered"] else "  <- DEMAND ABOVE SHIPPED K"
        if not d["covered"]:
            misses.append((shape, batch))
        print(f"  {shape:>13} B{batch:<3} {d['k_needed_calibration']:6.3f} / "
              f"{d['k_needed_independent']:6.3f}{flag}")
    p = demands["pooled"]
    print(f"  {'pooled':>13}      {p['k_needed_calibration']:6.3f} / "
          f"{p['k_needed_independent']:6.3f}")
    print(f"  pooled binding (calibration): {p['k_binding']}")
    print(f"  pooled binding (independent): {p['k_binding_independent']}")

    # -- step 5: gates at the shipped K -------------------------------------
    from calibrate_quant_device import CLASSES
    cells = cell_reports(records, K_SHIP)
    pooled_gates = standing_gates(records, K_SHIP)
    header = (f"{'cell':>18}{'records':>9}{'G1 FP':>7}{'G2 worst margin':>17}"
              f"{'G3 classes':>12}{'verdict':>13}")
    print()
    print("=" * len(header))
    print(f"STEP 5 GATES AT K = {K_SHIP}, WIDTH-POOLED FAULT EQUIVALENCE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    cell_count = {}
    for r in records:
        key = (r["shape"], r["batch"])
        cell_count[key] = cell_count.get(key, 0) + 1
    for (shape, batch), rep in sorted(cells.items()):
        sizes = "/".join(str(rep["class_size"][c]["distinct"]) for c in CLASSES)
        print(f"{shape:>13} B{batch:<4}{cell_count[(shape, batch)]:>9}"
              f"{len(rep['false_positives']):>7}{rep['worst_margin']:>16.1f}x"
              f"{sizes:>12}{rep['verdict']:>13}")
    sizes = "/".join(str(pooled_gates["class_size"][c]["distinct"])
                     for c in CLASSES)
    print(f"{'pooled':>18}{len(records):>9}"
          f"{len(pooled_gates['false_positives']):>7}"
          f"{pooled_gates['worst_margin']:>16.1f}x{sizes:>12}"
          f"{pooled_gates['verdict']:>13}")
    print("=" * len(header))
    for line in pooled_gates["duplicates"]:
        print(f"  value-duplicate members: {line}")
    for line in pooled_gates["false_positives"]:
        print(f"  FP {line}")
    if pooled_gates["equivalent_faults"]:
        print(f"  width-level equivalent faults: "
              f"{pooled_gates['equivalent_faults']}")

    # -- step 6: diagnostic and boundary, pooled ----------------------------
    print("\nSTEP 6 class-necessity diagnostic (ratios above 1 mean "
          "load-bearing, never a gate):")
    for class_name, c in pooled_gates["class_out"].items():
        print(f"  drop {class_name:<18} {c['remaining']} members left, "
              f"worst ratio {c['worst_ratio']:7.3f}, {c['failures']} exceed")
    b = pooled_gates["boundary"]
    print("fp16-dequant boundary, cases outside tolerance "
          "(CPU floor -> device-joined floor):")
    for dtype_name in ("float32", "float16"):
        d = b[dtype_name]
        print(f"  {dtype_name}: {d['cpu_floor']}/{d['of']} -> "
              f"{d['full_floor']}/{d['of']}")

    # -- verdict and records -------------------------------------------------
    attested = (not misses and demands["pooled"]["covered"]
                and all(rep["verdict"] == "ADEQUATE" for rep in cells.values())
                and pooled_gates["verdict"] == "ADEQUATE")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "k_ship": K_SHIP, "bits": BITS,
        "provenance": provenance, "pinned_hashes": hashes,
        "shapes": {n: list(s) for n, s in shapes},
        "base_batches": list(BASE_BATCHES), "grid_batches": list(batches),
        "probe": probe,
        "demands": {"cells": {f"{s}|B{b}": v for (s, b), v
                              in demands["cells"].items()},
                    "pooled": demands["pooled"]},
        "cells": {f"{s}|B{b}": rep for (s, b), rep in cells.items()},
        "pooled_gates": pooled_gates,
        "records": records,
    }, default=float))
    print(f"\nrecords: {OUT_PATH}")

    if misses:
        print(f"DEMAND MISS at {misses}: per the pre-registered branch, "
              f"interpretation STOPS here - membership before K, renegotiated "
              f"with the coordinator, never resolved inside this harness")
        return 1
    print(f"VERDICT: {'K=4 COVERS THE SERVING SHAPES - attestation holds' if attested else 'INADEQUATE somewhere - see the tables'}")
    return 0 if attested else 1


def _print_probe(probe: dict) -> None:
    for name, p in probe.items():
        if not p.get("g0_exact"):
            print(f"  {name}: G0 FAILED on the probe artefact")
            continue
        components = p["components"]
        weak = p["weak_prefix_covered"]
        print(f"  {name}: {len(components)} regime component(s) "
              f"{[f'{min(c)}..{max(c)}' for c in components]}, "
              f"joins {p['joins'] or 'none'}, "
              f"weak-prefix covered {weak or 'none'}")


if __name__ == "__main__":
    raise SystemExit(main())
