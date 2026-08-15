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

AMENDMENT, 2026-08-15: surviving STEP 2 (coordinator-authorized)
---------------------------------------------------------------
The rule above is unchanged; what follows changes only what the harness
SPENDS and what it REPORTS, and it changes no measured value. Three runs
died inside STEP 2, the third under instrumentation: exit 137, SIGKILL, with
RSS already 2.4 GB early in the step and no log line naming the shape that
was live. Authorized by the coordinator and recorded here before the rerun:

1. STEP 2 announces every shape and every (dtype, implementation) as it
   starts and finishes, each line carrying an RSS self-report (current and
   peak), and each shape's report carries ``rss_gb`` into the JSON. The
   memory hypothesis is then confirmed or refuted by the next run's log
   rather than argued about.

2. A per-step checkpoint at ``.cache/quant_serving_partial.json``, written
   atomically (temp then replace, so a SIGKILL mid-write cannot leave a file
   the next run chokes on) after each finished step, and after each (shape,
   draw) block inside the main grid. ``--resume`` reuses finished steps, at
   STEP granularity only - the pinned rng consumption order makes a mid-step
   restart unsound - and every reused step is named in the output JSON and
   on stdout, because a resumed run has not re-measured what it reports.

3. The fp64 reference is memory-bounded by ROW-CHUNKING THE DEQUANTIZATION,
   and it is now built lazily, on first read. Two separate findings forced
   this exact shape:

   (a) ``dequantize`` holds three whole-matrix arrays at once - the cast of
       q, the product, the sum - so at lm_head an fp64 dequant peaks near
       9.3 GB to return 3.1 GB, and the fp32 dequants near 4.7 GB to return
       1.6 GB, six fault artefacts included. Every element of a dequant is
       an independent elementwise expression of its own row's q, scale and
       bias: no reduction, no BLAS. A row block therefore computes exactly
       the bits the whole matrix computes, at EVERY block size, which
       ``test_chunked_dequant_is_bit_identical`` proves across dtypes, bit
       widths and seven chunk sizes (the lut gather and the fp16 round trip
       are per-row for the same reason and carry the same tests).

   (b) The amendment as first proposed row-chunked the fp64 reference
       MATMUL, on the argument that each output element's dot product is
       unchanged by an output-row partition. That argument is true of the
       arithmetic and FALSE of this machine: measured on numpy 2.5.2 over
       Accelerate, cutting the output-row dimension of ``x @ w64.T`` changes
       dgemm's blocking and moves results by up to 2e-14, at chunk sizes 1,
       7 and 3276, at shapes from (5, 128, 96) to (16, 2560, 20000). The
       shift is numerically negligible and disciplinarily fatal: it would
       have moved the fp64 anchor every error in this harness is measured
       against, silently, and STEP 0's continuity anchor could not have
       caught it because the standing measure never takes the chunked path.
       The matmul is therefore left whole - one call, exactly the standing
       one - and ``test_reference_matmul_is_never_row_partitioned`` pins it
       at a shape where a partitioned matmul demonstrably differs.

   Lazy construction is the other half: STEP 2 never reads the fp64
   reference, so at lm_head the probe was spending 3.1 GB resident (and a
   9.3 GB transient) on a matrix it does not use. STEP 3 is the step that
   DOES read it: eval_ref forces the build on the first record of each
   (shape, draw, seed) iteration and w64 stays resident for that whole
   iteration - 3.1 GB at lm_head - which the isolation amendment below
   bounds by ending the iteration's process.

AMENDMENT, 2026-08-15 (second): memory-truthfulness - budget, lock, isolation
-----------------------------------------------------------------------------
The first amendment made the run report and checkpoint honestly; it did not
stop the machine from collapsing. Three Jetsam kills in one night (66.7,
69.4 and 39.5 GB Python footprints on a 36 GB machine, all inside STEP 2)
force three protections, none of which changes a measured value:

1. FOOTPRINT BUDGET. The budget reads phys_footprint via proc_pid_rusage
   (ctypes) - the number Jetsam kills on; one dead run self-reported 2.4 GB
   RSS while dying at 39.5 GB footprint, so RSS survives as a secondary
   print only. Between cells, and between records inside a child, footprint
   over the CLI-overridable ``--budget-gb`` (default 24, well under machine
   RAM) means REFUSAL: checkpoint written, live cell named, distinct exit
   code. Never a silently shrunk grid.

2. MACHINE-GLOBAL LOCK + AVAILABLE-MEMORY GATE. One live instance of this
   harness per machine, keyed to the harness identity rather than the
   worktree (a stale dead-pid lock is reclaimed), and a machine-available-
   memory refusal (kern.memorystatus_level - Jetsam's own meter) before
   each cell. Distinct exit codes for each gate.

3. CHILD-PROCESS ISOLATION (ruling D3). Every measurement iteration -
   continuity, one probe shape, one (shape, draw, seed) grid block - runs
   in a child process; the iteration is rng-self-contained, so the records
   are identical by construction (proven bit-exact across the boundary in
   tests). The OS reclaims everything at child exit, bounding known AND
   unknown leak classes; the parent streams the child's log, enforces the
   budget on the child's self-reported footprint, and a child death marks
   the cell in the checkpoint and refuses onward. The DeviceMemberSession
   re-initializes per child (measured ~0.2 s; ruled acceptable).

AMENDMENT, 2026-08-15 (third): what a resume and a lock actually cover
-----------------------------------------------------------------------
The merge review of the second amendment found two of its protections were
narrower than they read. Neither changes a measured value; both change what
the harness is allowed to skip or share.

1. THE FINGERPRINT CARRIES CODE IDENTITY. ``checkpoint_fingerprint`` covered
   the question being asked (bits, K, provenance, batches, shapes) and
   nothing about the code answering it, so the run of 2026-08-15 resumed past
   STEP 0 - this pipeline's only end-to-end bit-exactness gate against a
   change in device arithmetic - on a continuity result recorded before the
   runner's buffer pool existed. The fingerprint now includes a sha256 over
   the modules a record's value passes through (``CODE_IDENTITY_PATHS``), so
   an edit to any of them invalidates a resume and STEP 0 is re-measured.
   ``--continuity-only`` runs that step alone, which is the check to run
   after any change to the arithmetic path.

2. ONE MACHINE-WIDE MEASUREMENT LOCK, taken with ``fcntl.flock``. The lock
   was keyed to THIS harness, so the pricing probe - a different harness -
   took a different lock and could run beside it, which is precisely the
   03:29 two-large-Pythons collapse the lock exists to prevent. It now lives
   in ``bench/machine_state.py`` as one lock every heavy harness shares. It
   is a flock rather than a pid file because the kernel releases it on death
   however the process died (Jetsam included), which deletes the stale-pid
   judgement and, with it, the race that let a reader landing between the
   file's creation and the pid's write call a LIVE holder stale and steal
   its lock.

Machine discipline
------------------
One measuring lane at a time: the full run executes only in the
coordinator-granted GPU slot, after the pack lane's boundary pricing. The
pre-slot allowance is the probe validated at (1024, 2560) plus unit-test
smoke dispatches. Reruns must reproduce ADR 0013's tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
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
CHECKPOINT_PATH = Path(__file__).with_name(".cache") / "quant_serving_partial.json"
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
    FAULTS,
    QuantArtefact,
    QuantContract,
)


def _f32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)


# The 2026-08-15 memory amendment. ``dequantize`` holds three whole-matrix
# arrays at once - the cast of q, the product, the sum - so at lm_head an
# fp64 dequant peaks near 9.3 GB to return 3.1 GB. Every element of it is an
# independent elementwise expression of its own row's q, scale and bias: no
# reduction, no BLAS. Row blocks therefore compute exactly the bits the whole
# matrix computes, at every block size, and the transient becomes proportional
# to a block. Bit-identity is proven, not assumed, in tests/test_serving_adequacy.py.
DEQUANT_CHUNK_BYTES = 1 << 27  # ~134 MB per row block of the working array


def dequant_chunk_rows(d_in: int, itemsize: int) -> int:
    return max(1, DEQUANT_CHUNK_BYTES // max(1, d_in * itemsize))


def _row_blocks(rows: int, chunk_rows: int):
    for start in range(0, rows, chunk_rows):
        yield start, min(start + chunk_rows, rows)


def dequant_chunked(a: QuantArtefact, dtype,
                    chunk_rows: int | None = None) -> np.ndarray:
    """``dequantize`` evaluated in output-row blocks, bit-identical to it."""
    g = a.contract.group_size
    rows, cols = a.q.shape
    if chunk_rows is None:
        chunk_rows = dequant_chunk_rows(cols, np.dtype(dtype).itemsize)
    out = np.empty((rows, cols), dtype=dtype)
    for start, stop in _row_blocks(rows, chunk_rows):
        block_rows = stop - start
        q = a.q[start:stop].reshape(block_rows, cols // g, g).astype(dtype)
        block = (a.scales[start:stop].astype(dtype)[:, :, None] * q
                 + a.biases[start:stop].astype(dtype)[:, :, None])
        out[start:stop] = block.reshape(block_rows, cols)
    return out


def lut_gather_chunked(a: QuantArtefact,
                       chunk_rows: int | None = None) -> np.ndarray:
    """member_lut_gather's weights in row blocks; the per-group table is a
    function of the row's own scale and bias, so the gather chunks exactly."""
    g = a.contract.group_size
    rows, cols = a.q.shape
    levels = np.arange(1 << a.contract.bits, dtype=np.float32)
    if chunk_rows is None:
        chunk_rows = dequant_chunk_rows(cols, np.dtype(np.float32).itemsize)
    out = np.empty((rows, cols), dtype=np.float32)
    for start, stop in _row_blocks(rows, chunk_rows):
        block_rows = stop - start
        table = (a.scales[start:stop].astype(np.float32)[:, :, None]
                 * levels[None, None, :]
                 + a.biases[start:stop].astype(np.float32)[:, :, None])
        q = a.q[start:stop].reshape(block_rows, cols // g, g)
        block = np.take_along_axis(table, q, axis=2)
        out[start:stop] = block.reshape(block_rows, cols)
    return out


def f16_roundtrip_chunked(w32: np.ndarray,
                          chunk_rows: int | None = None) -> np.ndarray:
    """The fp16-dequant boundary's weights: two elementwise casts, so chunked."""
    if chunk_rows is None:
        chunk_rows = dequant_chunk_rows(w32.shape[1], w32.dtype.itemsize)
    out = np.empty_like(w32)
    for start, stop in _row_blocks(w32.shape[0], chunk_rows):
        out[start:stop] = w32[start:stop].astype(np.float16).astype(np.float32)
    return out


class ArtefactHoists:
    """Per-(draw, seed) precomputation, so the lm_head shape pays for each
    pure function of the artefact once instead of once per record.

    Fault dequants are deliberately NOT stored here: six extra fp32 weight
    matrices would add ~9 GB at lm_head, so ``fault_dequants`` yields them
    one at a time instead. The fp64 reference is deferred for the same reason
    - it is the single largest array here and STEP 2 never reads it. STEP 3
    DOES read it: ``eval_ref`` forces the build on the first record and the
    reference then stays resident for the rest of that (shape, draw, seed)
    iteration - 3.1 GB at lm_head - which is part of why each iteration runs
    in its own child process, whose exit is what hands the memory back.
    """

    def __init__(self, artefact: QuantArtefact):
        rows, cols = artefact.q.shape
        g = artefact.contract.group_size
        self._artefact = artefact
        self._w64 = None
        self.w32 = dequant_chunked(artefact, np.float32)
        self.w_lut32 = lut_gather_chunked(artefact)
        # the int-domain members' operands
        self.qg32 = artefact.q.reshape(rows, cols // g, g).astype(np.float32)
        self.scales32 = artefact.scales.astype(np.float32)
        self.biases32 = artefact.biases.astype(np.float32)
        # the fp16-dequant boundary's weights
        self.w16_32 = f16_roundtrip_chunked(self.w32)

    @property
    def reference_built(self) -> bool:
        return self._w64 is not None

    @property
    def w64(self) -> np.ndarray:
        """The fp64 reference weights, built on first read. At lm_head they are
        3.1 GB and the batch-regime probe never asks for them, so building them
        with the rest spent the probe's whole budget on a matrix it never reads."""
        if self._w64 is None:
            self._w64 = dequant_chunked(self._artefact, np.float64)
        return self._w64


def fault_dequants(artefact: QuantArtefact):
    """(name, fp32 dequant of the faulted artefact), one at a time, in
    catalogue order; the consumer frees each before the next is built."""
    for name, fault in FAULTS.items():
        yield name, dequant_chunked(fault(artefact), np.float32)


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
# Run survival: the RSS self-report and the per-step checkpoint. Three runs
# died at STEP 2 with exit 137 (SIGKILL) and no log past the step header, so
# the run reports its own footprint as it goes and records each finished step
# before starting the next.
# ---------------------------------------------------------------------------
def rss_gb() -> tuple[float, float]:
    """(current, peak) resident set size in GB.

    ``ru_maxrss`` is bytes on Darwin and kilobytes elsewhere; current RSS has
    no stdlib reader, so it comes from ``ps``. Current separates a transient
    spike from a leak across shapes, peak says how close the spike came to the
    machine's limit, and only the pair answers the memory hypothesis.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / 1e9 if sys.platform == "darwin" else peak / 1e6
    try:
        ps = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                            capture_output=True, text=True, timeout=10)
        current_gb = int(ps.stdout.strip()) / 1e6  # ps reports kilobytes
    except (OSError, ValueError, subprocess.SubprocessError):
        current_gb = peak_gb
    return current_gb, peak_gb


def rss_line() -> str:
    current, peak = rss_gb()
    return f"rss {current:.2f} GB, peak {peak:.2f} GB"


def footprint_line() -> str:
    """The budget's own number first; RSS stays a secondary print only."""
    current, peak = phys_footprint_gb()
    return f"footprint {current:.2f} GB (peak {peak:.2f} GB); {rss_line()}"


# The modules a measured record's value passes through: the harness itself,
# the runner that dispatches on the GPU, the device members and the contract
# they implement, and the three standing bench modules whose functions this
# harness calls to produce and score records. A change to any of them can
# move a number, so a change to any of them must invalidate a resume.
CODE_IDENTITY_PATHS = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parents[1] / "kernelverify" / "runners" / "device.py",
    Path(__file__).resolve().parents[1] / "kernelverify" / "runners" / "spec.py",
    Path(__file__).resolve().parents[1] / "kernelverify" / "schemas" / "quant_device.py",
    Path(__file__).resolve().parents[1] / "kernelverify" / "schemas" / "quant_contract.py",
    Path(__file__).with_name("calibrate_quant_device.py"),
    Path(__file__).with_name("calibrate_quant_bits.py"),
    Path(__file__).with_name("phase0_contract_k.py"),
)


def code_identity() -> str:
    """A sha256 over the source of the code that produces the records.

    CONTENT hashes rather than the git HEAD sha, deliberately. HEAD is blind
    to an uncommitted edit, which is the state a lane actually measures in,
    and it moves on commits that touch nothing numeric, which would throw
    away a 40-minute step for a docstring. Content hashes bite exactly when
    the arithmetic path changes and never otherwise. A file that cannot be
    read raises here: a fingerprint that cannot see its own code must stop,
    not guess.
    """
    digest = hashlib.sha256()
    for path in CODE_IDENTITY_PATHS:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_fingerprint(provenance: str, shapes) -> dict:
    """What a checkpoint must agree with before a --resume may reuse it.

    Resuming across a different artifact or shape set would splice two
    different calibrations into one attestation. So would resuming across a
    change in the measuring CODE, and that one actually happened: on
    2026-08-15 a run resumed past STEP 0 - the only end-to-end bit-exactness
    gate this pipeline has against a change in device arithmetic - reusing a
    continuity result recorded before the runner's buffer pool existed,
    because the fingerprint described the question and nothing about the code
    answering it.
    """
    return {"bits": BITS, "k_ship": K_SHIP, "provenance": provenance,
            "base_batches": list(BASE_BATCHES),
            "shapes": {name: list(shape) for name, shape in shapes},
            "code": code_identity()}


def write_checkpoint(path: Path, fingerprint: dict, steps: dict) -> None:
    """Atomic: a SIGKILL landing mid-write must not leave a half-written file
    that the next run then fails to parse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"fingerprint": fingerprint, "steps": steps},
                              default=float))
    tmp.replace(path)


def read_checkpoint(path: Path, fingerprint: dict) -> dict:
    """The steps a --resume may reuse; empty when there is nothing sound to
    reuse, so an absent or truncated file is never a crash."""
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if saved.get("fingerprint") != fingerprint:
        return {}
    return saved.get("steps", {})


# ---------------------------------------------------------------------------
# The 2026-08-15 memory-truthfulness amendment: budget, lock, isolation.
# Three Jetsam kills in one night (66.7, 69.4 and 39.5 GB Python footprints on
# a 36 GB machine) - the harness now refuses over a footprint budget instead
# of letting Jetsam collapse the machine. All sizes are decimal GB (1e9),
# matching rss_gb above.
#
# The vocabulary itself - exit codes, the ctypes footprint reader, the budget
# and low-memory guards - is single-sourced from bench/memory_guard.py
# (ruling D1: every heavy harness imports it, none copies it), the same way
# the machine-wide lock is single-sourced from bench/machine_state.py.
# ---------------------------------------------------------------------------
from memory_guard import (  # noqa: E402
    EXIT_BUDGET_REFUSAL,
    EXIT_CHILD_DEATH,
    EXIT_LOCK_HELD,
    EXIT_LOW_MEMORY,
    EXIT_NO_DEVICE,
    BudgetExceeded,
    BudgetGuard,
    LowMemoryRefusal,
    available_memory_gb,
    budget_gb_arg,
    machine_ram_gb,
    phys_footprint_gb,
    positive_float_arg,
    require_available_memory,
)

# Well under the machine's 38.7 decimal-GB (36 GiB) of unified memory: the
# probes ruled 20-24 GB, and the default leaves the OS and the coordinator's
# other lanes ~15 GB of headroom.
DEFAULT_BUDGET_GB = 24.0


def refuse(code: int, reason: str, fingerprint: dict, steps: dict) -> int:
    """The shared refusal path: keep the checkpoint whole on disk, say what
    refused, exit with that gate's own code. OUT_PATH is never written here -
    the results JSON means a finished run and nothing less.

    Non-destructive by construction: finished steps already on disk under the
    SAME fingerprint are kept even when this invocation had not (yet) loaded
    them, so a refusal - resumed or not, however early - never costs a step.
    """
    kept = {**read_checkpoint(CHECKPOINT_PATH, fingerprint), **steps}
    write_checkpoint(CHECKPOINT_PATH, fingerprint, kept)
    print(f"REFUSAL (exit {code}): {reason}")
    print(f"checkpoint intact at {CHECKPOINT_PATH}; rerun with --resume once "
          f"the machine has room")
    return code


def _wall_cap_arg(text: str) -> float:
    return positive_float_arg(text, "wall cap", "seconds")


# The machine-wide measurement lock, shared with every other heavy harness
# through bench/machine_state.py rather than owned here. A lock keyed to THIS
# harness would have let the 03:29 collapse happen exactly as it did: the
# second large Python that night was the pricing probe, a different harness,
# which would have taken a different lock and run anyway.
from machine_state import (  # noqa: E402
    MEASUREMENT_LOCK_PATH,
    MeasurementLock,
)


# ---------------------------------------------------------------------------
# Child-process isolation (ruling D3). Each measurement iteration runs in a
# child process; the OS reclaims EVERYTHING at child exit, bounding the known
# leak classes and the unknown ones alike. The parent spawns, streams the
# child's log through (it inherits stdout/stderr - a foreground run stays a
# foreground run), reads the result file, and enforces the budget on the
# child's own footprint. In-process build-use-free surgery is deliberately
# not attempted.
# ---------------------------------------------------------------------------
CHILD_DIR = Path(__file__).with_name(".cache")  # durable, never /tmp
_CHILD_TASK_NAME = "quant_serving_child_task.json"
_CHILD_RESULT_NAME = "quant_serving_child_result.json"


def write_child_result(path: Path, result: dict) -> None:
    """Atomic (temp then replace), bit-exact for every float.

    json emits shortest-roundtrip decimal for finite doubles, which parses
    back to the identical bits (including -0.0 and subnormals); numpy scalars
    are widened by ``default=float``, exact for float64 and an exact widening
    for float32/16. NaN and +/-Infinity policy, explicitly: ``allow_nan``
    stays on, so json writes the ``NaN``/``Infinity`` literals and Python's
    parser restores them - err() on an overflowed fp16 fault path survives
    the boundary unchanged. A silent drift here would poison the continuity
    anchor, which compares records exactly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(result, default=float))
    tmp.replace(path)


def read_child_result(path: Path):
    """The child's result, or None when there is nothing sound to read - an
    absent or half-written file means the child died, never a crash here."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class ChildRefusal(RuntimeError):
    """A measurement child ended without a result. The parent marks the cell
    and refuses onward - a dead child means the machine or the cell is not
    safe to keep measuring."""

    def __init__(self, cell: str, exit_for_parent: int, reason: str):
        super().__init__(f"child {reason} at cell {cell}")
        self.cell = cell
        self.exit_for_parent = exit_for_parent
        self.reason = reason


def _child_death_reason(returncode: int) -> str:
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = "?"
        return f"died on signal {-returncode} ({name})"
    return f"died with exit {returncode}"


def spawn_measurement(cell: str, task: dict, budget_gb: float,
                      wall_cap_s: float | None = None,
                      command: list | None = None) -> tuple[dict, dict]:
    """Run one measurement iteration in a child process; (payload, meta),
    where meta carries the child's self-reported ``footprint_gb`` (and its
    mx numbers when present).

    The child inherits stdout/stderr so its progress lines land in the same
    foreground log. ``command`` is injectable for tests; production spawns
    this very file under the same interpreter. Raises ChildRefusal when the
    child dies, self-refuses, or has no Metal device, and BudgetExceeded when
    the child's own reported peak footprint crossed the budget - a child that
    survived while exceeding it would just die next time.
    """
    task_path = CHILD_DIR / _CHILD_TASK_NAME
    result_path = CHILD_DIR / _CHILD_RESULT_NAME
    task_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.unlink(missing_ok=True)
    task_path.write_text(json.dumps({**task, "cell": cell,
                                     "budget_gb": budget_gb}))
    argv = list(command) if command is not None else [
        sys.executable, str(Path(__file__).resolve())]
    argv += ["--child-task", str(task_path), "--child-out", str(result_path)]
    try:
        proc = subprocess.run(argv, timeout=wall_cap_s)
    except subprocess.TimeoutExpired:
        raise ChildRefusal(cell, EXIT_CHILD_DEATH,
                           f"hit the {wall_cap_s:.0f}s wall cap")
    finally:
        task_path.unlink(missing_ok=True)
    if proc.returncode == EXIT_BUDGET_REFUSAL:
        raise ChildRefusal(cell, EXIT_BUDGET_REFUSAL,
                           "refused over its own footprint budget")
    if proc.returncode == EXIT_NO_DEVICE:
        raise ChildRefusal(cell, EXIT_NO_DEVICE, "found no usable Metal device")
    if proc.returncode != 0:
        raise ChildRefusal(cell, EXIT_CHILD_DEATH,
                           _child_death_reason(proc.returncode))
    result = read_child_result(result_path)
    result_path.unlink(missing_ok=True)
    if result is None:
        raise ChildRefusal(cell, EXIT_CHILD_DEATH,
                           "exited 0 but left no result file")
    peak = result.get("footprint_gb", {}).get("peak", 0.0)
    if peak > budget_gb:
        raise BudgetExceeded(cell, peak, budget_gb)
    return result["payload"], {k: v for k, v in result.items()
                               if k != "payload"}


def refuse_child(exc: ChildRefusal, fingerprint: dict, steps: dict) -> int:
    """A child refusal, through the shared refusal path, with the cell MARKED
    in the checkpoint: the next run (or the coordinator) reads which cell
    died without parsing the log. The mark lives beside the finished steps
    and the next run's first own checkpoint write drops it."""
    marked = {**steps, "refused_cell": {"cell": exc.cell, "reason": exc.reason,
                                        "exit": exc.exit_for_parent}}
    return refuse(exc.exit_for_parent, str(exc), fingerprint, marked)


def child_main(args) -> int:
    """The child half of the isolation protocol: run ONE measurement
    iteration, self-guard the budget between records, self-report footprint
    and mx memory, write the result atomically, exit. No lock (the parent
    holds it), no checkpoint (the parent owns it), fresh DeviceMemberSession
    per child by design (seconds of compile, ruled acceptable under D3)."""
    task = json.loads(Path(args.child_task).read_text())
    guard = BudgetGuard(task["budget_gb"])
    cell = task.get("cell", task["kind"])
    verbose = task.get("verbose", False)
    print(f"  child {cell}: start ({footprint_line()})", flush=True)
    try:
        session = DeviceMemberSession()
    except Exception as error:  # no PyObjC Metal, no GPU: nothing to measure
        print(f"  child {cell}: no usable Metal device: {error}", flush=True)
        return EXIT_NO_DEVICE
    try:
        if task["kind"] == "continuity":
            ok, detail = continuity_anchor(session)
            payload = {"ok": ok, "detail": detail}
        elif task["kind"] == "probe":
            report = probe_batch_regimes(
                [(task["name"], (task["d_out"], task["d_in"]))], session,
                verbose=verbose, guard=guard)
            payload = report[task["name"]]
        elif task["kind"] == "grid":
            payload = grid_iteration(
                task["name"], task["d_out"], task["d_in"], task["draw"],
                task["seed"], tuple(task["batches"]), session, guard=guard,
                verbose=verbose)
        else:
            raise ValueError(f"unknown child task kind {task['kind']!r}")
    except BudgetExceeded as exc:
        print(f"  child {cell}: BUDGET REFUSAL - {exc}", flush=True)
        return EXIT_BUDGET_REFUSAL
    current, peak = phys_footprint_gb()
    write_child_result(Path(args.child_out), {
        "payload": payload,
        "footprint_gb": {"current": current, "peak": peak},
        "mx_gb": {"cache": mx.get_cache_memory() / 1e9,
                  "peak": mx.get_peak_memory() / 1e9},
    })
    print(f"  child {cell}: done ({footprint_line()})", flush=True)
    return 0


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
def probe_batch_regimes(shapes, session: DeviceMemberSession,
                        verbose: bool = True,
                        guard: BudgetGuard | None = None) -> dict:
    """The pre-registered batch-regime probe at every serving shape.

    Evaluates one (implementation, dtype) at a time across B = 1..16 and ANDs
    its pairwise matrix into the joint one, bounding residency at lm_head.

    Every stage announces itself with its RSS: this step died three times
    under SIGKILL with nothing in the log past the header, so the last line
    printed has to name the shape and the implementation that was live.
    ``guard`` (when given) is checked before each shape starts, so an
    over-budget process refuses before the next cell rather than mid-way.
    """
    contract = QuantContract(scheme="mlx-affine", bits=BITS, group_size=GROUP_SIZE)
    report = {}
    for name, (d_out, d_in) in shapes:
        if guard is not None:
            guard.check(f"probe {name} {d_out}x{d_in}")
        if verbose:
            print(f"  {name} {d_out}x{d_in}: quantizing ({rss_line()})",
                  flush=True)
        rng = np.random.default_rng(10_000 + PROBE_SEED)
        w = make_w(d_out, d_in, "normal-0.02", rng)
        artefact, exact = verify_against_mlx(w, contract)
        if not exact:
            report[name] = {"g0_exact": False}
            continue
        hoists = ArtefactHoists(artefact)
        carrier = _carrier(artefact)
        w_q, scales, biases = mlx_quantize_hoist(w)
        if verbose:
            print(f"    hoists + carrier built ({rss_line()})", flush=True)
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
                if verbose:
                    print(f"    {dtype_name} {impl_name} done ({rss_line()})",
                          flush=True)
        current, peak = rss_gb()
        report[name] = {
            "g0_exact": True,
            "matrix": matrix.tolist(),
            "components": [sorted(c) for c in match_components(matrix)],
            "joins": coverage_joins(matrix, BASE_BATCHES),
            "weak_prefix_covered": weak_prefix_cover(matrix, BASE_BATCHES),
            "rss_gb": {"current": current, "peak": peak},
        }
        del hoists, carrier, w_q, scales, biases
    return report


# ---------------------------------------------------------------------------
# Step 3: the main grid
# ---------------------------------------------------------------------------
def grid_iteration(name: str, d_out: int, d_in: int, draw: str, seed: int,
                   batches, session: DeviceMemberSession,
                   guard: BudgetGuard | None = None,
                   verbose: bool = False) -> dict:
    """One (shape, draw, seed) block of the main grid: the isolation unit.

    Rng-self-contained by construction - fresh ``default_rng(10_000 + seed)``,
    stream order pinned (make_w first, then make_x per (batch, mode, dtype),
    batch outermost) - which is exactly the property that makes running it in
    a child process sound without renumbering anything. Records are
    JSON-serializable pure-Python values throughout. ``guard`` is checked at
    the start and between records, so an over-budget process refuses at a
    boundary instead of meeting Jetsam mid-record.
    """
    if guard is not None:
        guard.check(f"grid {name} {d_out}x{d_in} {draw} s{seed}")
    contract = QuantContract(scheme="mlx-affine", bits=BITS, group_size=GROUP_SIZE)
    rng = np.random.default_rng(10_000 + seed)
    w = make_w(d_out, d_in, draw, rng)
    artefact, exact = verify_against_mlx(w, contract)
    xs = {}
    for batch in batches:
        for mode in MODES:
            for dtype_name in ("float32", "float16"):
                xs[(batch, mode, dtype_name)] = make_x(
                    batch, d_in, _np_dtype(dtype_name), mode, rng)
    if not exact:
        return {"records": [], "exact": False}
    hoists = ArtefactHoists(artefact)
    carrier = _carrier(artefact)
    w_q, scales, biases = mlx_quantize_hoist(w)
    block = []
    for (batch, mode, dtype_name), x in xs.items():
        label = (f"b{BITS} {name} {d_out}x{d_in} B{batch} {draw} "
                 f"s{seed} {mode} {dtype_name}")
        if guard is not None:
            guard.check(label)
        ref = eval_ref(x, hoists.w64)
        scale = float(np.abs(ref).max())
        chunk = serial_chunk_rows(batch, d_in)
        members = {n: err(out, ref)
                   for n, out in _cpu_members(x, hoists, chunk).items()}
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
        if verbose:
            print(f"    {label} done ({rss_line()})", flush=True)
    for fault_name, w_fault32 in fault_dequants(artefact):
        for record, x, ref in block:
            record["faults"][fault_name] = err(
                eval_pairwise(x, w_fault32), ref)
        del w_fault32
    return {"records": [record for record, _, _ in block], "exact": True}


def measure_serving(shapes, batches, session: DeviceMemberSession,
                    verbose: bool, on_block=None,
                    guard: BudgetGuard | None = None) -> dict:
    """Every record for the serving grid, iterated IN-PROCESS: the standing
    quantities plus the batch axis, one ``grid_iteration`` at a time. The
    production run executes each iteration in a child process instead (same
    ``grid_iteration``, same arithmetic - see the STEP 3 loop in
    ``_run_measured_steps``); this driver is the in-process seam the unit
    tests measure against.

    ``on_block`` receives the records so far after each (shape, draw), so a
    death inside this step leaves the records it already earned on disk. It is
    a recorder only: a resume never restarts mid-step, because the rng stream
    order is pinned across the whole step. ``guard`` (when given) is checked
    at every iteration and record boundary inside grid_iteration."""
    records, exactness = [], []
    for name, (d_out, d_in) in shapes:
        for draw in WEIGHT_DRAWS:
            for seed in CALIBRATION_SEEDS + HELDOUT_SEEDS:
                block = grid_iteration(name, d_out, d_in, draw, seed, batches,
                                       session, guard=guard)
                exactness.append(block["exact"])
                records.extend(block["records"])
            if verbose:
                print(f"    {name} {d_out}x{d_in} {draw} done "
                      f"({len(records)} records, {rss_line()})", flush=True)
            if on_block is not None:
                on_block(records)
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


# ---------------------------------------------------------------------------
# T4 probes: bounded, foreground, wall-capped, budget-guarded. The
# ATTRIBUTION probe reruns the STEP 2 workload's components IN-PROCESS -
# the pre-fix configuration that leaked - instrumented per stage, to convict
# the holder of the unattributed 30-60 GB gap. The STABILITY probe runs the
# post-fix path: the same shape, child-per-iteration, footprints per child.
# Both write durable JSON under .cache after every stage, so even a refusal
# mid-probe leaves the attribution numbers it earned.
# ---------------------------------------------------------------------------
ATTRIBUTION_PATH = Path(__file__).with_name(".cache") / "memfix_probe_attribution.json"
STABILITY_PATH = Path(__file__).with_name(".cache") / "memfix_probe_stability.json"


def _mem_snapshot() -> dict:
    current, peak = phys_footprint_gb()
    return {"footprint_gb": current, "footprint_peak_gb": peak,
            "mx_cache_gb": mx.get_cache_memory() / 1e9,
            "mx_peak_gb": mx.get_peak_memory() / 1e9}


def attribution_probe(name: str, d_out: int, d_in: int, iterations: int,
                      guard: BudgetGuard, deadline: float | None,
                      out_path: Path = ATTRIBUTION_PATH,
                      probe_batch_max: int = PROBE_BATCH_MAX) -> dict:
    """The pre-fix attribution instrument, in-process by design.

    Per iteration of the SAME shape: build the artefact hoists and carrier
    (structural share), one CPU member pass (numpy share), the mx heldout
    pass (mx-visible share), then the 96-call device-member sweep - the
    production dispatch pattern and the PRIME SUSPECT, since each call
    re-uploads the fp16 carrier through PyObjC Metal, invisible to
    mx.get_cache_memory(). After freeing everything, mx.clear_cache()
    separates the mx-cache share from what only the session (or the heap)
    still holds. Footprint and mx numbers are recorded at every stage
    boundary; growth across iterations is the leak the isolation fix bounds.
    """
    session = DeviceMemberSession()
    contract = QuantContract(scheme="mlx-affine", bits=BITS,
                             group_size=GROUP_SIZE)
    report = {"kind": "attribution", "shape": [name, d_out, d_in],
              "budget_gb": guard.budget_gb, "probe_batch_max": probe_batch_max,
              "stages": [], "sweep_delta_gb": [], "iteration_end_gb": [],
              "wall_capped_at": None}

    def note(iteration: int, stage: str) -> dict:
        snap = {"iteration": iteration, "stage": stage,
                "t": time.monotonic() - t0, **_mem_snapshot()}
        report["stages"].append(snap)
        write_child_result(out_path, report)  # atomic, durable, incremental
        print(f"  attr i{iteration} {stage:<24} footprint "
              f"{snap['footprint_gb']:6.2f} GB (peak "
              f"{snap['footprint_peak_gb']:6.2f}), mx cache "
              f"{snap['mx_cache_gb']:5.2f} GB (peak {snap['mx_peak_gb']:5.2f})",
              flush=True)
        return snap

    def capped(iteration: int, stage: str) -> bool:
        if deadline is not None and time.monotonic() > deadline:
            report["wall_capped_at"] = f"iteration {iteration} before {stage}"
            print(f"  attr: wall cap hit before i{iteration} {stage}; "
                  f"stopping with what was measured", flush=True)
            write_child_result(out_path, report)
            return True
        return False

    t0 = time.monotonic()
    note(0, "baseline")
    for i in range(1, iterations + 1):
        if capped(i, "build"):
            break
        guard.check(f"attribution i{i} build {name}")
        rng = np.random.default_rng(10_000 + PROBE_SEED)
        w = make_w(d_out, d_in, "normal-0.02", rng)
        artefact, exact = verify_against_mlx(w, contract)
        if not exact:
            report["g0_exact"] = False
            break
        hoists = ArtefactHoists(artefact)
        carrier = _carrier(artefact)
        w_q, scales, biases = mlx_quantize_hoist(w)
        x16 = {d: make_x(probe_batch_max, d_in, _np_dtype(d), "unit", rng)
               for d in ("float32", "float16")}
        note(i, "built")
        if capped(i, "cpu-pairwise"):
            break
        guard.check(f"attribution i{i} cpu {name}")
        for batch in range(1, probe_batch_max + 1):
            eval_pairwise(x16["float32"][:batch], hoists.w32)
        note(i, "cpu-pairwise")
        if capped(i, "mx-heldout"):
            break
        guard.check(f"attribution i{i} mx {name}")
        for batch in range(1, probe_batch_max + 1):
            mlx_qmm_heldout(x16["float32"][:batch], w_q, scales, biases)
        note(i, "mx-heldout")
        if capped(i, "device-sweep"):
            break
        before = _mem_snapshot()
        calls = 0
        for dtype_name in ("float32", "float16"):
            for member in DEVICE_MEMBERS:
                for batch in range(1, probe_batch_max + 1):
                    guard.check(f"attribution i{i} sweep {member} "
                                f"{dtype_name} B{batch}")
                    _device_member_output(session, member,
                                          x16[dtype_name][:batch], carrier,
                                          d_out, d_in, f"attr {member}")
                    calls += 1
        after = note(i, f"device-sweep-{calls}calls")
        delta = after["footprint_gb"] - before["footprint_gb"]
        report["sweep_delta_gb"].append(delta)
        print(f"  attr i{i} DEVICE-SWEEP DELTA {delta:+.2f} GB footprint, "
              f"{after['mx_cache_gb'] - before['mx_cache_gb']:+.2f} GB mx "
              f"cache ({calls} dispatches)", flush=True)
        del hoists, carrier, w_q, scales, biases, x16, w, artefact
        note(i, "freed")
        mx.clear_cache()
        end = note(i, "mx-cache-cleared")
        report["iteration_end_gb"].append(end["footprint_gb"])
        write_child_result(out_path, report)
    ends = report["iteration_end_gb"]
    if len(ends) >= 2:
        report["growth_per_iteration_gb"] = [
            round(b - a, 3) for a, b in zip(ends, ends[1:])]
        print(f"  attr: end-of-iteration footprints {ends} GB; growth "
              f"{report['growth_per_iteration_gb']} GB/iter", flush=True)
    write_child_result(out_path, report)
    print(f"  attr: written to {out_path}")
    return report


def _attribution_probe_main(args, shapes, guard) -> int:
    by_name = dict(shapes)
    d_out, d_in = by_name[args.probe_shape]
    require_available_memory(args.budget_gb,
                             f"attribution probe {args.probe_shape}")
    deadline = (time.monotonic() + args.wall_cap_s
                if args.wall_cap_s is not None else None)
    print(f"\nATTRIBUTION PROBE (pre-fix configuration, in-process): "
          f"{args.probe_iterations}x {args.probe_shape} {d_out}x{d_in}, "
          f"budget {args.budget_gb:.0f} GB"
          + (f", wall cap {args.wall_cap_s:.0f}s" if args.wall_cap_s else ""),
          flush=True)
    attribution_probe(args.probe_shape, d_out, d_in, args.probe_iterations,
                      guard, deadline)
    return 0


def _stability_probe_main(args, shapes, spawn) -> int:
    """POST-FIX: the same shape through the production child path, N times,
    plus one kv_proj cell; each child's peak footprint is the stability
    series, and the parent's own footprint must stay flat."""
    by_name = dict(shapes)
    runs = ([(args.probe_shape, by_name[args.probe_shape])]
            * args.probe_iterations)
    if args.probe_shape != "kv_proj":
        runs.append(("kv_proj", by_name["kv_proj"]))
    report = {"kind": "stability", "budget_gb": args.budget_gb, "children": []}
    print(f"\nSTABILITY PROBE (post-fix, child-per-iteration): "
          f"{args.probe_iterations}x {args.probe_shape} + kv_proj, "
          f"budget {args.budget_gb:.0f} GB", flush=True)
    for i, (name, (d_out, d_in)) in enumerate(runs, start=1):
        payload, meta = spawn(
            "probe", f"stability i{i} probe {name} {d_out}x{d_in}",
            {"name": name, "d_out": d_out, "d_in": d_in})
        parent = _mem_snapshot()
        report["children"].append({
            "iteration": i, "shape": [name, d_out, d_in],
            "child_footprint_gb": meta.get("footprint_gb", {}),
            "child_mx_gb": meta.get("mx_gb", {}),
            "child_rss_gb": payload.get("rss_gb"),
            "parent_footprint_gb": parent["footprint_gb"]})
        write_child_result(STABILITY_PATH, report)
    peaks = [c["child_footprint_gb"].get("peak") for c in report["children"]]
    print(f"  stability: child peak footprints {peaks} GB; "
          f"parent {footprint_line()}")
    print(f"  stability: written to {STABILITY_PATH}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="3-bit device adequacy at the Qwen3-4B serving shapes")
    parser.add_argument("--probe-validate", action="store_true",
                        help="pre-slot smoke: the probe at kv_proj (1024, 2560) "
                             "only, no continuity run, no main grid")
    parser.add_argument("--continuity-only", action="store_true",
                        help="run STEP 0 (the continuity anchor against the "
                             "ADR 0012 cache) and stop. The check to run "
                             "after ANY change to the arithmetic path, since "
                             "it is the pipeline's only end-to-end "
                             "bit-exactness gate and costs minutes rather "
                             "than the full run's 45")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="reuse the finished steps in the checkpoint left "
                             "by an earlier run of the SAME shapes and "
                             "artifact; step granularity only, and every "
                             "reused step is named in the output JSON")
    parser.add_argument("--budget-gb", type=budget_gb_arg,
                        default=DEFAULT_BUDGET_GB,
                        help="phys_footprint budget in decimal GB; crossing "
                             "it writes the checkpoint, names the live cell "
                             f"and exits {EXIT_BUDGET_REFUSAL} (default "
                             f"{DEFAULT_BUDGET_GB}, well under this machine's "
                             "RAM)")
    parser.add_argument("--wall-cap-s", type=_wall_cap_arg, default=None,
                        help="kill any measurement child running longer than "
                             "this many seconds and refuse onward; probes "
                             "always pass one, the full run defaults to "
                             "uncapped")
    parser.add_argument("--attribution-probe", action="store_true",
                        help="T4 pre-fix attribution: the STEP 2 workload's "
                             "components in-process, instrumented per stage, "
                             "to convict the holder of the unattributed gap; "
                             "bounded by --budget-gb and --wall-cap-s")
    parser.add_argument("--stability-probe", action="store_true",
                        help="T4 post-fix stability: the same shape through "
                             "the production child path, --probe-iterations "
                             "times, plus one kv_proj cell")
    parser.add_argument("--probe-shape", choices=["lm_head", "kv_proj"],
                        default="lm_head",
                        help="which serving shape the T4 probes exercise")
    parser.add_argument("--probe-iterations", type=int, default=3,
                        choices=range(1, 11), metavar="N",
                        help="same-shape iterations for the T4 probes")
    # The child half of the isolation protocol; spawned by this harness only.
    parser.add_argument("--child-task", help=argparse.SUPPRESS)
    parser.add_argument("--child-out", help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.child_task:
        return child_main(args)  # no lock: the parent holds it

    lock = MeasurementLock("calibrate_quant_serving")
    acquired, lock_detail = lock.acquire()
    if not acquired:
        print(f"REFUSAL (exit {EXIT_LOCK_HELD}): machine measurement lock "
              f"{MEASUREMENT_LOCK_PATH} {lock_detail}; one heavy measurement "
              f"per machine - whichever harness it is - and a refused run "
              f"touches nothing")
        return EXIT_LOCK_HELD
    try:
        return _locked_main(args)
    finally:
        lock.release()


def _locked_main(args) -> int:
    config, provenance = resolve_serving_source(ARTIFACT_DIR)
    shapes = derive_serving_shapes(config)
    hashes_path = ARTIFACT_DIR.parent / "PINNED-HASHES.txt"
    hashes = hashes_path.read_text() if hashes_path.exists() else ""

    print(f"contract: mlx-affine, bits={BITS}, group_size={GROUP_SIZE}; "
          f"shipped K={K_SHIP} (ADR 0012, three-class nine-name floor)")
    print(f"source: {provenance}")
    print("serving shapes: "
          + ", ".join(f"{n} {d_out}x{d_in}" for n, (d_out, d_in) in shapes))
    print(f"budget: {args.budget_gb:.1f} GB phys_footprint on a "
          f"{machine_ram_gb():.1f} GB machine; start {rss_line()}")

    fingerprint = checkpoint_fingerprint(provenance, shapes)
    saved = read_checkpoint(CHECKPOINT_PATH, fingerprint) if args.resume else {}
    steps, resumed = {}, []
    guard = BudgetGuard(args.budget_gb)

    def checkpoint(step: str, payload) -> None:
        steps[step] = payload
        write_checkpoint(CHECKPOINT_PATH, fingerprint, steps)

    def spawn(kind: str, cell: str, task: dict) -> tuple[dict, dict]:
        """One measurement child, gated: machine-available memory before it,
        the parent's own footprint before it, the child's reported peak
        after it. The parent holds no Metal state - the session lives and
        dies with each child."""
        require_available_memory(args.budget_gb, cell)
        guard.check(f"parent before {cell}")
        payload, meta = spawn_measurement(
            cell, {**task, "kind": kind, "verbose": not args.quiet},
            budget_gb=args.budget_gb, wall_cap_s=args.wall_cap_s)
        if not args.quiet and meta.get("footprint_gb"):
            print(f"  {cell}: child peak footprint "
                  f"{meta['footprint_gb'].get('peak', 0.0):.2f} GB "
                  f"(parent {footprint_line()})", flush=True)
        return payload, meta

    try:
        if args.attribution_probe:
            return _attribution_probe_main(args, shapes, guard)
        if args.stability_probe:
            return _stability_probe_main(args, shapes, spawn)
        return _run_measured_steps(args, shapes, spawn, saved, steps,
                                   resumed, checkpoint, provenance, hashes)
    except BudgetExceeded as exc:
        return refuse(EXIT_BUDGET_REFUSAL, str(exc), fingerprint, steps)
    except LowMemoryRefusal as exc:
        return refuse(EXIT_LOW_MEMORY, str(exc), fingerprint, steps)
    except ChildRefusal as exc:
        # Every child refusal, the missing device included, goes through the
        # one path: the checkpoint keeps the finished steps and the cell that
        # refused is marked in it. A gate that returns early leaves the next
        # run guessing which cell it was.
        return refuse_child(exc, fingerprint, steps)


def _run_measured_steps(args, shapes, spawn, saved, steps, resumed,
                        checkpoint, provenance, hashes) -> int:
    """Steps 0-6 and the verdict, every measurement in a child process via
    ``spawn``. Split from main() so every refusal raised anywhere inside the
    measured steps funnels through one handler."""
    if args.probe_validate:
        payload, _ = spawn("probe", "probe-validate kv_proj 1024x2560",
                           {"name": "kv_proj", "d_out": 1024, "d_in": 2560})
        _print_probe({"kv_proj": payload})
        return 0

    # -- step 0: the continuity anchor --------------------------------------
    # The marker stays out of `detail`: `detail` is what gets checkpointed, so
    # concatenating it there would stack a marker per resume.
    if "continuity" in saved:
        resumed.append("continuity")
        ok, detail, note = True, saved["continuity"], " [RESUMED, not re-measured]"
    else:
        payload, _ = spawn("continuity",
                           f"continuity anchor {CONTINUITY_SHAPE}", {})
        (ok, detail), note = (payload["ok"], payload["detail"]), ""
    print(f"\nSTEP 0 continuity anchor at {CONTINUITY_SHAPE}, bits={BITS}: "
          f"{'REPRODUCED - ' + detail + note if ok else 'FAILED - ' + detail}")
    if not ok:
        print("STOP: the pipeline does not reproduce ADR 0012's records; "
              "toolchain drift is a finding to raise with the coordinator, "
              "and nothing new is measured on top of it")
        return 1
    checkpoint("continuity", detail)
    if args.continuity_only:
        print("--continuity-only: STEP 0 alone, nothing further measured; the "
              "checkpoint now carries it for a later --resume under this "
              "exact code")
        return 0

    # -- steps 1+2: G0 on the probe artefacts, then the batch-regime probe --
    print("\nSTEP 2 batch-regime probe (G0 checked on each probe artefact):",
          flush=True)
    if "probe" in saved:
        resumed.append("probe")
        probe = saved["probe"]
        print("  [RESUMED from the checkpoint, not re-measured]")
    else:
        probe = {}
        for name, (d_out, d_in) in shapes:
            probe[name], _ = spawn("probe", f"probe {name} {d_out}x{d_in}",
                                   {"name": name, "d_out": d_out,
                                    "d_in": d_in})
    _print_probe(probe)
    if not all(p.get("g0_exact") for p in probe.values()):
        print("STOP: G0 failed on a probe artefact; nothing here describes MLX")
        return 1
    checkpoint("probe", probe)
    joins = sorted({b for p in probe.values() for b in p["joins"]})
    batches = tuple(sorted(set(BASE_BATCHES) | set(joins)))
    print(f"  grid batches: base {BASE_BATCHES}"
          + (f" + joined {joins} -> {batches}" if joins else " (no joins)"))

    # -- step 3: the main grid, one child per (shape, draw, seed) -----------
    print("\nSTEP 3 main grid:", flush=True)
    records, exactness = [], []
    for name, (d_out, d_in) in shapes:
        for draw in WEIGHT_DRAWS:
            for seed in CALIBRATION_SEEDS + HELDOUT_SEEDS:
                payload, _ = spawn(
                    "grid", f"grid {name} {d_out}x{d_in} {draw} s{seed}",
                    {"name": name, "d_out": d_out, "d_in": d_in,
                     "draw": draw, "seed": seed, "batches": list(batches)})
                exactness.append(payload["exact"])
                records.extend(payload["records"])
                checkpoint("grid_partial", records)
            if not args.quiet:
                print(f"    {name} {d_out}x{d_in} {draw} done "
                      f"({len(records)} records, {footprint_line()})",
                      flush=True)
    bit_exact = all(exactness)
    print(f"  G0: {'bit-exact on all' if bit_exact else 'FAILED'} "
          f"({len(exactness)} checks)")
    if not bit_exact:
        print("STOP: canonical_quantize is not bit-exact against mx.quantize "
              "at a serving shape; nothing here describes MLX")
        return 1
    steps.pop("grid_partial", None)
    checkpoint("grid", records)

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
        "resumed_steps": resumed,
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
    print(f"peak footprint: {rss_line()}")
    if resumed:
        print(f"RESUMED steps (not re-measured in this process): {resumed}; "
              f"ADR 0013 must say so")

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
        rss = p.get("rss_gb")
        rss_note = (f" [rss {rss['current']:.2f} GB, "
                    f"peak {rss['peak']:.2f} GB]" if rss else "")
        print(f"  {name}: {len(components)} regime component(s) "
              f"{[f'{min(c)}..{max(c)}' for c in components]}, "
              f"joins {p['joins'] or 'none'}, "
              f"weak-prefix covered {weak or 'none'}{rss_note}")


if __name__ == "__main__":
    raise SystemExit(main())
