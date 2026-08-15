"""The serving-shapes adequacy harness: derivation, exact-value machinery, and
the pre-registered probe and gate wrappers.

Every numerical helper here exists so the harness can run the lm_head serving
shape inside 36 GB without changing a single computed value; each one therefore
carries a bit-identity test against the untouched reference path, not a
tolerance test. A helper that is merely close is wrong.
"""

import json

import numpy as np
import pytest

from calibrate_quant_serving import (
    ARTIFACT_DIR,
    QWEN3_4B_DERIVED_CONFIG,
    ArtifactMismatch,
    derive_serving_shapes,
    resolve_serving_source,
    verify_artifact_quantization,
)


# ---------------------------------------------------------------------------
# Serving-shape derivation (ticket T3: artifact preferred, config fallback)
# ---------------------------------------------------------------------------
EXPECTED_SHAPES = {
    "q_proj": (4096, 2560),
    "kv_proj": (1024, 2560),
    "o_proj": (2560, 4096),
    "gate_up_proj": (9728, 2560),
    "down_proj": (2560, 9728),
    "lm_head": (151936, 2560),
}


def test_derivation_from_qwen3_4b_config():
    shapes = derive_serving_shapes(QWEN3_4B_DERIVED_CONFIG)
    assert dict(shapes) == EXPECTED_SHAPES


def test_derived_shapes_are_group_aligned():
    for _, (d_out, d_in) in derive_serving_shapes(QWEN3_4B_DERIVED_CONFIG):
        assert d_in % 64 == 0, (d_out, d_in)


def test_derivation_dedupes_kv_and_gate_up():
    """k/v and gate/up share shapes; the grid must not pay for them twice."""
    names = [name for name, _ in derive_serving_shapes(QWEN3_4B_DERIVED_CONFIG)]
    assert names.count("kv_proj") == 1 and names.count("gate_up_proj") == 1
    assert "k_proj" not in names and "up_proj" not in names


def test_untied_embeddings_are_rejected():
    """The lm_head shape rides tie_word_embeddings; an untied config would
    dispatch a separate unquantized head and the derivation must say so."""
    config = dict(QWEN3_4B_DERIVED_CONFIG, tie_word_embeddings=False)
    with pytest.raises(ArtifactMismatch):
        derive_serving_shapes(config)


# ---------------------------------------------------------------------------
# Artifact-config verification (gate finding: attestation is conditional)
# ---------------------------------------------------------------------------
def test_artifact_quantization_accepts_uniform_3bit_g64():
    verify_artifact_quantization({"group_size": 64, "bits": 3})
    # mlx-lm records the scheme as mode=affine, which IS the contract's scheme
    verify_artifact_quantization({"group_size": 64, "bits": 3, "mode": "affine"})


@pytest.mark.parametrize("quant", [
    {"group_size": 64, "bits": 4},
    {"group_size": 32, "bits": 3},
    {"group_size": 64, "bits": 3, "model.embed_tokens": {"group_size": 64, "bits": 4}},
    {"group_size": 64, "bits": 3, "lm_head": False},
    {"group_size": 64, "bits": 3, "mode": "mxfp4"},
    None,
])
def test_artifact_quantization_rejects_nonuniform_or_wrong(quant):
    """Any deviation from uniform 3-bit g64 means the attestation would cover
    a different artifact than the one serving; loud stop, never a footnote."""
    with pytest.raises(ArtifactMismatch):
        verify_artifact_quantization(quant)


def test_resolve_prefers_artifact_config(tmp_path):
    config = dict(QWEN3_4B_DERIVED_CONFIG)
    config["quantization"] = {"group_size": 64, "bits": 3}
    (tmp_path / "config.json").write_text(json.dumps(config))
    resolved, provenance = resolve_serving_source(tmp_path)
    assert resolved["vocab_size"] == 151936
    assert provenance.startswith("pinned-artifact:")
    assert str(tmp_path) in provenance


def test_resolve_falls_back_to_derived_config(tmp_path):
    resolved, provenance = resolve_serving_source(tmp_path / "absent")
    assert resolved is QWEN3_4B_DERIVED_CONFIG
    assert "assumption" in provenance


def test_resolve_rejects_artifact_with_wrong_quantization(tmp_path):
    config = dict(QWEN3_4B_DERIVED_CONFIG)
    config["quantization"] = {"group_size": 64, "bits": 4}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ArtifactMismatch):
        resolve_serving_source(tmp_path)


def test_canonical_artifact_path_is_absolute_and_shared():
    """bench/.models is gitignored, so the pinned artifact lives in the MAIN
    worktree; a relative path would silently probe this lane's empty dir."""
    assert ARTIFACT_DIR.is_absolute()
    assert str(ARTIFACT_DIR).startswith("/Users/vlad/kernelverify/")


# ---------------------------------------------------------------------------
# Exact-value engineering: every rewritten path must be bit-identical to the
# untouched reference path. Tolerance has no place in these assertions.
# ---------------------------------------------------------------------------
from calibrate_quant_serving import (  # noqa: E402
    ArtefactHoists,
    dequant_chunk_rows,
    dequant_chunked,
    eval_block_tiled,
    eval_boundary,
    eval_factored_groups,
    eval_factored_serial,
    eval_lut,
    eval_pairwise,
    eval_ref,
    eval_serial_chunked,
    f16_roundtrip_chunked,
    fault_dequants,
    lut_gather_chunked,
)
from calibrate_quant_bits import boundary_fp16_dequant  # noqa: E402
from kernelverify.schemas.quant_contract import (  # noqa: E402
    ENSEMBLE,
    FAULTS,
    QuantContract,
    canonical_quantize,
    dequantize,
    r_contract,
)
from phase0_contract_k import heldout_block_tiled, make_x  # noqa: E402


def _small_artefact(bits=3, d_out=96, d_in=128, seed=3):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((d_out, d_in)) * 0.02).astype(np.float16)
    return w, canonical_quantize(w, QuantContract(bits=bits, group_size=64))


@pytest.fixture(scope="module")
def small():
    w, artefact = _small_artefact()
    return w, artefact, ArtefactHoists(artefact)


def _xs(d_in, mode="unit", dtype=np.float32, batch=5, seed=17):
    return make_x(batch, d_in, dtype, mode, np.random.default_rng(seed))


@pytest.mark.parametrize("reverse,name", [(False, "dequant-serial"),
                                          (True, "dequant-reversed")])
@pytest.mark.parametrize("chunk", [1, 5, 64, 1000])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("mode", ["unit", "constant-rows"])
def test_chunked_serial_is_bit_identical(small, reverse, name, chunk, dtype, mode):
    _, artefact, hoists = small
    x = _xs(128, mode, dtype)
    ours = eval_serial_chunked(x, hoists.w32, chunk, reverse=reverse)
    theirs = ENSEMBLE[name](x, artefact)
    assert ours.dtype == theirs.dtype
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_hoisted_paths_are_bit_identical(small, dtype):
    w, artefact, hoists = small
    x = _xs(128, "unit", dtype)
    assert np.array_equal(eval_ref(x, hoists.w64), r_contract(x, artefact))
    assert np.array_equal(eval_pairwise(x, hoists.w32),
                          ENSEMBLE["dequant-pairwise"](x, artefact))
    assert np.array_equal(eval_lut(x, hoists.w_lut32),
                          ENSEMBLE["lut-gather"](x, artefact))
    assert np.array_equal(
        eval_factored_groups(x, hoists.qg32, hoists.scales32, hoists.biases32),
        ENSEMBLE["factored-groups"](x, artefact))
    assert np.array_equal(
        eval_factored_serial(x, hoists.qg32, hoists.scales32, hoists.biases32),
        ENSEMBLE["factored-serial"](x, artefact))
    assert np.array_equal(eval_block_tiled(x, hoists.w32),
                          heldout_block_tiled(x, artefact))
    assert np.array_equal(eval_boundary(x, hoists.w16_32),
                          boundary_fp16_dequant(x, artefact))


def test_hoisted_fault_paths_are_bit_identical(small):
    _, artefact, _ = small
    x = _xs(128, "corpus-scale", np.float32)
    seen = []
    for fault_name, w_fault32 in fault_dequants(artefact):
        seen.append(fault_name)
        theirs = ENSEMBLE["dequant-pairwise"](x, FAULTS[fault_name](artefact))
        assert np.array_equal(eval_pairwise(x, w_fault32), theirs), fault_name
    assert seen == list(FAULTS), "every catalogue fault, in catalogue order"


# ---------------------------------------------------------------------------
# The 2026-08-15 memory amendment: row-chunked dequantization.
#
# The chunked path must be bit-identical at EVERY chunk size, because the row
# partition is the whole safety argument: each dequantized element is an
# independent elementwise expression of its own row, so no reduction and no
# BLAS call can see the partition. A chunk size that changed a single bit
# would mean that argument is false.
# ---------------------------------------------------------------------------
CHUNK_SIZES = [1, 2, 7, 64, 95, 96, 10_000]


@pytest.mark.parametrize("chunk", CHUNK_SIZES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
@pytest.mark.parametrize("bits", [3, 4])
def test_chunked_dequant_is_bit_identical(chunk, dtype, bits):
    _, artefact = _small_artefact(bits=bits)
    ours = dequant_chunked(artefact, dtype, chunk)
    theirs = dequantize(artefact, dtype)
    assert ours.dtype == theirs.dtype and ours.shape == theirs.shape
    assert np.array_equal(ours, theirs)


def test_chunked_dequant_default_chunk_is_bit_identical():
    """The production call passes no chunk size and must be exact anyway."""
    _, artefact = _small_artefact()
    assert np.array_equal(dequant_chunked(artefact, np.float64),
                          dequantize(artefact, np.float64))


@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_chunked_lut_gather_is_bit_identical(chunk):
    """lut-gather's table is per-row too, so its gather chunks exactly."""
    _, artefact = _small_artefact()
    x = _xs(128, "unit", np.float32)
    ours = lut_gather_chunked(artefact, chunk)
    assert np.array_equal(eval_lut(x, ours), ENSEMBLE["lut-gather"](x, artefact))


@pytest.mark.parametrize("chunk", CHUNK_SIZES)
@pytest.mark.parametrize("mode", ["unit", "constant-rows"])
def test_chunked_factored_groups_is_bit_identical(chunk, mode, small):
    """The int-domain chain is elementwise in the output row, so its row block
    is exact at every size - unlike the fp64 reference matmul, whose output-row
    partition demonstrably is not (test_reference_matmul_is_never_row_partitioned).
    The member's own BLAS term stays whole inside the chunked path."""
    _, artefact, hoists = small
    x = _xs(128, mode, np.float32)
    ours = eval_factored_groups(x, hoists.qg32, hoists.scales32,
                                hoists.biases32, chunk)
    assert np.array_equal(ours, ENSEMBLE["factored-groups"](x, artefact))


@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_chunked_f16_roundtrip_is_bit_identical(chunk, small):
    """The fp16-dequant boundary's weights: two elementwise casts, so chunked."""
    _, artefact, hoists = small
    x = _xs(128, "unit", np.float32)
    ours = f16_roundtrip_chunked(hoists.w32, chunk)
    assert np.array_equal(eval_boundary(x, ours),
                          boundary_fp16_dequant(x, artefact))


def test_dequant_chunk_rows_bounds_the_block_and_never_returns_zero():
    """A row wider than the whole budget must still yield one row, not zero."""
    assert dequant_chunk_rows(2560, 8) >= 1
    assert dequant_chunk_rows(10 ** 9, 8) == 1
    assert dequant_chunk_rows(2560, 8) < dequant_chunk_rows(2560, 4)


def test_reference_weights_are_lazy(small):
    """STEP 2 never reads the fp64 reference, and at lm_head it is 3.1 GB of
    the probe's budget. Building it eagerly is what the amendment removes."""
    w, artefact, _ = small
    hoists = ArtefactHoists(artefact)
    assert not hoists.reference_built
    built = hoists.w64
    assert np.array_equal(built, dequantize(artefact, np.float64))
    assert hoists.reference_built
    assert hoists.w64 is built, "built once, then reused"


def test_reference_matmul_is_never_row_partitioned():
    """The amendment as first proposed row-chunked the fp64 reference MATMUL.
    Measured on this machine's Accelerate BLAS that is NOT bit-exact: cutting
    the output-row dimension changes dgemm's blocking and moves results by up
    to 2e-14. The harness therefore chunks the DEQUANTIZATION only and leaves
    the matmul whole, so eval_ref must still agree with r_contract bit for bit
    at a shape where a partitioned matmul demonstrably would not.

    Sized to a shape that actually separates the two (the fixture's 96 rows do
    not); the skip is the safety valve for a future BLAS that partitions
    invariantly, never a licence for the harness to rely on one that does.
    """
    w, artefact = _small_artefact(d_out=1024, d_in=1024)
    hoists = ArtefactHoists(artefact)
    x = _xs(1024, "unit", np.float32, batch=16)
    full = r_contract(x, artefact)
    x64 = x.astype(np.float64)
    partitioned = np.concatenate(
        [x64 @ hoists.w64[s:s + 1].T for s in range(hoists.w64.shape[0])],
        axis=1)
    if np.array_equal(partitioned, full):
        pytest.skip("this BLAS is row-partition invariant at this shape; the "
                    "harness never relies on it either way")
    assert np.array_equal(eval_ref(x, hoists.w64), full)


# ---------------------------------------------------------------------------
# Per-step checkpoint: a SIGKILL must cost one step, not the whole run
# ---------------------------------------------------------------------------
from calibrate_quant_serving import (  # noqa: E402
    checkpoint_fingerprint,
    read_checkpoint,
    rss_gb,
    write_checkpoint,
)


def _fingerprint(provenance="derived", shapes=(("tiny", (8, 64)),)):
    return checkpoint_fingerprint(provenance, shapes)


def test_checkpoint_round_trips_completed_steps(tmp_path):
    path = tmp_path / "partial.json"
    fingerprint = _fingerprint()
    write_checkpoint(path, fingerprint, {"probe": {"kv_proj": {"g0_exact": True}}})
    assert read_checkpoint(path, fingerprint) == {
        "probe": {"kv_proj": {"g0_exact": True}}}


def test_checkpoint_is_refused_when_the_run_identity_changed(tmp_path):
    """Resuming across a different provenance or shape set would splice two
    different calibrations into one attestation."""
    path = tmp_path / "partial.json"
    write_checkpoint(path, _fingerprint(), {"probe": {}})
    assert read_checkpoint(path, _fingerprint(provenance="pinned-artifact:x")) == {}
    assert read_checkpoint(path, _fingerprint(shapes=(("tiny", (16, 64)),))) == {}


def test_checkpoint_absent_or_corrupt_reads_as_nothing_to_resume(tmp_path):
    """A SIGKILL mid-write must not turn into a crash on the next run."""
    assert read_checkpoint(tmp_path / "absent.json", _fingerprint()) == {}
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text('{"fingerprint": {"bits": 3}, "steps": {"probe"')
    assert read_checkpoint(corrupt, _fingerprint()) == {}


def test_checkpoint_write_leaves_no_partial_file_behind(tmp_path):
    """The write is atomic (temp then replace), so the checkpoint on disk is
    always a whole one; the scratch file must not survive."""
    path = tmp_path / "partial.json"
    write_checkpoint(path, _fingerprint(), {"probe": {}})
    write_checkpoint(path, _fingerprint(), {"probe": {}, "grid": {"records": []}})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["partial.json"]
    assert set(read_checkpoint(path, _fingerprint())) == {"probe", "grid"}


def test_rss_self_report_returns_current_and_peak_gigabytes():
    """The line that confirms or refutes the memory hypothesis in the log."""
    current, peak = rss_gb()
    assert 0.0 < current < 1000.0
    assert peak >= current * 0.5  # peak is a high-water mark, never far below


# ---------------------------------------------------------------------------
# Batch-regime probe: match matrix, direct-match coverage, component reading
# ---------------------------------------------------------------------------
from calibrate_quant_serving import (  # noqa: E402
    BASE_BATCHES,
    coverage_joins,
    match_components,
    match_matrix,
    weak_prefix_cover,
)


def _outputs_from_rowfn(rowfn, batches=range(1, 17), d_out=6):
    """Synthetic per-B outputs: row r of the B-batch output is rowfn(r, B)."""
    return {B: np.array([rowfn(r, B) for r in range(B)], dtype=np.float64)
            .reshape(B, 1) * np.ones((1, d_out))
            for B in batches}


def test_match_matrix_all_invariant():
    outs = _outputs_from_rowfn(lambda r, B: float(r))
    matrix = match_matrix([outs])
    assert matrix.all()
    assert coverage_joins(matrix, BASE_BATCHES) == []
    assert match_components(matrix) == [set(range(1, 17))]


def test_regime_switch_at_six_yields_no_joins_with_base_set():
    """qmv-vs-qmm shape: rows differ between the two regimes; base {1,2,8,16}
    already holds a representative on each side, so nothing joins."""
    outs = _outputs_from_rowfn(lambda r, B: float(r) + (100.0 if B >= 6 else 0.0))
    matrix = match_matrix([outs])
    assert matrix[1 - 1][2 - 1] and matrix[8 - 1][16 - 1]
    assert not matrix[2 - 1][8 - 1]
    assert coverage_joins(matrix, BASE_BATCHES) == []
    assert match_components(matrix) == [set(range(1, 6)), set(range(6, 17))]


def test_uncovered_regime_joins_smallest_batch():
    """A middle regime at B in 3..5 has no grid representative: 3 joins."""
    outs = _outputs_from_rowfn(
        lambda r, B: float(r) + (50.0 if 3 <= B <= 5 else 0.0) + (100.0 if B >= 6 else 0.0))
    matrix = match_matrix([outs])
    assert coverage_joins(matrix, BASE_BATCHES) == [3]
    assert coverage_joins(matrix, BASE_BATCHES + (3,)) == []


def test_match_requires_joint_agreement_across_implementations():
    """One implementation B-invariant, another switching: the AND governs."""
    invariant = _outputs_from_rowfn(lambda r, B: float(r))
    switching = _outputs_from_rowfn(lambda r, B: float(r) + (9.0 if B >= 12 else 0.0))
    matrix = match_matrix([invariant, switching])
    assert not matrix[11 - 1][12 - 1]
    assert coverage_joins(matrix, BASE_BATCHES) == []  # 8 covers 1..11, 16 covers 12..16


def test_weak_prefix_coverage_is_flagged_not_absorbed():
    """A short-prefix link can chain two regimes into one component AND cover
    a batch through a smaller grid batch. The accepted rule counts that as
    covered; the weak-prefix report makes the thin evidence visible."""
    def rowfn(r, B):
        return 0.0 if r == 0 else float(r) + (5.0 if B > 8 else 0.0)
    # Row 0 identical everywhere; rows >= 1 differ between B<=8 and B>8, so
    # B=1 direct-matches every batch while 2..8 and 9..16 direct-mismatch.
    outs = {B: np.array([[rowfn(r, B)] for r in range(B)], dtype=np.float64)
            for B in range(1, 17)}
    matrix = match_matrix([outs])
    assert matrix[1 - 1][2 - 1] and matrix[1 - 1][9 - 1] and not matrix[2 - 1][9 - 1]
    assert match_components(matrix) == [set(range(1, 17))]  # absorbed reading
    assert coverage_joins(matrix, (1,)) == []               # accepted rule
    assert weak_prefix_cover(matrix, (1,)) == list(range(2, 17))  # visible


def test_weak_prefix_cover_names_batches_attested_only_by_row_invariance():
    """Switch at six with base {1,2,8,16}: 3..5 are covered only through the
    smaller batches 1 and 2, and the report must say so."""
    outs = _outputs_from_rowfn(lambda r, B: float(r) + (100.0 if B >= 6 else 0.0))
    matrix = match_matrix([outs])
    assert weak_prefix_cover(matrix, BASE_BATCHES) == [3, 4, 5]
    invariant = _outputs_from_rowfn(lambda r, B: float(r))
    assert weak_prefix_cover(match_matrix([invariant]), BASE_BATCHES) == []


# ---------------------------------------------------------------------------
# Per-cell gates with width-pooled fault equivalence
# ---------------------------------------------------------------------------
from calibrate_quant_device import ALL_MEMBERS, HELDOUTS  # noqa: E402
from calibrate_quant_serving import K_SHIP, cell_reports  # noqa: E402


def _record(cell, fault_errors, mode="unit", dtype="float32"):
    """A schema-complete record with a tiny floor and huge base-tol margins."""
    return {
        "shape": cell[0], "batch": cell[1], "draw": "normal-0.02", "seed": 0,
        "heldout_draw": False, "mode": mode, "dtype": dtype,
        "ref_scale": 1.0, "base_tol": 1e-6,
        "members": {name: 1e-5 * (i + 1) for i, name in enumerate(ALL_MEMBERS)},
        "heldout": {name: 1e-5 for name in HELDOUTS},
        "faults": dict(fault_errors),
        "boundary_fp16_dequant": 0.0,
    }


def _fault_set(**overrides):
    base = {name: 1.0 for name in FAULTS}  # caught at huge margin by default
    base.update(overrides)
    return base


def test_cell_gates_use_width_level_equivalence():
    """A fault caught in cell A but silent in cell B must FAIL cell B's G2
    rather than slipping into per-cell equivalence."""
    cell_a, cell_b = ("512x512", 1), ("512x512", 8)
    records = (
        [_record(cell_a, _fault_set()) for _ in range(4)]
        + [_record(cell_b, _fault_set(**{"bias-dropped": 0.0})) for _ in range(4)]
    )
    reports = cell_reports(records, K_SHIP)
    assert "bias-dropped" not in reports[cell_a]["equivalent_faults"]
    assert "bias-dropped" not in reports[cell_b]["equivalent_faults"]
    assert reports[cell_a]["gates"]["G2_faults_caught"]
    assert not reports[cell_b]["gates"]["G2_faults_caught"]
    assert reports[cell_b]["verdict"] == "INADEQUATE"


def test_cell_gates_keep_width_level_equivalents_excluded():
    """A fault caught nowhere at the width stays an equivalent everywhere and
    no cell's G2 pays for it."""
    cells = [("512x512", 1), ("512x512", 8)]
    records = [_record(c, _fault_set(**{"nibble-order-swapped": 0.0}))
               for c in cells for _ in range(4)]
    reports = cell_reports(records, K_SHIP)
    for cell in cells:
        assert "nibble-order-swapped" in reports[cell]["equivalent_faults"]
        assert reports[cell]["gates"]["G2_faults_caught"]
        assert reports[cell]["verdict"] == "ADEQUATE"


# ---------------------------------------------------------------------------
# Device-touching seams: the mlx quantize hoist and the measure-loop schema.
# Skipped wholesale without PyObjC Metal, matching the runner tests.
# ---------------------------------------------------------------------------
Metal = pytest.importorskip("Metal")

from calibrate_quant_bits import mlx_on_device  # noqa: E402
from calibrate_quant_device import gates as standing_gates  # noqa: E402
from calibrate_quant_device import k_demand  # noqa: E402
from calibrate_quant_serving import (  # noqa: E402
    measure_serving,
    mlx_qmm_heldout,
    mlx_quantize_hoist,
    probe_batch_regimes,
)
from kernelverify.runners.device import MetalDevice  # noqa: E402
from kernelverify.schemas.quant_device import DeviceMemberSession  # noqa: E402

try:
    _DEVICE = MetalDevice()
except RuntimeError:
    _DEVICE = None

needs_metal = pytest.mark.skipif(_DEVICE is None,
                                 reason="no Metal device on this machine")


@needs_metal
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_mlx_quantize_hoist_is_bit_identical(dtype):
    w, artefact = _small_artefact(d_out=128, d_in=256)
    x = _xs(256, "unit", dtype)
    hoisted = mlx_qmm_heldout(x, *mlx_quantize_hoist(w))
    verbatim = mlx_on_device(x, w, artefact.contract)
    assert hoisted.dtype == verbatim.dtype
    assert np.array_equal(hoisted, verbatim)


@needs_metal
def test_measure_serving_records_feed_the_standing_machinery():
    """A tiny grid end to end: records carry the batch axis and slot straight
    into the standing k_demand and gates plus the per-cell wrapper."""
    session = DeviceMemberSession(_DEVICE)
    measured = measure_serving([("tiny", (128, 128))], (1, 2), session,
                               verbose=False)
    records = measured["records"]
    assert measured["bit_exact"]
    assert len(records) == 2 * 4 * 2 * 4 * 2  # draws x seeds x B x modes x dtypes
    assert {r["batch"] for r in records} == {1, 2}
    assert all(r["proj"] == "tiny" and r["bits"] == 3 for r in records)
    needed, binding = k_demand(records)
    assert needed >= 0.0 and isinstance(binding, str)
    pooled = standing_gates(records, K_SHIP)
    assert set(pooled["gates"]) == {"G1_no_false_positives", "G2_faults_caught",
                                    "G3_class_coverage"}
    per_cell = cell_reports(records, K_SHIP)
    assert set(per_cell) == {("128x128", 1), ("128x128", 2)}
    for rep in per_cell.values():
        assert rep["verdict"] in ("ADEQUATE", "INADEQUATE")


@needs_metal
def test_measure_serving_checkpoints_after_every_block():
    """The recorder that makes a death inside the main grid cost one block
    instead of the whole step: it fires per (shape, draw), and each call sees
    every record earned so far."""
    session = DeviceMemberSession(_DEVICE)
    seen = []
    measured = measure_serving([("tiny", (128, 128))], (1,), session,
                               verbose=False,
                               on_block=lambda rs: seen.append(len(rs)))
    assert len(seen) == 2, "one checkpoint per (shape, draw)"
    assert seen == sorted(seen) and seen[-1] == len(measured["records"])


@needs_metal
def test_probe_reports_its_own_rss_per_shape():
    """The line that confirms or refutes the memory hypothesis has to reach
    the JSON, not only the log."""
    session = DeviceMemberSession(_DEVICE)
    probe = probe_batch_regimes([("tiny", (64, 64))], session, verbose=False)
    rss = probe["tiny"]["rss_gb"]
    assert rss["current"] > 0.0 and rss["peak"] >= rss["current"] * 0.5
