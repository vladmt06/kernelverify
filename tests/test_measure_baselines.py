"""The two gates that decide whether a baseline number may be published.

Both exist because this project has already been burned once each: the
first-pass baselines were hand-run on a shared machine, and kv-runner-e9
measured that sub-millisecond GPU timings move by up to 4x with power state.
Neither rule is allowed to be a comment.
"""

import json
from pathlib import Path

import pytest

import machine_state
import measure_baselines
from conftest import schema_row
from machine_state import binding_verdict, timing_verdict


def state(load1=0.5, source="AC", lpm="0", thermal=False, blockers=None):
    """An idle_check result, without touching the real machine."""
    if blockers is None:
        blockers = []
        if load1 > 3.0:
            blockers.append(f"load1 {load1:.2f} over threshold 3.00")
        if source != "AC":
            blockers.append(f"power source {source}")
        if lpm not in ("0", "?"):
            blockers.append("low power mode on")
        if thermal:
            blockers.append("thermal warning recorded")
    return {
        "load1": load1, "load5": load1, "load15": load1, "load_threshold": 3.0,
        "power": {"source": source, "low_power_mode": lpm, "thermal_warning": thermal},
        "idle": not blockers, "blockers": blockers,
    }


# --- the idle gate --------------------------------------------------------


def test_idle_threshold_scales_with_core_count(monkeypatch):
    monkeypatch.setattr(machine_state, "load_averages", lambda: (2.0, 2.0, 2.0))
    monkeypatch.setattr(
        machine_state, "power_state",
        lambda: {"source": "AC", "low_power_mode": "0", "thermal_warning": False},
    )
    monkeypatch.setattr(machine_state, "competing_processes", lambda *a, **k: [])
    # 2.0 is idle on 12 cores and busy on 4: a fixed threshold would call one
    # of those wrong, and this machine has both kinds of core.
    assert machine_state.idle_check(cores=12)["idle"] is True
    assert machine_state.idle_check(cores=4)["idle"] is False


@pytest.mark.parametrize(
    "kwargs, expected_blocker",
    [
        ({"load1": 9.0}, "load1"),
        ({"source": "battery"}, "power source"),
        ({"lpm": "1"}, "low power mode"),
        ({"thermal": True}, "thermal"),
    ],
)
def test_every_condition_blocks_binding(kwargs, expected_blocker):
    clean = timing_verdict(50.0)
    verdict = binding_verdict(state(**kwargs), state(), clean)
    assert verdict["binding"] is False
    assert any(expected_blocker in b for b in verdict["binding_blockers"])


def test_a_machine_that_goes_busy_mid_run_also_blocks():
    verdict = binding_verdict(state(), state(load1=9.0), timing_verdict(50.0))
    assert verdict["binding"] is False
    assert any(b.startswith("after:") for b in verdict["binding_blockers"])


def test_clean_machine_and_real_timescale_binds():
    verdict = binding_verdict(state(), state(), timing_verdict(50.0))
    assert verdict == {"binding": True, "binding_blockers": []}


# --- the dispersion gate --------------------------------------------------


def test_disagreeing_repeats_block_binding_on_a_quiet_machine():
    """The case that forced this gate.

    A run reported load average 1.93 from start to finish while one spec's
    third sample came in at 22.07 against 99.38 and 90.06. Load is averaged
    over a minute and cannot see a transient that lands inside one sample, so
    an idle machine is not evidence that a measurement succeeded.
    """
    dispersion = machine_state.dispersion_verdict(85.84)
    verdict = binding_verdict(state(), state(), timing_verdict(50.0), dispersion)
    assert verdict["binding"] is False
    assert any("repeats disagree" in b for b in verdict["binding_blockers"])


def test_tight_repeats_still_bind():
    dispersion = machine_state.dispersion_verdict(1.76)
    assert dispersion["over_spread_limit"] is False
    assert binding_verdict(state(), state(), timing_verdict(50.0), dispersion)["binding"]


def test_a_single_sample_has_no_spread_and_is_not_blocked_by_it():
    # one rep cannot disagree with itself; the other gates still apply
    assert machine_state.dispersion_verdict(None)["over_spread_limit"] is False


# --- the GPU contention gate ----------------------------------------------


def test_an_interactive_app_on_the_gpu_blocks_even_at_low_load(monkeypatch):
    """The case the kernels lane measured.

    An AC-powered machine at load average 1.79 produced 2.2x-9.7x spreads
    because Terminal and VS Code were driving the display stack. Load average
    and power state both passed it, so neither is sufficient on its own.
    """
    monkeypatch.setattr(machine_state, "load_averages", lambda: (1.79, 1.8, 1.8))
    monkeypatch.setattr(
        machine_state, "power_state",
        lambda: {"source": "AC", "low_power_mode": "0", "thermal_warning": False},
    )
    monkeypatch.setattr(
        machine_state, "competing_processes",
        lambda *a, **k: [{"name": "Terminal", "pcpu": 36.0},
                         {"name": "Code Helper (GPU)", "pcpu": 25.0}],
    )
    check = machine_state.idle_check(cores=12)
    assert check["idle"] is False
    assert any("Terminal" in b and "36%" in b for b in check["blockers"])


def test_the_harness_does_not_flag_its_own_subprocesses(monkeypatch):
    monkeypatch.setattr(machine_state, "load_averages", lambda: (1.0, 1.0, 1.0))
    monkeypatch.setattr(
        machine_state, "power_state",
        lambda: {"source": "AC", "low_power_mode": "0", "thermal_warning": False},
    )
    # competing_processes already excludes our own tree; with nothing else busy
    # the machine is idle even though our own benchmark pegs a core
    monkeypatch.setattr(machine_state, "competing_processes", lambda *a, **k: [])
    assert machine_state.idle_check(cores=12)["idle"] is True


def test_own_process_tree_contains_this_process():
    import os

    assert os.getpid() in machine_state._own_process_tree()


# --- the timing floor -----------------------------------------------------


def test_sub_millisecond_samples_are_refused_as_absolute_claims():
    # 200 us is the scale kv-runner-e9 measured moving 4x with power state.
    assert timing_verdict(0.2)["below_timing_floor"] is True
    assert timing_verdict(50.0)["below_timing_floor"] is False


def test_timing_floor_blocks_binding_on_an_otherwise_perfect_machine():
    verdict = binding_verdict(state(), state(), timing_verdict(0.2))
    assert verdict["binding"] is False
    assert any("timing floor" in b for b in verdict["binding_blockers"])


# --- the consumer contract ------------------------------------------------


def valid_row():
    return schema_row(version=3)


def test_a_complete_row_validates_and_round_trips_as_jsonl():
    row = measure_baselines.validate_row(valid_row())
    assert json.loads(json.dumps(row))["row_id"] == "r/x"


@pytest.mark.parametrize("field", ["run_id", "provenance_tier", "machine", "binding"])
def test_a_missing_field_fails_the_producer_not_the_consumer(field):
    row = valid_row()
    del row[field]
    with pytest.raises(ValueError):
        measure_baselines.validate_row(row)


def test_unknown_provenance_tier_is_rejected():
    row = valid_row() | {"provenance_tier": "vibes"}
    with pytest.raises(ValueError, match="provenance tier"):
        measure_baselines.validate_row(row)


def test_a_row_cannot_claim_to_bind_while_carrying_blockers():
    row = valid_row() | {"binding": True, "binding_blockers": ["load1 9.0"]}
    with pytest.raises(ValueError, match="cannot bind"):
        measure_baselines.validate_row(row)


def test_a_model_row_without_a_roofline_placement_is_rejected():
    row = valid_row()
    del row["roofline"]
    with pytest.raises(ValueError, match="no roofline placement"):
        measure_baselines.validate_row(row)


def test_a_row_that_cannot_say_how_it_was_sampled_is_rejected():
    """Comparability is a separate gate from bindingness.

    The kernels lane ran an A/B whose every dispatch was past 5 ms and still
    got the sign backwards, because the arms ran in separate passes. A row
    that cannot state it was interleaved must not be compared against another.
    """
    row = valid_row()
    del row["sampling"]
    with pytest.raises(ValueError, match="sampling missing"):
        measure_baselines.validate_row(row)


def test_a_binding_row_can_still_be_incomparable():
    # binding is about absolutes, interleaving is about comparisons; a row may
    # legitimately pass one gate and fail the other
    row = valid_row()
    row["sampling"] = {"interleaved": False, "group": None, "rounds": 1}
    assert measure_baselines.validate_row(row)["binding"] is True
    assert row["sampling"]["interleaved"] is False


# --- the v3 contract: constrained vocabulary ------------------------------


def test_v3_constrains_binding_resource_to_the_known_set():
    """The renderer picks the utilisation column by this value, so a producer
    writing 'bandwidth' would render every such row as an unknown column."""
    row = valid_row()
    row["roofline"]["binding_resource"] = "bandwidth"
    with pytest.raises(ValueError, match="binding_resource"):
        measure_baselines.validate_row(row)
    del row["roofline"]["binding_resource"]
    with pytest.raises(ValueError, match="binding_resource"):
        measure_baselines.validate_row(row)


def test_v3_requires_a_canonical_logical_model_name():
    """The logical name is the cross-stack cell identity; the artefact
    spelling would split one cell into two, one per stack."""
    row = valid_row()
    row["model"]["logical_name"] = "Qwen3-4B"
    with pytest.raises(ValueError, match="logical_name"):
        measure_baselines.validate_row(row)
    del row["model"]["logical_name"]
    with pytest.raises(ValueError, match="logical_name"):
        measure_baselines.validate_row(row)


def test_a_v2_row_is_not_held_to_v3_fields():
    """The shipped 2026-08-14 record is v2; the validator must not
    retroactively reject the contract those rows were written under."""
    row = valid_row()
    row["schema_version"] = 2
    del row["model"]
    row["roofline"] = {"bandwidth_utilisation_pct": 80.0}
    assert measure_baselines.validate_row(row)


def test_a_v3_ceiling_row_needs_no_model_or_binding_resource():
    row = valid_row()
    row["measurement"] = {"kind": "ceiling", "name": "bandwidth_read"}
    row["model"] = None
    del row["roofline"]
    del row["sampling"]
    assert measure_baselines.validate_row(row)


def test_the_producer_writes_a_logical_name_the_validator_accepts():
    assert measure_baselines.LOGICAL_MODEL in measure_baselines.LOGICAL_MODEL_NAMES


# --- the restructured loop: a sampling group is one A/B session (D1) ------


def spec(spec_id, kind, batch, stack="llama.cpp", n_prompt=0):
    return {"id": spec_id, "stack": stack, "kind": kind, "batch": batch,
            "n_prompt": n_prompt, "n_gen": 0, "model_path": None}


def cell_specs():
    return [
        spec("llamacpp/decode", "decode", 1),
        spec("llamacpp/prefill", "prefill", 1024, n_prompt=1024),
        spec("llamacpp/width8", "matmul_width", 8),
        spec("mlx/decode", "decode", 1, stack="mlx-lm", n_prompt=1024),
        spec("mlx/width8", "matmul_width", 8, stack="mlx-lm", n_prompt=32),
    ]


def fake_measure(_spec):
    return {"tokens_per_s": 1.0, "prompt_tps": 1.0, "generation_tps": 1.0,
            "sample_ms": 5.0}


def test_consecutive_samples_within_a_cell_alternate_arms():
    """The mandated D1 regression: the old loop rotated ALL specs round-robin,
    so the two arms of one comparison sat a full rotation - minutes - apart
    and a sampling group was a relabel of the run. The arms of a cell must be
    measured shoulder to shoulder: consecutive samples alternate arms.
    """
    specs = cell_specs()
    _, sequence = measure_baselines.measure_cells(specs, rounds=3,
                                                  measure=fake_measure)
    for label, members in measure_baselines.group_cells(specs):
        ids = {m["id"] for m in members}
        block = [i for i, spec_id in enumerate(sequence) if spec_id in ids]
        assert block == list(range(block[0], block[0] + len(block))), \
            f"cell {label} was interleaved with other cells, not run as one session"
        if len(ids) < 2:
            continue  # a lone arm has nothing to alternate with
        arms = [sequence[i] for i in block]
        assert all(first != second for first, second in zip(arms, arms[1:])), \
            f"cell {label} measured the same arm twice in a row"


def test_every_arm_gets_every_round():
    samples, sequence = measure_baselines.measure_cells(cell_specs(), rounds=3,
                                                        measure=fake_measure)
    assert all(len(v) == 3 for v in samples.values())
    assert len(sequence) == 3 * len(cell_specs())


def test_a_sampling_group_is_one_workload_cell_not_the_run():
    """Never a relabel: decode and width8 land in different groups, and each
    group holds exactly the arms of its own A/B."""
    cells = measure_baselines.group_cells(cell_specs())
    assert [(label, [m["id"] for m in members]) for label, members in cells] == [
        ("decode-w1", ["llamacpp/decode", "mlx/decode"]),
        ("prefill-w1024", ["llamacpp/prefill"]),
        ("matmul_width-w8", ["llamacpp/width8", "mlx/width8"]),
    ]


# --- the D3.7 regression rule: the shipped v3 record is protected ---------
#
# Ruled in the 2026-08-15 pivot review: every EXISTING v3 cell of the
# 2026-08-14 record must reproduce within the producer's repeat-disagreement
# limit, and new cells only add, never replace. The rule protects cells by
# IDENTITY, not by kind label: the mlx width-8/16 rows are batch-mechanism
# cells and are protected exactly like every other cell.

RECORD = Path(__file__).resolve().parents[1] / "bench" / ".baselines" / "2026-08-14.jsonl"
PROTECTED_RUN = "20260814T230251Z-b50ff2a8"


def rows_in(path):
    """Every row of one baseline file; a blank line is not a row."""
    return [json.loads(line)
            for line in path.read_text().splitlines() if line.strip()]


def protected_rows():
    """The v3 measurement rows of the binding 00:05 run: the protected cells."""
    return [r for r in rows_in(RECORD)
            if r["schema_version"] == 3 and r["run_id"] == PROTECTED_RUN
            and r["measurement"]["kind"] != "ceiling"]


@pytest.fixture
def specs(monkeypatch):
    # resolve_hf_model walks the HF cache on disk; the specs' workload identity
    # must not depend on what this machine happens to have downloaded.
    monkeypatch.setattr(measure_baselines.mlx_info, "resolve_hf_model",
                        lambda repo: Path("/mlx-model"))
    return measure_baselines.build_specs(None)


def test_every_protected_cell_is_still_produced_with_its_recorded_workload(specs):
    """A rerun can only reproduce a cell if the producer still measures the
    same workload under the same spec id; a silently changed n_prompt or depth
    would make 'reproduces' vacuous."""
    by_id = {s["id"]: s for s in specs}
    for row in protected_rows():
        spec_id = row["row_id"].split("/", 1)[1]
        assert spec_id in by_id, f"protected arm {spec_id} vanished from the producer"
        assert measure_baselines.measurement_fields(by_id[spec_id]) == row["measurement"], \
            f"protected arm {spec_id} no longer measures its recorded workload"


def test_protected_cells_keep_their_exact_arms(specs):
    """Adding an arm into an existing cell changes what that cell's A/B is,
    which is a replacement in disguise; the recorded membership is the cell."""
    recorded = {row["sampling"]["group"].split("/", 1)[1]:
                row["sampling"]["group_members"] for row in protected_rows()}
    current = {label: [m["id"] for m in members]
               for label, members in measure_baselines.group_cells(specs)}
    for label, members in recorded.items():
        assert current.get(label) == members, \
            f"protected cell {label} changed its arms: {current.get(label)}"


def test_new_cells_only_add_never_replace(specs):
    recorded = {row["sampling"]["group"].split("/", 1)[1]
                for row in protected_rows()}
    current = {label for label, _ in measure_baselines.group_cells(specs)}
    assert recorded <= current, f"protected cells vanished: {recorded - current}"
    assert current - recorded == {
        "batch_decode-w4", "batch_decode-w8", "batch_decode-w16",
    }, "this block adds exactly the three mlx-only serving cells (D6)"


def rerun_row(row, median):
    rerun = json.loads(json.dumps(row))
    rerun["run_id"] = "20260816T000000Z-00000000"
    rerun["row_id"] = f"{rerun['run_id']}/{row['row_id'].split('/', 1)[1]}"
    rerun["result"]["median"] = median
    return rerun


def test_a_rerun_within_the_repeat_disagreement_limit_reproduces():
    row = protected_rows()[0]  # llamacpp/decode, median 43.05
    ok, why = measure_baselines.reproduces(row, rerun_row(row, 46.92))  # +9.0%
    assert ok, why


def test_a_rerun_drifted_past_the_limit_is_a_regression_not_noise():
    """The limit is the producer's own repeat-disagreement limit: a drift the
    producer would refuse inside one run cannot be waved through between runs."""
    row = protected_rows()[0]
    ok, why = measure_baselines.reproduces(row, rerun_row(row, 48.22))  # +12.0%
    assert not ok
    assert str(machine_state.MAX_SPREAD_PCT) in why


def test_a_row_with_a_different_workload_is_not_a_rerun_at_all():
    """Equal medians on different workloads prove nothing; identity first."""
    row = protected_rows()[0]
    other = rerun_row(row, row["result"]["median"])
    other["measurement"] = {**other["measurement"], "n_gen": 999}
    ok, why = measure_baselines.reproduces(row, other)
    assert not ok
    assert "workload" in why


def test_every_later_binding_rerun_of_a_protected_cell_reproduces_the_record():
    """The standing D3.7 rule, live against the record itself.

    Vacuously true today, because no later v3 run exists; the moment the next
    binding run lands in bench/.baselines/, this test IS the regression gate.
    Non-binding rerun rows are skipped: a median that failed its own
    dispersion gate is not a claim and cannot fail a reproduction either.
    """
    protected = {row["row_id"].split("/", 1)[1]: row for row in protected_rows()}
    for path in sorted(RECORD.parent.glob("*.jsonl")):
        for r in rows_in(path):
            if (r.get("schema_version", 0) < 3 or r.get("run_id") == PROTECTED_RUN
                    or (r.get("measurement") or {}).get("kind") == "ceiling"
                    or not r.get("binding")):
                continue
            old = protected.get(r["row_id"].split("/", 1)[1])
            if old is None or old["measurement"] != r["measurement"]:
                continue  # a new cell: allowed to add, checked elsewhere
            ok, why = measure_baselines.reproduces(old, r)
            assert ok, f"{r['row_id']} fails to reproduce its protected cell: {why}"


# --- the D6 contract: serving cells are mlx-only this block ---------------
#
# The llama.cpp batched-serving baseline is DEFERRED (llama-bench has no
# parallel mode; TODOS.md carries the scoping). The producer therefore ships
# n_parallel decode cells for mlx-lm only, labeled so no cross-stack serving
# comparison can be misread from them.


def test_the_producer_ships_mlx_batch_decode_cells_at_the_ruled_points(specs):
    cells = [s for s in specs if s["kind"] == "batch_decode"]
    assert sorted(s["batch"] for s in cells) == [4, 8, 16]
    assert all(s["stack"] == "mlx-lm" for s in cells)
    # The per-stream shape mirrors the existing mlx/decode cell, so the B=1
    # anchor of the n_parallel series is the already-protected decode cell.
    assert all(s["n_prompt"] == measure_baselines.PREFILL_TOKENS for s in cells)
    assert all(s["n_gen"] == measure_baselines.DECODE_TOKENS for s in cells)


def test_no_llamacpp_batch_decode_spec_exists_this_block(specs):
    assert not [s for s in specs
                if s["stack"] == "llama.cpp" and s["kind"] == "batch_decode"]


def test_each_batch_decode_cell_is_its_own_single_arm_group(specs):
    """Per-cell groups with one arm each: with no second arm in any group, the
    renderer structurally cannot form a serving comparison from these rows."""
    labels = {label: members for label, members in measure_baselines.group_cells(specs)}
    for n in (4, 8, 16):
        members = labels[f"batch_decode-w{n}"]
        assert [m["id"] for m in members] == [f"mlx/batch_decode{n}"]


def test_a_batch_decode_row_carries_its_parallelism_and_its_scope(specs):
    spec = next(s for s in specs if s["kind"] == "batch_decode" and s["batch"] == 8)
    m = measure_baselines.measurement_fields(spec)
    assert m["n_parallel"] == 8
    assert m["matmul_width"] == 8
    assert m["stack_scope"] == "mlx-only"
    assert m["width_mechanism"] == "batch-size"


def test_batch_decode_bytes_charge_kv_per_stream_and_weights_once(monkeypatch):
    """n_parallel streams share one weight read per forward pass but each
    carries its own KV cache; charging KV once would overstate utilisation by
    nearly the stream count at serving depth."""
    monkeypatch.setattr(measure_baselines.mlx_info, "cost_model", lambda path: {})
    monkeypatch.setattr(measure_baselines.mlx_info, "gen_bytes",
                        lambda cost: 1_000_000_000)
    monkeypatch.setattr(measure_baselines.mlx_info, "kv_bytes",
                        lambda cost, ctx: int(ctx * 1_000_000))
    spec = {"id": "mlx/batch_decode4", "stack": "mlx-lm", "kind": "batch_decode",
            "batch": 4, "n_prompt": 1024, "n_gen": 128, "model_path": Path("/m")}
    out = measure_baselines.utilisation(spec, 200.0, {"read_gbs": 100.0,
                                                      "fp16_gflops": 1000.0})
    # per-stream average context 1024 + 128/2; weights once, KV four times
    assert out["bytes_per_pass"] == 1_000_000_000 + 4 * 1_088_000_000
    # 200 aggregate tokens/s across 4 streams is 50 forward passes/s
    assert out["achieved_gbs"] == round(5_352_000_000 * 50 / 1e9, 1)
    assert out["binding_resource"] == "memory"


def batch_decode_row(**over):
    row = valid_row()
    row["stack"] = {"name": "mlx-lm"}
    row["measurement"] = {"kind": "batch_decode", "matmul_width": 8,
                          "n_parallel": 8, "stack_scope": "mlx-only"} | over
    return row


def test_a_well_formed_batch_decode_row_validates():
    assert measure_baselines.validate_row(batch_decode_row())


def test_a_batch_decode_row_missing_its_mlx_only_scope_is_rejected():
    """The label is the D6 deliverable, not decoration: an unlabeled serving
    row is exactly the row a reader would set beside the other stack."""
    with pytest.raises(ValueError, match="mlx-only"):
        measure_baselines.validate_row(batch_decode_row(stack_scope=None))


def test_a_batch_decode_row_without_n_parallel_is_rejected():
    with pytest.raises(ValueError, match="n_parallel"):
        measure_baselines.validate_row(batch_decode_row(n_parallel=None))


def test_a_llamacpp_batch_decode_row_is_refused_this_block():
    """D6 made executable: the deferred llama.cpp serving arm cannot be
    written by this producer, only by the block that builds it honestly."""
    row = batch_decode_row()
    row["stack"] = {"name": "llama.cpp"}
    with pytest.raises(ValueError, match="deferred"):
        measure_baselines.validate_row(row)


def test_the_producer_console_table_carries_the_scope_too():
    """The scope renders wherever the kind appears (SCHEMA.md), and the
    operator reading a run's console summary is a reader too."""
    row = batch_decode_row()
    row["result"] |= {"spread_pct": 0.5}
    row["roofline"] |= {"achieved_gbs": 50.0}
    assert "batch_decode (mlx-only)" in measure_baselines.render([row])


# --- the operator console cell for a row no compute column can place --------


def unknown_resource_row(**roofline):
    """An mlx prefill row: the producer stamps binding_resource 'unknown' for
    every non-llama.cpp prefill, because the MLX checkpoint gives bytes but no
    unambiguous parameter count to place the row on the compute axis."""
    row = valid_row()
    row["stack"] = {"name": "mlx-lm"}
    row["measurement"] = {"kind": "prefill", "matmul_width": 1024,
                          "width_mechanism": "prompt-width"}
    row["result"] |= {"spread_pct": 0.5}
    row["roofline"] |= {"binding_resource": "unknown", "achieved_gbs": 12.3,
                        "roofline_utilisation_pct": None,
                        "bandwidth_utilisation_pct": 9.1, **roofline}
    return row


def test_an_unknown_binding_resource_row_prints_its_bandwidth_percentage():
    """The renderer returns no percentage for an 'unknown' row on purpose: the
    published matrix must not read a prefill row as a bandwidth catastrophe.
    The console summary is the operator card's own table and has always shown
    the bandwidth number there, so formatting the renderer's None straight into
    the cell put a literal 'None%' in front of the operator.
    """
    table = measure_baselines.render([unknown_resource_row()])
    assert "None%" not in table
    # The axis is named in the cell: a bare percentage under "% of that
    # ceiling" beside a "binds on" of unknown is the mixed reading
    # matrix.utilisation() refuses to publish, and two readers of the same
    # run must not disagree about which ceiling a number is measured against.
    assert "| unknown | 9.1% (bandwidth) |" in table


def test_a_cell_with_no_percentage_at_all_says_so_rather_than_printing_None():
    table = measure_baselines.render(
        [unknown_resource_row(bandwidth_utilisation_pct=None)])
    assert "| unknown | n/a |" in table


def test_the_shipped_v3_record_still_validates_under_todays_producer():
    """The other half of only-add: tightening the contract for new cells must
    not retroactively reject the rows already in the record."""
    for row in rows_in(RECORD):
        if row["schema_version"] == 3:
            assert measure_baselines.validate_row(row)
