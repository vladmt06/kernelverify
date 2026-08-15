"""Device-arithmetic members join the quant contract population; K re-derived.

PRE-REGISTRATION
================
This docstring is committed before the calibration runs. The rule below was
dispatched by the coordinating lane and confirmed before any measurement; if
the numbers force a deviation the run stops and the rule is renegotiated
before anything further is looked at, never after.

What this discharges
--------------------
Phase 0 calibrated K = 3 for the MLX-affine contract from five CPU members and
pre-registered the device-arithmetic branch: a correct Metal kernel exceeding
the CPU-calibrated tolerance triggers ensemble RE-DERIVATION with device
members, never a false-positive ship. The members now exist
(``kernelverify/schemas/quant_device.py``): three raw-MSL implementations,
correct by construction, spanning both evaluation domains on the device. This
harness evaluates the trigger, performs the membership join, and re-derives K,
in that order.

The rule
--------
For each bits in {2, 3, 4, 8}, on the ADR 0009 grid (shapes x weight draws x
seeds x input modes x activation dtypes, calibration seeds 0/1 and independent
draw 100/101):

1. G0 unchanged: `canonical_quantize` bit-exact against `mx.quantize`, or the
   width stops.

2. TRIGGER, measured before any repair: every device implementation (the
   three members and MLX's own `quantized_matmul`, which stays held out)
   against the CPU-calibrated tolerance `max(base, 3 * cpu_floor)`. Any
   exceedance on any case means the pre-registered branch fires at that
   width. The count, the members and the worst overshoot are reported either
   way, because "covered" is a measurement, not an assumption.

3. MEMBERSHIP BEFORE K, the repair order ADR 0005 and ADR 0009 both fixed:
   the device-arithmetic class joins the floor first, unconditionally - it is
   a distinct class of the promised population, and a class the floor holds no
   member of is exactly the artifact that manufactured K = 7.02 once. Only
   then is K re-derived, under Phase 0's own rule:

   - k_needed = the largest ratio, over calibration-draw records, of any
     member's error to the floor of the OTHER members (leave-one-member-out,
     full membership), and of any held-out implementation's error to the full
     floor, counting only cases where the error exceeds the base tolerance.
   - The same quantity measured on the independent draw must not exceed the
     grid value chosen from the calibration draw; if it does, the next grid
     value that covers it is taken and the miss is reported.
   - The shipped K stays 3 unless some width demands more AFTER the
     membership join; K moving while membership could still repair would
     invert the pre-registered order.

4. Gates at the shipped K with full membership, per width, exactly ADR 0009's
   amended gate: G1 zero false positives (held-out implementations, both
   draws); G2 every non-equivalent artefact fault caught on every iid-mode
   case at >= 10x margin, equivalence doctrine unchanged; G3 every class
   carries >= 2 distinct members, counted on values rather than names, now
   over THREE classes (dequant-domain, int-domain, device-arithmetic), with
   cross-class duplicates reported.

5. Leave-one-class-out stays the class-NECESSITY diagnostic and can never
   fail an ensemble. It now answers the new question: is the device class
   load-bearing, or does the CPU floor already cover device arithmetic?

6. The fp16-dequant boundary is re-measured under BOTH floors, CPU-only and
   device-joined, per width and per activation dtype. Joining a class can
   only raise the floor, so ADR 0009's enforcement table can only weaken;
   by how much is a certificate-spec input this run must put on the record.

7. KILL: at some width no K in the grid passes G1 and G2 with full
   membership. Then the device class cannot be admitted at a survivable
   tolerance and quant device work stops pending renegotiation.

Costs are grouped by spec from day one: one Metal compile per (member,
activation dtype), every case after that a ~ms dispatch, which is what makes
the full grid affordable (ADR 0006 measured ~180 ms fixed, ~1.4 ms marginal).

AMENDMENT, 2026-08-15: re-run under the repaired `factored-groups`
-----------------------------------------------------------------
This harness is being re-run because one ensemble member changed. The
`factored-groups` member formed its per-group sums with numpy's pairwise
reduction, which is EXACT on a constant row, so it was most accurate exactly
where a real int-accumulate kernel is least accurate; the floor it feeds was
too tight there and the shipped tolerance flagged a correct in-contract device
kernel on 10 of the 1,536 committed serving records. Both sums are now explicit
fp32 chains.

The rule above is unchanged and still binds. What this amendment fixes, before
the numbers are seen, is what the re-run's outcomes MEAN, since a looser member
can only move things in known directions:

a. K is expected to FALL, not rise. The repaired member has larger error on
   constant-rows cases, so the leave-one-out floor those cases produce is
   larger and the ratios that set k_needed are smaller. A fall is the expected
   reading and is adopted under step 3's own rule; K = 3 is the value the
   review's analysis predicts.

b. A FALL IS NOT AUTOMATICALLY SHIPPED. Step 4's gates decide. A smaller K
   tightens every tolerance the verifier ships, so G1 (zero false positives on
   held-out implementations, both draws) and G2 (every non-equivalent fault
   caught at >= 10x margin) must both pass at the new K with full membership,
   per width, exactly as before. If G1 or G2 fails at the re-derived K, the
   shipped K stays where it is and the gap is reported, never papered over.

c. K RISING is a stop. Nothing about lowering one member's accuracy can
   legitimately demand a larger K; a rise means the repair did something other
   than what it claims, and the run is read no further until that is explained.

d. The detection price is the headline of step 6, not a footnote. The repair
   raises the floor on constant-rows, so fault detection at the margin can only
   weaken. Every cell of the fp16-dequant boundary table that flips
   caught -> not-caught is named individually in ADR 0016 with its margin
   before and after, whatever the count turns out to be.

e. The five untouched CPU members must reproduce their committed values
   bit-identically (tests/test_quant_contract_members.py pins this on a 64
   record sample). Any other member moving means the environment moved, not
   the repair, and this run's numbers describe neither.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernelverify.schemas.quant_contract import (  # noqa: E402
    ENSEMBLE,
    FAULTS,
    QuantContract,
    r_contract,
)
from kernelverify.schemas.quant_device import (  # noqa: E402
    DEVICE_CLASS,
    DEVICE_MEMBERS,
    DeviceMemberSession,
    make_run_case,
    result_outputs,
)
from calibrate_quant_bits import (  # noqa: E402
    BATCH,
    CALIBRATION_SEEDS,
    CLASSES as CPU_CLASSES,
    GROUP_SIZE,
    HELDOUT_SEEDS,
    IID_MODES,
    K_GRID,
    K_QUANT,
    MARGIN_GATE,
    MODES,
    SHAPES,
    WEIGHT_DRAWS,
    base_tol,
    boundary_fp16_dequant,
    mlx_on_device,
    verify_against_mlx,
)
from phase0_contract_k import err, heldout_block_tiled, make_w, make_x  # noqa: E402

BITS = (2, 3, 4, 8)
CPU_MEMBERS = tuple(ENSEMBLE)
ALL_MEMBERS = CPU_MEMBERS + DEVICE_MEMBERS
CLASSES = {**CPU_CLASSES, DEVICE_CLASS: DEVICE_MEMBERS}
HELDOUTS = ("block-tiled", "mlx-on-device")

OUT_PATH = Path(__file__).with_name(".cache") / "quant_device_adequacy.json"


def floor_of(record: dict, members) -> float:
    return max(record["members"][m] for m in members)


def measure(bits: int, shapes: list, session: DeviceMemberSession,
            verbose: bool) -> dict:
    """Every record for one width: CPU members, device members, held-outs,
    faults and the boundary case, on the ADR 0009 grid and seeds."""
    contract = QuantContract(scheme="mlx-affine", bits=bits, group_size=GROUP_SIZE)
    records, exactness = [], []
    for d_out, d_in in shapes:
        for draw in WEIGHT_DRAWS:
            for seed in CALIBRATION_SEEDS + HELDOUT_SEEDS:
                rng = np.random.default_rng(10_000 + seed)
                w = make_w(d_out, d_in, draw, rng)
                artefact, exact = verify_against_mlx(w, contract)
                exactness.append(exact)
                carrier = {
                    "q": np.ascontiguousarray(artefact.q.astype(np.float16)),
                    "scales": np.ascontiguousarray(artefact.scales.astype(np.float32)),
                    "biases": np.ascontiguousarray(artefact.biases.astype(np.float32)),
                }
                params = {"D_IN": d_in, "D_OUT": d_out,
                          "N_GROUPS": d_in // GROUP_SIZE, "GROUP": GROUP_SIZE,
                          "BATCH": BATCH}
                for mode in MODES:
                    for dtype in ("float32", "float16"):
                        npdt = np.float32 if dtype == "float32" else np.float16
                        x = make_x(BATCH, d_in, npdt, mode, rng)
                        ref = r_contract(x, artefact)
                        scale = float(np.abs(ref).max())
                        members = {n: err(f(x, artefact), ref)
                                   for n, f in ENSEMBLE.items()}
                        case = make_run_case(
                            inputs={"x": np.ascontiguousarray(x), **carrier},
                            params=params,
                            output_shapes=[((BATCH, d_out), dtype)],
                            label=f"b{bits} {d_out}x{d_in} {draw} s{seed} {mode} {dtype}",
                        )
                        for member in DEVICE_MEMBERS:
                            kernel = session.compiled(member, dtype)
                            result = kernel.run(case, warmup=0, repeats=0)
                            if not result.ok:
                                raise RuntimeError(
                                    f"trusted member {member} failed on "
                                    f"{case.label}: {result.status.value} - "
                                    f"{result.detail}")
                            members[member] = err(result_outputs(result)[0], ref)
                        records.append({
                            "bits": bits, "shape": f"{d_out}x{d_in}",
                            "draw": draw, "seed": seed,
                            "heldout_draw": seed in HELDOUT_SEEDS,
                            "mode": mode, "dtype": dtype,
                            "ref_scale": scale,
                            "base_tol": base_tol(dtype, scale),
                            "members": members,
                            "heldout": {
                                "block-tiled": err(heldout_block_tiled(x, artefact), ref),
                                "mlx-on-device": err(mlx_on_device(x, w, contract), ref),
                            },
                            "faults": {n: err(ENSEMBLE["dequant-pairwise"](x, f(artefact)), ref)
                                       for n, f in FAULTS.items()},
                            "boundary_fp16_dequant": err(boundary_fp16_dequant(x, artefact), ref),
                        })
        if verbose:
            print(f"    bits={bits} {d_out}x{d_in} done ({len(records)} records)")
    return {"records": records, "bit_exact": all(exactness),
            "exact_checks": len(exactness)}


# ---------------------------------------------------------------------------
# Step 2: the trigger, before any repair
# ---------------------------------------------------------------------------
def trigger(records: list) -> dict:
    """Device implementations against the CPU-calibrated tolerance."""
    device_names = DEVICE_MEMBERS + ("mlx-on-device",)
    exceed, worst, worst_at = {n: 0 for n in device_names}, 0.0, ""
    for r in records:
        tol = max(r["base_tol"], K_QUANT * floor_of(r, CPU_MEMBERS))
        for name in device_names:
            e = r["members"][name] if name in r["members"] else r["heldout"][name]
            if e > tol:
                exceed[name] += 1
                if tol > 0 and e / tol > worst:
                    worst = e / tol
                    worst_at = (f"{name} @ {r['shape']} {r['draw']} "
                                f"s{r['seed']} {r['mode']} {r['dtype']}")
    total = sum(exceed.values())
    return {"cases": len(records), "exceedances": exceed, "total": total,
            "fires": total > 0, "worst_overshoot": worst, "worst_at": worst_at}


# ---------------------------------------------------------------------------
# Step 3: K re-derivation with full membership, Phase 0's rule
# ---------------------------------------------------------------------------
def k_demand(records: list) -> tuple[float, str]:
    """The K the full-membership class demands over these records."""
    needed, binding = 0.0, ""
    for r in records:
        base = r["base_tol"]
        for name in ALL_MEMBERS:
            e = r["members"][name]
            others = max(r["members"][m] for m in ALL_MEMBERS if m != name)
            if e > base and others > 0 and e / others > needed:
                needed = e / others
                binding = (f"member {name} @ {r['shape']} {r['draw']} "
                           f"s{r['seed']} {r['mode']} {r['dtype']}")
        full = floor_of(r, ALL_MEMBERS)
        for name in HELDOUTS:
            e = r["heldout"][name]
            if e > base and full > 0 and e / full > needed:
                needed = e / full
                binding = (f"heldout {name} @ {r['shape']} {r['draw']} "
                           f"s{r['seed']} {r['mode']} {r['dtype']}")
    return needed, binding


# ---------------------------------------------------------------------------
# Step 4-6: the amended gate at the shipped K, plus diagnostic and boundary
# ---------------------------------------------------------------------------
def gates(records: list, k: float) -> dict:
    report = {"false_positives": [], "class_out": {}, "faults": {},
              "class_size": {}, "duplicates": []}

    for r in records:
        tol = max(r["base_tol"], k * floor_of(r, ALL_MEMBERS))
        for name in HELDOUTS:
            if r["heldout"][name] > tol:
                report["false_positives"].append(
                    f"heldout {name} @ {r['shape']} {r['mode']} {r['dtype']} "
                    f"seed={r['seed']} ({r['heldout'][name]:.3g} > {tol:.3g})")

    # -- class necessity: drop a class, watch its members against the rest --
    for class_name, members in CLASSES.items():
        remaining = [m for m in ALL_MEMBERS if m not in members]
        worst, failures = 0.0, 0
        for r in records:
            tol = max(r["base_tol"], k * floor_of(r, remaining))
            for m in members:
                e = r["members"][m]
                if tol > 0:
                    worst = max(worst, e / tol)
                if e > tol:
                    failures += 1
        report["class_out"][class_name] = {"worst_ratio": worst,
                                           "failures": failures,
                                           "remaining": len(remaining)}

    # -- faults under the enlarged floor, equivalence doctrine unchanged ----
    iid = [r for r in records if r["mode"] in IID_MODES]
    for name in FAULTS:
        margins, caught, anywhere = [], 0, 0
        for r in records:
            tol = max(r["base_tol"], k * floor_of(r, ALL_MEMBERS))
            hit = r["faults"][name] > tol
            anywhere += hit
            if r["mode"] in IID_MODES:
                caught += hit
                margins.append(r["faults"][name] / tol if tol > 0 else float("inf"))
        report["faults"][name] = {"caught": caught, "of": len(iid),
                                  "worst_margin": min(margins) if margins else 0.0,
                                  "equivalent": anywhere == 0}

    # -- G3 on values, across classes as well as within ---------------------
    signature = {m: tuple(round(r["members"][m], 17) for r in records[:24])
                 for m in ALL_MEMBERS}
    for class_name, members in CLASSES.items():
        report["class_size"][class_name] = {
            "named": len(members),
            "distinct": len({signature[m] for m in members}),
        }
    seen: dict[tuple, str] = {}
    for m in ALL_MEMBERS:
        if signature[m] in seen:
            report["duplicates"].append(f"{m} == {seen[signature[m]]}")
        else:
            seen[signature[m]] = m

    # -- the boundary under both floors, per activation dtype ---------------
    boundary = {}
    for dtype in ("float32", "float16"):
        rows = [r for r in records if r["dtype"] == dtype]
        cpu = sum(1 for r in rows if r["boundary_fp16_dequant"]
                  > max(r["base_tol"], k * floor_of(r, CPU_MEMBERS)))
        full = sum(1 for r in rows if r["boundary_fp16_dequant"]
                   > max(r["base_tol"], k * floor_of(r, ALL_MEMBERS)))
        boundary[dtype] = {"cpu_floor": cpu, "full_floor": full, "of": len(rows)}
    report["boundary"] = boundary

    scored = {n: f for n, f in report["faults"].items() if not f["equivalent"]}
    report["equivalent_faults"] = [n for n, f in report["faults"].items()
                                  if f["equivalent"]]
    worst_margin = min((f["worst_margin"] for f in scored.values()), default=0.0)
    report["worst_margin"] = worst_margin
    report["gates"] = {
        "G1_no_false_positives": not report["false_positives"],
        "G2_faults_caught": all(f["caught"] == f["of"] for f in scored.values())
                            and worst_margin >= MARGIN_GATE,
        "G3_class_coverage": all(c["distinct"] >= 2
                                 for c in report["class_size"].values()),
    }
    report["verdict"] = ("ADEQUATE" if all(report["gates"].values())
                         else "INADEQUATE")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="device members join the quant ensemble; K re-derived")
    parser.add_argument("--bits", type=int, nargs="*", default=list(BITS))
    parser.add_argument("--full-shapes", action="store_true",
                        help="restore Phase 0's shape set, including 2048x11008")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        session = DeviceMemberSession()
    except Exception as error:  # no PyObjC Metal, no GPU: nothing to calibrate
        print(f"no usable Metal device for the device members: {error}")
        return 2

    shapes = SHAPES + [(2048, 11008)] if args.full_shapes else SHAPES
    print(f"contract: mlx-affine, group_size={GROUP_SIZE}; "
          f"CPU-calibrated K={K_QUANT} (Phase 0)")
    print(f"membership: {len(CPU_MEMBERS)} CPU members + "
          f"{len(DEVICE_MEMBERS)} device members ({', '.join(DEVICE_MEMBERS)})")
    print(f"bits: {args.bits}; shapes: {shapes}")
    print(f"calibration seeds {CALIBRATION_SEEDS}, independent draw {HELDOUT_SEEDS}\n")

    results, out = {}, {}
    for bits in args.bits:
        print(f"  measuring bits={bits} ...")
        measured = measure(bits, shapes, session, not args.quiet)
        results[bits] = measured
        if not measured["bit_exact"]:
            print(f"  GATE 0 FAILED at bits={bits}: canonical_quantize is not "
                  f"bit-exact against mx.quantize, so nothing here describes MLX")
            out[bits] = {"verdict": "GATE-0-FAIL"}
            continue
        records = measured["records"]
        cal = [r for r in records if not r["heldout_draw"]]
        indep = [r for r in records if r["heldout_draw"]]
        needed_cal, binding_cal = k_demand(cal)
        needed_indep, binding_indep = k_demand(indep)
        out[bits] = {
            "trigger": trigger(records),
            "k_needed_calibration": needed_cal, "k_binding": binding_cal,
            "k_needed_independent": needed_indep,
            "k_binding_independent": binding_indep,
        }

    # -- the shipped K, decided across widths after the membership join -----
    usable = [b for b in args.bits if out[b].get("verdict") != "GATE-0-FAIL"]
    demand = max((max(out[b]["k_needed_calibration"],
                      out[b]["k_needed_independent"]) for b in usable),
                 default=0.0)
    k_ship = K_QUANT if demand <= K_QUANT else next(
        (k for k in K_GRID if k >= demand), None)
    if k_ship is None:
        print(f"\nKILL: full membership demands K = {demand:.2f}, above the grid")
        return 1

    for bits in usable:
        out[bits].update(gates(results[bits]["records"], k_ship))

    # ------------------------------------------------------------------ print
    print("\nTRIGGER: device implementations vs the CPU-calibrated tolerance "
          f"(K={K_QUANT}, CPU floor)")
    for bits in usable:
        t = out[bits]["trigger"]
        state = ("FIRES" if t["fires"] else "covered")
        detail = (f", worst {t['worst_overshoot']:.2f}x at {t['worst_at']}"
                  if t["fires"] else "")
        by = {n: c for n, c in t["exceedances"].items() if c}
        print(f"  bits={bits:<3} {t['total']}/{t['cases']} case-exceedances "
              f"-> re-derivation branch {state}{detail} {by if by else ''}")

    print("\nK RE-DERIVATION, membership joined first (Phase 0 rule, full ensemble):")
    for bits in usable:
        o = out[bits]
        print(f"  bits={bits:<3} k_needed {o['k_needed_calibration']:.3f} "
              f"(calibration draw; binding: {o['k_binding']})")
        print(f"          k_needed {o['k_needed_independent']:.3f} "
              f"(independent draw)")
    moved = "" if k_ship == K_QUANT else "  <- K MOVED, membership could not repair"
    print(f"  shipped K: {k_ship} (demand {demand:.3f}, grid {K_GRID}){moved}")

    header = (f"{'bits':>6}{'records':>9}{'G1 FP':>7}{'G2 worst margin':>17}"
              f"{'G3 classes':>12}{'equiv':>7}{'verdict':>13}")
    print()
    print("=" * len(header))
    print(f"AMENDED GATE AT K = {k_ship}, THREE-CLASS MEMBERSHIP")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for bits in usable:
        rep = out[bits]
        sizes = "/".join(str(rep["class_size"][c]["distinct"]) for c in CLASSES)
        print(f"{bits:>6}{len(results[bits]['records']):>9}"
              f"{len(rep['false_positives']):>7}{rep['worst_margin']:>16.1f}x"
              f"{sizes:>12}{len(rep['equivalent_faults']):>7}"
              f"{rep['verdict']:>13}")
    print("=" * len(header))
    print(f"G3 classes = distinct members per class ({'/'.join(CLASSES)}), "
          "gate is >= 2 each")
    for bits in usable:
        for line in out[bits]["duplicates"]:
            print(f"  bits={bits} value-duplicate members: {line}")
        for line in out[bits]["false_positives"]:
            print(f"  bits={bits} FP {line}")

    print("\nCLASS-NECESSITY diagnostic (drop a class; ratios above 1 mean "
          "load-bearing, never a gate):")
    for bits in usable:
        for class_name, c in out[bits]["class_out"].items():
            print(f"  bits={bits:<3} drop {class_name:<18} "
                  f"{c['remaining']} members left, worst ratio {c['worst_ratio']:7.3f}, "
                  f"{c['failures']} exceed tolerance")

    print("\nfp16-dequant boundary, cases outside tolerance (CPU floor -> "
          "device-joined floor), per activation dtype:")
    for bits in usable:
        b = out[bits]["boundary"]
        f32, f16 = b["float32"], b["float16"]
        print(f"  bits={bits:<3} float32: {f32['cpu_floor']}/{f32['of']} -> "
              f"{f32['full_floor']}/{f32['of']}    "
              f"float16: {f16['cpu_floor']}/{f16['of']} -> "
              f"{f16['full_floor']}/{f16['of']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(
        {"k_ship": k_ship, "k_cpu": K_QUANT, "shapes": shapes,
         "classes": {c: list(m) for c, m in CLASSES.items()},
         "reports": out,
         "records": {b: results[b]["records"] for b in results}}, default=float))
    print(f"\nrecords: {OUT_PATH}")

    verdicts = {out[b].get("verdict") for b in args.bits}
    print(f"VERDICT: {'ADEQUATE at every width' if verdicts == {'ADEQUATE'} else sorted(verdicts)}")
    return 0 if verdicts == {"ADEQUATE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
