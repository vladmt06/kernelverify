"""Per-bits ensemble adequacy for the MLX-affine quant contract.

PRE-REGISTRATION
================
This docstring is committed before the calibration runs. The decision rule
below was agreed with the coordinating lane in advance, and if the numbers
force a deviation from it the run stops and the rule is renegotiated before
anything further is looked at, never after.

Why adequacy and not K
----------------------
Phase 0 calibrated K = 3 for MLX-affine at bits = 4. The corpus-side
measurement (ADR 0005) found that K is very nearly inert: over 9,678
(case, fault) pairs not one verdict moved anywhere between K = 1.0 and
K = 6.0, while the ensemble MEMBERSHIP moved the floor by up to 8.45x. The
same shape shows on this surface, where the smallest constructible artefact
fault sits hundreds of times above tolerance. So the deliverable here is
per-bits ensemble ADEQUACY, with K secondary.

The rule
--------
For each bits in {2, 3, 8}, with bits = 4 carried as the control:

1. Gate 0. `canonical_quantize` must be bit-exact against `mx.quantize` -
   codes, scales and biases - at that width. If it is not, the run reports
   the replica's divergence and stops, because a contract reference that is
   not MLX's own output cannot anchor a claim about MLX.

2. Score the ensemble by leave-one-CLASS-out, never leave-one-member-out.
   Removing a single member measures within-class spread; removing a class
   measures whether the class was needed. Phase 0 saw K blow up to 7.02
   purely because the int-domain class held one member, so leave-one-out
   deleted the class and measured its absence; a second member collapsed it
   to 2.27. The corpus side reproduced the same artifact independently, at
   K = 2.119 with a real false positive for a one-member ensemble against
   K = 1.000 for two.

3. Score also against INDEPENDENT DRAWS: the floor is calibrated on one seed
   stream and validated on another, so no ensemble is credited for covering
   the very cases it was measured on.

4. ADEQUACY at a given bits = zero false positives on held-out members and
   on independent draws, AND every artefact fault caught with margin >= 10x,
   at that width.

   AMENDED, after the bits=4 control and before any of widths 2, 3 and 8 ran.
   Clause 2 above originally scored leave-one-class-out as a pass/fail gate,
   and the control failed it 17 times. That gate was wrong: no ensemble can
   cover a class it holds no member of, which is what an unrepresented class
   means, so as a pass/fail test it fails every ensemble whose classes are
   genuinely distinct, and fails hardest when they are most distinct. It
   measures the opposite of adequacy. The control was a calibration case whose
   answer Phase 0 had already pinned, not an experimental width, and the
   amendment was settled before any experimental width ran. Both readings of
   the control are reported, so nobody has to take the timing on trust.

   The amended gate, in full:

     G0  the canonical quantizer replica is bit-exact against mx.quantize.
     G1  zero false positives for the FULL ensemble, on held-out correct
         implementations and on the independent seed draw.
     G2  every NON-EQUIVALENT artefact fault caught on every iid-mode case,
         at a margin of 10x or better. A fault caught on no case at all at
         this width is an equivalent mutant there, excluded with its reason
         stated, exactly as the corpus catalogue treats equivalents.
     G3  STRUCTURAL: every class carries at least two members. This stays a
         gate rather than folding into the diagnostic, because it is the
         precise artifact that produced K = 7.02 on this surface and K = 2.119
         on the corpus surface, and a diagnostic would let it drift.

   Leave-one-class-out is now the class-NECESSITY diagnostic. A ratio above 1
   means the class is load-bearing and must be kept. It can never fail an
   ensemble.

5. K stays fixed at 3 across all widths UNLESS adequacy fails somewhere. On a
   failure the FIRST repair is membership - add the missing class member -
   and K moves only if membership cannot repair it.

6. KILL: no membership set combined with any K in the grid passes both sides
   at some bits level.

7. Measure the fp16-dequant boundary PER BITS. At bits = 8 the quantization
   step may be fine enough that fp16 intermediates fall INSIDE tolerance,
   which would make the contract's intermediate-precision requirement
   bits-dependent. That is a certificate-spec input either way.

Ensemble classes, defined before the run
----------------------------------------
Classes are evaluation DOMAINS, because that is what changes the rounding
sequence rather than merely reordering it:

- dequant-domain: materialise W~ in fp32, then multiply and accumulate.
  Members: dequant-pairwise, dequant-serial, dequant-reversed, lut-gather.
- int-domain: accumulate against the integer codes and apply scale and bias
  per group, the shape every int-accumulate quantized GEMV kernel takes.
  Members: factored-groups, factored-serial.

Recorded in advance, because it changes how the leave-one-class-out result
must be read: `lut-gather` returns bit-identical output to `dequant-pairwise`
at bits 2, 3, 4 and 8, verified before this run. It is a distinct way of
BUILDING W~ but not a distinct way of rounding it, so the ensemble holds five
numerically distinct members, not six, and the dequant class holds three.
That is still two members per class, so the class-coverage rule holds; the
consequence is only that a leave-one-member-out on either of that pair would
have measured nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from kernelverify.schemas.quant_contract import (  # noqa: E402
    ENSEMBLE,
    FAULTS,
    QuantArtefact,
    QuantContract,
    canonical_quantize,
    dequantize,
    r_contract,
)
from phase0_contract_k import (  # noqa: E402
    BATCH,
    MODES,
    WEIGHT_DRAWS,
    err,
    heldout_block_tiled,
    make_w,
    make_x,
    unpack_mlx_q,
)

BITS = (2, 3, 4, 8)          # 4 is Phase 0's control width
CONTROL_BITS = 4
GROUP_SIZE = 64
K_QUANT = 3.0                # what Phase 0 shipped; fixed unless adequacy fails
K_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
MARGIN_GATE = 10.0           # every artefact fault, every width


def cover(demand: float) -> float:
    """The smallest grid value at or above `demand`.

    Lives here, beside the grid it reads, because every harness that derives a
    K needs the same lookup and two copies are how one of them keeps an old
    rule after the other is amended.

    A demand no grid value covers RAISES. That is the KILL branch every
    calibration in this repo pre-registers: clamping to the top of the grid
    would ship a tolerance already measured to be too tight, and say nothing
    about it.
    """
    for k in K_GRID:
        if k >= demand:
            return k
    raise ValueError(
        f"demand {demand:.3f} is above the K grid {K_GRID}: this is the KILL "
        f"branch, and no grid value may be clamped to cover it")

IID_MODES = ("unit", "corpus-scale")
CALIBRATION_SEEDS = (0, 1)
HELDOUT_SEEDS = (100, 101)   # the independent draw

# Smaller than Phase 0's set and stated rather than silently capped: this
# sweep runs four bit widths where Phase 0 ran one, and the serial members
# materialise a (batch, d_out, d_in) tensor. The largest Phase 0 shape is
# carried so the regime is represented; see --shapes to restore the full set.
SHAPES = [(512, 512), (1024, 4096), (4096, 4096)]

CLASSES = {
    "dequant-domain": ("dequant-pairwise", "dequant-serial",
                       "dequant-reversed", "lut-gather"),
    "int-domain": ("factored-groups", "factored-serial"),
}

OUT_PATH = Path(__file__).with_name(".cache") / "quant_bits_adequacy.json"


def boundary_fp16_dequant(x: np.ndarray, a: QuantArtefact) -> np.ndarray:
    """Dequantize to fp16, then accumulate in fp32: legal or fault, measured."""
    w16 = dequantize(a, np.float32).astype(np.float16).astype(np.float32)
    return (x.astype(np.float32) @ w16.T).astype(x.dtype)


def base_tol(dtype: str, ref_scale: float) -> float:
    """Phase 0's absolute floor, kept identical so widths stay comparable."""
    eps = 9.77e-4 if dtype == "float16" else 1.19e-7
    return 4.0 * eps * ref_scale


def verify_against_mlx(w: np.ndarray, contract: QuantContract):
    """Gate 0: the numpy replica must reproduce mx.quantize exactly."""
    ours = canonical_quantize(w, contract)
    w_q, scales, biases = mx.quantize(mx.array(w), group_size=contract.group_size,
                                      bits=contract.bits)
    mx.eval(w_q)
    theirs_q = unpack_mlx_q(np.array(w_q), w.shape[1], contract.bits)
    exact = (np.array_equal(ours.q, theirs_q)
             and np.array_equal(ours.scales.astype(np.float32),
                                np.array(scales).astype(np.float32))
             and np.array_equal(ours.biases.astype(np.float32),
                                np.array(biases).astype(np.float32)))
    return ours, exact


def mlx_on_device(x: np.ndarray, quantized, contract: QuantContract) -> np.ndarray:
    """MLX's own quantized_matmul: a correct implementation we did not write.

    `quantized` is the mx.quantize triplet of the case's weight matrix,
    hoisted to the per-seed loop (mx.quantize is deterministic, so quantizing
    once per weight draw instead of once per record changes nothing printed).
    """
    w_q, scales, biases = quantized
    out = mx.quantized_matmul(mx.array(x), w_q, scales, biases, transpose=True,
                              group_size=contract.group_size, bits=contract.bits)
    mx.eval(out)
    return np.array(out)


def measure(bits: int, shapes: list, verbose: bool) -> dict:
    """Every record for one bit width."""
    contract = QuantContract(scheme="mlx-affine", bits=bits, group_size=GROUP_SIZE)
    records, exactness = [], []
    for d_out, d_in in shapes:
        for draw in WEIGHT_DRAWS:
            for seed in CALIBRATION_SEEDS + HELDOUT_SEEDS:
                rng = np.random.default_rng(10_000 + seed)
                w = make_w(d_out, d_in, draw, rng)
                artefact, exact = verify_against_mlx(w, contract)
                exactness.append(exact)
                quantized = mx.quantize(mx.array(w), group_size=contract.group_size,
                                        bits=contract.bits)
                for mode in MODES:
                    for dtype in ("float32", "float16"):
                        npdt = np.float32 if dtype == "float32" else np.float16
                        x = make_x(BATCH, d_in, npdt, mode, rng)
                        ref = r_contract(x, artefact)
                        scale = float(np.abs(ref).max())
                        records.append({
                            "bits": bits,
                            "shape": f"{d_out}x{d_in}",
                            "draw": draw,
                            "seed": seed,
                            "heldout_draw": seed in HELDOUT_SEEDS,
                            "mode": mode,
                            "dtype": dtype,
                            "ref_scale": scale,
                            "base_tol": base_tol(dtype, scale),
                            "members": {n: err(f(x, artefact), ref)
                                        for n, f in ENSEMBLE.items()},
                            "heldout": {
                                "block-tiled": err(heldout_block_tiled(x, artefact), ref),
                                "mlx-on-device": err(mlx_on_device(x, quantized, contract), ref),
                            },
                            "faults": {n: err(ENSEMBLE["dequant-pairwise"](x, f(artefact)), ref)
                                       for n, f in FAULTS.items()},
                            "boundary_fp16_dequant": err(boundary_fp16_dequant(x, artefact), ref),
                        })
        if verbose:
            print(f"    bits={bits} {d_out}x{d_in} done ({len(records)} records)")
    return {"records": records, "bit_exact": all(exactness),
            "exact_checks": len(exactness)}


def floor_of(record: dict, members) -> float:
    return max(record["members"][n] for n in members)


def adequacy(records: list, k: float) -> dict:
    """The two-sided verdict at one bit width."""
    all_members = tuple(ENSEMBLE)
    report = {"false_positives": [], "class_out": {}, "faults": {}, "boundary": 0}

    # -- side one: does the floor cover every correct implementation? -------
    for r in records:
        tol = max(r["base_tol"], k * floor_of(r, all_members))
        for name, e in r["heldout"].items():
            if e > tol:
                report["false_positives"].append(
                    f"heldout {name} @ {r['shape']} {r['mode']} {r['dtype']} "
                    f"seed={r['seed']} ({e:.3g} > {tol:.3g})")

    # -- leave-one-CLASS-out, scored on the independent draw ---------------
    for class_name, members in CLASSES.items():
        remaining = [m for m in all_members if m not in members]
        worst, failures = 0.0, 0
        for r in records:
            tol = max(r["base_tol"], k * floor_of(r, remaining))
            for m in members:
                e = r["members"][m]
                if tol > 0:
                    worst = max(worst, e / tol)
                if e > tol:
                    failures += 1
        report["class_out"][class_name] = {
            "worst_ratio": worst, "failures": failures,
            "remaining": len(remaining),
        }

    # -- side two: is every non-equivalent artefact fault still caught? ----
    # A fault caught on no case anywhere at this width is an equivalent mutant
    # there and is excluded with its reason, the way the corpus catalogue
    # excludes mutations no oracle could ever see. A fault caught somewhere but
    # not on a particular structured case is a per-case equivalence, reported
    # and not scored, which is the doctrine the Phase 0 harness already used.
    iid = [r for r in records if r["mode"] in IID_MODES]
    for name in FAULTS:
        margins, caught, caught_anywhere = [], 0, 0
        for r in records:
            tol = max(r["base_tol"], k * floor_of(r, all_members))
            hit = r["faults"][name] > tol
            caught_anywhere += hit
            if r["mode"] in IID_MODES:
                caught += hit
                margins.append(r["faults"][name] / tol if tol > 0 else float("inf"))
        report["faults"][name] = {
            "caught": caught, "of": len(iid),
            "worst_margin": min(margins) if margins else 0.0,
            "equivalent": caught_anywhere == 0,
        }

    report["boundary"] = sum(
        1 for r in records
        if r["boundary_fp16_dequant"] > max(r["base_tol"], k * floor_of(r, tuple(ENSEMBLE))))
    report["boundary_of"] = len(records)

    # -- G3, structural: every class needs two members that actually differ --
    # Counted on DISTINCT members, not on names: lut-gather returns
    # bit-identical output to dequant-pairwise at every width, so a name count
    # would credit the dequant class with a member that cannot move the floor.
    report["class_size"] = {}
    for class_name, members in CLASSES.items():
        signatures = {tuple(round(r["members"][m], 17) for r in records[:8]) for m in members}
        report["class_size"][class_name] = {"named": len(members),
                                            "distinct": len(signatures)}

    scored = {n: f for n, f in report["faults"].items() if not f["equivalent"]}
    report["equivalent_faults"] = [n for n, f in report["faults"].items() if f["equivalent"]]
    worst_margin = min((f["worst_margin"] for f in scored.values()), default=0.0)

    gates = {
        "G1_no_false_positives": not report["false_positives"],
        "G2_faults_caught": all(f["caught"] == f["of"] for f in scored.values())
                            and worst_margin >= MARGIN_GATE,
        "G3_class_coverage": all(c["distinct"] >= 2 for c in report["class_size"].values()),
    }
    report["gates"] = gates
    report["worst_margin"] = worst_margin
    report["verdict"] = "ADEQUATE" if all(gates.values()) else "INADEQUATE"
    # The pre-amendment reading, kept so the ADR can report both without a rerun.
    report["verdict_as_first_written"] = (
        "ADEQUATE" if (all(gates.values())
                       and all(c["failures"] == 0 for c in report["class_out"].values()))
        else "INADEQUATE")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="per-bits quant ensemble adequacy")
    parser.add_argument("--bits", type=int, nargs="*", default=list(BITS))
    parser.add_argument("--full-shapes", action="store_true",
                        help="restore Phase 0's shape set, including 2048x11008")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    shapes = SHAPES + [(2048, 11008)] if args.full_shapes else SHAPES
    print(f"contract: mlx-affine, group_size={GROUP_SIZE}, K={K_QUANT} (Phase 0)")
    print(f"bits: {args.bits} (control {CONTROL_BITS})")
    print(f"shapes: {shapes}")
    print(f"calibration seeds {CALIBRATION_SEEDS}, independent draw {HELDOUT_SEEDS}")
    print(f"adequacy gate: zero false positives AND every fault >= {MARGIN_GATE:g}x\n")

    results, out = {}, {}
    for bits in args.bits:
        print(f"  measuring bits={bits} ...")
        measured = measure(bits, shapes, not args.quiet)
        results[bits] = measured
        if not measured["bit_exact"]:
            print(f"  GATE 0 FAILED at bits={bits}: canonical_quantize is not "
                  f"bit-exact against mx.quantize, so nothing here describes MLX")
            out[bits] = {"verdict": "GATE-0-FAIL"}
            continue
        out[bits] = adequacy(measured["records"], K_QUANT)
        out[bits]["gate0_checks"] = measured["exact_checks"]

    header = (f"{'bits':>6}{'records':>9}{'G0 exact':>10}{'G1 FP':>7}"
              f"{'G2 worst margin':>17}{'G3 classes':>12}{'equiv':>7}"
              f"{'fp16 boundary':>15}{'verdict':>13}")
    print()
    print("=" * len(header))
    print(f"PER-BITS ENSEMBLE ADEQUACY, K = {K_QUANT} fixed, amended gate")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for bits in args.bits:
        rep = out[bits]
        if rep.get("verdict") == "GATE-0-FAIL":
            print(f"{bits:>6}{len(results[bits]['records']):>9}{'FAIL':>10}"
                  f"{'-':>7}{'-':>17}{'-':>12}{'-':>7}{'-':>15}{'GATE-0-FAIL':>13}")
            continue
        sizes = "/".join(str(c["distinct"]) for c in rep["class_size"].values())
        boundary = f"{rep['boundary']}/{rep['boundary_of']}"
        print(f"{bits:>6}{len(results[bits]['records']):>9}{'pass':>10}"
              f"{len(rep['false_positives']):>7}{rep['worst_margin']:>16.1f}x"
              f"{sizes:>12}{len(rep['equivalent_faults']):>7}{boundary:>15}"
              f"{rep['verdict']:>13}")
    print("=" * len(header))
    print("G3 classes = distinct members per class (dequant-domain/int-domain), "
          "gate is >= 2 each")

    pre = {b: out[b].get("verdict_as_first_written") for b in args.bits
           if out[b].get("verdict") != "GATE-0-FAIL"}
    if any(v != out[b]["verdict"] for b, v in pre.items()):
        print()
        print("the same records under the gate AS FIRST WRITTEN, before the "
              "leave-one-class-out")
        print("clause was amended, reported so the amendment is auditable:")
        for bits, v in pre.items():
            print(f"  bits={bits:<3} {v:<12} (amended reading: {out[bits]['verdict']})")

    equivalents = {b: out[b].get("equivalent_faults") for b in args.bits
                   if out[b].get("equivalent_faults")}
    if equivalents:
        print()
        print("artefact faults excluded as equivalent, per width:")
        for bits, names in equivalents.items():
            for name in names:
                print(f"  bits={bits:<3} {name:<28} caught on no case at this width")

    print()
    print("leave-one-CLASS-out, worst member/tolerance ratio on the remaining classes:")
    for bits in args.bits:
        if out[bits].get("verdict") == "GATE-0-FAIL":
            continue
        for class_name, c in out[bits]["class_out"].items():
            print(f"  bits={bits:<3} drop {class_name:<16} "
                  f"{c['remaining']} members left, worst ratio {c['worst_ratio']:6.3f}, "
                  f"{c['failures']} exceed tolerance")

    print()
    print("fp16-dequant boundary per bits (the certificate-spec input):")
    for bits in args.bits:
        rep = out[bits]
        if rep.get("verdict") == "GATE-0-FAIL":
            continue
        inside = rep["boundary_of"] - rep["boundary"]
        state = ("outside tolerance everywhere: intermediate precision must be pinned"
                 if rep["boundary"] == rep["boundary_of"] else
                 f"INSIDE tolerance on {inside}/{rep['boundary_of']} cases: the "
                 f"requirement is bits-dependent")
        print(f"  bits={bits:<3} {state}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(
        {"k": K_QUANT, "shapes": shapes, "classes": CLASSES,
         "adequacy": out,
         "records": {b: results[b]["records"] for b in results}}, default=float))
    print(f"\nrecords: {OUT_PATH}")

    verdicts = [out[b].get("verdict") for b in args.bits]
    print(f"VERDICT: {'ADEQUATE at every width' if set(verdicts) == {'ADEQUATE'} else verdicts}")
    return 0 if set(verdicts) == {"ADEQUATE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
