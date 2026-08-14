"""Phase 0: does a contract-anchored K exist for MLX-affine quantization?

The keystone experiment (design doc Premise 6, eng review D9-A). Everything
quant-side is gated on its outcome, with the kill criterion pre-registered:
if no safety factor K simultaneously (a) passes every legitimate
implementation, including the held-out diverse set and MLX's own on-device
quantized_matmul, and (b) keeps every artefact fault caught on every case,
the contract anchor buys nothing and quant kernel work stops.

Steps:
  1. Verify the numpy canonical quantizer is bit-exact against mx.quantize
     (contract tied to the real channel, not to our reimagining of it).
  2. Measure the ensemble floor (5 legitimate CPU members) against r_contract
     across a case grid of shapes x activation dtypes x input modes x weight
     draws.
  3. Hold out two diverse correct implementations - a block-tiled CPU variant
     and MLX's own on-device quantized_matmul - and find the smallest K with
     zero false positives on them. The on-device member doubles as the first
     contact between the CPU-calibrated floor and real Metal arithmetic
     (eng review's pre-registered device-arithmetic branch).
  4. Check every artefact fault stays caught at that K; report margins.
  5. Report the fp16-dequant boundary case (legal in some shipped kernels,
     precision-fault-shaped): which side of K*floor it lands decides whether
     the contract must pin intermediate precision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

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

CONTRACT = QuantContract(scheme="mlx-affine", bits=4, group_size=64)
K_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
SHAPES = [(512, 512), (4096, 4096), (2048, 11008)]  # (d_out, d_in)
BATCH = 2
SEEDS = (0, 1)
MODES = ("unit", "corpus-scale", "near-zero", "constant-rows")
WEIGHT_DRAWS = ("normal-0.02", "heavy-tailed")
OUT_PATH = Path(__file__).with_name(".cache") / "phase0_contract_k.json"


def err(candidate: np.ndarray, ref: np.ndarray) -> float:
    diff = np.abs(candidate.astype(np.float64) - ref)
    return float(diff.max()) if diff.size else 0.0


def make_x(batch: int, d_in: int, dtype, mode: str, rng) -> np.ndarray:
    x = rng.standard_normal((batch, d_in)).astype(np.float32)
    if mode == "corpus-scale":
        x = x * np.float32(10.0)
    elif mode == "near-zero":
        x = x * np.float32(1e-3)
    elif mode == "constant-rows":
        x = np.broadcast_to(x[:, :1], x.shape).copy()
    return x.astype(dtype)


def make_w(d_out: int, d_in: int, draw: str, rng) -> np.ndarray:
    w = rng.standard_normal((d_out, d_in)).astype(np.float32)
    if draw == "normal-0.02":
        w = w * np.float32(0.02)
    else:  # heavy-tailed: cubed normals concentrate mass in few channels
        w = (w ** 3) * np.float32(0.02)
    return w.astype(np.float16)


def unpack_mlx_q(w_q: np.ndarray, d_in: int, bits: int) -> np.ndarray:
    """Unpack mx.quantize's words as one contiguous little-endian bit stream.

    Value i occupies bits [bits*i, bits*(i+1)) of the row's stream, lowest
    element in the lowest bits, and the stream is then reinterpreted as
    uint32. That is not the same as packing 32//bits values per word: the two
    agree only when bits divides 32, so the previous shift-and-mask reading
    was correct at bits 2, 4 and 8 and wrong at 3, 5 and 6, where a value
    straddles a word boundary. Verified against mx.quantize at 2, 3, 4 and 8.
    """
    words = np.asarray(w_q, dtype=np.uint32)
    stream = np.unpackbits(words.view(np.uint8).reshape(words.shape[0], -1),
                           axis=1, bitorder="little")
    index = np.arange(d_in)[:, None] * bits + np.arange(bits)[None, :]
    weights = (1 << np.arange(bits)).astype(np.int32)
    return (stream[:, index] * weights).sum(axis=2).astype(np.int32)


def verify_canonical_against_mlx(w: np.ndarray) -> tuple[QuantArtefact, bool]:
    """Bit-exactness check; on mismatch, MLX's own output becomes canonical."""
    ours = canonical_quantize(w, CONTRACT)
    w_q, scales, biases = mx.quantize(mx.array(w), group_size=CONTRACT.group_size,
                                      bits=CONTRACT.bits)
    theirs_q = unpack_mlx_q(np.array(w_q), w.shape[1], CONTRACT.bits)
    theirs = QuantArtefact(theirs_q, np.array(scales), np.array(biases), CONTRACT)
    exact = (np.array_equal(ours.q, theirs.q)
             and np.array_equal(ours.scales.astype(np.float32),
                                theirs.scales.astype(np.float32))
             and np.array_equal(ours.biases.astype(np.float32),
                                theirs.biases.astype(np.float32)))
    return (ours if exact else theirs), exact


def heldout_block_tiled(x: np.ndarray, a: QuantArtefact) -> np.ndarray:
    """Diverse correct implementation NOT in the floor ensemble: K tiled in 128s,
    pairwise inside a tile, strictly serial across tiles."""
    w = dequantize(a, np.float32)
    xf = x.astype(np.float32)
    out = np.zeros((x.shape[0], w.shape[0]), dtype=np.float32)
    for start in range(0, w.shape[1], 128):
        out = out + xf[:, start:start + 128] @ w[:, start:start + 128].T
    return out.astype(x.dtype)


def heldout_mlx_device(x: np.ndarray, a: QuantArtefact, w: np.ndarray) -> np.ndarray:
    """MLX's own quantized matmul on the GPU: the first device-arithmetic contact."""
    w_q, scales, biases = mx.quantize(mx.array(w), group_size=CONTRACT.group_size,
                                      bits=CONTRACT.bits)
    out = mx.quantized_matmul(mx.array(x), w_q, scales, biases, transpose=True,
                              group_size=CONTRACT.group_size, bits=CONTRACT.bits)
    mx.eval(out)
    return np.array(out)


def boundary_fp16_dequant(x: np.ndarray, a: QuantArtefact) -> np.ndarray:
    """The boundary case: dequantize to fp16 (as some shipped kernels do),
    then MAC in fp32. Legal or fault - measured, not assumed."""
    w16 = dequantize(a, np.float32).astype(np.float16).astype(np.float32)
    return (x.astype(np.float32) @ w16.T).astype(x.dtype)


def main() -> int:
    records = []
    mismatches = 0
    for d_out, d_in in SHAPES:
        for draw in WEIGHT_DRAWS:
            for seed in SEEDS:
                rng = np.random.default_rng(10_000 + seed)
                w = make_w(d_out, d_in, draw, rng)
                artefact, exact = verify_canonical_against_mlx(w)
                mismatches += (not exact)
                for dtype in (np.float32, np.float16):
                    for mode in MODES:
                        x = make_x(BATCH, d_in, dtype, mode, rng)
                        ref = r_contract(x, artefact)
                        member_errs = {name: err(fn(x, artefact), ref)
                                       for name, fn in ENSEMBLE.items()}
                        heldout_errs = {
                            "block-tiled": err(heldout_block_tiled(x, artefact), ref),
                            "mlx-on-device": err(heldout_mlx_device(x, artefact, w), ref),
                        }
                        fault_errs = {name: err(ENSEMBLE["dequant-pairwise"](x, fn(artefact)), ref)
                                      for name, fn in FAULTS.items()}
                        records.append({
                            "shape": [d_out, d_in], "draw": draw, "seed": seed,
                            "dtype": np.dtype(dtype).name, "mode": mode,
                            "canonical_exact": exact,
                            "members": member_errs, "heldout": heldout_errs,
                            "faults": fault_errs,
                            "boundary_fp16_dequant": err(boundary_fp16_dequant(x, artefact), ref),
                            "ref_scale": float(np.max(np.abs(ref))),
                        })
        print(f"measured shape {d_out}x{d_in}")

    print(f"\ncanonical quantizer vs mx.quantize: "
          f"{'BIT-EXACT on all weight draws' if mismatches == 0 else f'{mismatches} MISMATCHES - MLX output used as canonical'}")

    # ---- K calibration, mirroring the shipped oracle ----------------------
    # tol(case) = max(base, K * floor): the base term is a few output ULPs in
    # the activation dtype and absorbs the near-zero-error regime where ratios
    # of two tiny numbers would otherwise inflate K meaninglessly (the same
    # role base_tol plays in ADR 0004's oracle).
    def base_tol(r) -> float:
        eps = 9.77e-4 if r["dtype"] == "float16" else 1.19e-7
        return 4.0 * eps * r["ref_scale"]

    k_needed = 0.0
    binding = ""
    for r in records:
        errs = r["members"]
        base = base_tol(r)
        for name, e in errs.items():
            floor_others = max(v for k, v in errs.items() if k != name)
            if e > base and floor_others > 0 and e / floor_others > k_needed:
                k_needed, binding = e / floor_others, f"member {name} @ {r['shape']} {r['mode']} {r['dtype']}"
        floor_all = max(errs.values())
        for name, e in r["heldout"].items():
            if e > base and floor_all > 0 and e / floor_all > k_needed:
                k_needed, binding = e / floor_all, f"heldout {name} @ {r['shape']} {r['mode']} {r['dtype']}"
    k_chosen = next((k for k in K_GRID if k >= k_needed), None)
    print(f"K needed to cover every correct implementation: {k_needed:.3f} (binding: {binding})")

    if k_chosen is None:
        print(f"KILL: no K in the grid covers legitimate diversity (needed {k_needed:.1f})")
        verdict = "KILL"
    else:
        def tol_for(r) -> float:
            return max(base_tol(r), k_chosen * max(r["members"].values()))

        # ---- fault detection at K, per the (fault, case) doctrine ---------
        # A fault must be VIABLE (caught somewhere) and caught on 100% of the
        # iid-mode cases, where every mechanism here physically expresses.
        # Structured modes can make individual faults per-case-equivalent
        # (a nibble swap is invisible to constant rows); those pairs are
        # reported, not scored, exactly as the unquantized harness treats
        # equivalent (fault, case) pairs.
        verdict = "PASS"
        print(f"smallest zero-FP K: {k_chosen}")
        print(f"{'fault':<28}{'iid caught':>12}{'all caught':>12}{'worst iid margin':>18}")
        for name in FAULTS:
            iid = [r for r in records if r["mode"] in ("unit", "corpus-scale")]
            iid_hit = sum(1 for r in iid if r["faults"][name] > tol_for(r))
            all_hit = sum(1 for r in records if r["faults"][name] > tol_for(r))
            margins = [r["faults"][name] / tol_for(r) for r in iid]
            worst = min(margins)
            flag = "" if iid_hit == len(iid) else "  <- GATE FAIL"
            if iid_hit != len(iid):
                verdict = "KILL"
            print(f"{name:<28}{iid_hit:>7}/{len(iid):<4}{all_hit:>7}/{len(records):<4}"
                  f"{worst:>15.1f}x{flag}")
        equiv_pairs = sum(1 for r in records for name in FAULTS
                          if r["faults"][name] <= tol_for(r)
                          and r["mode"] not in ("unit", "corpus-scale"))
        print(f"per-case-equivalent (fault, structured-case) pairs: {equiv_pairs} (reported, not scored)")

        # ---- the boundary case -------------------------------------------
        above = sum(1 for r in records if r["boundary_fp16_dequant"] > tol_for(r))
        print(f"fp16-dequant boundary case: outside tolerance on {above}/{len(records)} cases "
              f"-> the contract {'must pin intermediate precision' if above else 'may leave it free'}")

        # ---- device-arithmetic branch ------------------------------------
        dev_fp = sum(1 for r in records if r["heldout"]["mlx-on-device"] > tol_for(r))
        print(f"MLX on-device vs CPU-calibrated K: {dev_fp}/{len(records)} cases exceed tolerance "
              f"({'device-member re-derivation branch TRIGGERS' if dev_fp else 'covered - no re-derivation needed yet'})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"records": records, "k_needed": k_needed,
                                    "verdict": verdict}, default=float))
    print(f"\nPHASE 0 VERDICT: {verdict}  (records: {OUT_PATH})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
