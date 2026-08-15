"""Does ANY battery case separate a half-precision accumulator from legitimate
fp16 output rounding under the shipped tolerance?

The question ADR 0009 left open (C1-at-fp16 enforceability) and the verdict
clause agreed with the phase0 lane: if any case separates, C1-at-fp16 stays
numeric for quantized_matmul; if none does, it moves to structural attestation.
Run BEFORE catalogue integration so the fleet gets the answer fast.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))

from kernelverify.battery.core import case_space, make_mode_inputs
from kernelverify.reference.native_kernels import quantized_matmul
from kernelverify.schemas.native_ops import NATIVE_OPS
from measure_escape import corpus_oracle_passes

op = NATIVE_OPS["quantized_matmul"]
cases = case_space(op.meta)
print(f"{len(cases)} cases in the battery space")

hits = defaultdict(lambda: [0, 0])   # key -> [detected, total]
fp16_hits = defaultdict(lambda: [0, 0])
first_separating = []
controls_failed = 0

for i, case in enumerate(cases):
    inputs = make_mode_inputs(op.meta, case.dim_map, case.dtype, case.seed,
                              case.distribution)
    if op.augment:
        inputs = op.augment(case, inputs)
    ref = op.reference(inputs)
    tol = op.tolerance(case, inputs, ref)
    good = quantized_matmul(inputs)
    if not corpus_oracle_passes(good, ref, tol):
        controls_failed += 1
        continue
    bad_tree = quantized_matmul(inputs, accum_dtype="float16")
    # The sequential sub-shape: a naive one-thread-per-output kernel adds the
    # whole row left to right in half precision. np.add.accumulate is truly
    # sequential where np.add.reduce is pairwise.
    x_, w_ = inputs["x"], inputs["w"]
    import kernelverify.reference.native_kernels as nk
    from kernelverify.schemas.quant_contract import QuantContract, dequantize
    art = nk.canonical_quantize(w_, QuantContract(bits=int(inputs["bits"][0]), group_size=64))
    wd = dequantize(art, np.float32)
    prods = (x_.astype(np.float16)[:, None, :] * wd.astype(np.float16)[None, :, :])
    bad_seq = np.add.accumulate(prods, axis=-1, dtype=np.float16)[..., -1].astype(x_.dtype)

    for name, bad in (("tree", bad_tree), ("seq", bad_seq)):
        detected = not corpus_oracle_passes(bad, ref, tol)
        key = (name, case.dtype)
        hits[key][0] += int(detected)
        hits[key][1] += 1
        if case.dtype == "float16":
            k2 = (name, case.distribution, case.dim_map.get("D_IN"))
            fp16_hits[k2][0] += int(detected)
            fp16_hits[k2][1] += 1
            if detected and len(first_separating) < 10:
                err = float(np.max(np.abs(bad.astype(np.float64) - ref)))
                first_separating.append((name, case.distribution, case.dim_map, tol, err))
    if (i + 1) % 200 == 0:
        print(f"  ... {i+1}/{len(cases)}")

print(f"\ncontrols failed: {controls_failed} (must be 0)")
print("\ndetection by activation dtype:")
for k, (d, t) in sorted(hits.items()):
    print(f"  {str(k):<28} {d:>4}/{t}")

print("\nfp16 activations, by (mode, D_IN, bits):")
for k, (d, t) in sorted(fp16_hits.items()):
    if d:
        print(f"  {str(k):<44} {d:>3}/{t}")
none = [k for k, (d, t) in fp16_hits.items() if d == 0]
print(f"  (modes/dims with zero detection: {len(none)} of {len(fp16_hits)})")

print("\nfirst separating fp16 cases:")
for name, mode, dims, tol, err in first_separating:
    print(f"  {name:<5} {mode:<18} {dims}  err={err:.3e} tol={tol:.3e} ratio={err/tol:.2f}x")
