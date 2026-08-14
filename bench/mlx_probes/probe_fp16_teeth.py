"""FINDING: at fp16 the shipped tolerance does not catch a contract violation
in the pack's own kernel.

The wide-tile matvec is verified at fp16 activations, which is what it ships
for. The contract's C1 clause requires every intermediate at fp32 or wider,
so a kernel that accumulates in half is out of contract by construction - the
quantized analogue of ADR 0004's fp16-score canary.

That kernel passes. In all four cases below base_tol exceeds K*floor by 3 to
10x, so the conditioning-aware half of the tolerance contributes nothing at
fp16, and the faulted kernel's error (2.4 to 3.1x the correct kernel's) still
lands under the absolute bound. Three of the four would not be caught by a
floor-only rule either.

This is the known hole from ADR 0004's consequences - absolute base tolerances
going vacuous - showing up not on near-zero edge cases but on the typical
operating point of the dtype the pack ships. Independently, kv-phase0-cd
measured the same boundary from the battery side at bits=4: fp16 activations
inside tolerance, fp32 outside, because base_tol at fp16 swamps dequant error.

Not claimed here: the fp32 side. This kernel reads x as half4, so the fp32
arm was not measured.
"""
import sys
sys.path.insert(0, "/Users/vlad/kv-kernels")
import numpy as np, mlx.core as mx
from kernelverify.pack.wide_qmv import WIDE_QMV_MSL, pack_nibbles, launch_config
from kernelverify.schemas.native_ops import K_QUANT
from kernelverify.schemas.quant_contract import (ENSEMBLE, QuantContract,
                                                 canonical_quantize, r_contract)

CONTRACT = QuantContract(bits=4, group_size=64)
FP16_EPS = 9.77e-4

# The fault: accumulate the running sum in half instead of float. Everything
# else identical. This is the quantized analogue of the fp16-score canary.
FAULTED = WIDE_QMV_MSL.replace(
    "    float acc[M][R];", "    half acc[M][R];").replace(
    "acc[m][r] = metal::fma(s[r], xq, metal::fma(b[r], xs, acc[m][r]));",
    "acc[m][r] = (half)((float)acc[m][r] + s[r] * xq + b[r] * xs);").replace(
    "float v = metal::simd_sum(acc[m][r]);",
    "float v = metal::simd_sum((float)acc[m][r]);")

good = mx.fast.metal_kernel(name="teeth_good", input_names=["x","w_q","scales","biases"],
                            output_names=["out"], source=WIDE_QMV_MSL)
bad  = mx.fast.metal_kernel(name="teeth_bad", input_names=["x","w_q","scales","biases"],
                            output_names=["out"], source=FAULTED)

def run(k, x, art, d_out):
    m = x.shape[0]
    grid, tg, r = launch_config(d_out, m)
    o = k(inputs=[mx.array(x), mx.array(pack_nibbles(art.q)), mx.array(art.scales),
                  mx.array(art.biases)],
          output_shapes=[(m, d_out)], output_dtypes=[mx.float16], grid=grid,
          threadgroup=tg, template=[("T", mx.float16), ("M", m), ("R", r)])[0]
    mx.eval(o)
    return np.array(o).astype(np.float64)

print(f"{'case':<26}{'base_tol':>11}{'K*floor':>11}{'which':>7}"
      f"{'good err':>11}{'FAULT err':>11}{'caught':>8}")
for label, d_out, d_in, m, xscale, wscale in (
    ("typical 4096, x~N(0,1)", 4096, 4096, 8, 1.0, 0.02),
    ("typical 2560, x~N(0,1)", 2560, 2560, 8, 1.0, 0.02),
    ("small outputs",          2560, 2560, 8, 0.02, 0.02),
    ("wide dynamic range",     2560, 2560, 8, 1.0, 0.02),
):
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((d_out, d_in)).astype(np.float32) * wscale).astype(np.float16)
    if label == "wide dynamic range":
        # one loud row, the rest quiet: peak sets base_tol for everyone
        w[0] *= 50.0
    art = canonical_quantize(w, CONTRACT)
    x = (rng.standard_normal((m, d_in)).astype(np.float32) * xscale).astype(np.float16)
    ref = r_contract(x, art)
    floor = max(float(np.max(np.abs(fn(x, art).astype(np.float64) - ref))) for fn in ENSEMBLE.values())
    base = 4.0 * FP16_EPS * float(np.max(np.abs(ref)))
    tol = max(base, K_QUANT * floor)
    g = float(np.max(np.abs(run(good, x, art, d_out) - ref)))
    b = float(np.max(np.abs(run(bad,  x, art, d_out) - ref)))
    which = "base" if base >= K_QUANT * floor else "floor"
    print(f"{label:<26}{base:>11.3e}{K_QUANT*floor:>11.3e}{which:>7}"
          f"{g:>11.3e}{b:>11.3e}{str(b > tol):>8}")
