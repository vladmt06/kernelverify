"""FALSIFIED: staging x in threadgroup memory is slower than re-reading it.

The shipped wide-tile matvec has each simdgroup read all M input vectors from
device memory for its own R rows, so the 8 simdgroups in a threadgroup issue
the same x loads 8 times and recompute the same per-word x sums 8 times. At
M=8 it runs at 81 GB/s against a 128 GB/s single-pass bound, so 1.56x looked
available and removing that redundancy looked like the way to get it.

It is not. This variant stages an x tile in threadgroup memory with one
cooperative coalesced load per threadgroup and computes the per-word sums
once, and it is SLOWER than the shipped kernel at every tile width and every
M measured: 0.77-0.94x at TW = 32, 64, 128 and 256.

Set TW in the environment to re-measure a tile width. Run it interleaved, as
here, because a non-interleaved comparison of the same two kernels showed the
staged version WINNING by up to 1.45x - that was run-to-run power-state drift,
not a result, and it is the reason every arm below is sampled in the same
round.
"""
import sys, statistics, time
sys.path.insert(0, "/Users/vlad/kv-kernels")
import numpy as np, mlx.core as mx
from kernelverify.schemas.quant_contract import QuantContract, canonical_quantize, r_contract
from kernelverify.pack.wide_qmv import pack_nibbles, WIDE_QMV_MSL

V3 = """
    threadgroup half4 xt[M * TW * 2];
    threadgroup float  xsum[M * TW];

    uint sg_row = thread_position_in_grid.y;
    uint lane   = thread_index_in_simdgroup;
    uint tid    = thread_position_in_threadgroup.y * 32 + thread_position_in_threadgroup.x;
    uint nthr   = SG * 32;
    uint d_in   = x_shape[1];
    uint d_out  = scales_shape[0];
    uint words  = d_in / 8;
    uint n_groups = d_in / 64;
    const device half4* x4 = (const device half4*)x;

    float acc[M][R];
    #pragma clang loop unroll(full)
    for (int m = 0; m < M; ++m)
        #pragma clang loop unroll(full)
        for (int r = 0; r < R; ++r) acc[m][r] = 0.0f;

    for (uint w0 = 0; w0 < words; w0 += TW) {
        uint nw = metal::min((uint)TW, words - w0);

        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        // Coalesced: consecutive threads take consecutive half4 of the same m.
        for (uint i = tid; i < M * nw * 2; i += nthr) {
            uint m  = i / (nw * 2);
            uint rest = i - m * (nw * 2);
            uint wi = rest >> 1, h = rest & 1;
            xt[m * (TW * 2) + wi * 2 + h] = x4[m * (d_in / 4) + (w0 + wi) * 2 + h];
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        for (uint i = tid; i < M * nw; i += nthr) {
            uint m = i / nw, wi = i - m * nw;
            float4 lo = float4(xt[m * (TW * 2) + wi * 2]);
            float4 hi = float4(xt[m * (TW * 2) + wi * 2 + 1]);
            xsum[m * TW + wi] = metal::dot(lo, float4(1.0f)) + metal::dot(hi, float4(1.0f));
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        for (uint wi = lane; wi < nw; wi += 32) {
            uint gwi = w0 + wi;
            uint g = gwi / 8;
            float4 qlo[R], qhi[R];
            float s[R], b[R];
            #pragma clang loop unroll(full)
            for (int r = 0; r < R; ++r) {
                uint row = metal::min(sg_row * R + r, d_out - 1);
                uint word = w_q[row * words + gwi];
                s[r] = (float)scales[row * n_groups + g];
                b[r] = (float)biases[row * n_groups + g];
                qlo[r] = float4(float(word & 0xF), float((word >> 4) & 0xF),
                                float((word >> 8) & 0xF), float((word >> 12) & 0xF));
                qhi[r] = float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                                float((word >> 24) & 0xF), float((word >> 28) & 0xF));
            }
            #pragma clang loop unroll(full)
            for (int m = 0; m < M; ++m) {
                float4 lo = float4(xt[m * (TW * 2) + wi * 2]);
                float4 hi = float4(xt[m * (TW * 2) + wi * 2 + 1]);
                float xs = xsum[m * TW + wi];
                #pragma clang loop unroll(full)
                for (int r = 0; r < R; ++r) {
                    float xq = metal::dot(lo, qlo[r]) + metal::dot(hi, qhi[r]);
                    acc[m][r] = metal::fma(s[r], xq, metal::fma(b[r], xs, acc[m][r]));
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) {
        uint row = sg_row * R + r;
        #pragma clang loop unroll(full)
        for (int m = 0; m < M; ++m) {
            float v = metal::simd_sum(acc[m][r]);
            if (lane == 0 && row < d_out) out[m * d_out + row] = (T)v;
        }
    }
"""

D_OUT, D_IN, SG = 4096, 4096, 8
WB = D_OUT*D_IN*0.5 + 2*(D_OUT*D_IN/64)*2
def disp(f,c):
    o=[f(i) for i in range(c)]; t0=time.perf_counter(); mx.eval(o); mx.synchronize(); return time.perf_counter()-t0
def calib(f):
    c=4
    while c<2048 and disp(f,c)*1e3<5.0: c*=2
    return c
def med(f):
    c=calib(f); return statistics.median([disp(f,c)/c for _ in range(7)])

rng=np.random.default_rng(7)
w=(rng.standard_normal((D_OUT,D_IN)).astype(np.float32)*0.02).astype(np.float16)
art=canonical_quantize(w,QuantContract(bits=4,group_size=64)); packed=pack_nibbles(art.q)
mp,ms,mb = mx.array(packed), mx.array(art.scales), mx.array(art.biases)
sets=[]
for s in range(55):
    ws=(np.random.default_rng(100+s).standard_normal((D_OUT,D_IN)).astype(np.float32)*0.02).astype(np.float16)
    sets.append(mx.quantize(mx.array(ws),group_size=64,bits=4))
mx.eval([a for t in sets for a in t]+[mp,ms,mb])

k3=mx.fast.metal_kernel(name="v3",input_names=["x","w_q","scales","biases"],output_names=["out"],source=V3)
k2=mx.fast.metal_kernel(name="v2c",input_names=["x","w_q","scales","biases"],output_names=["out"],source=WIDE_QMV_MSL)
print(f"single-pass bound at 128 GB/s: {WB/128e9*1e6:.1f} us\n")
import os
def pick_tw(M):
    return int(os.environ.get("TW", "64"))
print(f"{'M':>3} {'TW':>4} {'v2 us':>9} {'v3 us':>9} {'mlx us':>9} {'v3/v2':>7} {'v3 vs mlx':>10} {'v3 GB/s':>9}")
for M in (4,5,6,7,8,10,11):
    x=(rng.standard_normal((M,D_IN)).astype(np.float16)); ref=r_contract(x,art); mxx=mx.array(x); mx.eval(mxx)
    R=4 if M<=10 else 2
    TW=pick_tw(M)
    def v2(i):
        wq,sc,bi=sets[i%55]
        return k2(inputs=[mxx,wq,sc,bi],output_shapes=[(M,D_OUT)],output_dtypes=[mx.float16],
                  grid=(32,(D_OUT+R-1)//R,1),threadgroup=(32,SG,1),
                  template=[("T",mx.float16),("M",M),("R",R)])[0]
    def v3(i):
        wq,sc,bi=sets[i%55]
        return k3(inputs=[mxx,wq,sc,bi],output_shapes=[(M,D_OUT)],output_dtypes=[mx.float16],
                  grid=(32,(D_OUT+R-1)//R,1),threadgroup=(32,SG,1),
                  template=[("T",mx.float16),("M",M),("R",R),("TW",TW),("SG",SG)])[0]
    def mlxf(i):
        wq,sc,bi=sets[i%55]
        return mx.quantized_matmul(mxx,wq,sc,bi,transpose=True,group_size=64,bits=4)
    chk=k3(inputs=[mxx,mp,ms,mb],output_shapes=[(M,D_OUT)],output_dtypes=[mx.float16],
           grid=(32,(D_OUT+R-1)//R,1),threadgroup=(32,SG,1),
           template=[("T",mx.float16),("M",M),("R",R),("TW",TW),("SG",SG)])[0]
    mx.eval(chk)
    err=float(np.max(np.abs(np.array(chk).astype(np.float64)-ref)))
    assert err < 0.05*max(1.0,float(np.max(np.abs(ref)))), (M,err)
    for f in (v2,v3,mlxf): mx.eval(f(0))
    mx.synchronize()
    c=calib(v3)
    a,b,cc=[],[],[]
    for _ in range(7):
        a.append(disp(v2,c)/c); b.append(disp(v3,c)/c); cc.append(disp(mlxf,c)/c)
    t2,t3,tm=statistics.median(a),statistics.median(b),statistics.median(cc)
    print(f"{M:>3} {TW:>4} {t2*1e6:>9.1f} {t3*1e6:>9.1f} {tm*1e6:>9.1f} {t2/t3:>6.2f}x {tm/t3:>9.2f}x {WB/t3/1e9:>9.1f}")
