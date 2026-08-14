# ADR 0004: Conditioning-aware tolerance - the ensemble floor, and a falsified probe

Date: 2026-08-14
Status: Accepted

## Context

ADR 0003 found that fixed per-operator tolerances are wrong in ill-conditioned regimes: on constant-rows input at D=64, M=8, N=256, a provably correct fp32 attention kernel errs 2.22e-3 against the fp64 reference, past the published 1e-3, because fp32 rounding of score intermediates near magnitude 800 legitimately moves softmax outputs that far.
The scoring harness papered over this with a control-based no-evidence rule, but the shipped verifier has no known-correct control, so the product needed a per-case tolerance that adapts to conditioning.

## The falsified first design

The first proposal probed the fp64 reference with 1-ULP input perturbations and set tolerance to max(base, K * movement).
It was falsified before touching the repo, independently by a three-lens design panel and by a full-battery experiment, on its own validation plan:

- A correct kernel's error is input conditioning *plus* internal accumulation rounding; the probe measures only the first term, and the ratio between the two spans 1x to 64x across this battery, so no constant K bridges it.
  The motivating case needs K of about 64-301 depending on probe seed; every candidate K in 2..16 ships the false positive.
- Probing at fp16 epsilon legitimises fp16-internal arithmetic: a canary fault (attention with fp16-rounded scores, error 0.774 vs base tolerance 0.05) escapes at K>=4 on every fp16 case.
  That is precisely the precision-fault class the verifier exists to reject.
- The 3-probe floor is a random variable with 9x-15x spread across seeds, making near-threshold verdicts flaky.

## Decision 1: the shipped oracle is the working-precision ensemble floor

`kernelverify/tolerance/floor.py`:

    floor(case) = max over an ensemble of provably-correct working-precision
                  implementations of |impl(inputs) - ref_fp64(inputs)|
    tolerance   = max(base_tol, K_ENSEMBLE * floor),  K_ENSEMBLE = 1.5

Ensemble members are the reference kernel plus worst-legitimate-order variants: sequential (non-pairwise) accumulation for every reduction, and a second flash tile width for the attention family.
K = 1.5 is the smallest grid value with zero false positives over *held-out* diverse correct implementations (other tile widths, reversed summation order, the plain-vs-flash cross pair), which is what makes the gate non-circular.

Measured, full battery (4,100 cases):

| Gate | Result |
|---|---|
| False positives over held-out correct implementations | 0 at K=1.5 (2 at K=1.0, from the two constant-rows cases) |
| Fault detection vs the old fixed rule | nothing lost; 9 attention-family rows gain the formerly discarded case |
| The two formerly no-evidence cases | pass the control and detect 5/5 and 4/5 faults |
| fp16-score canary | stays detected (34/220 fp16 cases, same as the fixed rule) |
| Worst tolerance inflation | x3.3 over base, negligible against output scale |

## Decision 2: the harness scores with the shipped oracle

`bench/score_oracles.py` `build_verdicts` now applies `conditioned_tolerance` to every (case, fault) verdict and asserts that every correct control passes it on every case, structured modes included.
The no-evidence discard is gone; published policy tables now describe the verdict function the product runs.
The verdict-cache fingerprint carries an oracle tag, so an oracle change can never silently reuse stale verdicts.

## Decision 3: a permanent precision canary in the catalogue

`attention[scores_dtype=float16]` (scores held in half precision) joins the catalogue as fault 45.
Any future tolerance change that absolves it is a regression by definition; `tests/test_tolerance.py` additionally pins the motivating constant-rows case: the ill-conditioning is real, the ensemble floor covers the correct kernel and a held-out correct implementation, the canary stays caught, and the probe floor stays >8x too small to cover the case.

## Decision 4: the repaired probe FAILED its gates and does not ship

Some production references will be opaque (binary, remote, no numpy port), where the ensemble floor cannot be computed.
The repaired probe was the candidate for that path: probe at fp32 epsilon in fp64 input space regardless of working dtype; a separate additive `eps_work * max|ref|` output-rounding term; a derandomised probe core plus 8 random draws; and a transfer factor T per structural family, the measured max of ensemble_floor/probe_floor over family peers, validated leave-one-op-out:

    opaque tolerance = max(base_tol, K * T(family) * probe_floor + eps_work * max|ref|)

Measured leave-one-op-out, full battery: T = 0.7-1.2 (elementwise), 52-57 (row-reduction), 54-57 (bilinear).
False positives: zero at every K, so the transfer is conservative enough.
But the two-sided rule requires detection to survive, and it does not:

| Defect | Measurement | Mechanism |
|---|---|---|
| 23 fp16 detection losses on leaky_relu | alpha=0.011 fault absolved, err up to 9.0e-3 vs tol 9.3e-3 | the scalar eps term grants the max-magnitude element's ulp tensor-wide, but leaky_relu's max-magnitude outputs are exact pass-throughs with zero rounding |
| 7 fp16 detection losses on attention-family scale faults | err ~0.09-0.11 vs tol ~0.20-0.23 | T=54, earned on fp32 constant-rows conditioning, transfers to fp16 cases whose true gap is ~1x |
| fp16-score canary collapses 34 -> 10 | all 24 losses caused by T * probe_floor | same over-transfer: the opaque rule absolves the precision fault the oracle must never absolve |
| Vacuity in the tails | at K=1, tolerance exceeds half the output scale on 98/440 softmax and 72/360 matmul cases, near-zero modes worst | probe floors explode near softmax argmax ties (probe spread up to 10^287) and near-zero outputs inherit the absolute-base hole |

Verdict, per the decision rule pre-registered in the plan: no family passes, so the probe path ships nowhere.
An opaque reference gets an honest "requires a re-implementable reference for full verification", with the vacuity ceiling classifying the affected cases as no-evidence rather than pass.
The repairs that would revive it, pre-registered for when an opaque reference is an actual product need: per-element floors instead of scalar max terms, per-dtype and conditioning-bucketed transfer factors instead of a family max, and tie-aware probing for argmax-adjacent rows.

## The harness rerun under the shipped oracle

Old rule (fixed tolerance, control-based no-evidence discard) vs new rule (ensemble floor, no discards), same battery:

- Population: 44 -> 45 faults (the canary joins, detectable at 103/440 cases); the same 5 equivalents; 39 -> 40 viable.
- Per-fault detectability: every attention-family fault gains exactly the formerly discarded constant-rows case (+1); every other count is identical.
- Policy table (40 viable faults): boundary pairs + random holds 95.0% / 97.5% / 100.0% / 100.0% at B = 4/8/16/32, against 94.9% / 97.4% / 100.0% / 100.0% on the 39-fault population under the old rule; the 100.0% cells are exact (zero misses over 40 seeded runs), and the corpus-subset row stays 100% from B=4.
- The only systematic small-budget survivors remain l2norm[eps=1e-06] and flash init-max-zero at B=4, and l2norm[eps=1e-06] at B=8, both pair-located and covered from B=16.

The rerun also completed without a single control assertion, which is the integrated proof of gate (a): the shipped tolerance clears every correct implementation on all 4,100 cases with the no-evidence rule deleted.

## Consequences

The verifier's verdict is now defensible in regimes the published tolerances get wrong, without a known-correct control at verification time, and without absolving precision faults.
The remaining known holes, deliberately deferred as separate measured changes: absolute base tolerances are vacuous on near-zero cases (an 87% relative error passes fp16 attention at base 0.05) and want a relative component; per-element floors would localise leniency that the scalar floor grants tensor-wide; the shipped verifier should sample fresh input seeds per run so lenient cases are not predictable to a reward-hacking kernel generator.
