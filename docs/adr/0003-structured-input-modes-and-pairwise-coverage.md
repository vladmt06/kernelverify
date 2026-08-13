# ADR 0003: Structured input modes, the ill-conditioning rule, and pairwise boundary coverage

Date: 2026-08-13
Status: Accepted

## Context

Every verdict so far was computed on iid uniform inputs, at two scales.
The second-wave catalogue exposed what that misses: the flash-attention running-max fault (`init_max_zero`), a real production bug where a fully-masked row underflows exp and produces 0/0, was detectable at exactly 0% of iid cases at any scale.
An input model, not a shape or a tolerance, was the blind spot.

## Decision 1: input modes are the third battery axis

The case space now crosses shapes and dtypes with five input modes.

| Mode | Construction | Mechanism it can reach |
|---|---|---|
| corpus uniform[-10,10] | the corpus's own draw | saturation, overflow |
| unit-scale uniform[-1,1] | same draw at unit scale | near-tolerance drift |
| opposed signs | abs on even inputs, negated abs on odd | adversarial score sign patterns |
| near zero | unit draw times 1e-3 | epsilon and underflow faults |
| constant rows | first column broadcast across the row | degenerate reductions, zero variance |

The measurement that justifies the axis: `init_max_zero` went from 0% to 11.6% detectable, almost entirely where opposed signs meets the largest head dimension, and the l2norm epsilon fault from near-invisible to 11.1%, where near zero meets the smallest row.

## Finding: structured inputs break the published tolerances

On constant-rows input, a provably correct fp32 attention kernel exceeds the corpus tolerance against the corpus's own fp64 reference (first seen at D=64, M=8, N=256, where attention scores reach several hundred and fp32 rounding alone moves softmax outputs past 1e-3).
The published tolerances were calibrated on iid inputs and are simply wrong in these regimes.
Each attention-family fault had exactly one such case in its space; no other operator produced any.

The rule adopted: a case whose control (the correct kernel) fails the oracle is ill-conditioned and counts as no-evidence, never as a detection, because a verdict that also fires on the correct kernel is a false positive.
A control failure on an iid mode still aborts the run, because there it means the port itself is wrong.

## Population update

44 mutations, 39 viable, 5 equivalent.
The two mutations added as predicted-equivalent (l2norm without epsilon, LeakyReLU boundary rewrite) measured equivalent, which is the exclusion mechanism working.
Softmax without max-subtraction stays equivalent even with the new modes, because no mode pushes inputs to overflow.

## Decision 2: the singles core was falsified at mid budgets, so the core covers pairs

On the enlarged space, the ADR 0002 policy lost its lead at B=8 and B=16 to plain random sampling over all modes (96.3% vs 96.9%, and 99.0% vs 99.6%).
Its two systematic misses both live at feature *pairs* (opposed-signs times largest-D, near-zero times smallest-H), which single-feature cover visits separately but never guarantees jointly.
That is precisely the trigger ADR 0002 pre-registered for adopting pairwise coverage.

The ordering inside the pairwise core was itself decided by measurement, over the 39 viable faults:

| Core ordering | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| pure pair-greedy, then random | 66.7% | 94.9% | 100.0% | 100.0% |
| singles, then pairs, then random | 94.9% | 97.4% | 100.0% | 100.0% |
| singles, then pairs alternated with random | 94.9% | 95.8% | 100.0% | 100.0% |
| singles, then random (ADR 0002) | 94.9% | 96.3% | 99.0% | 99.8% |

Pure pair-greedy buys pair density before basic diversity and craters at B=4 with no exploration left.
Interleaving random into the pair phase displaces pairs that matter at B=8.
The winner at every budget: cover single features first, which reproduces the ADR 0002 core exactly, then extend to uncovered pairs, then explore at random.

## Result

Detection rate across the 39 viable faults, at equal budget B per operator:

| Policy | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| Single shape, fp32 (what benchmarks do) | 74.4% | 74.4% | 74.4% | 74.4% |
| Random schema (the corpus's own fuzzer) | 85.0% | 89.0% | 90.1% | 91.8% |
| Random schema, both input scales | 86.6% | 91.5% | 94.2% | 95.4% |
| Random schema, all input modes | 89.9% | 96.9% | 99.6% | 99.8% |
| Boundary singles + random (ADR 0002) | 94.9% | 96.3% | 99.0% | 99.8% |
| Boundary pairs + random (ours) | 94.9% | 97.4% | 100.0% | 100.0% |

The 100.0% cells at B=16 and B=32 are exact: zero misses across all 40 seeded runs, verified by the per-budget miss report rather than by a rounded cell.
On the 10 published corpus faults the policy stays at 100% from B=4.
The one systematic survivor at B=8 is the l2norm epsilon fault, whose pair is covered only once the deterministic prefix grows past 8 cases.

## Consequences

The sales claim sharpens: a single-shape harness catches 74.4% of this population no matter how much budget it burns, and the boundary-pair battery catches all of it at 16 evaluations per operator.
Tolerance calibration is now a known product problem, not just a scoring nuisance: any verifier that reuses published per-operator tolerances will false-positive on structured inputs, and ours must derive tolerance from a condition estimate instead; that is future work the ill-conditioning rule currently papers over.
The honest caveats compound: the fault population is still self-authored, and the core ordering was selected by measurement on that same population, so growing the catalogue faster than the policy adapts remains the discipline that keeps these numbers meaningful.
Triple-wise coverage is explicitly not adopted; like last time, it waits for a measured miss that demands it.
