# Codex review findings, 2026-08-24

<!-- Provenance: three Codex CLI runs dispatched in parallel, each in its own git
worktree, against commit 42ac312 of lane/metalrunner-sprint1. What follows is each
run's FINAL REPORT verbatim; the full transcripts (4.6 MB) were session scratch and
are not preserved. Codex has no Metal device, so every GPU-marked test skipped for
it and its executed reproductions ran on the CPU paths. Absolute paths in the
reports are worktree paths from the machine they ran on; read them relative to the
repo root. NOTHING in this file has been acted on yet. -->


## Codex A: adversarial review of the training-attention increment

Brief: read-only, worktree at 42ac312, commits 568540d..42ac312. Told to break the kernel, the fp64 references, the borrowed tolerance and the gate's verify-before-time ordering. Exit 1: the provider flagged the final message for cybersecurity review AFTER emitting it, so the findings below are complete and the nonzero exit is not a failure of the review.

1. BUG NEEDS FIXING: CONFIRMED by EXECUTION and LINE READING - [`geometry_ok()` is never called by `run_fwd()`](/Users/vlad/kv-attn-review/kernelverify/pack/train_attention.py:110); DH=65 dispatched with `DB=8` and left column 64 unwritten, while HQ=10/HKV=3 mapped query head 9 to nonexistent KV head 3, risking wrong output or out-of-bounds access instead of the promised stock fallback.

2. BUG NEEDS FIXING: CONFIRMED by EXECUTION and LINE READING - [`verify()` discards `_lse` and judges only `out`](/Users/vlad/kv-attn-review/bench/pack_train_attention.py:228), while [`screen_unwritten()` checks stores but not values](/Users/vlad/kv-attn-review/bench/pack_train_attention.py:175); a `PoisonLSE` reproduction passed all 145 cases and reached timing, so a schedule-specific bad LSE can corrupt the future backward while receiving green evidence.

3. BUG NEEDS FIXING: CONFIRMED by EXECUTION and LINE READING - [`_base_tol()` permits four bf16 ulps at the reference peak](/Users/vlad/kv-attn-review/kernelverify/schemas/native_ops.py:52); through the real `verify_output` path, a bf16-only `out *= 0.975` defect passed all 72 gate cases across four group ratios and 18 widths, with worst error/tolerance 0.9046, so a systematic 2.5% epilogue bias can be certified.

4. BUG NEEDS FIXING: CONFIRMED by EXECUTION and LINE READING - every forward and gradient ensemble member shares the same [`_attn_scores()` reduction](/Users/vlad/kv-attn-review/kernelverify/schemas/native_ops.py:468), despite [C2 admitting every summation order](/Users/vlad/kv-attn-review/kernelverify/tolerance/contract.py:32); an explicit sequential-fp32 QK reduction was rejected by 3.833x on the declared B=2, HKV=2, GQA=4, T=65, DH=64 float32 case, with gradient dK and dV rejected by 4.860x and 13.425x, so correct implementations can be discarded.

5. BUG NEEDS FIXING: CONFIRMED by EXECUTION and LINE READING - [`attn_inputs()` carries stored fp16/bf16 values as float32](/Users/vlad/kv-attn-review/kernelverify/pack/verify.py:115), but ensemble members cast results to that carrier dtype at [forward line 551 and gradient lines 662-663](/Users/vlad/kv-attn-review/kernelverify/schemas/native_ops.py:551); execution showed every routed member returned float32 and omitted C1's final storage rounding, understating member error by 2,029x for fp16 and 31,500x for bf16, so these are not legitimate routed-dtype ensemble members.

6. BUG NEEDS FIXING: CONFIRMED by EXECUTION and LINE READING - [`bench(space)` accepts no verification evidence](/Users/vlad/kv-attn-review/bench/pack_train_attention.py:300), despite the module claiming it refuses unverified work, and only [`main()` sequences the calls](/Users/vlad/kv-attn-review/bench/pack_train_attention.py:389); an imported, monkeypatched call timed SG1/BK8 without invoking `verify()`, making V1 a CLI convention rather than an enforced invariant.

7. BUG NEEDS FIXING: CONFIRMED by EXECUTION and LINE READING - the gate controls rejection through [`arms_agree()`](/Users/vlad/kv-attn-review/bench/pack_train_attention.py:326), whose separate hardcoded 5e-3 rule lives in [`interleave.py`](/Users/vlad/kv-attn-review/bench/interleave.py:68); composed output 1.0 versus an otherwise verified output 1.006 was rejected before timing, so `kernelverify.pack.verify` is not the gate's sole numerical judge.

8. INVESTIGATION: SUSPECTED by LINE READING - Amendment 20's [anti-drift claim](/Users/vlad/kv-attn-review/docs/research/2026-08-19-metalrunner-sprint1-prereg.md:3490) is not pinned to the real trainer path because [`composed_arm()` calls low-level `mx.fast` directly](/Users/vlad/kv-attn-review/bench/pack_train_attention.py:251), its [test checks only output bits at widths 65, 129, and 257](/Users/vlad/kv-attn-review/tests/test_pack_train_attention_live.py:295), and the [stack hashes omit both attention wrappers](/Users/vlad/kv-attn-review/metalrunner/versions.py:35); a wrapper change or shape-selective change at timed width 1057 can therefore move stock training without failing that test, potentially repeating the recorded 1.89x baseline error, but no current mismatch was reproduced without Metal.

9. BUG NEEDS FIXING: CONFIRMED by EXECUTION and LINE READING - verification records every specialization with hardcoded threadgroup `(32,1,1)` at [`pack_train_attention.py:218`](/Users/vlad/kv-attn-review/bench/pack_train_attention.py:218), while [`fwd_launch()` returns `(32,SGROUPS,1)`](/Users/vlad/kv-attn-review/kernelverify/pack/train_attention.py:370); execution produced SG2, SG4, and SG8 threadgroups `(32,2,1)`, `(32,4,1)`, and `(32,8,1)`, so downstream structured evidence misidentifies every launchable SGROUPS greater than one.

10. BUG NEEDS FIXING: CONFIRMED by LINE READING - the kernel module still claims a [286 MB fp32 score matrix](/Users/vlad/kv-attn-review/kernelverify/pack/train_attention.py:22), while Amendment 20 explicitly [withdraws that number and replaces it with 143 MB bf16](/Users/vlad/kv-attn-review/docs/research/2026-08-19-metalrunner-sprint1-prereg.md:3565), leaving the landed implementation rationale wrong by 2x.

11. HEADS UP: CONFIRMED by EXECUTION - independent PyTorch float64 autograd over GQA ratios 1, 2, 4, and 8 matched [`attn_reference`, `attn_lse_reference`, and `attn_grad_reference`](/Users/vlad/kv-attn-review/kernelverify/schemas/native_ops.py:484) within 5.4e-15, including independent per-group dK and dV sums, while 6,252 valid-domain source/index cases found no additional mask, padding, write-coverage, or LSE arithmetic defect.

12. BLOCKER: CONFIRMED by EXECUTION - the requested `.venv` does not exist in this worktree, the substitute full suite aborted during collection when [`test_custom_function_gradient_path.py`](/Users/vlad/kv-attn-review/tests/test_custom_function_gradient_path.py:36) imported `mlx.nn` before GPU skipping, and the focused review suite completed with 39 passed and 49 skipped; the exact range contains 14 commits at `42ac312`, and the worktree remained clean.
ERROR: This content was flagged for possible cybersecurity risk. If this seems wrong, try rephrasing your request. To get authorized for security work, join the Trusted Access for Cyber program: https://chatgpt.com/cyber


## Codex B: audit of pre-registration Amendments 19 and 20

Brief: read-only, worktree at 42ac312. Told to assume Amendment 20 contains at least one error, recompute every arithmetic result, check every claim it makes about code, and find any surviving sentence that still rests on a withdrawn number.

1. HEADS UP: “Every arithmetic result below was reproduced by execution.” I could verify derived arithmetic, but not the raw GPU medians, `ratio_lo` endpoints, canary spreads, `f (median)`, or peak-memory observations; no committed samples permit regeneration. **UNDERDETERMINED**

2. INVESTIGATION: Clause 77’s `dense/quant` table has two displayed divisions that do not reproduce: `3.116/3.012 = 1.035`, not `1.034`, and `1.171/1.117 = 1.048`, not `1.049`; hidden unrounded medians could explain both. **UNDERDETERMINED**

3. HEADS UP: “Stock is within 4.5 percent,” and gains `1.0186` at `f=.376,r=1.051` and `1.0516` at `r=1.15`; recomputation gives `4.4972%`, `1.0185846`, and `1.0515728`. **HOLDS**

4. HEADS UP: “The proxy overstates it by 2.7x”; `26.1/9.7 = 2.69072`. **HOLDS**

5. INVESTIGATION: “Candidate L’s ... share ... is about 0.10 ... formula gives 1.0464”; `1.0464439` follows if rounded `f=.10` is treated as exact, but the preceding figures give `f=.245×9.7/26.1=.09105` and gain `1.0421`. **UNDERDETERMINED**

6. HEADS UP: Clause 77’s attention arithmetic reproduces: ratios are `3.23877`, `5.80640`, and `6.67142`; composed times times 36 are `49.320`, `940.428`, and `1616.112 ms`; `f=.25,r=3` gives `1.20`; the fp32 matrix is `286,015,744 bytes = 286.016 MB`. **HOLDS**

7. BUG NEEDS FIXING: “Correcting it moves the ratios by up to a factor of 1.89”; the T=2048 composed median moves by `44.892/19.903 = 2.25554`, and the credited long ratio moves by `3.02/1.58 = 1.91139`. **CONFIRMED FAULT**

8. HEADS UP: Clause 80’s `1.11`, `1.89`, and `arms_agree` threshold reproduce: `.397/.358=1.10894`, `21.522/11.379=1.89138`, and [the function](/Users/vlad/kv-attn-text/bench/interleave.py:68) uses `.005×max(1,3.953)=.019765`, which admits `.01562`. **HOLDS**

9. INVESTIGATION: The corrected median quotients reproduce as `11.416/3.899=2.92793` and `19.903/6.729=2.95779`; however, credited `ratio_lo` values `2.87`, `2.92`, and A5’s `2.29` cannot be regenerated without samples, while `.234/.095=2.46316`. **UNDERDETERMINED**

10. BUG NEEDS FIXING: “Disagree by a factor of 2.25”; `44.892/19.903=2.255539`, which rounds to `2.26`, even after allowing for normal rounding of both displayed medians. **CONFIRMED FAULT**

11. BUG NEEDS FIXING: Clause 81’s raw deltas reproduce `f (min)` as `.1062719`, `.1076254`, and `.1084971`, but `f (median)` is not derivable, “min” is undefined, the registered `.105-.108` follows the median column, and the gain table switches without rule to the first row’s min `.1063`; literal `.1085` is also outside `.108`. **CONFIRMED FAULT**

12. HEADS UP: Conditional on `f=.1063`, the gain table gives `1.01221`, `1.04061`, `1.07627`, `1.09294`, and ceiling `1/(1-.1063)=1.11894`. **HOLDS**

13. BUG NEEDS FIXING: “`r=1.128`, the forward swapped and the backward left composed” has no stated derivation: using forward ratio `1.58` gives `37.611/(26.264+11.347/1.58)=1.12454`, while using fused forward `3.914 ms` gives `1.24631`; `1.128` implies an unstated forward ratio `1.60289`. **CONFIRMED FAULT**

14. BUG NEEDS FIXING: “The floor is reachable only at `r >= 6.9`”; the exact threshold is `6.9066745`, and `r=6.9` yields only `1.09998199`, below `1.10`. **CONFIRMED FAULT**

15. HEADS UP: “At `f=.1051` it needs `7.41` and at `.1076` it needs `6.45`”; exact results are `7.40615` and `6.44662`. **HOLDS**

16. BUG NEEDS FIXING: “37.611 ms ... 5.45 ms ... requirement is 1.53 ms”; `11.347+26.264=37.611` and `37.611/6.9=5.45087`, but subtracting `3.914` leaves `1.53687`, which rounds to `1.54`; `1.53` uses the exact `6.9066745` threshold while the sentence says `6.9`. **CONFIRMED FAULT**

17. HEADS UP: “A gain of 1.30 requires `f >= .2308`” and is unreachable at the stated share; the exact threshold is `.23076923`, more than twice the largest displayed share. **HOLDS**

18. HEADS UP: The memory conversions reproduce: bf16 is `143,007,872 bytes = 136.3829 MiB = 143.0079 MB`, and fp32 is `286,015,744 bytes = 272.7659 MiB = 286.0157 MB`. **HOLDS**

19. INVESTIGATION: The reported `195.3 MiB` and `561.2 MiB` peaks are measurements, not conversions, and no committed record permits independent recomputation. **UNDERDETERMINED**

20. BUG NEEDS FIXING: “Amendment 5’s clause 1 and the probe record already establish `f := (T_stock-T_ablated)/T_stock`”; Amendment 5 actually registers a dialled fitted slope, while [the cited probe](/Users/vlad/kv-attn-text/bench/mlx_probes/probe_attention_ablation.py:57) adds the non-free keepalive reduction: `attention=(stock-ablated)+(stock+reductions-stock)`. Amendment 20’s `.1063` is exactly the uncorrected raw delta. **CONFIRMED FAULT**

21. BUG NEEDS FIXING: The stand-in allegedly keeps K/V projection work from being charged to attention, but the standing Amendment 6 text and `test_profile_knobs_live.py` establish that it preserves K/V projection forward only and zeroes their backward cotangents, making the delta include part of K/V projection backward; Amendment 20 omits this registered high bias. **CONFIRMED FAULT**

22. BUG NEEDS FIXING: “Batch 2, width 1057” is called the “long registered band” and applied to section 4.3, but the deciding Cell B remains batch 4 and Amendment 5 says both widths apply to that deciding cell; no batch-2-to-batch-4 transfer rule exists. **CONFIRMED FAULT**

23. BUG NEEDS FIXING: “The operation that measurement named is attention” and “The first operation was selected” remain standing after Amendment 20 withdraws their supporting share and ratios; its corrected arithmetic requires about `r>=6.91`, gives only `1.076` at `r=3`, and reaches section 4.3’s no-single-operation branch, while Amendment 20 expressly says it does not re-select. This is the highest-value incomplete withdrawal. **CONFIRMED FAULT**

24. BUG NEEDS FIXING: Amendment 19 calls the attention ratio its “registered floor,” but standing Amendment 6 clause 27 says candidate A has “no floor and no credited ratio anywhere”; Amendment 19’s ledger says Amendments 5 through 18 remain unchanged. **CONFIRMED FAULT**

25. BUG NEEDS FIXING: “What removes candidate Q is section 4.3’s own branch”; section 4.3 applies its below-1.10 branch only after finding the largest gain among all candidates, and then selects no operation, so it is not a per-candidate elimination rule. **CONFIRMED FAULT**

26. BUG NEEDS FIXING: Amendment 19’s ledger says “no reading in clause 77 is derived from” Amendments 5 through 18, but its `.245`, `.045`, and `.376` shares are copied directly from Amendment 5’s table. **CONFIRMED FAULT**

27. BUG NEEDS FIXING: “What puts ... the 16 GB machine in reach” survives at `561.2 MiB`, contradicting section 10’s standing “Nothing about a 16 GB machine” and requirement to wait for a second Mac; [the kernel module](/Users/vlad/kv-attn-text/kernelverify/pack/train_attention.py:22) also still states the withdrawn `286 MB` fp32 figure and repeats the 16 GB conclusion. **CONFIRMED FAULT**

28. BUG NEEDS FIXING: “Exact equality is satisfiable exactly when the backward IS stock” and no rewrite can ever satisfy it; a rewrite may reproduce stock’s accumulation order or be exact on the fixed inputs, while a stock call surrounded by different casts need not be exact. **CONFIRMED FAULT**

29. BUG NEEDS FIXING: “For attention ... no stock implementation [exists]”; no fused backward primitive exists, but MLX’s composed stock backward does, Amendment 19 itself says it runs in every arrangement, and `tests/test_native_ops.py` computes it with `mx.vjp`. **CONFIRMED FAULT**

30. HEADS UP: Clause 79’s analytic formulas, grouped dK/dV sums, one tolerance per gradient, and borrowed `K_NATIVE=1.5` are implemented in `native_ops.py` and cross-checked against central differences and MLX autograd. **HOLDS**

31. BUG NEEDS FIXING: The ledger says the funnel remains unchanged while rewrite gradients use those verdicts, but the helpers are not wired into a gate: [the E2E funnel](/Users/vlad/kv-attn-text/bench/train_lora_e2e.py:1043) unconditionally requires `verify_backward_exact`, an `exact` boolean, and `exact is True`, so a rewrite cannot reach timing without changing the funnel and evidence schema. **CONFIRMED FAULT**

32. BUG NEEDS FIXING: “The class is read off the kernel’s registered vjp”; no total classifier is specified for a VJP that conditionally calls stock, calls and modifies it, or calls and discards it, and no current operation or certificate field performs this classification; the claimed quantized-matmul worked example also has only a forward implementation. **CONFIRMED FAULT**

33. HEADS UP: Clause 80’s training-attention implementation does obtain its composed arm from `mx.vjp`, and its written decomposition scales queries before the matmul and uses `mx.softmax(..., precise=True)`; the live equality test exists but could not be run without Metal. **HOLDS**

34. BUG NEEDS FIXING: “The arm a pack gate times as the stock path ... is never hand-written” is unqualified and false: `pack_kv_attention.py` hand-writes and times its incumbent composition, and `pack_moe_dispatch.py` hand-writes and times MLX routing plus gather. **CONFIRMED FAULT**

35. BUG NEEDS FIXING: “No value check could have caught it” and bit-equality makes every future decomposition change fail; exact equality would catch the reported nonzero difference, and the committed test covers only three fixed geometries, so bit-preserving or shape-specific graph changes can pass. **CONFIRMED FAULT**

36. BUG NEEDS FIXING: “The same interleaved sampler every pack certificate ... is measured under”; `emit_pack_certificates.py` explicitly records no timings and makes no performance claims, excludes training attention, and the MoE gate samples complete arms sequentially rather than through that interleaved sampler. **CONFIRMED FAULT**

37. BUG NEEDS FIXING: “Every pricing verdict ... is computed with `ratio_lo`”; the routing-boundary classifier uses `ratio_lo` for WIN, `ratio_hi` for LOSS, and both for REFUSED, while other gates report median ratios. **CONFIRMED FAULT**

38. INVESTIGATION: “Same three cells ... eleven rounds” cannot be reproduced through committed code: `pack_train_attention.py` fixes five rounds and only two timed cells; an uncommitted one-off protocol may have produced the table. **UNDERDETERMINED**

39. BUG NEEDS FIXING: The new rules are post-result: commit `29abffd` changed the baseline at 10:39, Amendment 20 followed at 10:48, clause 81 chooses its range and `.1063` after seeing the samples, and undefined “ruling F10” was actually committed after observing short-cell ratios `1.03` and `0.99`; this violates the document’s “BEFORE dependent code moves” discipline. **CONFIRMED FAULT**

40. HEADS UP: Clause 82’s schema description is accurate: [Certificate](/Users/vlad/kv-attn-text/kernelverify/report/certificate.py:128) has one `translation_unit`, hashes that one source, has no class field, and renders its existing free-form `domain` mapping under `validity()`. **HOLDS**

41. BUG NEEDS FIXING: “`verification_class` is a required key ... `retune` or `rewrite`”; neither `__post_init__`, `audit`, nor any producer requires or validates it, and no current emitted domain contains it. The mapping can store it, but the claimed requiredness is absent. **CONFIRMED FAULT**

42. INVESTIGATION: “One certificate per source ... follows from the schema”; it does not follow automatically because emission names files solely from `kernel_name`, so two certificates with one name overwrite, and no source-role naming rule is registered; distinct names could make it implementable. **UNDERDETERMINED**

43. HEADS UP: Every Amendment 20 ledger row was checked against the named text and code:

   | Ledger claim | Finding | Status |
   |---|---|---|
   | “Composed column and composed/fused column ... withdrawn” | The body withdraws and replaces all three cells and withholds the short cell. | **HOLDS** |
   | “Fused column unchanged and confirmed” | Old values are `.423`, `4.499`, `6.729`; replacements are `.107`, `3.899`, `6.729`. Only the last is confirmed. | **CONFIRMED FAULT** |
   | “Quarter ... and 1.20 ... withdrawn and replaced” | Withdrawal is complete, but the replacement range has the unresolved min/median reading. | **UNDERDETERMINED** |
   | “Candidate Q cannot reach the shipping floor ... unchanged” | For valid `0<=f<=1`, whole-step gain cannot exceed Q’s `r=1.051`, so it remains below `1.10`. | **HOLDS** |
   | “Memory finding ... halved; conclusion unchanged” | The matrix arithmetic halves, but preserving the 16 GB conclusion contradicts section 10. | **CONFIRMED FAULT** |
   | “Caveat ... superseded; share now measured” | The ablation is uncorrected and biased, and L/Q’s other shares remain estimates. | **CONFIRMED FAULT** |
   | “Certificate sentence ... withdrawn” | The sentence is withdrawn and `domain` can store the replacement key. | **HOLDS** |
   | “Rest of clause 78 and all of 79 unchanged” | Accurate bookkeeping, though it deliberately leaves their substantive faults standing. | **HOLDS** |
   | “Section 4.3’s 1.10 floor and R10 unchanged” | The rule remains, although Amendment 20 applies its input at the wrong batch. | **HOLDS** |
   | “R18 target unchanged; unreachable” | The target and conditional ceiling arithmetic remain unchanged. | **HOLDS** |
   | “Funnel, V1, tolerances unchanged; no verdict moves” | A rewrite cannot use clause 79 without changing the exact-only funnel. | **CONFIRMED FAULT** |
   | “Every price ... unchanged” | Clause 80 explicitly reprices `3.02→1.58` and `1.03→0.86`. | **CONFIRMED FAULT** |
1. HEADS UP: “Every arithmetic result below was reproduced by execution.” I could verify derived arithmetic, but not the raw GPU medians, `ratio_lo` endpoints, canary spreads, `f (median)`, or peak-memory observations; no committed samples permit regeneration. **UNDERDETERMINED**

2. INVESTIGATION: Clause 77’s `dense/quant` table has two displayed divisions that do not reproduce: `3.116/3.012 = 1.035`, not `1.034`, and `1.171/1.117 = 1.048`, not `1.049`; hidden unrounded medians could explain both. **UNDERDETERMINED**

3. HEADS UP: “Stock is within 4.5 percent,” and gains `1.0186` at `f=.376,r=1.051` and `1.0516` at `r=1.15`; recomputation gives `4.4972%`, `1.0185846`, and `1.0515728`. **HOLDS**

4. HEADS UP: “The proxy overstates it by 2.7x”; `26.1/9.7 = 2.69072`. **HOLDS**

5. INVESTIGATION: “Candidate L’s ... share ... is about 0.10 ... formula gives 1.0464”; `1.0464439` follows if rounded `f=.10` is treated as exact, but the preceding figures give `f=.245×9.7/26.1=.09105` and gain `1.0421`. **UNDERDETERMINED**

6. HEADS UP: Clause 77’s attention arithmetic reproduces: ratios are `3.23877`, `5.80640`, and `6.67142`; composed times times 36 are `49.320`, `940.428`, and `1616.112 ms`; `f=.25,r=3` gives `1.20`; the fp32 matrix is `286,015,744 bytes = 286.016 MB`. **HOLDS**

7. BUG NEEDS FIXING: “Correcting it moves the ratios by up to a factor of 1.89”; the T=2048 composed median moves by `44.892/19.903 = 2.25554`, and the credited long ratio moves by `3.02/1.58 = 1.91139`. **CONFIRMED FAULT**

8. HEADS UP: Clause 80’s `1.11`, `1.89`, and `arms_agree` threshold reproduce: `.397/.358=1.10894`, `21.522/11.379=1.89138`, and [the function](/Users/vlad/kv-attn-text/bench/interleave.py:68) uses `.005×max(1,3.953)=.019765`, which admits `.01562`. **HOLDS**

9. INVESTIGATION: The corrected median quotients reproduce as `11.416/3.899=2.92793` and `19.903/6.729=2.95779`; however, credited `ratio_lo` values `2.87`, `2.92`, and A5’s `2.29` cannot be regenerated without samples, while `.234/.095=2.46316`. **UNDERDETERMINED**

10. BUG NEEDS FIXING: “Disagree by a factor of 2.25”; `44.892/19.903=2.255539`, which rounds to `2.26`, even after allowing for normal rounding of both displayed medians. **CONFIRMED FAULT**

11. BUG NEEDS FIXING: Clause 81’s raw deltas reproduce `f (min)` as `.1062719`, `.1076254`, and `.1084971`, but `f (median)` is not derivable, “min” is undefined, the registered `.105-.108` follows the median column, and the gain table switches without rule to the first row’s min `.1063`; literal `.1085` is also outside `.108`. **CONFIRMED FAULT**

12. HEADS UP: Conditional on `f=.1063`, the gain table gives `1.01221`, `1.04061`, `1.07627`, `1.09294`, and ceiling `1/(1-.1063)=1.11894`. **HOLDS**

13. BUG NEEDS FIXING: “`r=1.128`, the forward swapped and the backward left composed” has no stated derivation: using forward ratio `1.58` gives `37.611/(26.264+11.347/1.58)=1.12454`, while using fused forward `3.914 ms` gives `1.24631`; `1.128` implies an unstated forward ratio `1.60289`. **CONFIRMED FAULT**

14. BUG NEEDS FIXING: “The floor is reachable only at `r >= 6.9`”; the exact threshold is `6.9066745`, and `r=6.9` yields only `1.09998199`, below `1.10`. **CONFIRMED FAULT**

15. HEADS UP: “At `f=.1051` it needs `7.41` and at `.1076` it needs `6.45`”; exact results are `7.40615` and `6.44662`. **HOLDS**

16. BUG NEEDS FIXING: “37.611 ms ... 5.45 ms ... requirement is 1.53 ms”; `11.347+26.264=37.611` and `37.611/6.9=5.45087`, but subtracting `3.914` leaves `1.53687`, which rounds to `1.54`; `1.53` uses the exact `6.9066745` threshold while the sentence says `6.9`. **CONFIRMED FAULT**

17. HEADS UP: “A gain of 1.30 requires `f >= .2308`” and is unreachable at the stated share; the exact threshold is `.23076923`, more than twice the largest displayed share. **HOLDS**

18. HEADS UP: The memory conversions reproduce: bf16 is `143,007,872 bytes = 136.3829 MiB = 143.0079 MB`, and fp32 is `286,015,744 bytes = 272.7659 MiB = 286.0157 MB`. **HOLDS**

19. INVESTIGATION: The reported `195.3 MiB` and `561.2 MiB` peaks are measurements, not conversions, and no committed record permits independent recomputation. **UNDERDETERMINED**

20. BUG NEEDS FIXING: “Amendment 5’s clause 1 and the probe record already establish `f := (T_stock-T_ablated)/T_stock`”; Amendment 5 actually registers a dialled fitted slope, while [the cited probe](/Users/vlad/kv-attn-text/bench/mlx_probes/probe_attention_ablation.py:57) adds the non-free keepalive reduction: `attention=(stock-ablated)+(stock+reductions-stock)`. Amendment 20’s `.1063` is exactly the uncorrected raw delta. **CONFIRMED FAULT**

21. BUG NEEDS FIXING: The stand-in allegedly keeps K/V projection work from being charged to attention, but the standing Amendment 6 text and `test_profile_knobs_live.py` establish that it preserves K/V projection forward only and zeroes their backward cotangents, making the delta include part of K/V projection backward; Amendment 20 omits this registered high bias. **CONFIRMED FAULT**

22. BUG NEEDS FIXING: “Batch 2, width 1057” is called the “long registered band” and applied to section 4.3, but the deciding Cell B remains batch 4 and Amendment 5 says both widths apply to that deciding cell; no batch-2-to-batch-4 transfer rule exists. **CONFIRMED FAULT**

23. BUG NEEDS FIXING: “The operation that measurement named is attention” and “The first operation was selected” remain standing after Amendment 20 withdraws their supporting share and ratios; its corrected arithmetic requires about `r>=6.91`, gives only `1.076` at `r=3`, and reaches section 4.3’s no-single-operation branch, while Amendment 20 expressly says it does not re-select. This is the highest-value incomplete withdrawal. **CONFIRMED FAULT**

24. BUG NEEDS FIXING: Amendment 19 calls the attention ratio its “registered floor,” but standing Amendment 6 clause 27 says candidate A has “no floor and no credited ratio anywhere”; Amendment 19’s ledger says Amendments 5 through 18 remain unchanged. **CONFIRMED FAULT**

25. BUG NEEDS FIXING: “What removes candidate Q is section 4.3’s own branch”; section 4.3 applies its below-1.10 branch only after finding the largest gain among all candidates, and then selects no operation, so it is not a per-candidate elimination rule. **CONFIRMED FAULT**

26. BUG NEEDS FIXING: Amendment 19’s ledger says “no reading in clause 77 is derived from” Amendments 5 through 18, but its `.245`, `.045`, and `.376` shares are copied directly from Amendment 5’s table. **CONFIRMED FAULT**

27. BUG NEEDS FIXING: “What puts ... the 16 GB machine in reach” survives at `561.2 MiB`, contradicting section 10’s standing “Nothing about a 16 GB machine” and requirement to wait for a second Mac; [the kernel module](/Users/vlad/kv-attn-text/kernelverify/pack/train_attention.py:22) also still states the withdrawn `286 MB` fp32 figure and repeats the 16 GB conclusion. **CONFIRMED FAULT**

28. BUG NEEDS FIXING: “Exact equality is satisfiable exactly when the backward IS stock” and no rewrite can ever satisfy it; a rewrite may reproduce stock’s accumulation order or be exact on the fixed inputs, while a stock call surrounded by different casts need not be exact. **CONFIRMED FAULT**

29. BUG NEEDS FIXING: “For attention ... no stock implementation [exists]”; no fused backward primitive exists, but MLX’s composed stock backward does, Amendment 19 itself says it runs in every arrangement, and `tests/test_native_ops.py` computes it with `mx.vjp`. **CONFIRMED FAULT**

30. HEADS UP: Clause 79’s analytic formulas, grouped dK/dV sums, one tolerance per gradient, and borrowed `K_NATIVE=1.5` are implemented in `native_ops.py` and cross-checked against central differences and MLX autograd. **HOLDS**

31. BUG NEEDS FIXING: The ledger says the funnel remains unchanged while rewrite gradients use those verdicts, but the helpers are not wired into a gate: [the E2E funnel](/Users/vlad/kv-attn-text/bench/train_lora_e2e.py:1043) unconditionally requires `verify_backward_exact`, an `exact` boolean, and `exact is True`, so a rewrite cannot reach timing without changing the funnel and evidence schema. **CONFIRMED FAULT**

32. BUG NEEDS FIXING: “The class is read off the kernel’s registered vjp”; no total classifier is specified for a VJP that conditionally calls stock, calls and modifies it, or calls and discards it, and no current operation or certificate field performs this classification; the claimed quantized-matmul worked example also has only a forward implementation. **CONFIRMED FAULT**

33. HEADS UP: Clause 80’s training-attention implementation does obtain its composed arm from `mx.vjp`, and its written decomposition scales queries before the matmul and uses `mx.softmax(..., precise=True)`; the live equality test exists but could not be run without Metal. **HOLDS**

34. BUG NEEDS FIXING: “The arm a pack gate times as the stock path ... is never hand-written” is unqualified and false: `pack_kv_attention.py` hand-writes and times its incumbent composition, and `pack_moe_dispatch.py` hand-writes and times MLX routing plus gather. **CONFIRMED FAULT**

35. BUG NEEDS FIXING: “No value check could have caught it” and bit-equality makes every future decomposition change fail; exact equality would catch the reported nonzero difference, and the committed test covers only three fixed geometries, so bit-preserving or shape-specific graph changes can pass. **CONFIRMED FAULT**

36. BUG NEEDS FIXING: “The same interleaved sampler every pack certificate ... is measured under”; `emit_pack_certificates.py` explicitly records no timings and makes no performance claims, excludes training attention, and the MoE gate samples complete arms sequentially rather than through that interleaved sampler. **CONFIRMED FAULT**

37. BUG NEEDS FIXING: “Every pricing verdict ... is computed with `ratio_lo`”; the routing-boundary classifier uses `ratio_lo` for WIN, `ratio_hi` for LOSS, and both for REFUSED, while other gates report median ratios. **CONFIRMED FAULT**

38. INVESTIGATION: “Same three cells ... eleven rounds” cannot be reproduced through committed code: `pack_train_attention.py` fixes five rounds and only two timed cells; an uncommitted one-off protocol may have produced the table. **UNDERDETERMINED**

39. BUG NEEDS FIXING: The new rules are post-result: commit `29abffd` changed the baseline at 10:39, Amendment 20 followed at 10:48, clause 81 chooses its range and `.1063` after seeing the samples, and undefined “ruling F10” was actually committed after observing short-cell ratios `1.03` and `0.99`; this violates the document’s “BEFORE dependent code moves” discipline. **CONFIRMED FAULT**

40. HEADS UP: Clause 82’s schema description is accurate: [Certificate](/Users/vlad/kv-attn-text/kernelverify/report/certificate.py:128) has one `translation_unit`, hashes that one source, has no class field, and renders its existing free-form `domain` mapping under `validity()`. **HOLDS**

41. BUG NEEDS FIXING: “`verification_class` is a required key ... `retune` or `rewrite`”; neither `__post_init__`, `audit`, nor any producer requires or validates it, and no current emitted domain contains it. The mapping can store it, but the claimed requiredness is absent. **CONFIRMED FAULT**

42. INVESTIGATION: “One certificate per source ... follows from the schema”; it does not follow automatically because emission names files solely from `kernel_name`, so two certificates with one name overwrite, and no source-role naming rule is registered; distinct names could make it implementable. **UNDERDETERMINED**

43. HEADS UP: Every Amendment 20 ledger row was checked against the named text and code:

   | Ledger claim | Finding | Status |
   |---|---|---|
   | “Composed column and composed/fused column ... withdrawn” | The body withdraws and replaces all three cells and withholds the short cell. | **HOLDS** |
   | “Fused column unchanged and confirmed” | Old values are `.423`, `4.499`, `6.729`; replacements are `.107`, `3.899`, `6.729`. Only the last is confirmed. | **CONFIRMED FAULT** |
   | “Quarter ... and 1.20 ... withdrawn and replaced” | Withdrawal is complete, but the replacement range has the unresolved min/median reading. | **UNDERDETERMINED** |
   | “Candidate Q cannot reach the shipping floor ... unchanged” | For valid `0<=f<=1`, whole-step gain cannot exceed Q’s `r=1.051`, so it remains below `1.10`. | **HOLDS** |
   | “Memory finding ... halved; conclusion unchanged” | The matrix arithmetic halves, but preserving the 16 GB conclusion contradicts section 10. | **CONFIRMED FAULT** |
   | “Caveat ... superseded; share now measured” | The ablation is uncorrected and biased, and L/Q’s other shares remain estimates. | **CONFIRMED FAULT** |
   | “Certificate sentence ... withdrawn” | The sentence is withdrawn and `domain` can store the replacement key. | **HOLDS** |
   | “Rest of clause 78 and all of 79 unchanged” | Accurate bookkeeping, though it deliberately leaves their substantive faults standing. | **HOLDS** |
   | “Section 4.3’s 1.10 floor and R10 unchanged” | The rule remains, although Amendment 20 applies its input at the wrong batch. | **HOLDS** |
   | “R18 target unchanged; unreachable” | The target and conditional ceiling arithmetic remain unchanged. | **HOLDS** |
   | “Funnel, V1, tolerances unchanged; no verdict moves” | A rewrite cannot use clause 79 without changing the exact-only funnel. | **CONFIRMED FAULT** |
   | “Every price ... unchanged” | Clause 80 explicitly reprices `3.02→1.58` and `1.03→0.86`. | **CONFIRMED FAULT** |


## Codex C: Lane B tuner fixes

Brief: workspace-write in the kv-anymac-mr worktree at 4c0fd8b. Fix 1 was written test-first, red then green, and left UNCOMMITTED because the full suite could not run on that host; Fix 2 was deferred for the same reason.

- `BLOCKER:` `KV_FORCE_NO_METAL=1 /Users/vlad/kernelverify/.venv/bin/python -m pytest -q` exits 134 while importing `mlx.nn`; its last non-empty line is `Extension modules: numpy._core._multiarray_umath, numpy.linalg._umath_linalg, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._pcg64, numpy.random._generator, numpy.random._mt19937, numpy.random._philox, numpy.random._sfc64, numpy.random.mtrand (total: 11)`.

- `HEADS UP:` Commits made: none, because committing would violate the mandatory green full-suite gate.

- `HEADS UP:` Fix 1 remains uncommitted in the working tree; its regression failed with `1 failed in 0.17s`, then passed after the accounting fix with `1 passed in 0.13s`.

- `DEFERRAL:` Fix 2, both commits, and final verification were not completed because the required suite cannot run on this host, while unrelated no-Metal gating changes are explicitly outside scope.

- `DEFERRAL:` The external lane report from the old plan was not written because the current brief permits changes only inside this worktree and only to files required by the two fixes.

- `WHATS NEXT:` Please provide a Metal-accessible test runner or authorize the additional no-Metal suite-gating work.
diff --git a/metalrunner/tuner.py b/metalrunner/tuner.py
index abbfa394396ace107dd12b8115b56801ed2fb009..a9454ba3bfc97314f0a070ae161d4752388f6c41
--- a/metalrunner/tuner.py
+++ b/metalrunner/tuner.py
@@ -268,13 +268,14 @@
 
     knob_map = {key: v["knobs"] for key, v in verdicts.items()}
     kept = sum(1 for value in knob_map.values() if value != "stock")
+    cells_considered = len(verdicts)
     report = {
         "cache_hit": False,
         "deferred": False,
         "candidate_sha256": candidate,
         "operation": OPERATION,
-        "cells_considered": len(cells),
-        "knob_census": {"kept": kept, "stock": len(cells) - kept},
+        "cells_considered": cells_considered,
+        "knob_census": {"kept": kept, "stock": cells_considered - kept},
         "verdicts": verdicts,
         "route_to_stock": kept == 0,
     }
diff --git a/tests/test_metalrunner_tuner.py b/tests/test_metalrunner_tuner.py
index 26a779c84c304bdc703db1311d1ce2052cd750f9..bace13bc90478f2e78a5953c7909950036abd10e
--- a/tests/test_metalrunner_tuner.py
+++ b/tests/test_metalrunner_tuner.py
@@ -198,6 +198,27 @@
     assert outcome.entry["knob_map"] == {"S1:M128": {}}
 
 
+def test_a_repeated_batch_m_that_collapses_to_one_cell_is_counted_once(
+        tmp_path, shapes):
+    path = tmp_path / "certified-v1.json"
+    outcome = tuner.ensure_tuned(
+        model_path="/m", bits=BITS, group_size=GROUP_SIZE, chip=CHIP,
+        mlx_version=MLX_VERSION, batches=[128, 128], store_path=path,
+        hardware=_hardware(
+            verify_cell=lambda shape, m, knobs: True,
+            time_cell=lambda shape, m, knobs: _timings(
+                [1.0, 1.0], [0.9, 0.9])))
+
+    assert set(outcome.report["verdicts"]) == {"S1:M128"}
+    assert outcome.entry["knob_map"] == {"S1:M128": {}}
+    assert outcome.report["cells_considered"] == 1
+    assert outcome.report["knob_census"] == {"kept": 1, "stock": 0}
+    census_total = sum(outcome.report["knob_census"].values())
+    assert census_total == len(outcome.entry["knob_map"]), (
+        "the report's own census must add up to the entries it actually "
+        "produced")
+
+
 def test_a_cell_falls_back_to_stock_when_the_canary_spread_is_too_wide(
         tmp_path, shapes):
     path = tmp_path / "certified-v1.json"

- `BLOCKER:` `KV_FORCE_NO_METAL=1 /Users/vlad/kernelverify/.venv/bin/python -m pytest -q` exits 134 while importing `mlx.nn`; its last non-empty line is `Extension modules: numpy._core._multiarray_umath, numpy.linalg._umath_linalg, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._pcg64, numpy.random._generator, numpy.random._mt19937, numpy.random._philox, numpy.random._sfc64, numpy.random.mtrand (total: 11)`.

- `HEADS UP:` Commits made: none, because committing would violate the mandatory green full-suite gate.

- `HEADS UP:` Fix 1 remains uncommitted in the working tree; its regression failed with `1 failed in 0.17s`, then passed after the accounting fix with `1 passed in 0.13s`.

- `DEFERRAL:` Fix 2, both commits, and final verification were not completed because the required suite cannot run on this host, while unrelated no-Metal gating changes are explicitly outside scope.

- `DEFERRAL:` The external lane report from the old plan was not written because the current brief permits changes only inside this worktree and only to files required by the two fixes.

- `WHATS NEXT:` Please provide a Metal-accessible test runner or authorize the additional no-Metal suite-gating work.
