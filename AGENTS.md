# AGENTS.md

## Project

kernelverify is a GPU-kernel correctness verifier.
Motivation: the paper "The Correctness Illusion in LLM-Generated GPU Kernels" (arXiv 2606.20128) and its gpuemu corpus.
Strategy: build the verifier first, then an optimiser that makes open-weight models (Llama, Qwen) run faster, with the Mac Metal backend as a first-class target alongside server GPUs.
Core reframing (ADR 0001): escape rate is a property of the pair (bug, test case), not of a bug alone.
Differentiator (ADR 0002): mutation scoring of test policies against a synthesised fault population that provably contains the published corpus faults.

## Layout

- `kernelverify/reference/kernels.py`: parameterised kernels, correct by default, keyword seams introduce faults. Single source for both corpus ports and mutations.
- `kernelverify/mutation/catalogue.py`: the fault catalogue (44 entries), corpus faults marked `from_corpus`.
- `bench/cpu_ports.py`: corpus kernel name -> parameterisation mapping, plus buggy-to-control pairs.
- `bench/measure_escape.py`: escape-rate measurement against the vendored corpus. Frozen: reruns must be regression-identical.
- `bench/score_oracles.py`: mutation scoring of test policies at equal budget B.
- `docs/adr/`: decisions with the measurements that forced them. Read these before changing method.
- `vendor/gpuemu-corpus/`: vendored unmodified at the commit pinned in `vendor/PINNED.txt`. Never edit.
- Empty packages (`battery`, `detectors`, ...) are planned components, not dead code.

## Running

```
cd /Users/vlad/kernelverify
.venv/bin/python bench/measure_escape.py   # ~1 min, must reproduce ADR 0001 tables
.venv/bin/python bench/score_oracles.py    # ~3-5 min cold, verdict cache at bench/.cache/
```

## Mistakes already encountered

- The shell cwd resets between tool calls, so `cd /Users/vlad/kernelverify` in every command.
- The verdict cache stores plain tuples, not dataclasses, because pickled dataclasses remember the defining module and break when loaded from an import context.
- The cache fingerprints the mutation names and input modes; changing either rebuilds it, but deleting `bench/.cache/` is the safe reset.
- Corpus `benchmark_verdict` fields are hardcoded "pass", never computed. Do not cite them as measurements.
- Structured (non-iid) input cases can make a correct fp32 kernel exceed the corpus tolerance against the fp64 reference.
  That is ill-conditioning, not a port bug: score such cases as no-evidence, never as detections, and only treat control failures on iid modes as port errors.
- Table cells rounded to whole percents once overstated a result (99.8% shown as 100%). Keep one decimal and check exact counts for any 100.0% cell.
- Never add AI co-author attribution to commits.

## Personality

Be extremely concise.
Sacrifice grammar for concision.

## Subagents

Delegate suitable tasks to subagents.
Keep simple tasks on the main agent.
Use the `fast_scan` agent for quick searches, codebase exploration, targeted reads, research, documentation lookups, or lightweight analysis.
Wait for all subagents to finish.
Consolidate findings, then make changes or recommendations.

## Research

Use the `fast_scan` agent for research tasks.
Verify external technical details before implementing or deciding.
Use available tools and skills, including Context7 and Exa.
Prefer official documentation, specifications, release notes, and other primary sources.
Verify APIs, library behavior, defaults, and best practices.
Do not guess.

## Defaults

Use official, documented, recommended approaches.
Prefer stable public APIs and conventional framework patterns.
Avoid monkey patches, brittle assumptions, timing hacks, and implementation-detail dependencies.
Do not bypass abstractions for speed.

## Documentation

Delegate any task that creates or updates technical documentation to the `documentation_expert` subagent.

Documentation work includes:

- Tutorials
- How-to guides
- Reference documentation
- Explanatory or conceptual documentation
- READMEs
- Installation and configuration guides
- API and CLI documentation
- Architecture and design documentation
- Troubleshooting guides
- Documentation audits and restructuring

## Skills

Output `🧢 Using Skill $<skill-name>` whenever using a skill.

## Browser

Use `$playwright-cli` when rendered UI is the evidence.
Covers interactive navigation, authenticated state, UI inspection, visual verification, behavior testing, debugging, snapshots, and screenshots.

## Questions

Use `request_user_input` for every question.
Never ask inline.
Ask one focused question per call.
Never ask more than three.
List the best option first and label it `(Recommended)`.
Use simple, technical language suitable for a junior developer.

## GitHub

Use the `gh` CLI for all GitHub operations.

## TypeScript

Use `pnpm`, or `vp` when Vite+ is configured.

## Python

Use `uv`.
Use `ruff` for linting and formatting.
