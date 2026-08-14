"""The per-chip matrix: measured rows in, publishable claims out.

This is the launch artefact (design decision D7: the tool and the matrix lead,
the pack follows), so its job is not to render numbers - it is to decide which
numbers a reader is allowed to believe, and to say why for the ones it refuses.

Two questions with different answers and different fields, per the row schema
(bench/.baselines/SCHEMA.md); conflating them is the failure this module
exists to prevent:

    publishable as an absolute   provenance_tier in (owner-run, rental-run)
                                 AND binding is true
    comparable against a row     both rows sampling.interleaved
                                 AND the same sampling.group

The second is not pedantry. The kernels lane ran an A/B whose every dispatch
was already past 5 ms and it still reversed sign against an interleaved rerun,
turning a measured 1.45x win into 0.77-0.94x, because the arms ran in separate
passes and the GPU changed clock underneath them. So two rows can both be
binding and still not be comparable to each other.

Everything refused is rendered WITH its reason rather than dropped: a rejected
row is the evidence for why a number was not published, and a matrix that
silently omits its failures is the kind of artefact this project exists to
criticise.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PUBLISHABLE_TIERS = ("owner-run", "rental-run")
UNATTESTED_TIER = "community-unattested"

# Decode streams weights and barely writes, so the copy ceiling understates the
# real limit and overstates utilisation by about 5% on this machine.
DECODE_DENOMINATOR = "bandwidth_read"


@dataclass(frozen=True)
class Claim:
    """One row, plus the verdict on what a reader may do with it."""

    row: dict
    publishable: bool
    reasons: tuple  # why not, empty iff publishable

    @property
    def attested(self) -> bool:
        """Measured on a machine the project controls.

        The trust question asked directly, not as a match against one literal:
        an unrecognised or future provenance value must be treated as
        unattested, or a row tagged "Owner-Run" (capitalised, unknown) walks
        into a comparison the known contributed tier is barred from.
        """
        return self.row.get("provenance_tier") in PUBLISHABLE_TIERS

    @property
    def unattested(self) -> bool:
        return not self.attested

    @property
    def group(self) -> str | None:
        return (self.row.get("sampling") or {}).get("group")

    @property
    def interleaved(self) -> bool:
        return bool((self.row.get("sampling") or {}).get("interleaved"))


def load_rows(directory: Path) -> list[dict]:
    """Every row from every dated file, oldest file first.

    A malformed line is a defect in the producer, not something to skip
    quietly: the schema says the producer validates before writing, so a line
    that will not parse means the record is damaged and the reader must say so.
    """
    rows: list[dict] = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} is not valid JSON: {exc}") from exc
    return rows


def binding_blockers(row: dict) -> tuple:
    """Why this row failed to bind, independent of its provenance tier."""
    if row.get("binding", False):
        return ()
    return tuple(row.get("binding_blockers")
                 or ["binding is false with no blocker recorded"])


def classify(row: dict) -> Claim:
    """Decide whether this row's absolute number may be published, and why not."""
    reasons: list[str] = []
    tier = row.get("provenance_tier")
    if tier == UNATTESTED_TIER:
        reasons.append("community-unattested: a claim, not a measurement we control")
    elif tier not in PUBLISHABLE_TIERS:
        reasons.append(f"unknown provenance tier {tier!r}")
    reasons.extend(binding_blockers(row))
    return Claim(row=row, publishable=not reasons, reasons=tuple(reasons))


def _workload(row: dict) -> tuple:
    """WHAT was measured. Two rows are an A/B only when this matches.

    Found live in the shipped record: the producer sets sampling.group to the
    run id, so one group held decode, prefill and four matmul widths, and the
    render anchored every ratio on the prefill row - a decode row read 0.07x
    against its own stack, a workload ratio published as a stack ratio.

    Deliberately excluded from the key, because requiring them would block the
    cross-stack comparison the table exists for:
    - width_mechanism (prompt-width on llama.cpp, batch-size on MLX by
      construction; rendered next to the width instead),
    - the model file (the same logical model is a different artefact per
      stack; rendered as its own column instead),
    - n_prompt outside prefill (stacks realise the same matmul width with
      different prompt lengths).
    """
    measurement = row.get("measurement") or {}
    kind = measurement.get("kind")
    return (kind,
            measurement.get("matmul_width"),
            measurement.get("n_prompt") if kind == "prefill" else None)


def _workload_label(row: dict) -> str:
    measurement = row.get("measurement") or {}
    return f"{measurement.get('kind', '?')} at width {measurement.get('matmul_width', '?')}"


def comparable(a: Claim, b: Claim) -> tuple[bool, str]:
    """May these two rows be put side by side as a ratio?

    Bindingness does not answer this and must not be used to. Two rows from
    different sampling groups may both be binding and still be a comparison of
    the clock rather than of the kernels.
    """
    if not (a.interleaved and b.interleaved):
        return False, "a row was not sampled interleaved; separate passes measure the clock"
    if a.group is None or a.group != b.group:
        return False, f"different sampling groups ({a.group!r} vs {b.group!r}); rows were never paired"
    if _workload(a.row) != _workload(b.row):
        return False, (f"different workloads ({_workload_label(a.row)} vs "
                       f"{_workload_label(b.row)}); a ratio across kinds divides "
                       "two different jobs, not two stacks")
    a_metric = (a.row.get("result") or {}).get("metric")
    b_metric = (b.row.get("result") or {}).get("metric")
    if a_metric != b_metric:
        return False, (f"different metrics ({a_metric!r} vs {b_metric!r}); "
                       "a ratio of two units is not a number")
    return True, ""


def utilisation(row: dict) -> tuple[str, float | None]:
    """The utilisation column this row is actually about.

    Reading a prefill row as a bandwidth percentage shows a 1% catastrophe
    where the truth is a 60% compute result, so the roofline says which
    resource binds and that choice is not the renderer's to make.
    """
    roofline = row.get("roofline") or {}
    resource = roofline.get("binding_resource")
    if resource == "compute":
        return "compute", roofline.get("roofline_utilisation_pct")
    if resource == "memory":
        return "bandwidth", roofline.get("bandwidth_utilisation_pct")
    # An unrecognised value is not a bandwidth row. The producer stamps MLX
    # prefill rows "unknown" precisely because no parameter count exists to
    # place them; defaulting those to bandwidth publishes the 1%-catastrophe
    # this function exists to prevent.
    return f"unknown ({resource!r})", None


def _fmt(value, suffix: str = "", places: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{places}f}{suffix}"
    return f"{value}{suffix}"


def _width_label(row: dict) -> str:
    """Width with the mechanism that produced it.

    Prompt width is one causal sequence; batch size is independent streams.
    Both widen the same matmul dimension and they are not the same workload,
    so the mechanism travels with the number or the two stacks look more
    comparable than they are.
    """
    measurement = row.get("measurement") or {}
    width = measurement.get("matmul_width")
    if width is None:
        return "-"
    mechanism = measurement.get("width_mechanism") or "unspecified"
    return f"{width} ({mechanism})"


def _stack_label(row: dict) -> str:
    stack = row.get("stack") or {}
    name = stack.get("name", "?")
    version = stack.get("version")
    return f"{name} {version}" if version else name


def render_ceilings(claims: list[Claim]) -> list[str]:
    ceilings = [c for c in claims
                if (c.row.get("measurement") or {}).get("kind") == "ceiling"]
    if not ceilings:
        return []
    lines = ["## Machine ceilings", "",
             "Everything below is a share of these, so a wrong ceiling makes every "
             "percentage wrong.", "",
             "| Ceiling | Value | Publishable |", "|---|---|---|"]
    for claim in sorted(ceilings, key=lambda c: (c.row["measurement"].get("name") or "")):
        row = claim.row
        result = row.get("result") or {}
        name = row["measurement"].get("name", "?")
        value = _fmt(result.get("median"), f" {result.get('metric', '')}".rstrip())
        verdict = "yes" if claim.publishable else "no: " + "; ".join(claim.reasons)
        lines.append(f"| `{name}` | {value} | {verdict} |")
    lines.append("")
    return lines


def render_measurements(claims: list[Claim]) -> list[str]:
    rows = [c for c in claims
            if (c.row.get("measurement") or {}).get("kind") != "ceiling"]
    publishable = [c for c in rows if c.publishable and c.attested]
    if not publishable:
        return ["## Measurements", "",
                "No row in the record is publishable as an absolute number yet.",
                "Every measured row appears below - refused rows under *Refused* "
                "with their reasons, contributed rows in their own section - "
                "which is the honest state rather than an empty table.", ""]

    lines = ["## Measurements", "",
             "Absolute numbers, each measured on a machine the project controls and "
             "each passing every bindingness gate.", "",
             "| Stack | Model | Kind | Width | Metric | Utilisation | Binds on | Spread | Run |",
             "|---|---|---|---|---|---|---|---|---|"]
    for claim in publishable:
        row = claim.row
        result = row.get("result") or {}
        model = (row.get("model") or {}).get("name", "-")
        kind = (row.get("measurement") or {}).get("kind", "-")
        resource, pct = utilisation(row)
        # The run identity keeps a second binding run from rendering as an
        # indistinguishable duplicate of the first: two absolutes for the same
        # workload must be attributable to their runs or neither is checkable.
        lines.append(
            f"| {_stack_label(row)} | {model} | {kind} | {_width_label(row)} "
            f"| {_fmt(result.get('median'))} {result.get('metric', '')} "
            f"| {_fmt(pct, '%')} | {resource} | {_fmt(result.get('spread_pct'), '%')} "
            f"| `{row.get('run_id', '-')}` |"
        )
    lines.append("")
    lines.append("Utilisation is measured against the resource that actually binds each "
                 "row; a compute-bound row read as bandwidth is meaningless.")
    lines.append("")
    return lines


def render_comparisons(claims: list[Claim]) -> list[str]:
    """Ratios, but only between rows the schema says may be compared.

    Two exclusions that are not the same rule:

    - An unattested row never enters a comparison. The schema says such rows
      are rendered separately and never merged into a binding result, and a
      ratio against one is exactly that merge.
    - A non-binding row MAY appear, because the schema keeps those rows
      explicitly for ratios measured in the same run - but only against a
      baseline from that same run, and it is labelled so no reader mistakes
      the ratio for a published absolute.
    """
    cells: dict[tuple, list[Claim]] = defaultdict(list)
    for claim in claims:
        if (claim.row.get("measurement") or {}).get("kind") == "ceiling":
            continue
        if not claim.attested:
            continue
        if claim.group:
            # A cell is one real A/B: one sampling group AND one workload.
            # Grouping on the sampling group alone rendered a decode row at
            # 0.07x of its own stack's prefill row on the first real record.
            cells[(claim.group, _workload(claim.row))].append(claim)

    pairs = {key: members for key, members in cells.items() if len(members) > 1}
    if not pairs:
        return []

    lines = ["## Comparisons", "",
             "Each table is one workload, measured interleaved within one sampling "
             "group. Rows from different groups are not paired even when both are "
             "binding, because an A/B run in separate passes measures the clock: "
             "one fixed shape drifted 131.7 to 93.1 us between runs minutes apart. "
             "Rows from different workloads are not paired because that ratio "
             "divides two different jobs, not two stacks.", ""]
    for (name, _workload_key), members in sorted(
            pairs.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        members = sorted(members, key=lambda c: (
            (c.row.get("measurement") or {}).get("kind", ""),
            (c.row.get("measurement") or {}).get("matmul_width") or 0,
            c.row.get("row_id", ""),
        ))
        # The baseline is the cell's own publishable row where one exists, so
        # a ratio is anchored to a number a reader is allowed to believe.
        baseline = next((c for c in members if c.publishable), members[0])
        base_median = (baseline.row.get("result") or {}).get("median")
        lines.append(f"### {_workload_label(baseline.row)}, group `{name}`")
        lines.append("")
        # The unit travels with the ratio: a table where "2.6x" might be a
        # throughput or a latency lets the reader pick the flattering reading.
        base_metric = (baseline.row.get("result") or {}).get("metric") or "?"
        lines.append(f"| Stack | Model | Width | Metric "
                     f"| Ratio vs {_stack_label(baseline.row)} ({base_metric}) | Note |")
        lines.append("|---|---|---|---|---|---|")
        for claim in members:
            result = claim.row.get("result") or {}
            median = result.get("median")
            notes: list[str] = []
            if claim is baseline:
                notes.append("baseline")
            ok, why = comparable(baseline, claim)
            same_run = claim.row.get("run_id") == baseline.row.get("run_id")
            if not ok:
                ratio, notes = "-", [why]
            elif not claim.publishable and not same_run:
                ratio = "-"
                notes = ["non-binding row from another run; the schema keeps these "
                         "for ratios inside their own run only"]
            elif not base_median or median is None:
                ratio, notes = "-", ["no median recorded"]
            else:
                value = median / base_median
                spread = (result.get("spread_pct") or 0.0) / 100.0
                base_spread = ((baseline.row.get("result") or {})
                               .get("spread_pct") or 0.0) / 100.0
                # A difference smaller than the pair's own noise is not a
                # difference, and the baseline's spread hides a ratio as surely
                # as the row's does - a tight row against a noisy baseline is
                # still a ratio of noise.
                if abs(value - 1.0) <= spread + base_spread and claim is not baseline:
                    notes.append("within the measurement's own spread, treat as no difference")
                if not claim.publishable:
                    notes.append("ratio only: this row is not publishable as an absolute")
                ratio = f"{value:.2f}x"
            model = (claim.row.get("model") or {}).get("name", "-")
            lines.append(f"| {_stack_label(claim.row)} | {model} "
                         f"| {_width_label(claim.row)} | {_fmt(median)} "
                         f"{result.get('metric', '')} | {ratio} | {'; '.join(notes)} |")
        lines.append("")
    return lines


def render_refusals(claims: list[Claim]) -> list[str]:
    refused = [c for c in claims if c.attested and not c.publishable]
    if not refused:
        return []
    lines = ["## Refused", "",
             "Measured, kept, and not published. These rows are still valid for "
             "ratios inside their own run, and they are the evidence for why a "
             "number was rejected.", "",
             "| Row | Kind | Reason |", "|---|---|---|"]
    for claim in refused:
        row = claim.row
        lines.append(f"| `{row.get('row_id', '?')}` "
                     f"| {(row.get('measurement') or {}).get('kind', '-')} "
                     f"| {'; '.join(claim.reasons)} |")
    lines.append("")
    return lines


def render_unattested(claims: list[Claim]) -> list[str]:
    unattested = [c for c in claims if not c.attested]
    if not unattested:
        return []
    lines = ["## Contributed claims (unattested or unrecognised provenance)", "",
             "Rows from machines the project does not control, including any row "
             "whose provenance value the schema does not recognise. They are "
             "rendered separately and never merged into a binding result: a "
             "contributed row is a claim about someone else's machine until we "
             "can reproduce it. The bindingness column carries the row's own "
             "gate results, because a contributed row that failed its own "
             "dispersion gate is a weaker claim than one that passed.", "",
             "| Row | Stack | Kind | Value | Bindingness |", "|---|---|---|---|---|"]
    for claim in unattested:
        row = claim.row
        result = row.get("result") or {}
        blockers = binding_blockers(row)
        verdict = ("passed its own gates" if not blockers
                   else "NOT binding: " + "; ".join(blockers))
        lines.append(f"| `{row.get('row_id', '?')}` | {_stack_label(row)} "
                     f"| {(row.get('measurement') or {}).get('kind', '-')} "
                     f"| {_fmt(result.get('median'))} {result.get('metric', '')} "
                     f"| {verdict} |")
    lines.append("")
    return lines


def render(rows: list[dict]) -> str:
    """The whole matrix as markdown."""
    claims = [classify(row) for row in rows]
    machine = next((r.get("machine") for r in rows if r.get("machine")), {}) or {}

    header = ["# Per-chip correctness and performance matrix", ""]
    if machine:
        header.append(
            f"Machine: {machine.get('chip', '?')}, {machine.get('hw_model', '?')}, "
            f"{machine.get('os', '?')} {machine.get('os_build', '')}".strip() + "."
        )
    published = sum(1 for c in claims if c.publishable and c.attested)
    header.append(
        f"{len(claims)} measured rows: {published} publishable, "
        f"{sum(1 for c in claims if c.attested and not c.publishable)} refused, "
        f"{sum(1 for c in claims if not c.attested)} contributed."
    )
    header.append("")

    sections = (header
                + render_ceilings(claims)
                + render_measurements(claims)
                + render_comparisons(claims)
                + render_refusals(claims)
                + render_unattested(claims))
    return "\n".join(sections).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse

    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, default=repo_root / "bench" / ".baselines",
                        help="directory of dated JSONL row files")
    parser.add_argument("--out", type=Path, default=None,
                        help="write markdown here instead of stdout")
    args = parser.parse_args(argv)

    if not args.rows.exists():
        print(f"no rows at {args.rows}; the binding run has not produced a record yet")
        return 1
    rows = load_rows(args.rows)
    if not rows:
        print(f"no rows in {args.rows}; nothing measured yet")
        return 1
    text = render(rows)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out} ({len(rows)} rows)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
