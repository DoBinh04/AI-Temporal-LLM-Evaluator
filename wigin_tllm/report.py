"""Human-readable rendering of evaluation output."""

from __future__ import annotations

from .check import FAIL, OK, SKIP, WARN, ModelReport
from .types import RoundResults

_MARK = {OK: "ok  ", WARN: "warn", FAIL: "FAIL", SKIP: "n/a "}

_COLUMNS = (
    ("rank", 5, "<"),
    ("submitter", 14, "<"),
    ("leak", 9, ">"),
    ("norm", 6, ">"),
    ("quality", 8, ">"),
    ("final", 7, ">"),
)


def format_results(results: RoundResults) -> str:
    """Render a results table, best submitter first."""
    header = "  ".join(f"{name:{align}{width}}" for name, width, align in _COLUMNS) + "  status"
    rule = "-" * max(len(header), 68)
    lines = [f"=== Round {results.round_id} ===", header, rule]

    if not results.submitters:
        lines.append("No submissions evaluated.")
        return "\n".join(lines)

    for s in results.submitters:
        status = "qualified" if s.qualified else (s.disqualified_reason or "not qualified")
        lines.append(
            "  ".join(
                [
                    f"{s.rank or '-':<5}",
                    f"{s.submitter_id:<14}",
                    f"{s.leak_score:>9.4f}",
                    f"{s.normalized_leak:>6.3f}",
                    f"{s.quality_win_rate:>8.3f}",
                    f"{s.final_score:>7.4f}",
                ]
            )
            + f"  {status}"
        )

    lines.append(rule)
    return "\n".join(lines)


def format_model_check(report: ModelReport) -> str:
    """Render a pre-flight report: what blocks submission, and how it scores."""
    rule = "-" * 72
    lines = [f"=== Model check: {report.ref} ===", "", "Artefact", rule]

    for item in report.artifact:
        lines.append(f"  [{_MARK[item.level]}]  {item.name:<32} {item.detail}")
        if item.hint:
            lines.append(f"            -> {item.hint}")

    if report.years:
        lines += ["", "Chronological consistency", rule]
        for year in report.years:
            lines.extend(_format_year(year))

    lines += ["", rule]
    if not report.ok:
        lines.append("NOT ACCEPTED — resolve the FAIL items above.")
    elif not report.years:
        lines.append("Artefact is fine. Pass --data to also score it against probe sets.")
    elif report.years_passed == 0:
        lines.append(
            "Artefact is fine, but no year passes the consistency check — "
            "see the diagnosis on each year above."
        )
    else:
        lines.append(
            f"Artefact is fine. {report.years_passed}/{len(report.years)} years "
            f"pass the consistency check."
        )
    return "\n".join(lines)


def _format_year(year) -> list[str]:
    assessment = year.assessment
    verdict = "PASS" if year.passed else "FAIL"
    lines = [f"  {year.year}  [{verdict}]  score {assessment.score:+.4f}"
             f"   normalised {year.normalized:.3f}"]

    for probe, want in ((assessment.known, "must recognise"), (assessment.unknown, "must not recognise")):
        lines.append(
            f"      {probe.kind:<8} median {probe.median:>9.4f}   "
            f"{probe.above_epsilon}/{probe.total} above epsilon {probe.epsilon:.2f}   "
            f"({want})"
        )
    lines.append(f"      {year.diagnosis}")
    return lines
