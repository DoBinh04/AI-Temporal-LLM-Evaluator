"""Rendering. Every number a model author reads comes through here."""

from __future__ import annotations

from .benchmark import BenchmarkReport
from .benchmark.artifact import FAIL, OK, SKIP, WARN, ArtifactReport
from .benchmark.consistency import ConsistencyReport
from .benchmark.quality import QualityReport
from .benchmark.similarity import SimilarityReport
from .corpus import CalibrationReport

RULE = "-" * 74
_MARK = {OK: "ok  ", WARN: "warn", FAIL: "FAIL", SKIP: "n/a "}


def _section(title: str) -> list[str]:
    return ["", title, RULE]


# ─── artefact ────────────────────────────────────────────────────────────


def format_artifact(report: ArtifactReport) -> list[str]:
    lines = []
    for item in report.items:
        lines.append(f"  [{_MARK[item.level]}]  {item.name:<32} {item.detail}")
        if item.hint:
            lines.append(f"            -> {item.hint}")
    return lines


# ─── consistency ─────────────────────────────────────────────────────────


def format_consistency(report: ConsistencyReport) -> list[str]:
    lines = []
    for year in report.years:
        assessment = year.assessment
        verdict = "PASS" if year.passed else "FAIL"
        lines.append(f"  {year.year}  [{verdict}]  score {year.score:+.4f}")
        if assessment is not None:
            for probe, want in ((assessment.known, "must recognise"),
                                (assessment.unknown, "must not recognise")):
                lines.append(
                    f"      {probe.kind:<8} median {probe.median:>9.4f}   "
                    f"{probe.above_epsilon}/{probe.total} above epsilon {probe.epsilon:.2f}   "
                    f"({want})"
                )
        lines.append(f"      {year.diagnosis}")

    for error in report.errors:
        lines.append(f"  [FAIL]  {error}")

    lines.append("")
    lines.append(
        f"  leak score      {report.leak_score:>9.4f}   "
        f"(threshold {report.min_eval_score:.4f}) — "
        f"{'clears it' if report.clears_threshold else 'below the bar'}"
    )
    lines.append(f"  normalised       {report.normalized:>8.3f}   0 = threshold, 1 = best")

    if report.saturated:
        lines.append(
            "      this half is maxed out — further consistency work earns nothing, "
            "quality is the only lever left"
        )
    if not report.covers_all_years:
        lines.append(
            f"      ! only {report.scored_years} of {report.total_years} years scored. "
            "Every missing year counts worst-possible,"
        )
        lines.append("        so a full run will score lower than this.")
    return lines


# ─── similarity ──────────────────────────────────────────────────────────


def format_similarity(report: SimilarityReport) -> list[str]:
    if report.error:
        return [f"  [FAIL]  could not read the model — {report.error}"]
    if not report.comparisons:
        return ["  nothing to compare against"]

    lines = []
    for comparison in report.comparisons:
        if not comparison.comparable:
            lines.append(f"  [n/a ]  {comparison.reference:<30} no matching weight matrices")
            continue
        mark = "TOO CLOSE" if comparison.too_close else "distinct "
        lines.append(
            f"  [{mark}]  {comparison.reference:<30} spectral distance "
            f"{comparison.distance:.6f}  (limit {comparison.threshold})"
        )
    if not report.ok:
        lines.append(
            "      a model this close to a published one reads as a copy — "
            "train from your own initialisation"
        )
    return lines


def format_pairwise(report) -> list[str]:
    lines = []
    for error in report.errors:
        lines.append(f"  [FAIL]  {error}")
    if not report.pairs:
        lines.append("  fewer than two readable models — nothing to compare")
        return lines

    for left, right, distance in report.pairs:
        if distance is None:
            lines.append(f"  [n/a ]  {left}  vs  {right}   no matching weight matrices")
            continue
        lines.append(f"  {left}  vs  {right}   spectral distance {distance:.6f}")

    if report.duplicates:
        lines.append("")
        for duplicate in report.duplicates:
            lines.append(f"  [COPY]  {duplicate} reads as a copy of an earlier-listed model")
    else:
        lines.append("")
        lines.append("  all models are spectrally distinct")
    return lines


# ─── quality ─────────────────────────────────────────────────────────────


def format_quality(report: QualityReport) -> list[str]:
    if not report.measured:
        return [f"  not measured — {report.skipped}"]

    lines = [
        f"  win rate         {report.win_rate:>8.3f}   "
        f"over {report.prompt_count} prompts against {len(report.opponents)} reference(s)"
    ]
    if report.ambiguous:
        lines.append(
            "      ! 0 against a single reference cannot be told from drawing every "
            "duel — a draw scores"
        )
        lines.append("        for neither side. Use two or more references.")
    return lines


# ─── the whole thing ─────────────────────────────────────────────────────


def format_benchmark(report: BenchmarkReport) -> str:
    lines = [f"=== Benchmark: {len(report.manifest)} model(s) ===", "", "Artefact", RULE]
    lines += format_artifact(report.artifact)

    if report.consistency:
        lines += _section("Consistency  (stage 1, weight %.2g)" % report.leak_weight)
        lines += format_consistency(report.consistency)

    if report.similarity:
        lines += _section("Similarity")
        lines += format_similarity(report.similarity)

    if report.quality:
        lines += _section("Quality  (stage 2, weight %.2g)" % report.quality_weight)
        lines += format_quality(report.quality)

    lines += ["", RULE]
    final = report.final_score
    if final is None:
        lines.append("Final score not available — both stages must be measured.")
        if report.consistency:
            lines.append(
                f"Consistency alone would contribute "
                f"{report.leak_weight * report.consistency.normalized:.4f}."
            )
    elif not report.qualified:
        lines.append(
            f"Final score  {final:.4f}"
            "   — stage 1 not cleared, so stage 2 never runs and the score is 0"
        )
    else:
        lines.append(
            f"Final score  {final:.4f}"
            f"   = {report.leak_weight:g} x {report.consistency.normalized:.3f}"
            f" + {report.quality_weight:g} x {report.quality.win_rate:.3f}"
        )
    if not report.ok:
        lines.append("NOT ACCEPTED — resolve the FAIL items above.")
    return "\n".join(lines)


# ─── corpus calibration ──────────────────────────────────────────────────


def format_calibration(report: CalibrationReport) -> str:
    lines = ["=== Corpus calibration ===", "", RULE,
             f"  {'year':<6} {'epsilon':>10} {'known':>8} {'unknown':>9} {'threshold':>10}  verdict"]
    for year in report.years:
        verdict = "separates" if year.separates else "DOES NOT SEPARATE"
        lines.append(
            f"  {year.year:<6} {year.epsilon:>10.4f} {year.known_rate:>7.1%} "
            f"{year.unknown_rate:>8.1%} {year.threshold:>9.1%}  {verdict}"
        )
    lines.append(RULE)

    tightest = report.tightest
    if not report.ok:
        lines.append(
            "This corpus cannot tell a clean model from a leaking one. The reference "
            "model must"
        )
        lines.append(
            "recognise its own era and not the future; if it does neither, the probes "
            "are too hard."
        )
    elif tightest and tightest.margin < 0.05:
        lines.append(
            f"Usable, but {tightest.year} sits {tightest.margin:.1%} from the threshold. "
            "A single probe"
        )
        lines.append(
            "crossing epsilon would flip that year — add probes or widen the threshold."
        )
    else:
        lines.append(
            f"Calibrated. Tightest margin {tightest.margin:.1%} "
            f"(year {tightest.year}) — comfortably clear of the threshold."
        )
    return "\n".join(lines)
