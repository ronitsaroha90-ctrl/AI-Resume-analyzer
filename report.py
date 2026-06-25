"""Console report generation for the AI Resume Analyzer."""

from __future__ import annotations

from pathlib import Path

from scorer import AnalysisReport
from utils import format_percentage


class ReportGenerator:
    """Display resume analysis results in a neat command-line report."""

    def __init__(self, use_symbols: bool = True, width: int = 70) -> None:
        self.use_symbols = use_symbols
        self.width = width

    def print_report(self, report: AnalysisReport) -> None:
        """Print the complete analysis report."""

        print(self.create_report_text(report))

    def create_report_text(self, report: AnalysisReport) -> str:
        """Build the report as a formatted string."""

        lines: list[str] = []
        lines.append(self._title("AI Resume Analyzer"))
        lines.append("")
        lines.append(f"Resume Match Score : {format_percentage(report.score)}")
        lines.append(f"Skill Match        : {format_percentage(report.skill_percentage)}")
        lines.append(f"Match Level        : {report.score_label()}")
        lines.append("")
        lines.extend(self._score_breakdown(report))
        lines.append("")
        lines.extend(self._skill_section("Matched Skills", report.match_result.matched_skills, True))
        lines.append("")
        lines.extend(self._skill_section("Missing Skills", report.match_result.missing_skills, False))
        lines.append("")
        lines.extend(self._bullet_section("Strengths", report.strengths))
        lines.append("")
        lines.extend(self._bullet_section("Weaknesses", report.weaknesses))
        lines.append("")
        lines.extend(self._bullet_section("Suggestions", report.suggestions))
        lines.append("")
        lines.append("Recommendation:")
        lines.append(report.recommendation)
        return "\n".join(lines)

    def save_report(self, report: AnalysisReport, output_path: str | Path) -> Path:
        """Save the report to a text file."""

        path = Path(output_path)
        path.write_text(self.create_report_text(report), encoding="utf-8")
        return path

    def _title(self, text: str) -> str:
        label = f" {text} "
        side = max((self.width - len(label)) // 2, 0)
        return "=" * side + label + "=" * side

    def _score_breakdown(self, report: AnalysisReport) -> list[str]:
        lines = ["Score Breakdown:"]
        for label, value in report.breakdown.as_rows():
            lines.append(f"  {label:<18} {value}")
        return lines

    def _skill_section(self, title: str, skills: list[str], positive: bool) -> list[str]:
        lines = [f"{title}:"]
        if not skills:
            lines.append("  None detected")
            return lines
        symbol = self._symbol(positive)
        for skill in skills:
            lines.append(f"  {symbol} {skill}")
        return lines

    def _bullet_section(self, title: str, items: list[str]) -> list[str]:
        lines = [f"{title}:"]
        if not items:
            lines.append("  None")
            return lines
        bullet = "*" if not self.use_symbols else "•"
        for item in items:
            lines.append(f"  {bullet} {item}")
        return lines

    def _symbol(self, positive: bool) -> str:
        if not self.use_symbols:
            return "+" if positive else "-"
        return "✔" if positive else "✖"


class PlainReportGenerator(ReportGenerator):
    """ASCII-only report generator for terminals without symbol support."""

    def __init__(self, width: int = 70) -> None:
        super().__init__(use_symbols=False, width=width)

    def _bullet_section(self, title: str, items: list[str]) -> list[str]:
        lines = [f"{title}:"]
        if not items:
            lines.append("  None")
            return lines
        for item in items:
            lines.append(f"  - {item}")
        return lines

