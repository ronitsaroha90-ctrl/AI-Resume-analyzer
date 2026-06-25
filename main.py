"""Command-line entry point for the AI Resume Analyzer project."""

from __future__ import annotations

from pathlib import Path

from job_description import JobDescription
from report import PlainReportGenerator, ReportGenerator
from resume import PlainTextResume, Resume
from scorer import ResumeScorer
from text_processor import TextProcessor
from utils import (
    EmptyTextError,
    FileLoadError,
    ResumeAnalyzerError,
    UnsupportedFileTypeError,
    get_sample_job_description,
)


class ResumeAnalyzerApp:
    """Coordinates user input, analysis, and report display."""

    def __init__(self) -> None:
        self.text_processor = TextProcessor()
        self.scorer = ResumeScorer()
        self.report_generator = ReportGenerator()

    def run(self) -> None:
        """Start the interactive command-line program."""

        self._print_welcome()
        try:
            resume = self._load_resume_flow()
            job_description = self._load_job_description_flow()
            report = self.scorer.analyze(resume, job_description.get_text())
            self.report_generator.print_report(report)
            self._optional_save(report)
        except ResumeAnalyzerError as error:
            print(f"\nError: {error}")
        except KeyboardInterrupt:
            print("\nProgram stopped by user.")

    def analyze_files(
        self,
        resume_path: str | Path,
        job_description_path: str | Path,
        ascii_only: bool = False,
    ) -> str:
        """Analyze two files and return report text.

        This method is useful for demos, tests, and GitHub screenshots because it
        avoids interactive input.
        """

        resume = self._create_resume(resume_path)
        resume.load_resume()
        job_description = JobDescription(text_processor=self.text_processor)
        job_description.load_from_file(job_description_path)
        report = self.scorer.analyze(resume, job_description.get_text())
        generator = PlainReportGenerator() if ascii_only else self.report_generator
        return generator.create_report_text(report)

    def analyze_text(
        self,
        resume_path: str | Path,
        job_description_text: str,
        ascii_only: bool = False,
    ) -> str:
        """Analyze a resume file against typed job description text."""

        resume = self._create_resume(resume_path)
        resume.load_resume()
        job_description = JobDescription(
            text=job_description_text,
            text_processor=self.text_processor,
        )
        job_description.preprocess()
        report = self.scorer.analyze(resume, job_description.get_text())
        generator = PlainReportGenerator() if ascii_only else self.report_generator
        return generator.create_report_text(report)

    def _load_resume_flow(self) -> Resume:
        while True:
            path_text = input("\nEnter resume path (.pdf or .txt): ").strip()
            if not path_text:
                print("Please enter a resume file path.")
                continue
            try:
                resume = self._create_resume(path_text)
                resume.load_resume()
                print("Resume loaded successfully.")
                return resume
            except (FileLoadError, UnsupportedFileTypeError, EmptyTextError) as error:
                print(f"Could not load resume: {error}")

    def _load_job_description_flow(self) -> JobDescription:
        print("\nJob Description Options")
        print("1. Enter job description manually")
        print("2. Load job description from .txt file")
        print("3. Use bundled sample job description")
        while True:
            choice = input("Choose option (1/2/3): ").strip()
            if choice == "1":
                return self._manual_job_description()
            if choice == "2":
                return self._file_job_description()
            if choice == "3":
                jd = JobDescription(
                    text=get_sample_job_description(),
                    text_processor=self.text_processor,
                )
                jd.preprocess()
                return jd
            print("Please choose 1, 2, or 3.")

    def _manual_job_description(self) -> JobDescription:
        print("\nPaste the job description. Enter a blank line when finished:")
        lines: list[str] = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        jd = JobDescription(
            text="\n".join(lines),
            text_processor=self.text_processor,
        )
        jd.preprocess()
        return jd

    def _file_job_description(self) -> JobDescription:
        while True:
            path_text = input("Enter job description .txt path: ").strip()
            try:
                jd = JobDescription(text_processor=self.text_processor)
                jd.load_from_file(path_text)
                print("Job description loaded successfully.")
                return jd
            except (FileLoadError, EmptyTextError) as error:
                print(f"Could not load job description: {error}")

    def _optional_save(self, report) -> None:
        choice = input("\nSave report to a text file? (y/n): ").strip().lower()
        if choice != "y":
            return
        output = input("Enter output path [analysis_report.txt]: ").strip()
        output_path = Path(output or "analysis_report.txt")
        saved_path = self.report_generator.save_report(report, output_path)
        print(f"Report saved to {saved_path}")

    def _create_resume(self, resume_path: str | Path) -> Resume:
        path = Path(resume_path)
        if path.suffix.lower() == ".txt":
            return PlainTextResume(path, text_processor=self.text_processor)
        return Resume(path, text_processor=self.text_processor)

    def _print_welcome(self) -> None:
        print("=" * 43)
        print("           AI Resume Analyzer")
        print("=" * 43)
        print("Upload a resume, compare it with a job description,")
        print("and get a match score with improvement suggestions.")


def main() -> None:
    """Run the application."""

    app = ResumeAnalyzerApp()
    app.run()


if __name__ == "__main__":
    main()

