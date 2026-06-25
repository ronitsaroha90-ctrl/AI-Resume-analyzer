"""Resume loading and resume-specific analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from text_processor import TextProcessor, TextStats
from utils import (
    ACTION_VERBS,
    EmptyTextError,
    FileLoadError,
    SECTION_ALIASES,
    SectionStatus,
    ensure_file_exists,
    read_text_file,
    validate_resume_extension,
)


class TextDocument(ABC):
    """Abstract parent for objects that load and expose text."""

    def __init__(self, source: str | Path | None = None) -> None:
        self.source = Path(source) if source else None
        self._text = ""

    @abstractmethod
    def load(self) -> None:
        """Load text into the object."""

    def get_text(self) -> str:
        """Return the loaded text."""

        return self._text


class Resume(TextDocument):
    """Represents a resume uploaded as a PDF or TXT file."""

    def __init__(
        self,
        file_path: str | Path,
        text_processor: TextProcessor | None = None,
    ) -> None:
        super().__init__(file_path)
        self.text_processor = text_processor or TextProcessor()
        self.file_type = self.source.suffix.lower() if self.source else ""
        self.metadata: dict[str, str | int | float] = {}
        self.sections: dict[str, SectionStatus] = {}

    def load(self) -> None:
        """Load the resume text from the configured file path."""

        self.load_resume()

    def load_resume(self) -> None:
        """Read resume content and validate that text was extracted."""

        if self.source is None:
            raise FileLoadError("No resume path was provided")
        ensure_file_exists(self.source)
        validate_resume_extension(self.source)
        if self.source.suffix.lower() == ".pdf":
            self._text = self._load_pdf(self.source)
        else:
            self._text = read_text_file(self.source)
        self._text = self._text.strip()
        if not self._text:
            raise EmptyTextError("The resume file did not contain readable text")
        self._update_metadata()
        self.sections = self.detect_sections()

    def get_text(self) -> str:
        """Return resume text."""

        return super().get_text()

    def get_clean_text(self) -> str:
        """Return cleaned resume text for NLP operations."""

        return self.text_processor.clean_text(self._text)

    def get_tokens(self) -> list[str]:
        """Return processed resume tokens."""

        return self.text_processor.tokenize(self._text)

    def get_stats(self) -> TextStats:
        """Return word and sentence statistics for the resume."""

        return self.text_processor.get_stats(self._text)

    def detect_sections(self) -> dict[str, SectionStatus]:
        """Detect common resume sections using headings and keywords."""

        lines = self.text_processor.extract_lines(self._text)
        cleaned_lines = [self.text_processor.clean_text(line) for line in lines]
        statuses: dict[str, SectionStatus] = {}
        for section_name, aliases in SECTION_ALIASES.items():
            evidence = self._find_section_evidence(cleaned_lines, aliases)
            statuses[section_name] = SectionStatus(
                name=section_name,
                found=bool(evidence),
                evidence=evidence,
            )
        return statuses

    def has_section(self, section_name: str) -> bool:
        """Return whether a section was detected."""

        status = self.sections.get(section_name.lower())
        return bool(status and status.found)

    def get_strength_signals(self) -> list[str]:
        """Return positive resume signals found from rules."""

        signals: list[str] = []
        if self.has_section("education"):
            signals.append("Clear education section")
        if self.has_section("projects"):
            signals.append("Project experience is visible")
        if self.has_section("skills"):
            signals.append("Technical skills section is easy to find")
        if self.has_quantified_achievements():
            signals.append("Includes measurable achievements")
        if self.uses_action_verbs():
            signals.append("Uses strong action verbs")
        return signals

    def get_weakness_signals(self) -> list[str]:
        """Return resume weaknesses based on simple resume quality rules."""

        weaknesses: list[str] = []
        stats = self.get_stats()
        if stats.word_count < 250:
            weaknesses.append("Resume is short and may need more project detail")
        if not self.has_section("education"):
            weaknesses.append("Education section was not detected")
        if not self.has_section("projects"):
            weaknesses.append("Project section was not detected")
        if not self.has_section("skills"):
            weaknesses.append("Technical skills section was not detected")
        if not self.has_quantified_achievements():
            weaknesses.append("Few measurable results or numbers were detected")
        if not self.uses_action_verbs():
            weaknesses.append("Could use stronger action verbs")
        return weaknesses

    def has_quantified_achievements(self) -> bool:
        """Return True if the resume includes numbers or percentages."""

        return self.text_processor.count_numbers(self._text) >= 2

    def uses_action_verbs(self) -> bool:
        """Return True when common action verbs appear in the resume."""

        return self.text_processor.contains_any(self._text, ACTION_VERBS)

    def _load_pdf(self, path: Path) -> str:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            pages: list[str] = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n".join(pages)
        except ModuleNotFoundError as error:
            raise FileLoadError(
                "PDF support requires PyPDF2. Install dependencies with "
                "'pip install -r requirements.txt'."
            ) from error
        except Exception as error:
            raise FileLoadError(f"Could not read PDF resume: {path}") from error

    def _find_section_evidence(
        self,
        cleaned_lines: Iterable[str],
        aliases: set[str],
    ) -> list[str]:
        evidence: list[str] = []
        for line in cleaned_lines:
            for alias in aliases:
                cleaned_alias = self.text_processor.clean_text(alias)
                if line == cleaned_alias or line.startswith(f"{cleaned_alias} "):
                    evidence.append(alias)
        return evidence

    def _update_metadata(self) -> None:
        stats = self.get_stats()
        self.metadata = {
            "file_name": self.source.name if self.source else "",
            "file_type": self.file_type,
            "word_count": stats.word_count,
            "sentence_count": stats.sentence_count,
            "reading_time_minutes": round(stats.word_count / 200.0, 2) if stats.word_count else 0.0,
        }


class PlainTextResume(Resume):
    """A Resume subclass that only accepts TXT files.

    This class demonstrates inheritance and polymorphism by overriding the
    loading behavior from Resume.
    """

    def load_resume(self) -> None:
        if self.source is None:
            raise FileLoadError("No resume path was provided")
        ensure_file_exists(self.source)
        if self.source.suffix.lower() != ".txt":
            raise FileLoadError("PlainTextResume accepts only .txt files")
        self._text = read_text_file(self.source).strip()
        if not self._text:
            raise EmptyTextError("The text resume is empty")
        self._update_metadata()
        self.sections = self.detect_sections()
