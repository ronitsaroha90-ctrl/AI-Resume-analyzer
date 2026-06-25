"""Job description model and preprocessing behavior."""

from __future__ import annotations

from pathlib import Path

from text_processor import TextProcessor, TextStats
from utils import EmptyTextError, FileLoadError, read_text_file


class JobDescription:
    """Stores and cleans a job description entered by the user."""

    def __init__(
        self,
        text: str = "",
        source_path: str | Path | None = None,
        text_processor: TextProcessor | None = None,
    ) -> None:
        self.raw_text = text
        self.source_path = Path(source_path) if source_path else None
        self.text_processor = text_processor or TextProcessor()
        self.cleaned_text = ""
        self.tokens: list[str] = []
        self.metadata: dict[str, int | str | float] = {}

    def load_from_file(self, file_path: str | Path) -> None:
        """Load job description text from a TXT file."""

        path = Path(file_path)
        if not path.exists():
            raise FileLoadError(f"Job description file not found: {path}")
        if path.suffix.lower() != ".txt":
            raise FileLoadError("Job descriptions should be uploaded as .txt files")
        self.source_path = path
        self.raw_text = read_text_file(path)
        self.preprocess()

    def set_text(self, text: str) -> None:
        """Set job description text typed by the user."""

        self.raw_text = text
        self.source_path = None
        self.preprocess()

    def preprocess(self) -> str:
        """Clean and tokenize the job description."""

        if not self.raw_text or not self.raw_text.strip():
            raise EmptyTextError("The job description is empty")
        self.cleaned_text = self.text_processor.clean_text(self.raw_text)
        self.tokens = self.text_processor.tokenize(self.raw_text)
        self._update_metadata()
        return self.cleaned_text

    def get_text(self) -> str:
        """Return original job description text."""

        return self.raw_text

    def get_clean_text(self) -> str:
        """Return cleaned job description text."""

        if not self.cleaned_text:
            self.preprocess()
        return self.cleaned_text

    def get_tokens(self) -> list[str]:
        """Return processed job description tokens."""

        if not self.tokens:
            self.preprocess()
        return self.tokens

    def get_stats(self) -> TextStats:
        """Return text statistics for the job description."""

        return self.text_processor.get_stats(self.raw_text)

    def get_required_phrases(self) -> list[str]:
        """Extract lines that probably list requirements."""

        lines = self.text_processor.extract_lines(self.raw_text)
        selected: list[str] = []
        markers = (
            "required",
            "requirements",
            "skills",
            "qualification",
            "qualifications",
            "preferred",
            "responsibilities",
        )
        for line in lines:
            cleaned = self.text_processor.clean_text(line)
            if any(marker in cleaned for marker in markers):
                selected.append(line)
        return selected

    def _update_metadata(self) -> None:
        stats = self.get_stats()
        self.metadata = {
            "source": str(self.source_path) if self.source_path else "manual input",
            "word_count": stats.word_count,
            "sentence_count": stats.sentence_count,
            "required_line_count": len(self.get_required_phrases()),
        }
