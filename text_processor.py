"""Text preprocessing tools used by resume and job description analysis."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
except Exception:  # pragma: no cover - fallback is tested through behavior
    stopwords = None
    PorterStemmer = None
    word_tokenize = None


FALLBACK_STOP_WORDS = set(
    """
    a an and are as at be by for from has have in is it its of on or that the this to was were will with we you
    your our their can using use used into about also such may than then there these those within without role
    candidate student intern internship responsibilities requirements required preferred skills experience work team teams
    """.split()
)


@dataclass
class TextStats:
    """Simple measurements for a block of text."""

    word_count: int
    unique_words: int
    sentence_count: int
    average_word_length: float
    lexical_diversity: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "word_count": self.word_count,
            "unique_words": self.unique_words,
            "sentence_count": self.sentence_count,
            "average_word_length": self.average_word_length,
            "lexical_diversity": self.lexical_diversity,
        }


class TextProcessor:
    """Clean, tokenize, normalize, and summarize resume-related text."""

    def __init__(self, use_stemming: bool = False) -> None:
        self.use_stemming = use_stemming
        self._stop_words = self._load_stop_words()
        self._stemmer = PorterStemmer() if PorterStemmer and use_stemming else None

    def clean_text(self, text: str) -> str:
        """Lowercase text and remove noisy characters while preserving skills."""

        if not text:
            return ""
        normalized = self.normalize_whitespace(text)
        normalized = self._protect_special_terms(normalized)
        normalized = normalized.lower()
        normalized = normalized.replace("/", " ")
        normalized = normalized.replace("-", " ")
        normalized = re.sub(r"[^a-z0-9+#.\s]", " ", normalized)
        normalized = self._restore_special_terms(normalized)
        return self.normalize_whitespace(normalized)

    def tokenize(self, text: str, remove_stop_words: bool = True) -> list[str]:
        """Split text into useful word tokens."""

        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        tokens = self._tokenize_words(cleaned)
        tokens = [token for token in tokens if self._is_meaningful_token(token)]
        if remove_stop_words:
            tokens = [token for token in tokens if token not in self._stop_words]
        if self.use_stemming and self._stemmer:
            tokens = [self._stemmer.stem(token) for token in tokens]
        return tokens

    def tokenize_sentences(self, text: str) -> list[str]:
        """Split text into sentence-like chunks without requiring NLTK data."""

        chunks = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        sentences = [self.normalize_whitespace(chunk) for chunk in chunks]
        return [sentence for sentence in sentences if sentence]

    def remove_stop_words(self, tokens: Iterable[str]) -> list[str]:
        """Remove common words that usually do not describe skills."""

        return [token for token in tokens if token.lower() not in self._stop_words]

    def word_frequencies(self, text: str, top_n: int = 20) -> list[tuple[str, int]]:
        """Return the most frequent meaningful words."""

        counter = Counter(self.tokenize(text))
        return counter.most_common(top_n)

    def ngrams(self, tokens: list[str], size: int) -> list[str]:
        """Create n-gram phrases from a token list."""

        if size <= 0 or len(tokens) < size:
            return []
        phrases: list[str] = []
        for index in range(len(tokens) - size + 1):
            phrases.append(" ".join(tokens[index : index + size]))
        return phrases

    def keyword_candidates(self, text: str) -> list[str]:
        """Return unigram, bigram, and trigram candidates for matching."""

        tokens = self.tokenize(text)
        candidates = list(tokens)
        candidates.extend(self.ngrams(tokens, 2))
        candidates.extend(self.ngrams(tokens, 3))
        return candidates

    def extract_lines(self, text: str) -> list[str]:
        """Return non-empty lines from raw text."""

        lines = [self.normalize_whitespace(line) for line in text.splitlines()]
        return [line for line in lines if line]

    def count_words(self, text: str) -> int:
        """Count meaningful words in text."""

        return len(self.tokenize(text, remove_stop_words=False))

    def count_numbers(self, text: str) -> int:
        """Count numeric values, useful for checking quantified achievements."""

        return len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))

    def contains_any(self, text: str, keywords: Iterable[str]) -> bool:
        """Check whether any keyword appears in cleaned text."""

        cleaned = f" {self.clean_text(text)} "
        for keyword in keywords:
            pattern = f" {self.clean_text(keyword)} "
            if pattern in cleaned:
                return True
        return False

    def find_present_keywords(self, text: str, keywords: Iterable[str]) -> list[str]:
        """Return keywords that appear in the text."""

        cleaned = f" {self.clean_text(text)} "
        found: list[str] = []
        for keyword in sorted(keywords, key=len, reverse=True):
            normalized = self.clean_text(keyword)
            if normalized and f" {normalized} " in cleaned:
                found.append(keyword)
        return found

    def get_stats(self, text: str) -> TextStats:
        """Compute simple text statistics."""

        all_tokens = self.tokenize(text, remove_stop_words=False)
        unique_words = len(set(all_tokens))
        word_count = len(all_tokens)
        sentence_count = len(self.tokenize_sentences(text))
        if word_count:
            average_word_length = sum(len(token) for token in all_tokens) / word_count
            lexical_diversity = unique_words / word_count
        else:
            average_word_length = 0.0
            lexical_diversity = 0.0
        return TextStats(
            word_count=word_count,
            unique_words=unique_words,
            sentence_count=sentence_count,
            average_word_length=average_word_length,
            lexical_diversity=lexical_diversity,
        )

    def normalize_whitespace(self, text: str) -> str:
        """Collapse repeated whitespace into single spaces."""

        return re.sub(r"\s+", " ", text).strip()

    def strip_punctuation(self, token: str) -> str:
        """Remove punctuation around a token while keeping C++ and C# readable."""

        protected = token.replace("c++", "cplusplus").replace("c#", "csharp")
        stripped = protected.strip(string.punctuation)
        return stripped.replace("cplusplus", "c++").replace("csharp", "c#")

    def _load_stop_words(self) -> set[str]:
        if stopwords is None:
            return set(FALLBACK_STOP_WORDS)
        try:
            return set(stopwords.words("english")) | set(FALLBACK_STOP_WORDS)
        except LookupError:
            return set(FALLBACK_STOP_WORDS)

    def _tokenize_words(self, cleaned: str) -> list[str]:
        if word_tokenize is not None:
            try:
                return [self.strip_punctuation(token) for token in word_tokenize(cleaned)]
            except LookupError:
                pass
        return [self.strip_punctuation(token) for token in re.findall(r"[a-z0-9+#.]+", cleaned)]

    def _is_meaningful_token(self, token: str) -> bool:
        if not token:
            return False
        if len(token) == 1 and token not in {"c", "r"}:
            return False
        if token.isdigit():
            return False
        return True

    def _protect_special_terms(self, text: str) -> str:
        protected = text
        protected = re.sub(r"\bC\+\+\b", "cplusplus", protected, flags=re.IGNORECASE)
        protected = re.sub(r"\bC#\b", "csharp", protected, flags=re.IGNORECASE)
        protected = re.sub(r"\bNode\.js\b", "nodejs", protected, flags=re.IGNORECASE)
        protected = re.sub(r"\bCI/CD\b", "cicd", protected, flags=re.IGNORECASE)
        return protected

    def _restore_special_terms(self, text: str) -> str:
        restored = text
        restored = restored.replace("cplusplus", "c++")
        restored = restored.replace("csharp", "c#")
        restored = restored.replace("nodejs", "node.js")
        restored = restored.replace("cicd", "ci/cd")
        return restored
