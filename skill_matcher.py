"""Skill extraction and similarity comparison logic."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_processor import TextProcessor
from utils import (
    DEFAULT_TOP_KEYWORDS,
    KeywordMatch,
    TECHNICAL_SKILLS,
    normalize_skill,
    safe_ratio,
    unique_preserve_order,
)


@dataclass
class MatchResult:
    """Container for all matching outputs."""

    similarity_score: float
    skill_percentage: float
    matched_skills: list[str]
    missing_skills: list[str]
    resume_only_skills: list[str]
    jd_keywords: list[str]
    resume_keywords: list[str]
    keyword_matches: list[KeywordMatch] = field(default_factory=list)

    def matched_count(self) -> int:
        return len(self.matched_skills)

    def missing_count(self) -> int:
        return len(self.missing_skills)

    def total_required_skills(self) -> int:
        return len(self.matched_skills) + len(self.missing_skills)


class SkillMatcher:
    """Extract skills and compare resume text with job description text."""

    def __init__(
        self,
        text_processor: TextProcessor | None = None,
        skill_catalog: set[str] | None = None,
    ) -> None:
        self.text_processor = text_processor or TextProcessor()
        self.skill_catalog = skill_catalog or set(TECHNICAL_SKILLS)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 3),
            max_features=600,
        )
        self.last_result: MatchResult | None = None

    def compare(self, resume_text: str, job_description_text: str) -> MatchResult:
        """Run the complete matching pipeline."""

        similarity_score = self.calculate_similarity(resume_text, job_description_text)
        resume_skills = self.extract_skills(resume_text)
        jd_skills = self.extract_skills(job_description_text)
        matched = self.match_skills(resume_skills, jd_skills)
        missing = self.missing_skills(resume_skills, jd_skills)
        resume_only = self.resume_only_skills(resume_skills, jd_skills)
        jd_keywords = self.extract_keywords(job_description_text, DEFAULT_TOP_KEYWORDS)
        resume_keywords = self.extract_keywords(resume_text, DEFAULT_TOP_KEYWORDS)
        keyword_matches = self.build_keyword_matches(resume_text, jd_keywords)
        skill_percentage = safe_ratio(len(matched), len(jd_skills))
        result = MatchResult(
            similarity_score=similarity_score,
            skill_percentage=skill_percentage,
            matched_skills=matched,
            missing_skills=missing,
            resume_only_skills=resume_only,
            jd_keywords=jd_keywords,
            resume_keywords=resume_keywords,
            keyword_matches=keyword_matches,
        )
        self.last_result = result
        return result

    def calculate_similarity(self, resume_text: str, job_description_text: str) -> float:
        """Calculate TF-IDF cosine similarity between resume and JD."""

        documents = [
            self.text_processor.clean_text(resume_text),
            self.text_processor.clean_text(job_description_text),
        ]
        if not documents[0] or not documents[1]:
            return 0.0
        try:
            matrix = self.vectorizer.fit_transform(documents)
        except ValueError:
            return 0.0
        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return round(float(score) * 100.0, 2)

    def extract_skills(self, text: str) -> list[str]:
        """Extract known skills from text using a lightweight catalog."""

        cleaned = f" {self.text_processor.clean_text(text)} "
        found: list[str] = []
        for skill in sorted(self.skill_catalog, key=len, reverse=True):
            normalized = self.text_processor.clean_text(skill)
            if not normalized:
                continue
            if self._phrase_exists(cleaned, normalized):
                found.append(normalize_skill(skill))
        return unique_preserve_order(found)

    def match_skills(
        self,
        resume_skills: list[str],
        job_description_skills: list[str],
    ) -> list[str]:
        """Return required skills found in the resume."""

        resume_lookup = {skill.lower(): skill for skill in resume_skills}
        matched: list[str] = []
        for skill in job_description_skills:
            if skill.lower() in resume_lookup:
                matched.append(skill)
        return unique_preserve_order(matched)

    def missing_skills(
        self,
        resume_skills: list[str],
        job_description_skills: list[str],
    ) -> list[str]:
        """Return required skills that are absent from the resume."""

        resume_lookup = {skill.lower() for skill in resume_skills}
        missing = [skill for skill in job_description_skills if skill.lower() not in resume_lookup]
        return unique_preserve_order(missing)

    def resume_only_skills(
        self,
        resume_skills: list[str],
        job_description_skills: list[str],
    ) -> list[str]:
        """Return skills that appear in the resume but not in the JD."""

        jd_lookup = {skill.lower() for skill in job_description_skills}
        extra = [skill for skill in resume_skills if skill.lower() not in jd_lookup]
        return unique_preserve_order(extra)

    def extract_keywords(self, text: str, top_n: int = 20) -> list[str]:
        """Extract important terms using TF-IDF on sentence chunks."""

        sentences = self.text_processor.tokenize_sentences(text)
        if not sentences:
            return []
        cleaned_sentences = [self.text_processor.clean_text(sentence) for sentence in sentences]
        cleaned_sentences = [sentence for sentence in cleaned_sentences if sentence]
        if not cleaned_sentences:
            return []
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=120,
            )
            matrix = vectorizer.fit_transform(cleaned_sentences)
        except ValueError:
            return self._frequency_keywords(text, top_n)
        scores = matrix.sum(axis=0).A1
        features = vectorizer.get_feature_names_out()
        weighted = sorted(zip(features, scores), key=lambda item: item[1], reverse=True)
        keywords = [feature for feature, _ in weighted[:top_n]]
        return [normalize_skill(keyword) for keyword in unique_preserve_order(keywords)]

    def build_keyword_matches(
        self,
        resume_text: str,
        jd_keywords: list[str],
    ) -> list[KeywordMatch]:
        """Build keyword match records for report details."""

        cleaned_resume = f" {self.text_processor.clean_text(resume_text)} "
        matches: list[KeywordMatch] = []
        total = max(len(jd_keywords), 1)
        for index, keyword in enumerate(jd_keywords):
            normalized = self.text_processor.clean_text(keyword)
            present = self._phrase_exists(cleaned_resume, normalized)
            weight = round((total - index) / total, 3)
            matches.append(KeywordMatch(keyword=keyword, present=present, weight=weight))
        return matches

    def keyword_coverage(self, keyword_matches: list[KeywordMatch]) -> float:
        """Return weighted keyword coverage."""

        if not keyword_matches:
            return 0.0
        total_weight = sum(match.weight for match in keyword_matches)
        present_weight = sum(match.weight for match in keyword_matches if match.present)
        return safe_ratio(present_weight, total_weight)

    def _frequency_keywords(self, text: str, top_n: int) -> list[str]:
        counter = Counter(self.text_processor.tokenize(text))
        common = [word for word, _ in counter.most_common(top_n)]
        return [normalize_skill(word) for word in common]

    def _phrase_exists(self, cleaned_text_with_spaces: str, normalized_phrase: str) -> bool:
        if not normalized_phrase:
            return False
        return f" {normalized_phrase} " in cleaned_text_with_spaces
