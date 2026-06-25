"""Scoring and rule-based AI suggestions."""

from __future__ import annotations

from dataclasses import dataclass, field

from resume import Resume
from skill_matcher import MatchResult, SkillMatcher
from utils import (
    AVERAGE_MATCH_THRESHOLD,
    GOOD_MATCH_THRESHOLD,
    LOW_SKILL_THRESHOLD,
    MINIMUM_SHORT_RESUME_WORDS,
    clamp,
    format_percentage,
    pluralize,
)


@dataclass
class ScoreBreakdown:
    """Explains how the final score was calculated."""

    similarity_component: float
    skill_component: float
    keyword_component: float
    structure_component: float
    final_score: float

    def as_rows(self) -> list[tuple[str, str]]:
        return [
            ("Similarity", format_percentage(self.similarity_component)),
            ("Skill Match", format_percentage(self.skill_component)),
            ("Keyword Coverage", format_percentage(self.keyword_component)),
            ("Resume Structure", format_percentage(self.structure_component)),
            ("Final Score", format_percentage(self.final_score)),
        ]


@dataclass
class AnalysisReport:
    """Full report produced by ResumeScorer."""

    score: float
    skill_percentage: float
    recommendation: str
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    match_result: MatchResult
    breakdown: ScoreBreakdown

    def score_label(self) -> str:
        if self.score >= GOOD_MATCH_THRESHOLD:
            return "Strong Match"
        if self.score >= AVERAGE_MATCH_THRESHOLD:
            return "Moderate Match"
        return "Needs Improvement"


class ResumeScorer:
    """Generate match score, strengths, weaknesses, and suggestions."""

    def __init__(self, skill_matcher: SkillMatcher | None = None) -> None:
        self.skill_matcher = skill_matcher or SkillMatcher()

    def analyze(
        self,
        resume: Resume,
        job_description_text: str,
    ) -> AnalysisReport:
        """Run matching and scoring for one resume and one job description."""

        match_result = self.skill_matcher.compare(resume.get_text(), job_description_text)
        breakdown = self.calculate_score(resume, match_result)
        strengths = self.identify_strengths(resume, match_result)
        weaknesses = self.identify_weaknesses(resume, match_result)
        suggestions = self.generate_feedback(resume, match_result, breakdown)
        recommendation = self.generate_recommendation(breakdown.final_score)
        return AnalysisReport(
            score=breakdown.final_score,
            skill_percentage=match_result.skill_percentage,
            recommendation=recommendation,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            match_result=match_result,
            breakdown=breakdown,
        )

    def calculate_score(
        self,
        resume: Resume,
        match_result: MatchResult,
    ) -> ScoreBreakdown:
        """Calculate weighted resume match score."""

        similarity_component = match_result.similarity_score
        skill_component = match_result.skill_percentage
        keyword_component = self.skill_matcher.keyword_coverage(match_result.keyword_matches)
        structure_component = self._structure_score(resume)
        final_score = (
            similarity_component * 0.35
            + skill_component * 0.35
            + keyword_component * 0.15
            + structure_component * 0.15
        )
        return ScoreBreakdown(
            similarity_component=clamp(similarity_component),
            skill_component=clamp(skill_component),
            keyword_component=clamp(keyword_component),
            structure_component=clamp(structure_component),
            final_score=round(clamp(final_score), 2),
        )

    def generate_feedback(
        self,
        resume: Resume,
        match_result: MatchResult,
        breakdown: ScoreBreakdown,
    ) -> list[str]:
        """Generate rule-based AI suggestions from the analysis."""

        suggestions: list[str] = []
        suggestions.extend(self._skill_suggestions(match_result))
        suggestions.extend(self._structure_suggestions(resume))
        suggestions.extend(self._content_suggestions(resume, match_result, breakdown))
        suggestions.extend(self._keyword_suggestions(match_result))
        return self._dedupe_with_limit(suggestions, limit=10)

    def identify_strengths(
        self,
        resume: Resume,
        match_result: MatchResult,
    ) -> list[str]:
        """Identify positive resume signals."""

        strengths: list[str] = []
        if match_result.skill_percentage >= 75:
            strengths.append("Good alignment with required technical skills")
        elif match_result.skill_percentage >= 50:
            strengths.append("Some important required skills are already present")
        if match_result.similarity_score >= 65:
            strengths.append("Resume language is relevant to the job description")
        if match_result.matched_skills:
            count_text = pluralize(len(match_result.matched_skills), "matched skill")
            strengths.append(f"Includes {count_text} from the job description")
        strengths.extend(resume.get_strength_signals())
        if not strengths:
            strengths.append("Resume has a starting foundation that can be improved")
        return self._dedupe_with_limit(strengths, limit=8)

    def identify_weaknesses(
        self,
        resume: Resume,
        match_result: MatchResult,
    ) -> list[str]:
        """Identify areas that are lowering the score."""

        weaknesses: list[str] = []
        if match_result.skill_percentage < LOW_SKILL_THRESHOLD:
            weaknesses.append("Low overlap with required job skills")
        if match_result.similarity_score < 40:
            weaknesses.append("Resume wording does not strongly match the role")
        if match_result.missing_skills:
            top_missing = ", ".join(match_result.missing_skills[:4])
            weaknesses.append(f"Missing important skills: {top_missing}")
        weaknesses.extend(resume.get_weakness_signals())
        return self._dedupe_with_limit(weaknesses, limit=8)

    def generate_recommendation(self, score: float) -> str:
        """Return final recommendation based on the match score."""

        if score >= 85:
            return "Excellent fit for this role. Apply with small polishing changes."
        if score >= GOOD_MATCH_THRESHOLD:
            return "Good fit for this role with minor improvements."
        if score >= AVERAGE_MATCH_THRESHOLD:
            return "Possible fit, but the resume should be tailored before applying."
        if score >= 40:
            return "Needs improvement before applying to this specific role."
        return "Low match. Build more relevant skills or choose a closer role."

    def _skill_suggestions(self, match_result: MatchResult) -> list[str]:
        suggestions: list[str] = []
        if match_result.skill_percentage < LOW_SKILL_THRESHOLD:
            suggestions.append("Add missing technical skills that you genuinely know.")
        if match_result.missing_skills:
            first_missing = match_result.missing_skills[:5]
            suggestions.append(
                "Prioritize learning or showcasing: " + ", ".join(first_missing) + "."
            )
        if len(match_result.matched_skills) < 4:
            suggestions.append("Add a dedicated skills section with role-specific tools.")
        return suggestions

    def _structure_suggestions(self, resume: Resume) -> list[str]:
        suggestions: list[str] = []
        if not resume.has_section("education"):
            suggestions.append("Add an Education section with degree, college, CGPA, and coursework.")
        if not resume.has_section("projects"):
            suggestions.append("Add 2-3 projects that clearly connect to the target role.")
        if not resume.has_section("skills"):
            suggestions.append("Add a Technical Skills section grouped by language, tools, and frameworks.")
        if not resume.has_section("experience"):
            suggestions.append("Include internships, volunteering, club work, or open-source contributions.")
        return suggestions

    def _content_suggestions(
        self,
        resume: Resume,
        match_result: MatchResult,
        breakdown: ScoreBreakdown,
    ) -> list[str]:
        suggestions: list[str] = []
        stats = resume.get_stats()
        if stats.word_count < MINIMUM_SHORT_RESUME_WORDS:
            suggestions.append("Expand project descriptions with problem, approach, tech stack, and result.")
        if not resume.has_quantified_achievements():
            suggestions.append("Add measurable outcomes such as accuracy, users, speed, marks, or percentages.")
        if not resume.uses_action_verbs():
            suggestions.append("Start bullets with action verbs like built, optimized, automated, or tested.")
        if breakdown.similarity_component < 45:
            suggestions.append("Rewrite project bullets using important phrases from the job description.")
        if match_result.resume_only_skills and match_result.missing_skills:
            suggestions.append("Keep extra skills, but move the most job-relevant skills higher.")
        return suggestions

    def _keyword_suggestions(self, match_result: MatchResult) -> list[str]:
        missing_keywords = [match.keyword for match in match_result.keyword_matches if not match.present]
        suggestions: list[str] = []
        if len(missing_keywords) >= 6:
            suggestions.append(
                "Naturally include more JD keywords such as "
                + ", ".join(missing_keywords[:5])
                + "."
            )
        return suggestions

    def _structure_score(self, resume: Resume) -> float:
        important_sections = ("education", "skills", "projects", "experience")
        found_count = sum(1 for section in important_sections if resume.has_section(section))
        score = found_count / len(important_sections) * 70.0
        if resume.has_quantified_achievements():
            score += 15.0
        if resume.uses_action_verbs():
            score += 15.0
        return clamp(score)

    def _dedupe_with_limit(self, values: list[str], limit: int) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = value.lower()
            if key not in seen:
                seen.add(key)
                result.append(value)
            if len(result) >= limit:
                break
        return result

