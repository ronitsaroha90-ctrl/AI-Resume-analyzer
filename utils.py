"""Utility helpers and shared constants for the AI Resume Analyzer.

This module keeps small reusable functions, validation helpers, and keyword
catalogs away from the core classes. That makes the project easier to explain
in interviews because every class can stay focused on one responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SUPPORTED_RESUME_EXTENSIONS = {".pdf", ".txt"}
DEFAULT_TOP_KEYWORDS = 35
MINIMUM_SHORT_RESUME_WORDS = 250
GOOD_MATCH_THRESHOLD = 75.0
AVERAGE_MATCH_THRESHOLD = 55.0
LOW_SKILL_THRESHOLD = 50.0


TECHNICAL_SKILLS = set(
    """
    python|java|javascript|typescript|c|c++|c#|go|golang|rust|ruby|php|swift|kotlin|scala|r|matlab|sql|mysql
    |postgresql|sqlite|mongodb|redis|firebase|html|css|sass|bootstrap|tailwind|react|angular|vue|node|node.js
    |express|django|flask|fastapi|spring|spring boot|rest|rest api|rest apis|graphql|api|apis|git|github
    |gitlab|bitbucket|linux|unix|bash|shell|powershell|docker|kubernetes|aws|azure|gcp|cloud|terraform|jenkins
    |ci|cd|ci/cd|devops|oop|object oriented|data structures|algorithms|dsa|machine learning|deep learning
    |nlp|natural language processing|computer vision|data science|pandas|numpy|scikit-learn|sklearn
    |matplotlib|seaborn|tensorflow|pytorch|statistics|probability|linear algebra|excel|tableau|power bi|spark
    |hadoop|airflow|etl|web scraping|beautifulsoup|selenium|playwright|testing|unit testing|pytest|unittest|junit
    |debugging|agile|scrum|jira|figma|ui|ux|android|ios|mobile|react native|flutter|microservices|system design
    |distributed systems|networking|operating systems|security|cybersecurity|encryption|authentication
    |authorization|oauth|jwt|json|xml|yaml|csv|regex|command line|cli|streamlit|tkinter|pyspark|nosql
    |relational database|database design|normalization|query optimization|data visualization
    |feature engineering|classification|regression|clustering|recommendation systems|time series|big data
    |software engineering|clean code|design patterns|mvc|frontend|backend|full stack|serverless|lambda|s3
    |ec2|rds|cloudwatch|postgres|mssql|oracle|snowflake|dbt|redshift|databricks|data mining|prompt engineering
    |generative ai|llm|chatbot|automation|scripting|performance tuning|responsive design|accessibility
    |web development
    """.replace('"', "")
    .replace("\n", "")
    .split("|")
)


EDUCATION_KEYWORDS = set(
    "education|degree|bachelor|bachelors|master|masters|university|college|school|cgpa|gpa|coursework|"
    "computer science|engineering".split("|")
)


PROJECT_KEYWORDS = set(
    "project projects built developed implemented designed created deployed application system tool website platform".split()
)


ACTION_VERBS = set(
    "achieved analyzed automated built collaborated created debugged deployed designed developed enhanced "
    "implemented improved increased launched led optimized reduced refactored researched tested".split()
)


SECTION_ALIASES = {
    "summary": {"summary", "profile", "objective", "about"},
    "education": {"education", "academic background", "academics"},
    "skills": {"skills", "technical skills", "technologies", "tools"},
    "projects": {"projects", "academic projects", "personal projects"},
    "experience": {"experience", "work experience", "internship", "internships"},
    "achievements": {"achievements", "awards", "certifications", "certificates"},
}


@dataclass
class SectionStatus:
    """Represents whether a common resume section appears to be present."""

    name: str
    found: bool
    evidence: list[str] = field(default_factory=list)

    def as_sentence(self) -> str:
        if self.found:
            return f"{self.name.title()} section found"
        return f"{self.name.title()} section not detected"


@dataclass
class KeywordMatch:
    """Stores one keyword and whether it appeared in the resume."""

    keyword: str
    present: bool
    weight: float = 0.0


class ResumeAnalyzerError(Exception):
    """Base exception for this project."""


class FileLoadError(ResumeAnalyzerError):
    """Raised when a resume or job description cannot be loaded."""


class UnsupportedFileTypeError(ResumeAnalyzerError):
    """Raised when the user gives a file type the project does not support."""


class EmptyTextError(ResumeAnalyzerError):
    """Raised when a file loads successfully but contains no useful text."""


def normalize_skill(skill: str) -> str:
    """Return a display-friendly skill name with common spellings preserved."""

    replacements = {
        "aws": "AWS",
        "gcp": "GCP",
        "api": "API",
        "apis": "APIs",
        "rest": "REST",
        "rest api": "REST API",
        "rest apis": "REST APIs",
        "sql": "SQL",
        "mysql": "MySQL",
        "postgresql": "PostgreSQL",
        "sqlite": "SQLite",
        "mongodb": "MongoDB",
        "html": "HTML",
        "css": "CSS",
        "oop": "OOP",
        "nlp": "NLP",
        "ci/cd": "CI/CD",
        "ui": "UI",
        "ux": "UX",
        "json": "JSON",
        "xml": "XML",
        "yaml": "YAML",
        "csv": "CSV",
        "llm": "LLM",
        "jwt": "JWT",
        "oauth": "OAuth",
    }
    cleaned = skill.strip().lower()
    if cleaned in replacements:
        return replacements[cleaned]
    return " ".join(part.capitalize() for part in cleaned.split())


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    """Remove duplicates while keeping the original order."""

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result


def format_percentage(value: float) -> str:
    """Format a numeric score as a whole-number percentage."""

    return f"{round(value)}%"


def safe_ratio(part: int | float, whole: int | float) -> float:
    """Return a percentage ratio without raising ZeroDivisionError."""

    if whole == 0:
        return 0.0
    return float(part) / float(whole) * 100.0


def read_text_file(path: Path) -> str:
    """Read a text file using a few common encodings."""

    encodings = ("utf-8", "utf-8-sig", "latin-1")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise FileLoadError(f"Could not read {path} as a text file") from last_error


def ensure_file_exists(path: Path) -> None:
    """Validate that a path exists and points to a file."""

    if not path.exists():
        raise FileLoadError(f"File not found: {path}")
    if not path.is_file():
        raise FileLoadError(f"Expected a file, but got: {path}")


def validate_resume_extension(path: Path) -> None:
    """Validate resume file extension."""

    extension = path.suffix.lower()
    if extension not in SUPPORTED_RESUME_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_RESUME_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported resume file type '{extension}'. Use {supported}."
        )


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Keep a score inside a fixed range."""

    return max(low, min(high, value))


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """Return a simple pluralized phrase."""

    if count == 1:
        return f"{count} {singular}"
    return f"{count} {plural or singular + 's'}"


def get_sample_resume_text() -> str:
    """Return text used to create the sample resume PDF."""

    return """
    AARAV SHARMA
    Computer Science Student
    Email: aarav.sharma@example.com | GitHub: github.com/aaravsharma

    SUMMARY
    Second-year Computer Science student interested in backend development,
    data analysis, and practical machine learning projects.

    EDUCATION
    B.Tech in Computer Science, City Engineering College
    CGPA: 8.5/10
    Relevant Coursework: Data Structures, OOP, Database Systems, Statistics

    TECHNICAL SKILLS
    Python, SQL, Git, GitHub, OOP, Data Structures, Algorithms, Flask,
    Pandas, NumPy, Matplotlib, HTML, CSS, JavaScript, Linux

    PROJECTS
    AI Resume Analyzer
    Built a Python application using TF-IDF and cosine similarity to compare
    resumes with job descriptions and generate improvement suggestions.

    Student Result Dashboard
    Developed a Streamlit dashboard to visualize student marks, attendance,
    and subject-wise performance using Pandas and Matplotlib.

    EXPERIENCE
    Coding Club Volunteer
    Helped beginners debug Python programs and explained OOP basics.

    ACHIEVEMENTS
    Solved 150+ programming problems on coding platforms.
    """.strip()


def get_sample_job_description() -> str:
    """Return the bundled sample internship job description."""

    return """
    Software Engineering Intern

    We are looking for a motivated Computer Science student with strong Python
    programming skills and understanding of object-oriented programming.
    The intern will work on backend APIs, SQL queries, Git workflows, and
    data-driven features.

    Required Skills:
    Python, SQL, Git, OOP, REST APIs, Flask or Django, Data Structures,
    Algorithms, unit testing, basic cloud knowledge, Docker basics.

    Responsibilities:
    Build clean backend modules, write readable code, test features, document
    project decisions, and collaborate with senior engineers.
    """.strip()
