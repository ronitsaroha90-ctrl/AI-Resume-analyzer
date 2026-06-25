# AI Resume Analyzer

This is a beginner-friendly Python project for comparing a resume with a job
description using basic NLP ideas. It is designed for a second-year Computer
Science student to understand, run, and explain in internship interviews.

## What Each Module Does

`resume.py`
: Defines `TextDocument`, `Resume`, and `PlainTextResume`. It loads PDF/TXT
resumes, extracts text, detects common sections, and demonstrates inheritance
with method overriding.

`job_description.py`
: Stores a job description typed by the user or loaded from a TXT file. It
cleans the text and keeps useful metadata.

`text_processor.py`
: Handles text cleaning, tokenization, stop-word removal, n-grams, word counts,
and keyword-friendly preprocessing.

`skill_matcher.py`
: Extracts skills, calculates TF-IDF cosine similarity, finds matched and
missing skills, and measures keyword coverage.

`scorer.py`
: Combines similarity, skill match, keyword coverage, and resume structure into
a final score. It also creates rule-based AI suggestions.

`report.py`
: Prints and saves a clean command-line report.

`main.py`
: Runs the application and connects all classes together.

## OOP Concepts Demonstrated

- Classes and objects: each module contains focused classes.
- Encapsulation: internal text and metadata are managed through methods.
- Inheritance: `Resume` inherits from `TextDocument`.
- Polymorphism: `PlainTextResume` overrides `load_resume`.
- Composition: `ResumeAnalyzerApp` owns processor, scorer, and report objects.
- Exception handling: custom exceptions are raised and handled gracefully.

## How to Run

```bash
cd resume_analyzer
pip install -r requirements.txt
python main.py
```

Use `sample_resume.pdf` and `sample_jd.txt` for a quick demo.

## AI/NLP Techniques Used

- Text cleaning
- Tokenization
- Stop-word removal
- Keyword extraction
- TF-IDF vectorization
- Cosine similarity
- Rule-based improvement suggestions

