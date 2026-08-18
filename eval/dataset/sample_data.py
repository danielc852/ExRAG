"""Rules for the small development and smoke-test evaluation dataset."""

from __future__ import annotations

from data import BenchmarkQuestion
from data.artifacts import StageManifest


def get_sample_questions(
    questions: list[BenchmarkQuestion], source: StageManifest
) -> list[BenchmarkQuestion]:
    limit = source.metadata.get("sample_question_limit")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError(
            "Sample source artifacts require a positive sample_question_limit"
        )
    if len(questions) < limit:
        raise ValueError(
            f"Sample source requests {limit} questions but only {len(questions)} exist"
        )
    return questions[:limit]
