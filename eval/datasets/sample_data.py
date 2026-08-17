"""Rules for the small development and smoke-test evaluation dataset."""

from __future__ import annotations

from typing import Any, Final

from data import BenchmarkQuestion
from data.artifacts import StageManifest

from .utils import build_question_example


DATASET_TYPE: Final = "sample"


def select_questions(
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


def question_to_example(
    question: BenchmarkQuestion,
    *,
    ordinal: int,
    dataset_name: str,
    source_fingerprint: str,
) -> dict[str, Any]:
    return build_question_example(
        question,
        ordinal=ordinal,
        dataset_name=dataset_name,
        source_fingerprint=source_fingerprint,
        dataset_type=DATASET_TYPE,
    )
