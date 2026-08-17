"""Rules for the complete frozen held-out evaluation dataset."""

from __future__ import annotations

from typing import Any, Final

from data import BenchmarkQuestion
from data.artifacts import StageManifest

from .utils import build_question_example


DATASET_TYPE: Final = "test"


def select_questions(
    questions: list[BenchmarkQuestion], _source: StageManifest
) -> list[BenchmarkQuestion]:
    return questions


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
