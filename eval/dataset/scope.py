"""Define the evaluation role and question scope for frozen source artifacts."""

from __future__ import annotations

from data import BenchmarkQuestion
from data.artifacts import StageManifest

from . import sample_data
from .utils import EvaluationDatasetType


def get_eval_mode(source: StageManifest) -> EvaluationDatasetType:
    """Map the frozen corpus mode to its evaluation dataset role."""
    corpus_mode = source.metadata.get("corpus_mode")
    if corpus_mode == "sample":
        return "sample"
    if corpus_mode == "full":
        return "test"
    raise ValueError(
        "Source artifacts do not declare a valid corpus_mode; expected sample or full"
    )


def get_eval_questions(
    questions: list[BenchmarkQuestion], source: StageManifest
) -> list[BenchmarkQuestion]:
    """Return only the questions that belong to the frozen evaluation dataset."""
    if get_eval_mode(source) == "sample":
        return sample_data.get_sample_questions(questions, source)
    return questions
