"""Shared helpers for sample and held-out evaluation datasets."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID, NAMESPACE_URL, uuid5

from data import BenchmarkQuestion
from data.artifacts import SCHEMA_VERSION


EvaluationDatasetType = Literal["sample", "test"]


def create_snapshot_name(
    base_name: str,
    source_fingerprint: str,
    dataset_type: EvaluationDatasetType | None = None,
) -> str:
    base_name = base_name.strip()
    if not base_name:
        raise ValueError("dataset name must not be empty")
    if not source_fingerprint:
        raise ValueError("source fingerprint must not be empty")
    role = f"-{dataset_type}" if dataset_type else ""
    return f"{base_name}{role}-{source_fingerprint[:12]}"


def get_snapshot_id(dataset_name: str, question_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"langsmith://{dataset_name}/{question_id}")


def question_to_example(
    question: BenchmarkQuestion,
    *,
    ordinal: int,
    dataset_name: str,
    source_fingerprint: str,
    dataset_type: EvaluationDatasetType = "test",
) -> dict[str, Any]:
    """Format one question without exposing gold data in its inputs."""
    return {
        "id": get_snapshot_id(dataset_name, question.question_id),
        "inputs": {
            "question_id": question.question_id,
            "question": question.question,
        },
        "outputs": {
            "gold_answer": question.gold_answer,
            "expected_doc_ids": question.expected_doc_ids,
            "answer_facts": question.answer_facts,
        },
        "metadata": {
            "question_type": question.question_type,
            "source_types": question.source_types,
            "ordinal": ordinal,
            "source_fingerprint": source_fingerprint,
            "schema_version": SCHEMA_VERSION,
            "dataset_type": dataset_type,
        },
        "split": dataset_type,
    }
