"""Synchronize frozen EnterpriseRAG questions to immutable LangSmith datasets."""

from __future__ import annotations

from typing import Any
from uuid import UUID, NAMESPACE_URL, uuid5

from data import ArtifactLayout, BenchmarkQuestion, load_frozen_questions
from data.artifacts import SCHEMA_VERSION, StageManifest, load_manifest

from .models import DatasetSyncResult, LangSmithDatasetConfig


def dataset_snapshot_name(base_name: str, source_fingerprint: str) -> str:
    base_name = base_name.strip()
    if not base_name:
        raise ValueError("dataset name must not be empty")
    if not source_fingerprint:
        raise ValueError("source fingerprint must not be empty")
    return f"{base_name}-{source_fingerprint[:12]}"


def deterministic_example_id(dataset_name: str, question_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"langsmith://{dataset_name}/{question_id}")


def question_to_example(
    question: BenchmarkQuestion,
    *,
    ordinal: int,
    dataset_name: str,
    source_fingerprint: str,
) -> dict[str, Any]:
    return {
        "id": deterministic_example_id(dataset_name, question.question_id),
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
        },
        "split": "test",
    }


def _source_manifest(config: LangSmithDatasetConfig) -> StageManifest:
    layout = ArtifactLayout(config.artifact_root)
    source = load_manifest(layout, "download")
    if source.status != "complete" or not source.output_fingerprint:
        raise ValueError(
            "Source artifacts are incomplete; run `python main.py prepare download`"
        )
    return source


def _dataset_metadata(source: StageManifest, question_count: int) -> dict[str, Any]:
    return {
        "source_fingerprint": source.output_fingerprint,
        "dataset_revision": source.metadata.get("dataset_revision"),
        "questions_fingerprint": source.metadata.get("questions_fingerprint"),
        "question_count": question_count,
        "artifact_schema_version": SCHEMA_VERSION,
    }


def _validate_dataset_metadata(dataset: Any, expected: dict[str, Any]) -> None:
    actual = dict(getattr(dataset, "metadata", None) or {})
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError(
            "LangSmith dataset snapshot metadata does not match local artifacts: "
            + ", ".join(mismatches)
            + ". Use a different --dataset-name."
        )


def _example_matches(existing: Any, expected: dict[str, Any]) -> bool:
    return (
        dict(getattr(existing, "inputs", None) or {}) == expected["inputs"]
        and dict(getattr(existing, "outputs", None) or {}) == expected["outputs"]
        and dict(getattr(existing, "metadata", None) or {}) == expected["metadata"]
    )


def _expected_examples(
    questions: list[BenchmarkQuestion],
    *,
    dataset_name: str,
    source_fingerprint: str,
) -> dict[UUID, dict[str, Any]]:
    examples = [
        question_to_example(
            question,
            ordinal=ordinal,
            dataset_name=dataset_name,
            source_fingerprint=source_fingerprint,
        )
        for ordinal, question in enumerate(questions)
    ]
    return {example["id"]: example for example in examples}


def load_dataset_snapshot(
    client: Any,
    config: LangSmithDatasetConfig,
) -> tuple[Any, StageManifest, list[BenchmarkQuestion], str]:
    source = _source_manifest(config)
    questions = load_frozen_questions(config.artifact_root)
    snapshot_name = dataset_snapshot_name(config.dataset_name, source.output_fingerprint)
    if not client.has_dataset(dataset_name=snapshot_name):
        raise FileNotFoundError(
            f"LangSmith dataset {snapshot_name!r} is missing. "
            "Run `python main.py langsmith sync` first."
        )
    dataset = client.read_dataset(dataset_name=snapshot_name)
    _validate_dataset_metadata(dataset, _dataset_metadata(source, len(questions)))
    return dataset, source, questions, snapshot_name


def sync_frozen_dataset(
    client: Any,
    config: LangSmithDatasetConfig,
) -> DatasetSyncResult:
    source = _source_manifest(config)
    questions = load_frozen_questions(config.artifact_root)
    snapshot_name = dataset_snapshot_name(config.dataset_name, source.output_fingerprint)
    expected_metadata = _dataset_metadata(source, len(questions))
    created_dataset = not client.has_dataset(dataset_name=snapshot_name)
    if created_dataset:
        dataset = client.create_dataset(
            snapshot_name,
            description=(
                "Frozen EnterpriseRAG-Bench questions for reproducible RAG experiments"
            ),
            metadata=expected_metadata,
        )
    else:
        dataset = client.read_dataset(dataset_name=snapshot_name)
        _validate_dataset_metadata(dataset, expected_metadata)

    expected_examples = _expected_examples(
        questions,
        dataset_name=snapshot_name,
        source_fingerprint=source.output_fingerprint,
    )
    existing_examples = {
        example.id: example
        for example in client.list_examples(dataset_id=dataset.id)
    }
    unexpected = sorted(set(existing_examples).difference(expected_examples), key=str)
    if unexpected:
        raise ValueError(
            f"LangSmith dataset contains {len(unexpected)} unexpected examples. "
            "Use a different --dataset-name."
        )
    mismatched = [
        example_id
        for example_id in set(existing_examples).intersection(expected_examples)
        if not _example_matches(
            existing_examples[example_id], expected_examples[example_id]
        )
    ]
    if mismatched:
        raise ValueError(
            f"LangSmith dataset contains {len(mismatched)} conflicting examples. "
            "Use a different --dataset-name."
        )
    missing = [
        payload
        for example_id, payload in expected_examples.items()
        if example_id not in existing_examples
    ]
    if missing:
        client.create_examples(dataset_id=dataset.id, examples=missing)

    if created_dataset:
        status = "created"
    elif missing:
        status = "updated"
    else:
        status = "unchanged"
    return DatasetSyncResult(
        dataset_id=str(dataset.id),
        dataset_name=snapshot_name,
        source_fingerprint=source.output_fingerprint,
        total_examples=len(expected_examples),
        created_examples=len(missing),
        status=status,
    )
