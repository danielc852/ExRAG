"""Build and validate immutable local representations of LangSmith datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from data import ArtifactLayout, BenchmarkQuestion, load_frozen_questions
from data.artifacts import SCHEMA_VERSION, StageManifest, load_manifest

from ..models import LangSmithDatasetConfig
from .scope import get_eval_mode, get_questions
from .utils import EvaluationDatasetType, create_snapshot_name, question_to_example


@dataclass(frozen=True)
class DatasetSnapshot:
    source: StageManifest
    dataset_type: EvaluationDatasetType
    questions: list[BenchmarkQuestion]
    name: str


def _source_manifest(config: LangSmithDatasetConfig) -> StageManifest:
    layout = ArtifactLayout(config.artifact_root)
    source = load_manifest(layout, "download")
    if source.status != "complete" or not source.output_fingerprint:
        raise ValueError(
            "Source artifacts are incomplete; run `python main.py download sample|full`"
        )
    return source


def build_dataset_snapshot(config: LangSmithDatasetConfig) -> DatasetSnapshot:
    source = _source_manifest(config)
    dataset_type = get_eval_mode(source)
    questions = get_questions(
        load_frozen_questions(config.artifact_root), source
    )
    name = create_snapshot_name(
        config.dataset_name, source.output_fingerprint, dataset_type
    )
    return DatasetSnapshot(
        source=source,
        dataset_type=dataset_type,
        questions=questions,
        name=name,
    )


def dataset_metadata(snapshot: DatasetSnapshot) -> dict[str, Any]:
    source = snapshot.source
    return {
        "source_fingerprint": source.output_fingerprint,
        "dataset_revision": source.metadata.get("dataset_revision"),
        "questions_fingerprint": source.metadata.get("questions_fingerprint"),
        "question_count": len(snapshot.questions),
        "artifact_schema_version": SCHEMA_VERSION,
        "dataset_type": snapshot.dataset_type,
    }


def validate_dataset_metadata(dataset: Any, snapshot: DatasetSnapshot) -> None:
    expected = dataset_metadata(snapshot)
    actual = dict(getattr(dataset, "metadata", None) or {})
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError(
            "LangSmith dataset snapshot metadata does not match local artifacts: "
            + ", ".join(mismatches)
            + ". Use a different --dataset-name."
        )


def example_matches(existing: Any, expected: dict[str, Any]) -> bool:
    return (
        dict(getattr(existing, "inputs", None) or {}) == expected["inputs"]
        and dict(getattr(existing, "outputs", None) or {}) == expected["outputs"]
        and dict(getattr(existing, "metadata", None) or {}) == expected["metadata"]
    )


def expected_examples(snapshot: DatasetSnapshot) -> dict[UUID, dict[str, Any]]:
    examples = [
        question_to_example(
            question,
            ordinal=ordinal,
            dataset_name=snapshot.name,
            source_fingerprint=snapshot.source.output_fingerprint,
            dataset_type=snapshot.dataset_type,
        )
        for ordinal, question in enumerate(snapshot.questions)
    ]
    return {example["id"]: example for example in examples}
