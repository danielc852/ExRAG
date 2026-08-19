"""Load and synchronize frozen evaluation dataset snapshots with LangSmith."""

from __future__ import annotations

from typing import Any

from data import BenchmarkQuestion
from data.artifacts import StageManifest

from ..models import DatasetSyncResult, LangSmithDatasetConfig
from .snapshot import (
    build_dataset_snapshot,
    dataset_metadata,
    example_matches,
    expected_examples,
    validate_dataset_metadata,
)


def load_dataset_snapshot(
    client: Any,
    config: LangSmithDatasetConfig,
) -> tuple[Any, StageManifest, list[BenchmarkQuestion], str]:
    snapshot = build_dataset_snapshot(config)
    if not client.has_dataset(dataset_name=snapshot.name):
        raise FileNotFoundError(
            f"LangSmith dataset {snapshot.name!r} is missing. "
            "Call sync_frozen_dataset() first."
        )
    dataset = client.read_dataset(dataset_name=snapshot.name)
    validate_dataset_metadata(dataset, snapshot)
    return dataset, snapshot.source, snapshot.questions, snapshot.name


def sync_frozen_dataset(
    client: Any,
    config: LangSmithDatasetConfig,
) -> DatasetSyncResult:
    snapshot = build_dataset_snapshot(config)
    metadata = dataset_metadata(snapshot)
    created_dataset = not client.has_dataset(dataset_name=snapshot.name)
    if created_dataset:
        dataset = client.create_dataset(
            snapshot.name,
            description=(
                f"Frozen EnterpriseRAG-Bench {snapshot.dataset_type} questions for "
                "reproducible RAG experiments"
            ),
            metadata=metadata,
        )
    else:
        dataset = client.read_dataset(dataset_name=snapshot.name)
        validate_dataset_metadata(dataset, snapshot)

    expected = expected_examples(snapshot)
    existing_examples = {
        example.id: example
        for example in client.list_examples(dataset_id=dataset.id)
    }
    unexpected = sorted(set(existing_examples).difference(expected), key=str)
    if unexpected:
        raise ValueError(
            f"LangSmith dataset contains {len(unexpected)} unexpected examples. "
            "Use a different --dataset-name."
        )
    mismatched = [
        example_id
        for example_id in set(existing_examples).intersection(expected)
        if not example_matches(
            existing_examples[example_id], expected[example_id]
        )
    ]
    if mismatched:
        raise ValueError(
            f"LangSmith dataset contains {len(mismatched)} conflicting examples. "
            "Use a different --dataset-name."
        )
    missing = [
        payload
        for example_id, payload in expected.items()
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
        dataset_name=snapshot.name,
        dataset_type=snapshot.dataset_type,
        source_fingerprint=snapshot.source.output_fingerprint,
        total_examples=len(expected),
        created_examples=len(missing),
        status=status,
    )
