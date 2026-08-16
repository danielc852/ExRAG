"""Validate frozen source artifacts before cleaning and chunking."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from ..artifacts import (
    ArtifactLayout,
    StageManifest,
    load_manifest,
    verify_manifest_files,
)
from ..models import BenchmarkQuestion


REQUIRED_COLUMNS = {
    "documents": {"dataset_row_index", "doc_id", "source_type", "title", "content"},
    "questions": {
        "question_id",
        "question_type",
        "source_types",
        "question",
        "expected_doc_ids",
        "gold_answer",
        "answer_facts",
    },
}


def review_download(artifact_root: Path | str) -> StageManifest:
    """Confirm the snapshot is complete, readable, and internally consistent."""
    layout = ArtifactLayout(artifact_root)
    manifest = load_manifest(layout, "download")
    if manifest.status != "complete":
        raise ValueError("Downloaded source artifacts are incomplete")
    verify_manifest_files(layout, manifest)

    grouped_shards = {
        kind: [shard for shard in manifest.shards if shard.kind == kind]
        for kind in REQUIRED_COLUMNS
    }
    if len(grouped_shards["questions"]) != 1:
        raise ValueError("Source manifest must contain exactly one questions artifact")

    for kind, shards in grouped_shards.items():
        manifest_count = sum(shard.row_count for shard in shards)
        expected_count = int(manifest.stats.get(f"{kind[:-1]}_count", -1))
        if manifest_count != expected_count:
            raise ValueError(f"Downloaded {kind} count does not match its manifest")
        for shard in shards:
            path = layout.source / shard.path
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != shard.row_count:
                raise ValueError(f"Downloaded row count does not match manifest: {path}")
            missing = REQUIRED_COLUMNS[kind] - set(parquet.schema_arrow.names)
            if missing:
                columns = ", ".join(sorted(missing))
                raise ValueError(f"Downloaded {kind} artifact is missing columns: {columns}")
    return manifest


def load_frozen_questions(artifact_root: Path | str) -> list[BenchmarkQuestion]:
    layout = ArtifactLayout(artifact_root)
    manifest = review_download(artifact_root)
    question_shard = next(shard for shard in manifest.shards if shard.kind == "questions")
    table = pq.read_table(layout.source / question_shard.path)
    return [
        BenchmarkQuestion(
            question_id=str(row["question_id"]),
            question_type=str(row["question_type"]),
            source_types=list(row["source_types"] or []),
            question=str(row["question"]),
            expected_doc_ids=list(row["expected_doc_ids"] or []),
            gold_answer=str(row["gold_answer"] or ""),
            answer_facts=list(row["answer_facts"] or []),
        )
        for row in table.to_pylist()
    ]
