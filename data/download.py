"""Download and freeze a deterministic EnterpriseRAG-Bench corpus snapshot."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import (
    ArtifactLayout,
    StageManifest,
    begin_stage,
    finalize_manifest,
    load_manifest,
    shard_info,
    verify_manifest_files,
    write_manifest_atomic,
)
from .models import BenchmarkQuestion, DownloadConfig


DATASET_NAME = "onyx-dot-app/EnterpriseRAG-Bench"


def load_hf_dataset(config_name: str, *, revision: str, cache_dir: Path | None):
    from datasets import load_dataset

    return load_dataset(
        DATASET_NAME,
        config_name,
        split="test",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def _require_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dataset row has an invalid {field!r} field")
    return value


def _document_payload(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    return {
        "dataset_row_index": row_index,
        "doc_id": _require_text(row, "doc_id"),
        "source_type": _require_text(row, "source_type"),
        "title": _require_text(row, "title"),
        "content": _require_text(row, "content"),
    }


def _question_payload(row: dict[str, Any]) -> dict[str, Any]:
    for field in ("source_types", "expected_doc_ids", "answer_facts"):
        if not isinstance(row.get(field), list):
            raise ValueError(f"Question row has an invalid {field!r} field")
    return {
        "question_id": _require_text(row, "question_id"),
        "question_type": _require_text(row, "question_type"),
        "source_types": [str(value) for value in row["source_types"]],
        "question": _require_text(row, "question"),
        "expected_doc_ids": [str(value) for value in row["expected_doc_ids"]],
        "gold_answer": str(row.get("gold_answer", "")),
        "answer_facts": [str(value) for value in row["answer_facts"]],
    }


def _write_parquet_atomic(table: pa.Table, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(destination)


def _selected_indices(total: int, config: DownloadConfig) -> list[int]:
    if config.full_corpus:
        return list(range(total))
    if config.document_limit is None or config.document_limit < 1:
        raise ValueError("document limit must be at least 1")
    count = min(config.document_limit, total)
    return random.Random(config.seed).sample(range(total), count)


def download_dataset(
    config: DownloadConfig, *, resume: bool = True, rebuild: bool = False
) -> StageManifest:
    if config.shard_size < 1:
        raise ValueError("shard_size must be at least 1")
    layout = ArtifactLayout(config.artifact_root)
    manifest = begin_stage(
        layout, "download", config, upstream=None, resume=resume, rebuild=rebuild
    )
    if manifest.status == "complete":
        verify_manifest_files(layout, manifest)
        return manifest

    documents = load_hf_dataset(
        "documents", revision=config.dataset_revision, cache_dir=config.cache_dir
    )
    questions = load_hf_dataset(
        "questions", revision=config.dataset_revision, cache_dir=config.cache_dir
    )
    indices = _selected_indices(len(documents), config)
    source_dir = layout.source

    if "questions" not in manifest.completed_units:
        question_rows = [_question_payload(dict(row)) for row in questions]
        question_path = source_dir / "questions.parquet"
        _write_parquet_atomic(pa.Table.from_pylist(question_rows), question_path)
        manifest.shards.append(
            shard_info(
                kind="questions",
                path=question_path,
                base_dir=source_dir,
                row_count=len(question_rows),
            )
        )
        manifest.completed_units.append("questions")
        write_manifest_atomic(layout, manifest)

    shard_count = math.ceil(len(indices) / config.shard_size)
    for part_number in range(shard_count):
        unit = f"part-{part_number:06d}"
        if unit in manifest.completed_units:
            continue
        part_indices = indices[
            part_number * config.shard_size : (part_number + 1) * config.shard_size
        ]
        rows = [
            _document_payload(dict(documents[row_index]), row_index)
            for row_index in part_indices
        ]
        destination = source_dir / "documents" / f"{unit}.parquet"
        _write_parquet_atomic(pa.Table.from_pylist(rows), destination)
        manifest.shards.append(
            shard_info(
                kind="documents",
                path=destination,
                base_dir=source_dir,
                row_count=len(rows),
                min_id=min(part_indices) if part_indices else None,
                max_id=max(part_indices) if part_indices else None,
            )
        )
        manifest.completed_units.append(unit)
        manifest.stats["document_count"] = sum(
            shard.row_count for shard in manifest.shards if shard.kind == "documents"
        )
        write_manifest_atomic(layout, manifest)

    manifest.stats.update(
        {
            "document_count": len(indices),
            "question_count": len(questions),
            "document_shard_count": shard_count,
        }
    )
    manifest.metadata.update(
        {
            "dataset_name": DATASET_NAME,
            "dataset_revision": config.dataset_revision,
            "documents_fingerprint": str(getattr(documents, "_fingerprint", "unknown")),
            "questions_fingerprint": str(getattr(questions, "_fingerprint", "unknown")),
            "corpus_mode": "full" if config.full_corpus else "sample",
            "document_limit": None if config.full_corpus else len(indices),
            "seed": config.seed,
        }
    )
    return finalize_manifest(layout, manifest)


def load_frozen_questions(artifact_root: Path | str) -> list[BenchmarkQuestion]:
    layout = ArtifactLayout(artifact_root)
    manifest = load_manifest(layout, "download")
    if manifest.status != "complete":
        raise ValueError("Downloaded source artifacts are incomplete")
    verify_manifest_files(layout, manifest)
    question_shards = [shard for shard in manifest.shards if shard.kind == "questions"]
    if len(question_shards) != 1:
        raise ValueError("Source manifest must contain exactly one questions artifact")
    table = pq.read_table(layout.source / question_shards[0].path)
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
