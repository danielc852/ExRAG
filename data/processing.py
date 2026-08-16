"""Normalize and chunk frozen source documents into Parquet shards."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import (
    ArtifactLayout,
    StageManifest,
    begin_stage,
    finalize_manifest,
    shard_info,
    validate_upstream,
    verify_manifest_files,
    write_manifest_atomic,
)
from .models import ChunkRecord, DocumentRecord, ProcessingConfig


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
    return normalized.strip()


def create_text_splitter(model_name: str, chunk_size: int, overlap: int):
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n# ", "\n## ", "\n\n", "\n", ". ", " ", ""],
    )


def chunk_document(document: DocumentRecord, splitter) -> Iterator[ChunkRecord]:
    content = normalize_text(document.content)
    title = normalize_text(document.title)
    seen_hashes: set[str] = set()
    chunk_number = 0
    for raw_chunk in splitter.split_text(content):
        chunk = normalize_text(raw_chunk)
        if not chunk:
            continue
        digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        yield ChunkRecord(
            integer_id=chunk_number,
            chunk_id=f"{document.doc_id}::{chunk_number:06d}",
            doc_id=document.doc_id,
            source_type=document.source_type,
            title=title,
            content=chunk,
            content_hash=digest,
        )
        chunk_number += 1


def _write_parquet_atomic(rows: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    schema = pa.schema(
        [
            ("integer_id", pa.int64()),
            ("chunk_id", pa.string()),
            ("doc_id", pa.string()),
            ("source_type", pa.string()),
            ("title", pa.string()),
            ("content", pa.string()),
            ("content_hash", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
    temporary.replace(destination)


def process_documents(
    config: ProcessingConfig, *, resume: bool = True, rebuild: bool = False
) -> StageManifest:
    layout = ArtifactLayout(config.artifact_root)
    upstream = validate_upstream(layout, "process")
    manifest = begin_stage(
        layout, "process", config, upstream=upstream, resume=resume, rebuild=rebuild
    )
    if manifest.status == "complete":
        verify_manifest_files(layout, manifest)
        return manifest

    splitter = create_text_splitter(
        config.tokenizer_model, config.chunk_size, config.chunk_overlap
    )
    source_shards = sorted(
        (shard for shard in upstream.shards if shard.kind == "documents"),
        key=lambda shard: shard.path,
    )
    next_integer_id = sum(
        shard.row_count for shard in manifest.shards if shard.kind == "chunks"
    )
    for source_shard in source_shards:
        unit = Path(source_shard.path).stem
        if unit in manifest.completed_units:
            continue
        table = pq.read_table(layout.source / source_shard.path)
        rows: list[dict] = []
        for payload in table.to_pylist():
            document = DocumentRecord(
                doc_id=str(payload["doc_id"]),
                source_type=str(payload["source_type"]),
                title=str(payload["title"]),
                content=str(payload["content"]),
                dataset_row_index=int(payload["dataset_row_index"]),
            )
            for chunk in chunk_document(document, splitter):
                rows.append(
                    {
                        "integer_id": next_integer_id,
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "source_type": chunk.source_type,
                        "title": chunk.title,
                        "content": chunk.content,
                        "content_hash": chunk.content_hash,
                    }
                )
                next_integer_id += 1
        destination = layout.processed / "chunks" / f"{unit}.parquet"
        _write_parquet_atomic(rows, destination)
        manifest.shards.append(
            shard_info(
                kind="chunks",
                path=destination,
                base_dir=layout.processed,
                row_count=len(rows),
                min_id=rows[0]["integer_id"] if rows else None,
                max_id=rows[-1]["integer_id"] if rows else None,
            )
        )
        manifest.completed_units.append(unit)
        manifest.stats["chunk_count"] = next_integer_id
        write_manifest_atomic(layout, manifest)

    manifest.stats.update(
        {
            "document_count": upstream.stats.get("document_count", 0),
            "chunk_count": next_integer_id,
            "chunk_shard_count": len(source_shards),
        }
    )
    manifest.metadata = {
        **upstream.metadata,
        "source_fingerprint": upstream.output_fingerprint,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "tokenizer_model": config.tokenizer_model,
    }
    return finalize_manifest(layout, manifest)
