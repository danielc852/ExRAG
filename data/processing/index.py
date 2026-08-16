"""Build and validate the final FAISS and SQLite retrieval index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..artifacts import (
    ArtifactLayout,
    StageManifest,
    begin_stage,
    finalize_manifest,
    load_manifest,
    shard_info,
    validate_upstream,
    verify_manifest_files,
    write_manifest_atomic,
)
from ..models import IndexConfig
from .store import (
    FAISS_NAME,
    SQLITE_NAME,
    connect_metadata_store,
    load_or_create_vector_store,
    save_vector_store,
)


def load_index_manifest(artifact_root: Path | str) -> StageManifest:
    return load_manifest(ArtifactLayout(artifact_root), "index")


def validate_index(artifact_root: Path | str) -> StageManifest:
    import faiss

    layout = ArtifactLayout(artifact_root)
    manifest = load_manifest(layout, "index")
    if manifest.status != "complete":
        raise ValueError("Index stage is incomplete; run `python main.py prepare index`")
    source = load_manifest(layout, "download")
    processed = load_manifest(layout, "process")
    embeddings = load_manifest(layout, "embed")
    lineage_matches = (
        source.status == processed.status == embeddings.status == "complete"
        and processed.upstream_fingerprint == source.output_fingerprint
        and embeddings.upstream_fingerprint == processed.output_fingerprint
        and manifest.upstream_fingerprint == embeddings.output_fingerprint
        and manifest.metadata.get("source_fingerprint") == source.output_fingerprint
        and manifest.metadata.get("processed_fingerprint") == processed.output_fingerprint
        and manifest.metadata.get("embedding_fingerprint") == embeddings.output_fingerprint
    )
    if not lineage_matches:
        raise ValueError("Index artifacts do not share one complete pipeline lineage")
    faiss_path = layout.index / FAISS_NAME
    sqlite_path = layout.index / SQLITE_NAME
    if not faiss_path.exists() or not sqlite_path.exists():
        raise FileNotFoundError("Index is missing its FAISS or SQLite file")
    index = faiss.read_index(str(faiss_path))
    with sqlite3.connect(sqlite_path) as connection:
        row_count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    expected = int(manifest.stats.get("chunk_count", -1))
    processed_count = int(processed.stats.get("chunk_count", -1))
    embedding_count = int(embeddings.stats.get("vector_count", -1))
    if len({int(index.ntotal), row_count, expected, processed_count, embedding_count}) != 1:
        raise ValueError(
            "Processed, embedding, FAISS, SQLite, and manifest counts do not match; rebuild index"
        )
    return manifest


def _vector_pairs(manifest: StageManifest) -> dict[str, tuple[Path, Path]]:
    vectors = {
        Path(item.path).stem: Path(item.path)
        for item in manifest.shards
        if item.kind == "vectors"
    }
    ids = {
        Path(item.path).stem: Path(item.path)
        for item in manifest.shards
        if item.kind == "ids"
    }
    if vectors.keys() != ids.keys():
        raise ValueError("Embedding vector and ID shard sets do not match")
    return {unit: (vectors[unit], ids[unit]) for unit in sorted(vectors)}


def build_faiss_index(
    config: IndexConfig, *, resume: bool = True, rebuild: bool = False
) -> StageManifest:
    if config.batch_size < 1:
        raise ValueError("index batch size must be at least 1")
    layout = ArtifactLayout(config.artifact_root)
    upstream = validate_upstream(layout, "index")
    processed = load_manifest(layout, "process")
    if processed.status != "complete":
        raise ValueError(
            "Processed artifacts are incomplete; run `python main.py prepare process`"
        )
    verify_manifest_files(layout, processed)
    if upstream.upstream_fingerprint != processed.output_fingerprint:
        raise ValueError("Embedding artifacts do not belong to the current processed artifacts")
    manifest = begin_stage(
        layout, "index", config, upstream=upstream, resume=resume, rebuild=rebuild
    )
    if manifest.status == "complete":
        return validate_index(config.artifact_root)

    layout.index.mkdir(parents=True, exist_ok=True)
    faiss_path = layout.index / FAISS_NAME
    sqlite_path = layout.index / SQLITE_NAME
    dimension = int(upstream.metadata.get("embedding_dimension", 0))
    if dimension < 1:
        raise ValueError("Embedding manifest has no valid vector dimension")
    index = load_or_create_vector_store(faiss_path, dimension)
    connection = connect_metadata_store(sqlite_path)
    row_count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    checkpoint_count = int(manifest.stats.get("chunk_count", 0))
    if index.ntotal != row_count or row_count != checkpoint_count:
        connection.close()
        raise ValueError("Partial index checkpoint is inconsistent; use --rebuild")

    chunk_shards = {
        Path(item.path).stem: item for item in processed.shards if item.kind == "chunks"
    }
    pairs = _vector_pairs(upstream)
    if chunk_shards.keys() != pairs.keys():
        connection.close()
        raise ValueError("Processed chunk and embedding shard sets do not match")

    try:
        for unit, (vector_relative, id_relative) in pairs.items():
            if unit in manifest.completed_units:
                continue
            chunk_table = pq.read_table(layout.processed / chunk_shards[unit].path)
            rows = chunk_table.to_pylist()
            expected_ids = np.asarray(
                [int(row["integer_id"]) for row in rows], dtype=np.int64
            )
            ids = np.load(layout.embeddings / id_relative, mmap_mode="r", allow_pickle=False)
            vectors = np.load(
                layout.embeddings / vector_relative, mmap_mode="r", allow_pickle=False
            )
            if vectors.dtype != np.float32 or ids.dtype != np.int64:
                raise ValueError(f"Unexpected vector or ID dtype in embedding shard {unit}")
            if vectors.shape != (len(rows), dimension) or not np.array_equal(
                ids, expected_ids
            ):
                raise ValueError(f"Processed and embedding rows are misaligned for shard {unit}")
            for start in range(0, len(ids), config.batch_size):
                end = min(start + config.batch_size, len(ids))
                index.add_with_ids(
                    np.asarray(vectors[start:end], dtype=np.float32),
                    np.asarray(ids[start:end], dtype=np.int64),
                )
            connection.executemany(
                """
                INSERT INTO chunks
                (integer_id, chunk_id, doc_id, source_type, title, content, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(row["integer_id"]),
                        row["chunk_id"],
                        row["doc_id"],
                        row["source_type"],
                        row["title"],
                        row["content"],
                        row["content_hash"],
                    )
                    for row in rows
                ],
            )
            connection.commit()
            save_vector_store(index, faiss_path)
            manifest.completed_units.append(unit)
            manifest.stats["chunk_count"] = int(index.ntotal)
            write_manifest_atomic(layout, manifest)
    finally:
        connection.close()

    manifest.shards = [
        shard_info(
            kind="faiss",
            path=faiss_path,
            base_dir=layout.index,
            row_count=int(index.ntotal),
        ),
        shard_info(
            kind="sqlite",
            path=sqlite_path,
            base_dir=layout.index,
            row_count=int(index.ntotal),
        ),
    ]
    manifest.stats.update(
        {
            "chunk_count": int(index.ntotal),
            "indexed_shard_count": len(pairs),
        }
    )
    manifest.metadata = {
        **upstream.metadata,
        "embedding_fingerprint": upstream.output_fingerprint,
        "index_type": "faiss.IndexIDMap2(IndexFlatIP)",
        "similarity": "cosine",
    }
    completed = finalize_manifest(layout, manifest)
    return validate_index(config.artifact_root) if completed.status == "complete" else completed
