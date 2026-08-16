"""Encode processed chunk shards into normalized NumPy vector shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..artifacts import (
    ArtifactLayout,
    StageManifest,
    begin_stage,
    finalize_manifest,
    shard_info,
    validate_upstream,
    verify_manifest_files,
    write_manifest_atomic,
)
from ..models import EmbeddingConfig
from .embed import create_embedding_model, encode_chunk_shard


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def embed_chunks(
    config: EmbeddingConfig, *, resume: bool = True, rebuild: bool = False
) -> StageManifest:
    if config.dtype != "float32":
        raise ValueError("Pipeline v2 supports only float32 embedding artifacts")
    if config.batch_size < 1:
        raise ValueError("embedding batch size must be at least 1")
    layout = ArtifactLayout(config.artifact_root)
    upstream = validate_upstream(layout, "embed")
    manifest = begin_stage(
        layout, "embed", config, upstream=upstream, resume=resume, rebuild=rebuild
    )
    if manifest.status == "complete":
        verify_manifest_files(layout, manifest)
        return manifest

    model = create_embedding_model(config.model_name, config.model_revision)
    chunk_shards = sorted(
        (shard for shard in upstream.shards if shard.kind == "chunks"),
        key=lambda shard: shard.path,
    )
    total_vectors = sum(
        shard.row_count for shard in manifest.shards if shard.kind == "vectors"
    )
    dimension = int(manifest.metadata.get("embedding_dimension", 0))
    for chunk_shard in chunk_shards:
        unit = Path(chunk_shard.path).stem
        if unit in manifest.completed_units:
            continue
        table = pq.read_table(layout.processed / chunk_shard.path)
        payloads = table.to_pylist()
        texts = [f"Title: {row['title']}\n\n{row['content']}" for row in payloads]
        ids = np.asarray([int(row["integer_id"]) for row in payloads], dtype=np.int64)
        vectors = encode_chunk_shard(
            model, texts, batch_size=config.batch_size, normalize=config.normalize
        )
        if vectors.ndim != 2 or vectors.shape[0] != ids.shape[0]:
            raise ValueError(f"Embedding output shape does not match chunk shard {unit}")
        if dimension and vectors.shape[1] != dimension:
            raise ValueError("Embedding dimension changed between shards")
        dimension = int(vectors.shape[1])
        vector_path = layout.embeddings / "vectors" / f"{unit}.npy"
        id_path = layout.embeddings / "ids" / f"{unit}.npy"
        _save_npy_atomic(vector_path, vectors.astype(np.float32, copy=False))
        _save_npy_atomic(id_path, ids)
        min_id = int(ids.min()) if ids.size else None
        max_id = int(ids.max()) if ids.size else None
        manifest.shards.extend(
            [
                shard_info(
                    kind="vectors",
                    path=vector_path,
                    base_dir=layout.embeddings,
                    row_count=len(ids),
                    min_id=min_id,
                    max_id=max_id,
                ),
                shard_info(
                    kind="ids",
                    path=id_path,
                    base_dir=layout.embeddings,
                    row_count=len(ids),
                    min_id=min_id,
                    max_id=max_id,
                ),
            ]
        )
        total_vectors += len(ids)
        manifest.completed_units.append(unit)
        manifest.stats["vector_count"] = total_vectors
        manifest.metadata["embedding_dimension"] = dimension
        write_manifest_atomic(layout, manifest)

    manifest.stats.update(
        {
            "chunk_count": upstream.stats.get("chunk_count", 0),
            "vector_count": total_vectors,
            "vector_shard_count": len(chunk_shards),
        }
    )
    manifest.metadata = {
        **upstream.metadata,
        "source_fingerprint": upstream.metadata.get("source_fingerprint"),
        "processed_fingerprint": upstream.output_fingerprint,
        "embedding_model": config.model_name,
        "embedding_revision": config.model_revision,
        "embedding_dimension": dimension,
        "embedding_dtype": config.dtype,
        "embedding_normalized": config.normalize,
    }
    return finalize_manifest(layout, manifest)
