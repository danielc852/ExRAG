"""Stable public API for the EnterpriseRAG data preparation pipeline."""

from .artifacts import ArtifactLayout, ShardInfo, StageManifest
from .models import (
    BenchmarkQuestion,
    ChunkRecord,
    DocumentRecord,
    DownloadConfig,
    EmbeddingConfig,
    IndexConfig,
    ProcessingConfig,
)
from .pipeline import get_status, pre_data, pre_store, run_process
from .preprocessing import (
    clean_data,
    chunk_document,
    download_dataset,
    load_frozen_questions,
    normalize_text,
    review_download,
)
from .processing import (
    build_faiss_index,
    embed_chunks,
    load_index_manifest,
    validate_index,
)

__all__ = [
    "ArtifactLayout",
    "BenchmarkQuestion",
    "ChunkRecord",
    "DocumentRecord",
    "DownloadConfig",
    "EmbeddingConfig",
    "IndexConfig",
    "ProcessingConfig",
    "ShardInfo",
    "StageManifest",
    "build_faiss_index",
    "clean_data",
    "chunk_document",
    "download_dataset",
    "embed_chunks",
    "get_status",
    "load_frozen_questions",
    "load_index_manifest",
    "normalize_text",
    "pre_data",
    "pre_store",
    "review_download",
    "run_process",
    "validate_index",
]
