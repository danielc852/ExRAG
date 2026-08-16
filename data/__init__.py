"""Stable public API for the EnterpriseRAG data preparation pipeline."""

from .artifacts import ArtifactLayout, ShardInfo, StageManifest
from .download import download_dataset, load_frozen_questions
from .embedding import embed_chunks
from .indexing import build_faiss_index, load_index_manifest, validate_index
from .models import (
    BenchmarkQuestion,
    ChunkRecord,
    DocumentRecord,
    DownloadConfig,
    EmbeddingConfig,
    IndexConfig,
    ProcessingConfig,
)
from .pipeline import get_pipeline_status, run_all, run_stage
from .processing import chunk_document, normalize_text, process_documents

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
    "chunk_document",
    "download_dataset",
    "embed_chunks",
    "get_pipeline_status",
    "load_frozen_questions",
    "load_index_manifest",
    "normalize_text",
    "process_documents",
    "run_all",
    "run_stage",
    "validate_index",
]
