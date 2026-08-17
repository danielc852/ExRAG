"""Records and configuration models for the data preparation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    source_type: str
    title: str
    content: str
    dataset_row_index: int | None = None


@dataclass(frozen=True)
class BenchmarkQuestion:
    question_id: str
    question_type: str
    source_types: list[str]
    question: str
    expected_doc_ids: list[str]
    gold_answer: str
    answer_facts: list[str]


@dataclass(frozen=True)
class ChunkRecord:
    integer_id: int
    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    content: str
    content_hash: str


@dataclass(frozen=True)
class DownloadConfig:
    artifact_root: Path = Path("artifacts")
    dataset_revision: str = "main"
    full_corpus: bool = False
    document_limit: int | None = 1_000
    sample_question_limit: int | None = 10
    seed: int = 42
    shard_size: int = 1_000
    cache_dir: Path | None = None


@dataclass(frozen=True)
class ProcessingConfig:
    artifact_root: Path = Path("artifacts")
    chunk_size: int = 512
    chunk_overlap: int = 64
    tokenizer_model: str = "BAAI/bge-base-en-v1.5"


@dataclass(frozen=True)
class EmbeddingConfig:
    artifact_root: Path = Path("artifacts")
    model_name: str = "BAAI/bge-base-en-v1.5"
    model_revision: str | None = None
    batch_size: int = 32
    dtype: str = "float32"
    normalize: bool = True


@dataclass(frozen=True)
class IndexConfig:
    artifact_root: Path = Path("artifacts")
    batch_size: int = 10_000
