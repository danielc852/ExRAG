"""Dataset ingestion, chunking, and FAISS index construction."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


DATASET_NAME = "onyx-dot-app/EnterpriseRAG-Bench"
MANIFEST_NAME = "manifest.json"
FAISS_NAME = "chunks.faiss"
SQLITE_NAME = "chunks.sqlite3"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    source_type: str
    title: str
    content: str


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
class IndexBuildConfig:
    index_dir: Path
    dataset_revision: str = "main"
    full_corpus: bool = False
    document_limit: int = 1_000
    seed: int = 42
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_batch_size: int = 32
    checkpoint_every: int = 1_000
    cache_dir: Path | None = None


@dataclass
class IndexManifest:
    schema_version: int
    status: str
    dataset_revision: str
    dataset_fingerprint: str
    corpus_mode: str
    document_limit: int | None
    processed_documents: int
    chunk_count: int
    embedding_model: str
    embedding_dimension: int
    chunk_size: int
    chunk_overlap: int
    seed: int


def _require_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dataset row has an invalid {field!r} field")
    return value


def _document_from_row(row: dict[str, Any]) -> DocumentRecord:
    return DocumentRecord(
        doc_id=_require_text(row, "doc_id"),
        source_type=_require_text(row, "source_type"),
        title=_require_text(row, "title"),
        content=_require_text(row, "content"),
    )


def _question_from_row(row: dict[str, Any]) -> BenchmarkQuestion:
    list_fields = ("source_types", "expected_doc_ids", "answer_facts")
    for field in list_fields:
        if not isinstance(row.get(field), list):
            raise ValueError(f"Question row has an invalid {field!r} field")
    return BenchmarkQuestion(
        question_id=_require_text(row, "question_id"),
        question_type=_require_text(row, "question_type"),
        source_types=[str(value) for value in row["source_types"]],
        question=_require_text(row, "question"),
        expected_doc_ids=[str(value) for value in row["expected_doc_ids"]],
        gold_answer=str(row.get("gold_answer", "")),
        answer_facts=[str(value) for value in row["answer_facts"]],
    )


def load_documents(revision: str = "main", cache_dir: Path | None = None):
    """Download and return the benchmark document split from Hugging Face."""
    from datasets import load_dataset

    return load_dataset(
        DATASET_NAME,
        "documents",
        split="test",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def load_questions(
    revision: str = "main", cache_dir: Path | None = None
) -> list[BenchmarkQuestion]:
    """Download and validate all benchmark questions."""
    from datasets import load_dataset

    rows = load_dataset(
        DATASET_NAME,
        "questions",
        split="test",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return [_question_from_row(dict(row)) for row in rows]


def select_documents(dataset, *, full: bool, limit: int, seed: int):
    """Return the full corpus or a deterministic shuffled sample."""
    if full:
        return dataset
    if limit < 1:
        raise ValueError("document limit must be at least 1")
    count = min(limit, len(dataset))
    return dataset.shuffle(seed=seed).select(range(count))


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while retaining paragraph structure."""
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
    return normalized.strip()


def create_text_splitter(model_name: str, chunk_size: int, overlap: int):
    """Create a token-aware recursive splitter using the embedding tokenizer."""
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
    """Yield stable chunks, removing exact duplicates only within this document."""
    content = normalize_text(document.content)
    title = normalize_text(document.title)
    seen_hashes: set[str] = set()
    output_number = 0
    for raw_chunk in splitter.split_text(content):
        chunk = normalize_text(raw_chunk)
        if not chunk:
            continue
        digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        yield ChunkRecord(
            integer_id=output_number,
            chunk_id=f"{document.doc_id}::{output_number:06d}",
            doc_id=document.doc_id,
            source_type=document.source_type,
            title=title,
            content=chunk,
            content_hash=digest,
        )
        output_number += 1


def iter_unique_chunks(documents: Iterable[Any], splitter) -> Iterator[ChunkRecord]:
    """Yield globally numbered chunks without deduplicating across documents."""
    integer_id = 0
    for row in documents:
        document = row if isinstance(row, DocumentRecord) else _document_from_row(dict(row))
        for chunk in chunk_document(document, splitter):
            yield ChunkRecord(integer_id=integer_id, **{k: v for k, v in asdict(chunk).items() if k != "integer_id"})
            integer_id += 1


def _manifest_path(index_dir: Path) -> Path:
    return index_dir / MANIFEST_NAME


def _write_manifest(index_dir: Path, manifest: IndexManifest) -> None:
    destination = _manifest_path(index_dir)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def load_manifest(index_dir: Path) -> IndexManifest:
    path = _manifest_path(Path(index_dir))
    if not path.exists():
        raise FileNotFoundError(f"Index manifest not found: {path}")
    return IndexManifest(**json.loads(path.read_text(encoding="utf-8")))


def _connect_store(index_dir: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(index_dir / SQLITE_NAME)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            integer_id INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            doc_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_type)")
    return connection


def validate_index(index_dir: Path) -> IndexManifest:
    """Validate manifest, FAISS, and SQLite cardinality."""
    import faiss

    index_dir = Path(index_dir)
    manifest = load_manifest(index_dir)
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported index schema version: {manifest.schema_version}")
    faiss_path = index_dir / FAISS_NAME
    sqlite_path = index_dir / SQLITE_NAME
    if not faiss_path.exists() or not sqlite_path.exists():
        raise FileNotFoundError("Index is missing its FAISS or SQLite data file")
    index = faiss.read_index(str(faiss_path))
    with sqlite3.connect(sqlite_path) as connection:
        row_count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    if index.ntotal != row_count or row_count != manifest.chunk_count:
        raise ValueError(
            "Index checkpoint is inconsistent; run `python main.py index --rebuild`"
        )
    return manifest


def _validate_resume_config(config: IndexBuildConfig, manifest: IndexManifest, fingerprint: str) -> None:
    expected_limit = None if config.full_corpus else config.document_limit
    checks = {
        "dataset revision": (manifest.dataset_revision, config.dataset_revision),
        "dataset fingerprint": (manifest.dataset_fingerprint, fingerprint),
        "corpus mode": (manifest.corpus_mode, "full" if config.full_corpus else "sample"),
        "document limit": (manifest.document_limit, expected_limit),
        "embedding model": (manifest.embedding_model, config.embedding_model),
        "chunk size": (manifest.chunk_size, config.chunk_size),
        "chunk overlap": (manifest.chunk_overlap, config.chunk_overlap),
        "seed": (manifest.seed, config.seed),
    }
    mismatches = [name for name, values in checks.items() if values[0] != values[1]]
    if mismatches:
        raise ValueError(f"Cannot resume because these settings changed: {', '.join(mismatches)}")


def _encode_chunks(model, chunks: list[ChunkRecord], batch_size: int):
    texts = [f"Title: {chunk.title}\n\n{chunk.content}" for chunk in chunks]
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def build_index(
    config: IndexBuildConfig, *, rebuild: bool = False, resume: bool = False
) -> IndexManifest:
    """Build or resume a disk-backed FAISS/SQLite index."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    index_dir = Path(config.index_dir)
    if rebuild and index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(config.dataset_revision, config.cache_dir)
    fingerprint = str(getattr(documents, "_fingerprint", "unknown"))
    selected = select_documents(
        documents, full=config.full_corpus, limit=config.document_limit, seed=config.seed
    )
    manifest_path = _manifest_path(index_dir)
    if manifest_path.exists():
        manifest = load_manifest(index_dir)
        if not resume:
            if manifest.status == "complete":
                return validate_index(index_dir)
            raise FileExistsError("A partial index exists; use --resume or --rebuild")
        _validate_resume_config(config, manifest, fingerprint)
        if manifest.status == "complete":
            return validate_index(index_dir)
    else:
        manifest = IndexManifest(
            schema_version=SCHEMA_VERSION,
            status="building",
            dataset_revision=config.dataset_revision,
            dataset_fingerprint=fingerprint,
            corpus_mode="full" if config.full_corpus else "sample",
            document_limit=None if config.full_corpus else len(selected),
            processed_documents=0,
            chunk_count=0,
            embedding_model=config.embedding_model,
            embedding_dimension=0,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            seed=config.seed,
        )
        _write_manifest(index_dir, manifest)

    connection = _connect_store(index_dir)
    faiss_path = index_dir / FAISS_NAME
    index = faiss.read_index(str(faiss_path)) if faiss_path.exists() else None
    row_count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    if index is not None:
        if index.ntotal != row_count or row_count != manifest.chunk_count:
            connection.close()
            raise ValueError("Partial checkpoint is inconsistent; use --rebuild")
    elif row_count or manifest.chunk_count:
        connection.close()
        raise ValueError("Partial checkpoint is missing its FAISS index; use --rebuild")

    splitter = create_text_splitter(config.embedding_model, config.chunk_size, config.chunk_overlap)
    embedding_model = SentenceTransformer(config.embedding_model)
    next_integer_id = manifest.chunk_count
    total_documents = len(selected)
    checkpoint_size = max(1, config.checkpoint_every)

    try:
        for start in range(manifest.processed_documents, total_documents, checkpoint_size):
            end = min(start + checkpoint_size, total_documents)
            chunks: list[ChunkRecord] = []
            for row in selected.select(range(start, end)):
                document = _document_from_row(dict(row))
                for chunk in chunk_document(document, splitter):
                    chunks.append(
                        ChunkRecord(
                            integer_id=next_integer_id,
                            chunk_id=chunk.chunk_id,
                            doc_id=chunk.doc_id,
                            source_type=chunk.source_type,
                            title=chunk.title,
                            content=chunk.content,
                            content_hash=chunk.content_hash,
                        )
                    )
                    next_integer_id += 1

            if chunks:
                vectors = np.asarray(
                    _encode_chunks(embedding_model, chunks, config.embedding_batch_size),
                    dtype="float32",
                )
                if index is None:
                    manifest.embedding_dimension = int(vectors.shape[1])
                    index = faiss.IndexIDMap2(faiss.IndexFlatIP(manifest.embedding_dimension))
                index.add_with_ids(
                    vectors,
                    np.asarray([chunk.integer_id for chunk in chunks], dtype="int64"),
                )
                connection.executemany(
                    """
                    INSERT INTO chunks
                    (integer_id, chunk_id, doc_id, source_type, title, content, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.integer_id,
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.source_type,
                            chunk.title,
                            chunk.content,
                            chunk.content_hash,
                        )
                        for chunk in chunks
                    ],
                )

            manifest.processed_documents = end
            manifest.chunk_count = next_integer_id
            connection.commit()
            if index is not None:
                temporary_index = faiss_path.with_suffix(".tmp")
                faiss.write_index(index, str(temporary_index))
                temporary_index.replace(faiss_path)
            _write_manifest(index_dir, manifest)

        if index is None:
            raise ValueError("No indexable chunks were produced from the selected documents")
        manifest.status = "complete"
        _write_manifest(index_dir, manifest)
        return validate_index(index_dir)
    finally:
        connection.close()
