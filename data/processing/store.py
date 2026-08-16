"""Initialize and persist the FAISS vector store and SQLite metadata store."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


FAISS_NAME = "chunks.faiss"
SQLITE_NAME = "chunks.sqlite3"


def connect_metadata_store(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
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
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_type)"
    )
    return connection


def load_or_create_vector_store(path: Path, dimension: int) -> Any:
    import faiss

    if path.exists():
        return faiss.read_index(str(path))
    return faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))


def save_vector_store(index: Any, path: Path) -> None:
    import faiss

    temporary = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(temporary))
    temporary.replace(path)
