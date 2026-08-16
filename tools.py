"""Retrieval implementation and LangChain tool adapter."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from data.artifacts import ArtifactLayout
from data.indexing import FAISS_NAME, SQLITE_NAME, validate_index


class RetrievalFilters(BaseModel):
    source_types: list[str] | None = None
    document_ids: list[str] | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_type: str
    title: str
    content: str
    similarity_score: float


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    latency_ms: float


class FaissRetriever:
    """A normalized dense retriever backed by FAISS and SQLite."""

    def __init__(
        self,
        *,
        index: Any,
        connection: sqlite3.Connection,
        embedding_model: Any,
        max_top_k: int = 20,
    ) -> None:
        self.index = index
        self.connection = connection
        self.embedding_model = embedding_model
        self.max_top_k = max_top_k
        self._database_lock = threading.RLock()

    @classmethod
    def load(cls, artifact_root: Path) -> "FaissRetriever":
        import faiss
        from sentence_transformers import SentenceTransformer

        layout = ArtifactLayout(artifact_root)
        manifest = validate_index(artifact_root)
        index = faiss.read_index(str(layout.index / FAISS_NAME))
        connection = sqlite3.connect(
            layout.index / SQLITE_NAME,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        model_name = str(manifest.metadata.get("embedding_model", ""))
        if not model_name:
            connection.close()
            raise ValueError("Index manifest does not identify its embedding model")
        revision = manifest.metadata.get("embedding_revision")
        kwargs = {"revision": revision} if revision else {}
        embedding_model = SentenceTransformer(model_name, **kwargs)
        return cls(index=index, connection=connection, embedding_model=embedding_model)

    def _matching_rows(
        self,
        integer_ids: list[int],
        scores: dict[int, float],
        filters: RetrievalFilters | None,
    ) -> list[RetrievedChunk]:
        if not integer_ids:
            return []
        placeholders = ",".join("?" for _ in integer_ids)
        clauses = [f"integer_id IN ({placeholders})"]
        parameters: list[Any] = list(integer_ids)
        if filters and filters.source_types:
            source_placeholders = ",".join("?" for _ in filters.source_types)
            clauses.append(f"source_type IN ({source_placeholders})")
            parameters.extend(filters.source_types)
        if filters and filters.document_ids:
            doc_placeholders = ",".join("?" for _ in filters.document_ids)
            clauses.append(f"doc_id IN ({doc_placeholders})")
            parameters.extend(filters.document_ids)
        with self._database_lock:
            rows = self.connection.execute(
                f"SELECT * FROM chunks WHERE {' AND '.join(clauses)}", parameters
            ).fetchall()
        by_id = {int(row["integer_id"]): row for row in rows}
        output: list[RetrievedChunk] = []
        for integer_id in integer_ids:
            row = by_id.get(integer_id)
            if row is None:
                continue
            output.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["doc_id"],
                    source_type=row["source_type"],
                    title=row["title"],
                    content=row["content"],
                    similarity_score=scores[integer_id],
                )
            )
        return output

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        import numpy as np

        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if top_k < 1 or top_k > self.max_top_k:
            raise ValueError(f"top_k must be between 1 and {self.max_top_k}")

        started = time.perf_counter()
        vector = self.embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        vector = np.asarray(vector, dtype="float32")
        candidate_count = min(max(top_k * 5, top_k), int(self.index.ntotal))
        chunks: list[RetrievedChunk] = []
        while candidate_count:
            raw_scores, raw_ids = self.index.search(vector, candidate_count)
            integer_ids = [int(value) for value in raw_ids[0] if int(value) >= 0]
            scores = {
                int(integer_id): float(score)
                for integer_id, score in zip(raw_ids[0], raw_scores[0])
                if int(integer_id) >= 0
            }
            chunks = self._matching_rows(integer_ids, scores, filters)[:top_k]
            if len(chunks) >= top_k or candidate_count >= self.index.ntotal:
                break
            candidate_count = min(candidate_count * 2, int(self.index.ntotal))

        return RetrievalResult(
            query=query,
            chunks=chunks,
            latency_ms=(time.perf_counter() - started) * 1_000,
        )

    def close(self) -> None:
        with self._database_lock:
            self.connection.close()

    def __enter__(self) -> "FaissRetriever":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def format_chunks_for_agent(result: RetrievalResult) -> str:
    if not result.chunks:
        return "No matching documents were found."
    sections = []
    for position, chunk in enumerate(result.chunks, start=1):
        sections.append(
            "\n".join(
                [
                    f"[Result {position}]",
                    f"Document ID: {chunk.document_id}",
                    f"Chunk ID: {chunk.chunk_id}",
                    f"Source: {chunk.source_type}",
                    f"Title: {chunk.title}",
                    f"Similarity: {chunk.similarity_score:.4f}",
                    "Content:",
                    chunk.content,
                ]
            )
        )
    return "\n\n---\n\n".join(sections)


def create_retrieval_tool(
    retriever: FaissRetriever, *, default_top_k: int = 5, max_top_k: int = 20
):
    """Create a tool whose artifact retains structured retrieval evidence."""
    from langchain.tools import tool

    if default_top_k < 1 or default_top_k > max_top_k:
        raise ValueError("default_top_k must be within the allowed top-k range")
    retriever.max_top_k = max_top_k

    @tool("retrieve_documents", response_format="content_and_artifact")
    def retrieve_documents(
        query: str,
        top_k: int = Field(default=default_top_k, ge=1, le=max_top_k),
        source_types: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Search enterprise documents for evidence relevant to a question.

        Use source_types or document_ids only when the user explicitly supplies those
        constraints. Never invent filters.
        """
        filters = RetrievalFilters(source_types=source_types, document_ids=document_ids)
        result = retriever.search(query, top_k=top_k, filters=filters)
        return format_chunks_for_agent(result), result.model_dump()

    return retrieve_documents
