from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pydantic import ValidationError

from tools import (
    FaissRetriever,
    RetrievalFilters,
    RetrievalResult,
    RetrieveDocumentsInput,
    RetrieveDocumentsOutput,
    RetrievedChunk,
    create_retrieval_tool,
    format_chunks_for_agent,
    load_tool_description,
)


class FakeEmbedding:
    def encode(self, _texts, **_kwargs):
        return np.asarray([[1.0, 0.0]], dtype="float32")


class FakeIndex:
    ntotal = 3

    def search(self, _vector, count):
        ids = np.asarray([[0, 1, 2][:count]], dtype="int64")
        scores = np.asarray([[0.9, 0.8, 0.7][:count]], dtype="float32")
        return scores, ids


def make_retriever(*, thread_safe=False):
    connection = sqlite3.connect(":memory:", check_same_thread=not thread_safe)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE chunks (
        integer_id INTEGER PRIMARY KEY, chunk_id TEXT, doc_id TEXT,
        source_type TEXT, title TEXT, content TEXT, content_hash TEXT)"""
    )
    connection.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (0, "d1::000000", "d1", "slack", "One", "alpha", "h1"),
            (1, "d2::000000", "d2", "gmail", "Two", "beta", "h2"),
            (2, "d3::000000", "d3", "slack", "Three", "gamma", "h3"),
        ],
    )
    return FaissRetriever(
        index=FakeIndex(), connection=connection, embedding_model=FakeEmbedding()
    )


def test_search_preserves_rank_and_applies_metadata_filter():
    retriever = make_retriever()
    result = retriever.search(
        "query", top_k=2, filters=RetrievalFilters(source_types=["slack"])
    )
    assert [chunk.document_id for chunk in result.chunks] == ["d1", "d3"]
    assert result.chunks[0].similarity_score == pytest.approx(0.9)
    retriever.close()


@pytest.mark.parametrize("query,top_k", [("", 1), ("valid", 0), ("valid", 21)])
def test_search_rejects_invalid_arguments(query, top_k):
    retriever = make_retriever()
    with pytest.raises(ValueError):
        retriever.search(query, top_k=top_k)
    retriever.close()


def test_agent_format_contains_citation_metadata():
    result = RetrievalResult(
        query="query",
        latency_ms=1,
        chunks=[
            RetrievedChunk(
                chunk_id="d1::000000",
                document_id="d1",
                source_type="slack",
                title="Title",
                content="Evidence",
                similarity_score=0.75,
            )
        ],
    )
    formatted = format_chunks_for_agent(result)
    assert "Document ID: d1" in formatted
    assert "Chunk ID: d1::000000" in formatted
    assert "Evidence" in formatted


def test_metadata_reads_are_safe_across_worker_threads():
    retriever = make_retriever(thread_safe=True)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(lambda _index: retriever.search("query", top_k=1), range(8))
            )
    finally:
        retriever.close()
    assert [result.chunks[0].document_id for result in results] == ["d1"] * 8


def test_tool_schema_defines_input_and_output_contracts():
    tool_input = RetrieveDocumentsInput(query="benefits policy")
    assert tool_input.top_k == 5
    assert RetrieveDocumentsOutput.model_validate(
        {"query": "benefits policy", "chunks": [], "latency_ms": 1.5}
    ).chunks == []
    with pytest.raises(ValidationError):
        RetrieveDocumentsInput(query="benefits policy", top_k=21)


def test_tool_uses_markdown_description_and_configured_schema():
    retriever = make_retriever()
    try:
        retrieval_tool = create_retrieval_tool(
            retriever, default_top_k=2, max_top_k=3
        )
        assert retrieval_tool.description == load_tool_description()
        schema = retrieval_tool.args_schema.model_json_schema()
        assert schema["properties"]["top_k"]["default"] == 2
        assert schema["properties"]["top_k"]["maximum"] == 3
    finally:
        retriever.close()
