"""Input and output contracts for the document retrieval tool."""

from __future__ import annotations

from pydantic import BaseModel, Field, create_model


class RetrievalFilters(BaseModel):
    """Optional metadata constraints supplied by the user."""

    source_types: list[str] | None = None
    document_ids: list[str] | None = None


class RetrieveDocumentsInput(BaseModel):
    """Default input contract exposed to the agent."""

    query: str = Field(description="The question or search phrase to retrieve evidence for.")
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of matching chunks to return.",
    )
    source_types: list[str] | None = Field(
        default=None,
        description="Source types explicitly requested by the user.",
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Document IDs explicitly requested by the user.",
    )


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_type: str
    title: str
    content: str
    similarity_score: float


class RetrieveDocumentsOutput(BaseModel):
    """Structured retrieval evidence returned as the tool artifact."""

    query: str
    chunks: list[RetrievedChunk]
    latency_ms: float


# Preserve the public name used before tools became a package.
RetrievalResult = RetrieveDocumentsOutput


def create_input_schema(
    *, default_top_k: int, max_top_k: int
) -> type[RetrieveDocumentsInput]:
    """Build an input contract matching a configured tool instance."""
    return create_model(
        "RetrieveDocumentsInput",
        __base__=RetrieveDocumentsInput,
        top_k=(
            int,
            Field(
                default=default_top_k,
                ge=1,
                le=max_top_k,
                description="Maximum number of matching chunks to return.",
            ),
        ),
    )
