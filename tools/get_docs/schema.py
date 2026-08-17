"""Input and output contracts for the document retrieval tool."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model


SourceType = Literal[
    "confluence",
    "fireflies",
    "github",
    "gmail",
    "google_drive",
    "hubspot",
    "jira",
    "linear",
    "slack",
]


class RetrievalFilters(BaseModel):
    """Optional metadata constraints supplied by the user."""

    source_types: list[SourceType] | None = None
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
    source_types: list[SourceType] | None = Field(
        default=None,
        description=(
            "Source types explicitly requested by the user. Omit unless the user "
            "names an exact supported source type."
        ),
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Document IDs explicitly requested by the user.",
    )


class RetrieveDocumentsQueryInput(BaseModel):
    """Retrieval contract for benchmark questions without user-supplied filters."""

    query: str = Field(description="The question or search phrase to retrieve evidence for.")
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of matching chunks to return.",
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
    *, default_top_k: int, max_top_k: int, include_filters: bool = True
) -> type[BaseModel]:
    """Build an input contract matching a configured tool instance."""
    base = RetrieveDocumentsInput if include_filters else RetrieveDocumentsQueryInput
    return create_model(
        "RetrieveDocumentsInput",
        __base__=base,
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
