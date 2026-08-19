"""State and result contracts shared by the agent runtime."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AgentMode = Literal["baseline", "simple", "deep"]


class ToolCallTrace(BaseModel):
    """Normalized trace data for one tool invocation."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    latency_ms: float | None = None


class AgentRunResult(BaseModel):
    """Stable result returned by every agent run."""

    question_id: str | None = None
    question: str
    answer: str
    document_ids: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    latency_ms: float
    model_name: str
    agent_mode: AgentMode
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
