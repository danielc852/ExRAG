"""Tool registry exposed to ExRAG agents."""

from __future__ import annotations

from typing import Any


def create_agent_tools(retrieval_tool: Any) -> list[Any]:
    """Return the tools available to the agent for one configured run."""
    return [retrieval_tool]
