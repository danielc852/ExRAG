"""Public API for constructing and running Enterprise RAG agents."""

from agent.agent import (
    SYSTEM_PROMPT,
    create_deep_agent_baseline,
    create_rag_agent,
    create_simple_agent,
)
from agent.llm import DEFAULT_OLLAMA_MODEL, create_ollama_model
from agent.runtime import agent_result, run_agent
from agent.state import AgentMode, AgentRunResult, ToolCallTrace
from agent.tools import create_agent_tools

__all__ = [
    "AgentMode",
    "AgentRunResult",
    "DEFAULT_OLLAMA_MODEL",
    "SYSTEM_PROMPT",
    "ToolCallTrace",
    "create_agent_tools",
    "create_deep_agent_baseline",
    "create_ollama_model",
    "create_rag_agent",
    "create_simple_agent",
    "agent_result",
    "run_agent",
]
