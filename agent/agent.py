"""Factories for the simple and deep Enterprise RAG agents."""

from __future__ import annotations

from agent.state import AgentMode
from agent.tools import create_agent_tools


SYSTEM_PROMPT = """You are an enterprise knowledge assistant being evaluated on grounded RAG.
Always search the enterprise corpus before answering. Base the answer only on retrieved
evidence. If sources conflict, describe the conflict and identify which source says what.
If evidence is insufficient, say that the information was not found. Do not invent facts,
document identifiers, or citations. Answer directly and preserve important qualifiers.
"""


def create_simple_agent(model, retrieval_tool):
    """Create the bounded LangChain baseline agent."""
    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolCallLimitMiddleware

    return create_agent(
        model=model,
        tools=create_agent_tools(retrieval_tool),
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="retrieve_documents", run_limit=3, exit_behavior="continue"
            )
        ],
        name="enterprise_rag_simple",
    )


def create_deep_agent_baseline(model, retrieval_tool):
    """Create the bounded Deep Agents baseline."""
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

    return create_deep_agent(
        model=model,
        tools=create_agent_tools(retrieval_tool),
        system_prompt=SYSTEM_PROMPT,
        backend=StateBackend(),
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="retrieve_documents", run_limit=8, exit_behavior="continue"
            ),
            ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
        ],
    )


def create_rag_agent(mode: AgentMode, model, retrieval_tool):
    """Create the agent implementation selected by ``mode``."""
    if mode == "simple":
        return create_simple_agent(model, retrieval_tool)
    if mode == "deep":
        return create_deep_agent_baseline(model, retrieval_tool)
    raise ValueError(f"Unsupported agent mode: {mode}")
