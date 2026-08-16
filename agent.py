"""LangChain and Deep Agents RAG baselines."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


AgentMode = Literal["simple", "deep"]

SYSTEM_PROMPT = """You are an enterprise knowledge assistant being evaluated on grounded RAG.
Always search the enterprise corpus before answering. Base the answer only on retrieved
evidence. If sources conflict, describe the conflict and identify which source says what.
If evidence is insufficient, say that the information was not found. Do not invent facts,
document identifiers, or citations. Answer directly and preserve important qualifiers.
"""


class ToolCallTrace(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    latency_ms: float | None = None


class AgentRunResult(BaseModel):
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


def create_ollama_model(
    model_name: str = "qwen3:8b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0,
):
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model_name, base_url=base_url, temperature=temperature)


def create_simple_agent(model, retrieval_tool):
    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolCallLimitMiddleware

    return create_agent(
        model=model,
        tools=[retrieval_tool],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="retrieve_documents", run_limit=3, exit_behavior="continue"
            )
        ],
        name="enterprise_rag_simple",
    )


def create_deep_agent_baseline(model, retrieval_tool):
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

    return create_deep_agent(
        model=model,
        tools=[retrieval_tool],
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
    if mode == "simple":
        return create_simple_agent(model, retrieval_tool)
    if mode == "deep":
        return create_deep_agent_baseline(model, retrieval_tool)
    raise ValueError(f"Unsupported agent mode: {mode}")


def _message_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip() if content is not None else ""


def _artifact_payload(artifact: Any) -> dict[str, Any] | None:
    if artifact is None:
        return None
    if isinstance(artifact, dict):
        return artifact
    if hasattr(artifact, "model_dump"):
        return artifact.model_dump()
    return None


def extract_agent_result(
    messages: list[Any],
    *,
    question: str,
    mode: AgentMode,
    model_name: str,
    latency_ms: float,
    question_id: str | None = None,
) -> AgentRunResult:
    """Normalize provider messages and structured tool artifacts."""
    call_arguments: dict[str, tuple[str, dict[str, Any]]] = {}
    traces: list[ToolCallTrace] = []
    document_ids: list[str] = []
    input_tokens = 0
    output_tokens = 0
    usage_seen = False
    answer = ""

    for message in messages:
        for tool_call in getattr(message, "tool_calls", []) or []:
            if isinstance(tool_call, dict):
                call_arguments[str(tool_call.get("id", ""))] = (
                    str(tool_call.get("name", "")),
                    dict(tool_call.get("args") or {}),
                )

        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            usage_seen = True

        if getattr(message, "type", None) in {"ai", "assistant"}:
            text = _message_content(message)
            if text:
                answer = text

        if getattr(message, "type", None) != "tool":
            continue
        tool_call_id = str(getattr(message, "tool_call_id", ""))
        tool_name, arguments = call_arguments.get(
            tool_call_id, (str(getattr(message, "name", "")), {})
        )
        artifact = _artifact_payload(getattr(message, "artifact", None)) or {}
        retrieved_ids = []
        for chunk in artifact.get("chunks", []) or []:
            if not isinstance(chunk, dict):
                continue
            document_id = str(chunk.get("document_id", ""))
            if not document_id:
                continue
            retrieved_ids.append(document_id)
        document_ids.extend(retrieved_ids)
        traces.append(
            ToolCallTrace(
                tool_name=tool_name,
                arguments=arguments,
                retrieved_document_ids=list(dict.fromkeys(retrieved_ids)),
                latency_ms=float(artifact["latency_ms"]) if "latency_ms" in artifact else None,
            )
        )

    if not answer:
        raise ValueError("Agent returned no final text answer")
    return AgentRunResult(
        question_id=question_id,
        question=question,
        answer=answer,
        document_ids=list(dict.fromkeys(document_ids)),
        tool_calls=traces,
        latency_ms=latency_ms,
        model_name=model_name,
        agent_mode=mode,
        input_tokens=input_tokens if usage_seen else None,
        output_tokens=output_tokens if usage_seen else None,
    )


def run_agent(
    agent,
    question: str,
    *,
    mode: AgentMode,
    model_name: str,
    question_id: str | None = None,
) -> AgentRunResult:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    started = time.perf_counter()
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": 16 if mode == "deep" else 10},
        )
        messages = response.get("messages", []) if isinstance(response, dict) else []
        return extract_agent_result(
            list(messages),
            question=question,
            mode=mode,
            model_name=model_name,
            latency_ms=(time.perf_counter() - started) * 1_000,
            question_id=question_id,
        )
    except Exception as exc:
        return AgentRunResult(
            question_id=question_id,
            question=question,
            answer="",
            latency_ms=(time.perf_counter() - started) * 1_000,
            model_name=model_name,
            agent_mode=mode,
            error=f"{type(exc).__name__}: {exc}",
        )
