"""Agent execution and provider-message normalization."""

from __future__ import annotations

import time
from typing import Any

from agent.helpers import _artifact_payload, _message_content
from agent.state import AgentMode, AgentRunResult, ToolCallTrace


def agent_result(
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
    """Run one question and return a stable success or error result."""
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
        return agent_result(
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
