from __future__ import annotations

from types import SimpleNamespace

from agent import AgentRunResult, create_agent_tools, extract_agent_result
from agent.runtime import extract_agent_result as runtime_extract_agent_result
from agent.state import AgentRunResult as StateAgentRunResult


def test_agent_package_preserves_public_api_and_tool_registry():
    retrieval_tool = object()

    assert AgentRunResult is StateAgentRunResult
    assert extract_agent_result is runtime_extract_agent_result
    assert create_agent_tools(retrieval_tool) == [retrieval_tool]


def test_extract_agent_result_collects_stable_document_union_and_usage():
    messages = [
        SimpleNamespace(
            type="ai",
            content="",
            tool_calls=[{"id": "call-1", "name": "retrieve_documents", "args": {"query": "x"}}],
            usage_metadata={"input_tokens": 10, "output_tokens": 2},
        ),
        SimpleNamespace(
            type="tool",
            name="retrieve_documents",
            tool_call_id="call-1",
            artifact={
                "latency_ms": 4.2,
                "chunks": [
                    {"document_id": "d1"},
                    {"document_id": "d1"},
                    {"document_id": "d2"},
                ],
            },
        ),
        SimpleNamespace(
            type="ai",
            content="Grounded answer",
            tool_calls=[],
            usage_metadata={"input_tokens": 20, "output_tokens": 5},
        ),
    ]
    result = extract_agent_result(
        messages,
        question="Question?",
        mode="simple",
        model_name="model",
        latency_ms=8,
    )
    assert result.answer == "Grounded answer"
    assert result.document_ids == ["d1", "d2"]
    assert result.tool_calls[0].retrieved_document_ids == ["d1", "d2"]
    assert result.input_tokens == 30
    assert result.output_tokens == 7
