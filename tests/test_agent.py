from __future__ import annotations

from types import SimpleNamespace

from agent import (
    BASELINE_SYSTEM_PROMPT,
    AgentRunResult,
    agent_result,
    create_agent_tools,
    run_baseline,
)
from agent.runtime import agent_result as runtime_agent_result
from agent.state import AgentRunResult as StateAgentRunResult


def test_agent_package_preserves_public_api_and_tool_registry():
    retrieval_tool = object()

    assert AgentRunResult is StateAgentRunResult
    assert agent_result is runtime_agent_result
    assert create_agent_tools(retrieval_tool) == [retrieval_tool]


def test_agent_result_collects_stable_document_union_and_usage():
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
    result = agent_result(
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


def test_baseline_calls_model_directly_without_tools_or_documents():
    class FakeModel:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(
                type="ai",
                content="Direct answer",
                tool_calls=[],
                usage_metadata={"input_tokens": 12, "output_tokens": 3},
            )

    model = FakeModel()
    result = run_baseline(
        model,
        "  Question?  ",
        model_name="model",
        question_id="q1",
    )

    assert model.messages == [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": "Question?"},
    ]
    assert result.answer == "Direct answer"
    assert result.agent_mode == "baseline"
    assert result.document_ids == []
    assert result.tool_calls == []
    assert result.input_tokens == 12
    assert result.output_tokens == 3


def test_baseline_normalizes_model_errors():
    class FailingModel:
        def invoke(self, _messages):
            raise RuntimeError("provider unavailable")

    result = run_baseline(FailingModel(), "Question?", model_name="model")

    assert result.answer == ""
    assert result.agent_mode == "baseline"
    assert result.error == "RuntimeError: provider unavailable"
