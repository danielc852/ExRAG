from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import AgentRunResult
from data import BenchmarkQuestion, StageManifest
from eval import (
    EvaluationConfig,
    run_evaluation,
    _write_run_config,
    append_answer,
    document_recall,
    load_completed_question_ids,
    select_questions,
    strict_extra_document_count,
)


def question(question_id="q1", question_type="basic"):
    return BenchmarkQuestion(
        question_id=question_id,
        question_type=question_type,
        source_types=["slack"],
        question="Question?",
        expected_doc_ids=["d1", "d2"],
        gold_answer="Answer",
        answer_facts=["Fact"],
    )


def test_retrieval_metrics_deduplicate_and_handle_empty_gold():
    assert document_recall(["d1", "d1"], ["d1", "d2"]) == 0.5
    assert strict_extra_document_count(["d1", "d3", "d3"], ["d1", "d2"]) == 1
    assert document_recall(["d1"], []) is None
    assert strict_extra_document_count(["d1"], []) is None


def test_question_selection_filters_before_limiting():
    questions = [question("q1", "basic"), question("q2", "semantic"), question("q3", "semantic")]
    selected = select_questions(questions, limit=1, question_types=["semantic"])
    assert [item.question_id for item in selected] == ["q2"]


def test_question_selection_rejects_unknown_type():
    with pytest.raises(ValueError, match="Available types"):
        select_questions([question()], limit=1, question_types=["unknown"])


def test_completed_ids_ignore_truncated_jsonl(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    result = AgentRunResult(
        question_id="q1",
        question="Question?",
        answer="Answer",
        document_ids=["d1"],
        latency_ms=1,
        model_name="model",
        agent_mode="simple",
    )
    append_answer(path, question(), result)
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"question_id":')
    assert load_completed_question_ids(path) == {"q1"}


def index_manifest():
    return StageManifest(
        stage="index",
        status="complete",
        config={"batch_size": 10},
        config_hash="config",
        upstream_fingerprint="embed",
        output_fingerprint="index",
        stats={"chunk_count": 1},
        metadata={
            "documents_fingerprint": "fingerprint",
            "corpus_mode": "sample",
            "embedding_model": "embed",
            "chunk_size": 512,
            "chunk_overlap": 64,
        },
    )


def test_resume_rejects_changed_model(tmp_path: Path):
    manifest = index_manifest()
    first = EvaluationConfig(agent_mode="simple", output_dir=tmp_path, model_name="first")
    _write_run_config(first, manifest, 1)
    changed = EvaluationConfig(agent_mode="simple", output_dir=tmp_path, model_name="second")
    with pytest.raises(ValueError, match="model"):
        _write_run_config(changed, manifest, 1)


def test_resume_rejects_changed_llm_provider(tmp_path: Path):
    manifest = index_manifest()
    first = EvaluationConfig(
        agent_mode="simple",
        output_dir=tmp_path,
        llm_provider="ollama",
        model_name="shared-model-name",
    )
    _write_run_config(first, manifest, 1)
    changed = first.model_copy(update={"llm_provider": "openrouter"})

    with pytest.raises(ValueError, match="LLM provider"):
        _write_run_config(changed, manifest, 1)


def test_resume_treats_legacy_config_as_ollama(tmp_path: Path):
    manifest = index_manifest()
    config = EvaluationConfig(
        agent_mode="simple", output_dir=tmp_path, model_name="legacy-model"
    )
    _write_run_config(config, manifest, 1)
    path = tmp_path / "config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["evaluation"]["llm_provider"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    _write_run_config(config, manifest, 1)


def test_evaluation_resumes_completed_answers(tmp_path: Path):
    class FakeAgent:
        def __init__(self):
            self.calls = 0

        def invoke(self, _payload, config):
            self.calls += 1
            assert config["recursion_limit"] == 24
            return {
                "messages": [
                    SimpleNamespace(
                        type="ai",
                        content="Grounded answer",
                        tool_calls=[],
                        usage_metadata=None,
                    )
                ]
            }

    manifest = index_manifest()
    config = EvaluationConfig(
        agent_mode="simple", output_dir=tmp_path, question_limit=1, model_name="model"
    )
    fake_agent = FakeAgent()
    first = run_evaluation(config, fake_agent, [question()], manifest)
    second = run_evaluation(config, fake_agent, [question()], manifest)
    assert first.completed == 1
    assert second.attempted == 0
    assert fake_agent.calls == 1
    assert load_completed_question_ids(tmp_path / "answers.jsonl") == {"q1"}
