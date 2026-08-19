from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

import eval.dataset.snapshot as dataset_snapshot_module
import eval.experiment as experiment_module
from agent import AgentRunResult
from data import BenchmarkQuestion, StageManifest
from eval.dataset import sample_data
from eval import (
    LangSmithDatasetConfig,
    LangSmithExperimentConfig,
    build_langsmith_target,
    compare_experiments,
    create_snapshot_name,
    get_eval_mode,
    get_questions,
    question_to_example,
    run_langsmith_experiment,
    sync_frozen_dataset,
)


def question(question_id: str = "q1") -> BenchmarkQuestion:
    return BenchmarkQuestion(
        question_id=question_id,
        question_type="semantic",
        source_types=["slack"],
        question=f"Question {question_id}?",
        expected_doc_ids=["d1", "d2"],
        gold_answer="Gold answer",
        answer_facts=["Fact one"],
    )


def source_manifest() -> StageManifest:
    return StageManifest(
        stage="download",
        status="complete",
        config={},
        config_hash="source-config",
        output_fingerprint="abcdef1234567890",
        metadata={
            "dataset_revision": "commit",
            "questions_fingerprint": "questions",
            "corpus_mode": "sample",
            "sample_question_limit": 2,
        },
    )


class FakeDatasetClient:
    def __init__(self):
        self.datasets: dict[str, SimpleNamespace] = {}
        self.examples: dict[UUID, SimpleNamespace] = {}

    def has_dataset(self, *, dataset_name):
        return dataset_name in self.datasets

    def create_dataset(self, dataset_name, *, description, metadata):
        dataset = SimpleNamespace(
            id=uuid4(),
            name=dataset_name,
            description=description,
            metadata=dict(metadata),
        )
        self.datasets[dataset_name] = dataset
        return dataset

    def read_dataset(self, *, dataset_name):
        return self.datasets[dataset_name]

    def create_examples(self, *, dataset_id, examples):
        for payload in examples:
            self.examples[payload["id"]] = SimpleNamespace(
                id=payload["id"],
                dataset_id=dataset_id,
                inputs=dict(payload["inputs"]),
                outputs=dict(payload["outputs"]),
                metadata=dict(payload["metadata"]),
            )

    def list_examples(self, dataset_id=None, example_ids=None):
        allowed = set(example_ids or [])
        return iter(
            example
            for example in self.examples.values()
            if (dataset_id is None or example.dataset_id == dataset_id)
            and (not allowed or example.id in allowed)
        )


@pytest.fixture
def frozen_questions(monkeypatch):
    questions = [question("q1"), question("q2"), question("q3")]
    manifest = source_manifest()
    monkeypatch.setattr(
        dataset_snapshot_module, "_source_manifest", lambda _config: manifest
    )
    monkeypatch.setattr(
        dataset_snapshot_module, "load_frozen_questions", lambda _root: questions
    )
    return questions, manifest


def test_question_example_keeps_gold_data_out_of_inputs():
    payload = question_to_example(
        question(),
        ordinal=3,
        dataset_name="dataset-snapshot",
        source_fingerprint="fingerprint",
    )
    assert payload["inputs"] == {"question_id": "q1", "question": "Question q1?"}
    assert "gold_answer" not in payload["inputs"]
    assert payload["outputs"]["gold_answer"] == "Gold answer"
    assert payload["metadata"]["ordinal"] == 3
    assert payload["metadata"]["dataset_type"] == "test"
    assert payload["split"] == "test"


def test_create_snapshot_name_includes_dataset_type_and_fingerprint():
    assert (
        create_snapshot_name("EnterpriseRAG-Bench", "abcdef1234567890", "sample")
        == "EnterpriseRAG-Bench-sample-abcdef123456"
    )


def test_eval_mode_and_questions_follow_source_manifest():
    questions = [question("q1"), question("q2"), question("q3")]
    sample = source_manifest()
    sample.metadata["sample_question_limit"] = 2
    assert get_eval_mode(sample) == "sample"
    assert [item.question_id for item in get_questions(questions, sample)] == [
        "q1",
        "q2",
    ]

    full = source_manifest()
    full.metadata["corpus_mode"] = "full"
    full.metadata["sample_question_limit"] = None
    assert get_eval_mode(full) == "test"
    assert get_questions(questions, full) == questions


def test_sample_module_returns_its_question_scope():
    questions = [question("q1"), question("q2"), question("q3")]
    source = source_manifest()

    assert sample_data.get_sample_questions(questions, source) == questions[:2]


def test_dataset_sync_is_idempotent_and_repairs_partial_sync(
    tmp_path, frozen_questions
):
    client = FakeDatasetClient()
    config = LangSmithDatasetConfig(
        artifact_root=tmp_path,
        dataset_name="EnterpriseRAG-Bench",
    )
    created = sync_frozen_dataset(client, config)
    unchanged = sync_frozen_dataset(client, config)
    assert created.status == "created"
    assert created.created_examples == 2
    assert created.dataset_name == "EnterpriseRAG-Bench-sample-abcdef123456"
    assert created.dataset_type == "sample"
    assert unchanged.status == "unchanged"
    assert unchanged.created_examples == 0

    client.examples.pop(next(iter(client.examples)))
    repaired = sync_frozen_dataset(client, config)
    assert repaired.status == "updated"
    assert repaired.created_examples == 1


def test_dataset_sync_rejects_conflicting_examples(tmp_path, frozen_questions):
    client = FakeDatasetClient()
    config = LangSmithDatasetConfig(artifact_root=tmp_path)
    sync_frozen_dataset(client, config)
    existing = next(iter(client.examples.values()))
    existing.inputs["question"] = "changed"
    with pytest.raises(ValueError, match="conflicting examples"):
        sync_frozen_dataset(client, config)


def test_langsmith_target_returns_structured_agent_result(monkeypatch):
    captured = {}

    def fake_run_agent(_agent, question_text, **kwargs):
        captured.update({"question": question_text, **kwargs})
        return AgentRunResult(
            question_id=kwargs["question_id"],
            question=question_text,
            answer="Answer",
            document_ids=["d1"],
            latency_ms=4.0,
            model_name=kwargs["model_name"],
            agent_mode=kwargs["mode"],
        )

    monkeypatch.setattr(experiment_module, "run_agent", fake_run_agent)
    config = LangSmithExperimentConfig(question_limit=1)
    target = build_langsmith_target(object(), config)
    output = target({"question_id": "q1", "question": "Question?"})
    assert output["answer"] == "Answer"
    assert output["document_ids"] == ["d1"]
    assert captured["question_id"] == "q1"


def test_langsmith_target_dispatches_direct_model_baseline(monkeypatch):
    captured = {}

    def fake_run_baseline(_model, question_text, **kwargs):
        captured.update({"question": question_text, **kwargs})
        return AgentRunResult(
            question_id=kwargs["question_id"],
            question=question_text,
            answer="Baseline answer",
            latency_ms=3.0,
            model_name=kwargs["model_name"],
            agent_mode="baseline",
        )

    monkeypatch.setattr(experiment_module, "run_baseline", fake_run_baseline)
    config = LangSmithExperimentConfig(agent_mode="baseline", question_limit=1)
    target = build_langsmith_target(object(), config)
    output = target({"question_id": "q1", "question": "Question?"})

    assert output["answer"] == "Baseline answer"
    assert output["agent_mode"] == "baseline"
    assert captured == {
        "question": "Question?",
        "model_name": config.model_name,
        "question_id": "q1",
    }


class FakeExperimentResults:
    def __init__(self, rows):
        self.rows = rows
        self.experiment_name = "exrag-simple-test-1234"
        self.experiment_id = uuid4()
        self.url = "https://smith.example/experiment"

    def __iter__(self):
        return iter(self.rows)


class FakeExperimentClient(FakeDatasetClient):
    def __init__(self, dataset, example):
        super().__init__()
        self.datasets[dataset.name] = dataset
        self.examples[example.id] = example
        self.evaluate_kwargs = None

    def evaluate(self, target, **kwargs):
        self.evaluate_kwargs = kwargs
        example = kwargs["data"][0]
        outputs = target(example.inputs)
        run = SimpleNamespace(
            id=uuid4(),
            inputs=example.inputs,
            outputs=outputs,
            reference_example_id=example.id,
            error=None,
        )
        return FakeExperimentResults(
            [
                {
                    "run": run,
                    "evaluation_results": {"results": []},
                }
            ]
        )


def test_experiment_runner_wires_langsmith_and_writes_artifacts(
    tmp_path, monkeypatch
):
    source = source_manifest()
    benchmark_question = question()
    snapshot_name = "EnterpriseRAG-Bench-sample-abcdef123456"
    dataset = SimpleNamespace(
        id=uuid4(), name=snapshot_name, metadata={"dataset_type": "sample"}
    )
    example_payload = question_to_example(
        benchmark_question,
        ordinal=0,
        dataset_name=snapshot_name,
        source_fingerprint=source.output_fingerprint,
        dataset_type="sample",
    )
    example = SimpleNamespace(
        id=example_payload["id"],
        dataset_id=dataset.id,
        inputs=example_payload["inputs"],
        outputs=example_payload["outputs"],
        metadata=example_payload["metadata"],
    )
    client = FakeExperimentClient(dataset, example)
    index = StageManifest(
        stage="index",
        status="complete",
        config={},
        config_hash="index-config",
        output_fingerprint="index-fingerprint",
        metadata={
            "source_fingerprint": source.output_fingerprint,
            "embedding_fingerprint": "embedding",
            "embedding_model": "fake-embedding",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "corpus_mode": "sample",
        },
    )
    monkeypatch.setattr(
        experiment_module,
        "load_dataset_snapshot",
        lambda _client, _config: (
            dataset,
            source,
            [benchmark_question],
            snapshot_name,
        ),
    )
    monkeypatch.setattr(experiment_module, "validate_index", lambda _root: index)
    monkeypatch.setattr(
        experiment_module,
        "run_agent",
        lambda _agent, question_text, **kwargs: AgentRunResult(
            question_id=kwargs["question_id"],
            question=question_text,
            answer="Answer",
            document_ids=["d1"],
            latency_ms=5.0,
            model_name=kwargs["model_name"],
            agent_mode=kwargs["mode"],
        ),
    )
    config = LangSmithExperimentConfig(
        artifact_root=tmp_path / "artifacts",
        question_limit=1,
        output_root=tmp_path / "runs",
    )
    result = run_langsmith_experiment(client, config, object())
    assert client.evaluate_kwargs["blocking"] is True
    assert client.evaluate_kwargs["max_concurrency"] == 1
    assert client.evaluate_kwargs["metadata"]["dataset_type"] == "sample"
    assert client.evaluate_kwargs["metadata"]["llm_provider"] == "ollama"
    assert "evaluators" not in client.evaluate_kwargs
    assert "summary_evaluators" not in client.evaluate_kwargs
    assert result.summary.completed == 1
    assert (result.output_dir / "answers.jsonl").exists()
    assert (result.output_dir / "records.jsonl").exists()
    assert (result.output_dir / "summary.json").exists()


class FakeComparisonClient:
    def __init__(self, *, mismatch=False):
        base_metadata = {
            "dataset_name": "dataset",
            "source_fingerprint": "source",
            "index_fingerprint": "index",
            "model_name": "model",
            "top_k": 5,
        }
        self.projects = {
            "a": SimpleNamespace(id=uuid4(), extra={"metadata": base_metadata}),
            "b": SimpleNamespace(
                id=uuid4(),
                extra={
                    "metadata": {
                        **base_metadata,
                        "top_k": 10 if mismatch else 5,
                    }
                },
            ),
        }
        self.runs = {
            "a": [self._run("q1", None)],
            "b": [self._run("q1", "failed")],
        }
        self.feedback = {
            str(self.runs["a"][0].id): 1.0,
            str(self.runs["b"][0].id): 0.5,
        }

    @staticmethod
    def _run(question_id, error):
        return SimpleNamespace(
            id=uuid4(),
            inputs={"question_id": question_id, "question": "Question?"},
            outputs={
                "question_id": question_id,
                "answer": "Answer" if not error else "",
                "document_ids": ["d1"],
                "tool_calls": [],
                "latency_ms": 10.0,
                "input_tokens": 5,
                "output_tokens": 2,
                "error": error,
            },
            reference_example_id=uuid4(),
            error=None,
        )

    def read_project(self, *, project_name, include_stats):
        assert include_stats is True
        return self.projects[project_name]

    def list_runs(self, *, project_name, is_root):
        assert is_root is True
        return iter(self.runs[project_name])

    def list_feedback(self, *, run_ids):
        return iter(
            SimpleNamespace(
                run_id=run_id,
                key="document_recall",
                score=self.feedback[str(run_id)],
                value=None,
            )
            for run_id in run_ids
        )


def test_compare_experiments_reports_deltas_and_regressions(tmp_path):
    report = compare_experiments(
        FakeComparisonClient(),
        "a",
        "b",
        output_root=tmp_path,
    )
    assert report.metrics["mean_document_recall"].delta_b_minus_a == -0.5
    assert report.regression_question_ids == ["q1"]
    assert report.output_path and report.output_path.exists()


def test_compare_experiments_rejects_incompatible_metadata(tmp_path):
    with pytest.raises(ValueError, match="top_k"):
        compare_experiments(
            FakeComparisonClient(mismatch=True),
            "a",
            "b",
            output_root=tmp_path,
        )
