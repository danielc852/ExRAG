"""Build and execute LangSmith experiments against the RAG agents."""

from __future__ import annotations

import subprocess
from typing import Any, Callable

from agent import run_agent
from data import validate_index

from .datasets import get_snapshot_id, load_dataset_snapshot
from .evaluators import deterministic_evaluator, deterministic_summary_evaluator
from .runner import select_questions
from .models import (
    LangSmithDatasetConfig,
    LangSmithExperimentConfig,
    LangSmithExperimentResult,
)
from .results import normalize_experiment_results, safe_name, write_experiment_artifacts


def build_langsmith_target(
    agent: Any,
    config: LangSmithExperimentConfig,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question_id = str(inputs.get("question_id") or "")
        question = str(inputs.get("question") or "")
        result = run_agent(
            agent,
            question,
            mode=config.agent_mode,
            model_name=config.model_name,
            question_id=question_id or None,
        )
        return result.model_dump(mode="json")

    return target


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _experiment_metadata(
    config: LangSmithExperimentConfig,
    *,
    dataset: Any,
    source: Any,
    index: Any,
    question_ids: list[str],
) -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "dataset_id": str(dataset.id),
        "dataset_name": str(dataset.name),
        "dataset_type": dataset.metadata.get("dataset_type"),
        "source_fingerprint": source.output_fingerprint,
        "index_fingerprint": index.output_fingerprint,
        "embedding_fingerprint": index.metadata.get("embedding_fingerprint"),
        "documents_fingerprint": index.metadata.get("documents_fingerprint"),
        "dataset_revision": index.metadata.get("dataset_revision"),
        "corpus_mode": index.metadata.get("corpus_mode"),
        "embedding_model": index.metadata.get("embedding_model"),
        "chunk_size": index.metadata.get("chunk_size"),
        "chunk_overlap": index.metadata.get("chunk_overlap"),
        "agent_mode": config.agent_mode,
        "llm_provider": config.llm_provider,
        "model_name": config.model_name,
        "top_k": config.top_k,
        "question_limit": config.question_limit,
        "question_types": config.question_types,
        "question_ids": question_ids,
        "question_count": len(question_ids),
        "max_concurrency": config.max_concurrency,
    }


def run_langsmith_experiment(
    client: Any,
    config: LangSmithExperimentConfig,
    agent: Any,
) -> LangSmithExperimentResult:
    dataset_config = LangSmithDatasetConfig(
        artifact_root=config.artifact_root,
        dataset_name=config.dataset_name,
    )
    dataset, source, questions, snapshot_name = load_dataset_snapshot(
        client, dataset_config
    )
    index = validate_index(config.artifact_root)
    if index.metadata.get("source_fingerprint") != source.output_fingerprint:
        raise ValueError("Index and LangSmith dataset do not share the same source lineage")
    selected = select_questions(
        questions,
        limit=config.question_limit,
        question_types=config.question_types,
    )
    example_ids = [
        get_snapshot_id(snapshot_name, question.question_id)
        for question in selected
    ]
    examples_by_id = {
        example.id: example
        for example in client.list_examples(
            dataset_id=dataset.id,
            example_ids=example_ids,
        )
    }
    missing = [
        example_id for example_id in example_ids if example_id not in examples_by_id
    ]
    if missing:
        raise ValueError(
            f"LangSmith dataset is missing {len(missing)} selected examples; run sync again"
        )
    examples = [examples_by_id[example_id] for example_id in example_ids]
    question_ids = [question.question_id for question in selected]
    metadata = _experiment_metadata(
        config,
        dataset=dataset,
        source=source,
        index=index,
        question_ids=question_ids,
    )
    prefix = config.experiment_prefix or (
        f"exrag-{config.agent_mode}-{safe_name(config.model_name, 'model')}"
    )
    result = client.evaluate(
        build_langsmith_target(agent, config),
        data=examples,
        evaluators=[deterministic_evaluator],
        summary_evaluators=[deterministic_summary_evaluator],
        metadata=metadata,
        experiment_prefix=prefix,
        description=(
            "EnterpriseRAG-Bench deterministic LangSmith evaluation; answer quality "
            "remains subject to the official benchmark evaluator"
        ),
        max_concurrency=config.max_concurrency,
        blocking=True,
        error_handling="log",
    )
    ordinal_by_question_id = {
        question.question_id: ordinal for ordinal, question in enumerate(selected)
    }
    records = normalize_experiment_results(
        list(result), ordinal_by_question_id=ordinal_by_question_id
    )
    output_dir, summary, experiment_url = write_experiment_artifacts(
        result=result,
        dataset=dataset,
        config=config,
        metadata=metadata,
        records=records,
    )
    return LangSmithExperimentResult(
        experiment_name=summary.experiment_name,
        experiment_id=summary.experiment_id,
        experiment_url=experiment_url,
        output_dir=output_dir,
        summary=summary,
    )
