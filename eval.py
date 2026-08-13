"""Resumable EnterpriseRAG-Bench answer generation and local metrics."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from pydantic import BaseModel, Field

from agent import AgentMode, AgentRunResult, run_agent
from dataset import BenchmarkQuestion, IndexManifest


class EvaluationConfig(BaseModel):
    agent_mode: AgentMode
    output_dir: Path
    question_limit: int | None = None
    question_types: list[str] | None = None
    resume: bool = True
    model_name: str = "qwen3:8b"


class EvaluationSummary(BaseModel):
    attempted: int
    completed: int
    failed: int
    mean_document_recall: float | None
    mean_strict_extra_documents: float | None
    mean_latency_ms: float
    mean_tool_calls: float
    benchmark_comparable: bool
    note: str = Field(
        default=(
            "strict_extra_documents only compares against gold IDs and is not the "
            "official LLM-judged Invalid Extra Documents metric"
        )
    )


def select_questions(
    questions: list[BenchmarkQuestion],
    *,
    limit: int | None,
    question_types: list[str] | None,
) -> list[BenchmarkQuestion]:
    if limit is not None and limit < 1:
        raise ValueError("question limit must be at least 1")
    allowed = set(question_types or [])
    selected = [question for question in questions if not allowed or question.question_type in allowed]
    if allowed and not selected:
        available = ", ".join(sorted({question.question_type for question in questions}))
        raise ValueError(f"No questions matched {sorted(allowed)}. Available types: {available}")
    return selected[:limit] if limit is not None else selected


def evaluate_question(
    agent,
    question: BenchmarkQuestion,
    *,
    mode: AgentMode,
    model_name: str,
) -> AgentRunResult:
    return run_agent(
        agent,
        question.question,
        mode=mode,
        model_name=model_name,
        question_id=question.question_id,
    )


def document_recall(retrieved_ids: list[str], expected_ids: list[str]) -> float | None:
    expected = set(expected_ids)
    if not expected:
        return None
    return len(expected.intersection(retrieved_ids)) / len(expected)


def strict_extra_document_count(
    retrieved_ids: list[str], expected_ids: list[str]
) -> int | None:
    if not expected_ids:
        return None
    return len(set(retrieved_ids).difference(expected_ids))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def load_completed_question_ids(path: Path) -> set[str]:
    return {
        str(row["question_id"])
        for row in _read_jsonl(path)
        if isinstance(row.get("question_id"), str)
    }


def append_answer(
    path: Path, question: BenchmarkQuestion, result: AgentRunResult
) -> None:
    _append_jsonl(
        path,
        {
            "question_id": question.question_id,
            "answer": result.answer,
            "document_ids": result.document_ids,
        },
    )


def append_run_detail(
    path: Path, question: BenchmarkQuestion, result: AgentRunResult
) -> None:
    payload = result.model_dump(mode="json")
    payload.update(
        {
            "question_type": question.question_type,
            "expected_document_ids": question.expected_doc_ids,
            "document_recall": document_recall(result.document_ids, question.expected_doc_ids),
            "strict_extra_documents": strict_extra_document_count(
                result.document_ids, question.expected_doc_ids
            ),
        }
    )
    _append_jsonl(path, payload)


def write_summary(path: Path, summary: EvaluationSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def _mean_optional(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return fmean(values) if values else None


def _write_run_config(
    config: EvaluationConfig,
    manifest: IndexManifest,
    selected_count: int,
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": config.model_dump(mode="json"),
        "selected_question_count": selected_count,
        "index": asdict(manifest),
    }
    path = config.output_dir / "config.json"
    if config.resume and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_evaluation = existing.get("evaluation", {})
        existing_index = existing.get("index", {})
        checks = {
            "agent mode": (existing_evaluation.get("agent_mode"), config.agent_mode),
            "model": (existing_evaluation.get("model_name"), config.model_name),
            "question limit": (
                existing_evaluation.get("question_limit"),
                config.question_limit,
            ),
            "question types": (
                existing_evaluation.get("question_types"),
                config.question_types,
            ),
            "selected question count": (
                existing.get("selected_question_count"),
                selected_count,
            ),
            "dataset fingerprint": (
                existing_index.get("dataset_fingerprint"),
                manifest.dataset_fingerprint,
            ),
            "embedding model": (
                existing_index.get("embedding_model"),
                manifest.embedding_model,
            ),
            "chunk size": (existing_index.get("chunk_size"), manifest.chunk_size),
            "chunk overlap": (
                existing_index.get("chunk_overlap"),
                manifest.chunk_overlap,
            ),
        }
        mismatches = [name for name, values in checks.items() if values[0] != values[1]]
        if mismatches:
            raise ValueError(
                "Cannot resume into this output directory because these settings changed: "
                + ", ".join(mismatches)
            )
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_evaluation(
    config: EvaluationConfig,
    agent,
    questions: list[BenchmarkQuestion],
    manifest: IndexManifest,
) -> EvaluationSummary:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = config.output_dir / "answers.jsonl"
    details_path = config.output_dir / "run_details.jsonl"
    errors_path = config.output_dir / "errors.jsonl"
    selected = select_questions(
        questions, limit=config.question_limit, question_types=config.question_types
    )
    _write_run_config(config, manifest, len(selected))

    completed_ids = load_completed_question_ids(answers_path) if config.resume else set()
    if not config.resume:
        for path in (answers_path, details_path, errors_path):
            path.unlink(missing_ok=True)

    attempted = 0
    failed = 0
    for question in selected:
        if question.question_id in completed_ids:
            continue
        attempted += 1
        result = evaluate_question(
            agent,
            question,
            mode=config.agent_mode,
            model_name=config.model_name,
        )
        if result.error:
            failed += 1
            _append_jsonl(
                errors_path,
                {
                    "question_id": question.question_id,
                    "question": question.question,
                    "error": result.error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            continue
        append_answer(answers_path, question, result)
        append_run_detail(details_path, question, result)

    detail_rows = _read_jsonl(details_path)
    latest_by_id = {
        str(row["question_id"]): row
        for row in detail_rows
        if isinstance(row.get("question_id"), str)
    }
    rows = [latest_by_id[key] for key in sorted(latest_by_id)]
    benchmark_comparable = (
        manifest.corpus_mode == "full"
        and config.question_limit is None
        and not config.question_types
        and len(rows) == len(questions)
        and failed == 0
    )
    summary = EvaluationSummary(
        attempted=attempted,
        completed=len(rows),
        failed=failed,
        mean_document_recall=_mean_optional(rows, "document_recall"),
        mean_strict_extra_documents=_mean_optional(rows, "strict_extra_documents"),
        mean_latency_ms=_mean_optional(rows, "latency_ms") or 0.0,
        mean_tool_calls=(
            fmean(len(row.get("tool_calls", [])) for row in rows) if rows else 0.0
        ),
        benchmark_comparable=benchmark_comparable,
    )
    write_summary(config.output_dir / "summary.json", summary)
    return summary
