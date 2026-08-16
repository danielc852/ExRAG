"""Normalize LangSmith experiments and write reproducible local artifacts."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from .evaluators import percentile
from .models import (
    ComparisonMetric,
    ComparisonReport,
    ExperimentRecord,
    LangSmithExperimentConfig,
    LangSmithSummary,
)


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _feedback_value(result: Any) -> Any:
    score = _attribute(result, "score")
    return score if score is not None else _attribute(result, "value")


def _feedback_from_result(item: Any) -> dict[str, Any]:
    evaluation_results = _attribute(item, "evaluation_results", {}) or {}
    results = _attribute(evaluation_results, "results", []) or []
    return {
        str(_attribute(result, "key")): _feedback_value(result)
        for result in results
        if _attribute(result, "key")
    }


def normalize_experiment_results(
    items: Iterable[Any],
    *,
    ordinal_by_question_id: dict[str, int],
) -> list[ExperimentRecord]:
    records: list[ExperimentRecord] = []
    for item in items:
        run = _attribute(item, "run", item)
        inputs = dict(_attribute(run, "inputs", {}) or {})
        outputs = dict(_attribute(run, "outputs", {}) or {})
        question_id = str(outputs.get("question_id") or inputs.get("question_id") or "")
        if not question_id:
            raise ValueError("LangSmith experiment run has no question_id")
        run_error = _attribute(run, "error")
        output_error = outputs.get("error")
        records.append(
            ExperimentRecord(
                run_id=str(_attribute(run, "id")),
                reference_example_id=(
                    str(_attribute(run, "reference_example_id"))
                    if _attribute(run, "reference_example_id")
                    else None
                ),
                question_id=question_id,
                ordinal=ordinal_by_question_id.get(
                    question_id, len(ordinal_by_question_id)
                ),
                inputs=inputs,
                outputs=outputs,
                feedback=_feedback_from_result(item),
                error=str(run_error or output_error) if run_error or output_error else None,
            )
        )
    return sorted(records, key=lambda record: (record.ordinal, record.question_id))


def _numeric_values(records: list[ExperimentRecord], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.outputs.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
    return values


def _numeric_feedback(records: list[ExperimentRecord], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.feedback.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def summarize_records(
    records: list[ExperimentRecord],
    *,
    experiment_name: str,
    experiment_id: str,
    dataset_name: str,
) -> LangSmithSummary:
    failures = sum(record.error is not None for record in records)
    latency_values = _numeric_values(records, "latency_ms")
    tool_counts = [
        float(len(record.outputs.get("tool_calls", []) or [])) for record in records
    ]
    return LangSmithSummary(
        experiment_name=experiment_name,
        experiment_id=experiment_id,
        dataset_name=dataset_name,
        attempted=len(records),
        completed=len(records) - failures,
        failed=failures,
        mean_document_recall=_mean(_numeric_feedback(records, "document_recall")),
        mean_strict_extra_documents=_mean(
            _numeric_feedback(records, "strict_extra_documents")
        ),
        failure_rate=failures / len(records) if records else 0.0,
        mean_latency_ms=_mean(latency_values),
        p95_latency_ms=percentile(latency_values, 0.95),
        mean_tool_calls=_mean(tool_counts),
        mean_input_tokens=_mean(_numeric_values(records, "input_tokens")),
        mean_output_tokens=_mean(_numeric_values(records, "output_tokens")),
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return normalized or "experiment"


def write_experiment_artifacts(
    *,
    result: Any,
    dataset: Any,
    config: LangSmithExperimentConfig,
    metadata: dict[str, Any],
    records: list[ExperimentRecord],
) -> tuple[Path, LangSmithSummary, str]:
    experiment_name = str(_attribute(result, "experiment_name"))
    experiment_id = str(_attribute(result, "experiment_id"))
    experiment_url = str(_attribute(result, "url", ""))
    output_dir = config.output_root / _safe_name(experiment_name)
    summary = summarize_records(
        records,
        experiment_name=experiment_name,
        experiment_id=experiment_id,
        dataset_name=str(_attribute(dataset, "name")),
    )
    experiment_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_url": experiment_url,
        "dataset_id": str(_attribute(dataset, "id")),
        "dataset_name": str(_attribute(dataset, "name")),
        "config": config.model_dump(mode="json"),
        "metadata": metadata,
    }
    successful = [record for record in records if record.error is None]
    answers = [
        {
            "question_id": record.question_id,
            "answer": str(record.outputs.get("answer") or ""),
            "document_ids": list(record.outputs.get("document_ids", []) or []),
        }
        for record in successful
    ]
    _atomic_json(output_dir / "experiment.json", experiment_payload)
    _atomic_jsonl(output_dir / "answers.jsonl", answers)
    _atomic_jsonl(
        output_dir / "records.jsonl",
        [record.model_dump(mode="json") for record in records],
    )
    _atomic_json(output_dir / "summary.json", summary.model_dump(mode="json"))
    return output_dir, summary, experiment_url


def _project_metadata(project: Any) -> dict[str, Any]:
    extra = dict(_attribute(project, "extra", {}) or {})
    metadata = extra.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else extra


def _question_id_from_run(run: Any) -> str:
    outputs = _attribute(run, "outputs", {}) or {}
    inputs = _attribute(run, "inputs", {}) or {}
    return str(outputs.get("question_id") or inputs.get("question_id") or "")


def load_cloud_experiment(
    client: Any, experiment_name: str
) -> tuple[Any, list[ExperimentRecord], dict[str, Any]]:
    project = client.read_project(project_name=experiment_name, include_stats=True)
    runs = list(client.list_runs(project_name=experiment_name, is_root=True))
    run_ids = [_attribute(run, "id") for run in runs]
    feedback_by_run: dict[str, dict[str, Any]] = {}
    for start in range(0, len(run_ids), 100):
        for feedback in client.list_feedback(run_ids=run_ids[start : start + 100]):
            run_id = str(_attribute(feedback, "run_id"))
            feedback_by_run.setdefault(run_id, {})[str(_attribute(feedback, "key"))] = (
                _feedback_value(feedback)
            )
    question_ids = sorted(_question_id_from_run(run) for run in runs)
    ordinals = {question_id: ordinal for ordinal, question_id in enumerate(question_ids)}
    records = normalize_experiment_results(runs, ordinal_by_question_id=ordinals)
    for record in records:
        record.feedback = feedback_by_run.get(record.run_id, {})
    return project, records, _project_metadata(project)


def _summary_metrics(summary: LangSmithSummary) -> dict[str, float | None]:
    return {
        "mean_document_recall": summary.mean_document_recall,
        "mean_strict_extra_documents": summary.mean_strict_extra_documents,
        "failure_rate": summary.failure_rate,
        "mean_latency_ms": summary.mean_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
        "mean_tool_calls": summary.mean_tool_calls,
        "mean_input_tokens": summary.mean_input_tokens,
        "mean_output_tokens": summary.mean_output_tokens,
    }


def compare_experiments(
    client: Any,
    experiment_a: str,
    experiment_b: str,
    *,
    output_root: Path = Path("runs/langsmith/comparisons"),
) -> ComparisonReport:
    project_a, records_a, metadata_a = load_cloud_experiment(client, experiment_a)
    project_b, records_b, metadata_b = load_cloud_experiment(client, experiment_b)
    compatibility_keys = (
        "dataset_name",
        "source_fingerprint",
        "index_fingerprint",
        "model_name",
        "top_k",
    )
    mismatches = [
        key for key in compatibility_keys if metadata_a.get(key) != metadata_b.get(key)
    ]
    ids_a = {record.question_id for record in records_a}
    ids_b = {record.question_id for record in records_b}
    if ids_a != ids_b:
        mismatches.append("question set")
    if mismatches:
        raise ValueError(
            "Cannot compare experiments because these settings differ: "
            + ", ".join(mismatches)
        )

    dataset_name = str(metadata_a.get("dataset_name") or "")
    summary_a = summarize_records(
        records_a,
        experiment_name=experiment_a,
        experiment_id=str(_attribute(project_a, "id")),
        dataset_name=dataset_name,
    )
    summary_b = summarize_records(
        records_b,
        experiment_name=experiment_b,
        experiment_id=str(_attribute(project_b, "id")),
        dataset_name=dataset_name,
    )
    directions = {
        "mean_document_recall": "higher",
        "mean_strict_extra_documents": "lower",
        "failure_rate": "lower",
        "mean_latency_ms": "lower",
        "p95_latency_ms": "lower",
        "mean_tool_calls": "diagnostic",
        "mean_input_tokens": "lower",
        "mean_output_tokens": "lower",
    }
    values_a = _summary_metrics(summary_a)
    values_b = _summary_metrics(summary_b)
    metrics = {}
    for key, direction in directions.items():
        value_a = values_a[key]
        value_b = values_b[key]
        delta = value_b - value_a if value_a is not None and value_b is not None else None
        metrics[key] = ComparisonMetric(
            experiment_a=value_a,
            experiment_b=value_b,
            delta_b_minus_a=delta,
            direction=direction,
        )

    by_id_a = {record.question_id: record for record in records_a}
    by_id_b = {record.question_id: record for record in records_b}
    regressions = []
    for question_id in sorted(ids_a):
        record_a = by_id_a[question_id]
        record_b = by_id_b[question_id]
        recall_a = record_a.feedback.get("document_recall")
        recall_b = record_b.feedback.get("document_recall")
        recall_regressed = (
            isinstance(recall_a, (int, float))
            and not isinstance(recall_a, bool)
            and isinstance(recall_b, (int, float))
            and not isinstance(recall_b, bool)
            and recall_b < recall_a
        )
        success_regressed = record_a.error is None and record_b.error is not None
        if recall_regressed or success_regressed:
            regressions.append(question_id)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = output_root / (
        f"{timestamp}-{_safe_name(experiment_a)}-vs-{_safe_name(experiment_b)}.json"
    )
    report = ComparisonReport(
        experiment_a=experiment_a,
        experiment_b=experiment_b,
        dataset_name=dataset_name,
        question_count=len(ids_a),
        metrics=metrics,
        regression_question_ids=regressions,
        output_path=output_path,
    )
    _atomic_json(output_path, report.model_dump(mode="json"))
    return report
