"""Deterministic LangSmith evaluators for ExRAG experiments."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any

from .local import document_recall, strict_extra_document_count


def deterministic_evaluator(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> list[dict[str, Any]]:
    document_ids = [str(value) for value in outputs.get("document_ids", []) or []]
    expected_ids = [
        str(value) for value in reference_outputs.get("expected_doc_ids", []) or []
    ]
    error = outputs.get("error")
    answer = str(outputs.get("answer") or "").strip()
    tool_calls = outputs.get("tool_calls", []) or []
    recall = document_recall(document_ids, expected_ids)
    extras = strict_extra_document_count(document_ids, expected_ids)
    return [
        {
            "key": "document_recall",
            "score": recall,
            "comment": "Not applicable when the benchmark has no gold document IDs"
            if recall is None
            else None,
        },
        {
            "key": "strict_extra_documents",
            "score": extras,
            "comment": (
                "Set difference against gold IDs; not the official LLM-judged "
                "Invalid Extra Documents metric"
            ),
        },
        {"key": "retrieved_document_count", "score": len(set(document_ids))},
        {"key": "tool_call_count", "score": len(tool_calls)},
        {"key": "run_success", "score": not bool(error)},
        {"key": "answer_present", "score": bool(answer)},
        {"key": "latency_ms", "score": _number_or_none(outputs.get("latency_ms"))},
        {"key": "input_tokens", "score": _number_or_none(outputs.get("input_tokens"))},
        {"key": "output_tokens", "score": _number_or_none(outputs.get("output_tokens"))},
    ]


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _mean(values: list[int | float]) -> float | None:
    return fmean(values) if values else None


def _metric_values(
    feedback_rows: list[dict[str, Any]], key: str
) -> list[int | float]:
    values = (_number_or_none(row.get(key)) for row in feedback_rows)
    return [value for value in values if value is not None]


def percentile(values: list[int | float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def aggregate_outputs(
    outputs: list[dict[str, Any]],
    reference_outputs: list[dict[str, Any]],
) -> dict[str, float | None]:
    feedback_rows = [
        {item["key"]: item["score"] for item in deterministic_evaluator(actual, expected)}
        for actual, expected in zip(outputs, reference_outputs)
    ]
    recalls = _metric_values(feedback_rows, "document_recall")
    extras = _metric_values(feedback_rows, "strict_extra_documents")
    latencies = _metric_values(feedback_rows, "latency_ms")
    count = len(outputs)
    return {
        "mean_document_recall": _mean(recalls),
        "mean_strict_extra_documents": _mean(extras),
        "failure_rate": (
            sum(not row["run_success"] for row in feedback_rows) / count
            if count
            else 0.0
        ),
        "mean_latency_ms": _mean(latencies),
        "p95_latency_ms": percentile(latencies, 0.95),
        "mean_tool_calls": _mean(_metric_values(feedback_rows, "tool_call_count")),
        "mean_input_tokens": _mean(_metric_values(feedback_rows, "input_tokens")),
        "mean_output_tokens": _mean(_metric_values(feedback_rows, "output_tokens")),
    }


def deterministic_summary_evaluator(
    outputs: list[dict[str, Any]],
    reference_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = aggregate_outputs(outputs, reference_outputs)
    return [
        {"key": key, "score": value}
        for key, value in metrics.items()
    ]
