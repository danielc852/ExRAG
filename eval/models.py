"""Typed configuration and result models for local and LangSmith evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from agent import AgentMode


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


class LangSmithDatasetConfig(BaseModel):
    artifact_root: Path = Path("artifacts")
    dataset_name: str = "EnterpriseRAG-Bench"

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("dataset_name must not be empty")
        return value


class DatasetSyncResult(BaseModel):
    dataset_id: str
    dataset_name: str
    source_fingerprint: str
    total_examples: int
    created_examples: int
    status: Literal["created", "updated", "unchanged"]


class LangSmithExperimentConfig(BaseModel):
    artifact_root: Path = Path("artifacts")
    dataset_name: str = "EnterpriseRAG-Bench"
    agent_mode: AgentMode = "simple"
    model_name: str = "qwen3:8b"
    ollama_url: str = "http://localhost:11434"
    top_k: int = Field(default=5, ge=1, le=20)
    question_limit: int | None = Field(default=10, ge=1)
    question_types: list[str] | None = None
    max_concurrency: int = Field(default=1, ge=0)
    experiment_prefix: str | None = None
    output_root: Path = Path("runs/langsmith")

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("dataset_name must not be empty")
        return value


FeedbackValue = bool | int | float | str | dict[str, Any] | None


class ExperimentRecord(BaseModel):
    run_id: str
    reference_example_id: str | None = None
    question_id: str
    ordinal: int
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    feedback: dict[str, FeedbackValue] = Field(default_factory=dict)
    error: str | None = None


class LangSmithSummary(BaseModel):
    experiment_name: str
    experiment_id: str
    dataset_name: str
    attempted: int
    completed: int
    failed: int
    mean_document_recall: float | None
    mean_strict_extra_documents: float | None
    failure_rate: float
    mean_latency_ms: float | None
    p95_latency_ms: float | None
    mean_tool_calls: float | None
    mean_input_tokens: float | None
    mean_output_tokens: float | None


class LangSmithExperimentResult(BaseModel):
    experiment_name: str
    experiment_id: str
    experiment_url: str
    output_dir: Path
    summary: LangSmithSummary


class ComparisonMetric(BaseModel):
    experiment_a: float | None
    experiment_b: float | None
    delta_b_minus_a: float | None
    direction: Literal["higher", "lower", "diagnostic"]


class ComparisonReport(BaseModel):
    experiment_a: str
    experiment_b: str
    dataset_name: str
    question_count: int
    metrics: dict[str, ComparisonMetric]
    regression_question_ids: list[str]
    output_path: Path | None = None
