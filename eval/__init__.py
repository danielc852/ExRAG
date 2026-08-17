"""Stable public API for local and LangSmith evaluation."""

from .dataset import (
    create_snapshot_name,
    deterministic_example_id,
    evaluation_dataset_type,
    evaluation_questions,
    question_to_example,
    sync_frozen_dataset,
)
from .evaluators import (
    aggregate_outputs,
    deterministic_evaluator,
    deterministic_summary_evaluator,
)
from .experiment import build_langsmith_target, run_langsmith_experiment
from .local import (
    _write_run_config,
    append_answer,
    append_run_detail,
    document_recall,
    evaluate_question,
    load_completed_question_ids,
    run_evaluation,
    select_questions,
    strict_extra_document_count,
    write_summary,
)
from .models import (
    ComparisonMetric,
    ComparisonReport,
    DatasetSyncResult,
    EvaluationConfig,
    EvaluationSummary,
    ExperimentRecord,
    LangSmithDatasetConfig,
    LangSmithExperimentConfig,
    LangSmithExperimentResult,
    LangSmithSummary,
)
from .results import compare_experiments

__all__ = [
    "ComparisonMetric",
    "ComparisonReport",
    "DatasetSyncResult",
    "EvaluationConfig",
    "EvaluationSummary",
    "ExperimentRecord",
    "LangSmithDatasetConfig",
    "LangSmithExperimentConfig",
    "LangSmithExperimentResult",
    "LangSmithSummary",
    "aggregate_outputs",
    "append_answer",
    "append_run_detail",
    "build_langsmith_target",
    "compare_experiments",
    "create_snapshot_name",
    "deterministic_evaluator",
    "deterministic_example_id",
    "evaluation_dataset_type",
    "evaluation_questions",
    "deterministic_summary_evaluator",
    "document_recall",
    "evaluate_question",
    "load_completed_question_ids",
    "question_to_example",
    "run_evaluation",
    "run_langsmith_experiment",
    "select_questions",
    "strict_extra_document_count",
    "sync_frozen_dataset",
    "write_summary",
]
