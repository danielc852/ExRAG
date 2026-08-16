"""Command-line entry point for the EnterpriseRAG-Bench baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

from langsmith.utils import LangSmithError

from agent import create_ollama_model, create_rag_agent, run_agent
from data import (
    ArtifactLayout,
    DownloadConfig,
    EmbeddingConfig,
    IndexConfig,
    ProcessingConfig,
    build_faiss_index,
    clean_data,
    download_dataset,
    embed_chunks,
    get_status,
    load_frozen_questions,
    run_process,
    validate_index,
)
from data.artifacts import load_manifest
from eval import (
    EvaluationConfig,
    LangSmithDatasetConfig,
    LangSmithExperimentConfig,
    compare_experiments,
    run_evaluation,
    run_langsmith_experiment,
    sync_frozen_dataset,
)
from tools import FaissRetriever, create_retrieval_tool


DEFAULT_ARTIFACT_ROOT = Path(os.getenv("RAG_ARTIFACT_ROOT", "artifacts"))
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBEDDING = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
DEFAULT_REVISION = os.getenv("DATASET_REVISION", "main")
DEFAULT_LANGSMITH_DATASET = os.getenv("LANGSMITH_DATASET", "EnterpriseRAG-Bench")


def _add_artifact_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)


def _add_lifecycle_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--rebuild", action="store_true")


def _add_download_arguments(parser: argparse.ArgumentParser) -> None:
    corpus = parser.add_mutually_exclusive_group()
    corpus.add_argument("--full", action="store_true")
    corpus.add_argument("--limit-documents", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=1_000)
    parser.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path)


def _add_processing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--tokenizer-model", default=DEFAULT_EMBEDDING)


def _add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    parser.add_argument("--embedding-revision")
    parser.add_argument("--embedding-batch-size", type=int, default=32)


def _add_index_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-batch-size", type=int, default=10_000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Run data preparation stages")
    stages = prepare.add_subparsers(dest="prepare_stage", required=True)
    for stage in ("download", "process", "embed", "index", "all"):
        stage_parser = stages.add_parser(stage)
        _add_artifact_argument(stage_parser)
        _add_lifecycle_arguments(stage_parser)
        if stage in {"download", "all"}:
            _add_download_arguments(stage_parser)
        if stage in {"process", "all"}:
            _add_processing_arguments(stage_parser)
        if stage in {"embed", "all"}:
            _add_embedding_arguments(stage_parser)
        if stage in {"index", "all"}:
            _add_index_arguments(stage_parser)
    status_parser = stages.add_parser("status")
    _add_artifact_argument(status_parser)

    ask_parser = commands.add_parser("ask", help="Ask one question")
    ask_parser.add_argument("question")
    _add_agent_arguments(ask_parser)
    ask_parser.add_argument("--json", action="store_true", dest="as_json")

    eval_parser = commands.add_parser("eval", help="Run resumable benchmark questions")
    _add_agent_arguments(eval_parser)
    _add_question_scope_arguments(eval_parser)
    eval_parser.add_argument("--output-dir", type=Path)
    eval_parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )

    langsmith_parser = commands.add_parser(
        "langsmith", help="Sync and evaluate with LangSmith"
    )
    langsmith_commands = langsmith_parser.add_subparsers(
        dest="langsmith_command", required=True
    )
    sync_parser = langsmith_commands.add_parser("sync", help="Sync frozen questions")
    _add_artifact_argument(sync_parser)
    sync_parser.add_argument("--dataset-name", default=DEFAULT_LANGSMITH_DATASET)

    run_parser = langsmith_commands.add_parser("run", help="Run a cloud experiment")
    _add_agent_arguments(run_parser)
    run_parser.add_argument("--dataset-name", default=DEFAULT_LANGSMITH_DATASET)
    _add_question_scope_arguments(run_parser)
    run_parser.add_argument("--max-concurrency", type=int, default=1)
    run_parser.add_argument("--experiment-prefix")
    run_parser.add_argument("--output-root", type=Path, default=Path("runs/langsmith"))

    compare_parser = langsmith_commands.add_parser(
        "compare", help="Compare two deterministic experiments"
    )
    compare_parser.add_argument("experiment_a")
    compare_parser.add_argument("experiment_b")
    compare_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/langsmith/comparisons"),
    )
    return parser


def _add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent", choices=("simple", "deep"), default="simple")
    _add_artifact_argument(parser)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)


def _add_question_scope_arguments(parser: argparse.ArgumentParser) -> None:
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--limit-questions", type=int, default=10)
    scope.add_argument("--all-questions", action="store_true")
    parser.add_argument("--question-type", action="append", dest="question_types")


def validate_ollama(base_url: str, model_name: str) -> None:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}. Start Ollama and try again. ({exc})"
        ) from exc
    installed = {
        str(item.get("name") or item.get("model"))
        for item in payload.get("models", [])
        if isinstance(item, dict)
    }
    requested = model_name if ":" in model_name else f"{model_name}:latest"
    normalized = {name if ":" in name else f"{name}:latest" for name in installed}
    if requested not in normalized:
        raise RuntimeError(f"Ollama model {model_name!r} is missing. Run: ollama pull {model_name}")


def _download_config(args: argparse.Namespace) -> DownloadConfig:
    return DownloadConfig(
        artifact_root=args.artifact_root,
        dataset_revision=args.dataset_revision,
        full_corpus=args.full,
        document_limit=None if args.full else args.limit_documents,
        seed=args.seed,
        shard_size=args.shard_size,
        cache_dir=args.cache_dir,
    )


def _processing_config(args: argparse.Namespace) -> ProcessingConfig:
    return ProcessingConfig(
        artifact_root=args.artifact_root,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        tokenizer_model=args.tokenizer_model,
    )


def _embedding_config(args: argparse.Namespace) -> EmbeddingConfig:
    return EmbeddingConfig(
        artifact_root=args.artifact_root,
        model_name=args.embedding_model,
        model_revision=args.embedding_revision,
        batch_size=args.embedding_batch_size,
    )


def _index_config(args: argparse.Namespace) -> IndexConfig:
    return IndexConfig(
        artifact_root=args.artifact_root,
        batch_size=args.index_batch_size,
    )


def run_prepare(args: argparse.Namespace) -> int:
    if args.prepare_stage == "status":
        print(json.dumps(get_status(args.artifact_root), indent=2, sort_keys=True))
        return 0
    stages = {
        "download": (download_dataset, _download_config),
        "process": (clean_data, _processing_config),
        "embed": (embed_chunks, _embedding_config),
        "index": (build_faiss_index, _index_config),
    }
    if args.prepare_stage == "all":
        manifests = run_process(
            _download_config(args),
            _processing_config(args),
            _embedding_config(args),
            _index_config(args),
            resume=args.resume,
            rebuild=args.rebuild,
        )
        payload = {
            stage: manifest.model_dump(mode="json") for stage, manifest in manifests.items()
        }
    else:
        runner, config_factory = stages[args.prepare_stage]
        manifest = runner(
            config_factory(args),
            resume=args.resume,
            rebuild=args.rebuild,
        )
        payload = manifest.model_dump(mode="json")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


@contextmanager
def _agent_stack(args: argparse.Namespace) -> Iterator[Any]:
    validate_ollama(args.ollama_url, args.model)
    with FaissRetriever.load(args.artifact_root) as retriever:
        retrieval_tool = create_retrieval_tool(retriever, default_top_k=args.top_k)
        model = create_ollama_model(args.model, args.ollama_url)
        yield create_rag_agent(args.agent, model, retrieval_tool)


def run_ask(args: argparse.Namespace) -> int:
    with _agent_stack(args) as rag_agent:
        result = run_agent(rag_agent, args.question, mode=args.agent, model_name=args.model)
    if args.as_json:
        print(result.model_dump_json(indent=2))
    elif result.error:
        print(f"Error: {result.error}", file=sys.stderr)
    else:
        print(result.answer)
        if result.document_ids:
            print(f"\nDocuments: {', '.join(result.document_ids)}")
    return 1 if result.error else 0


def _evaluation_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir
    runs_dir = Path("runs")
    if args.resume and runs_dir.exists():
        matches = sorted(runs_dir.glob(f"*-{args.agent}"))
        if matches:
            return matches[-1]
    return runs_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.agent}"


def run_eval(args: argparse.Namespace) -> int:
    manifest = validate_index(args.artifact_root)
    layout = ArtifactLayout(args.artifact_root)
    source = load_manifest(layout, "download")
    if manifest.metadata.get("source_fingerprint") != source.output_fingerprint:
        raise ValueError("Index and frozen question artifacts do not share the same source lineage")
    questions = load_frozen_questions(args.artifact_root)
    with _agent_stack(args) as rag_agent:
        config = EvaluationConfig(
            agent_mode=args.agent,
            output_dir=_evaluation_output_dir(args),
            question_limit=None if args.all_questions else args.limit_questions,
            question_types=args.question_types,
            resume=args.resume,
            model_name=args.model,
        )
        summary = run_evaluation(config, rag_agent, questions, manifest)
    print(summary.model_dump_json(indent=2))
    return 0 if summary.failed == 0 else 1


def create_langsmith_client():
    from langsmith import Client

    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "LANGSMITH_API_KEY is required. Create a LangSmith API key and export it first."
        )
    endpoint = os.getenv("LANGSMITH_ENDPOINT")
    workspace_id = os.getenv("LANGSMITH_WORKSPACE_ID")
    return Client(
        api_url=endpoint or None,
        api_key=api_key,
        workspace_id=workspace_id or None,
    )


def _run_langsmith_command(client, args: argparse.Namespace) -> int:
    if args.langsmith_command == "sync":
        result = sync_frozen_dataset(
            client,
            LangSmithDatasetConfig(
                artifact_root=args.artifact_root,
                dataset_name=args.dataset_name,
            ),
        )
        print(result.model_dump_json(indent=2))
        return 0
    if args.langsmith_command == "compare":
        report = compare_experiments(
            client,
            args.experiment_a,
            args.experiment_b,
            output_root=args.output_root,
        )
        print(report.model_dump_json(indent=2))
        return 0
    if args.langsmith_command == "run":
        with _agent_stack(args) as rag_agent:
            result = run_langsmith_experiment(
                client,
                LangSmithExperimentConfig(
                    artifact_root=args.artifact_root,
                    dataset_name=args.dataset_name,
                    agent_mode=args.agent,
                    model_name=args.model,
                    ollama_url=args.ollama_url,
                    top_k=args.top_k,
                    question_limit=(
                        None if args.all_questions else args.limit_questions
                    ),
                    question_types=args.question_types,
                    max_concurrency=args.max_concurrency,
                    experiment_prefix=args.experiment_prefix,
                    output_root=args.output_root,
                ),
                rag_agent,
            )
        print(result.model_dump_json(indent=2))
        return 0 if result.summary.failed == 0 else 1
    raise ValueError(f"Unknown LangSmith command: {args.langsmith_command}")


def run_langsmith(args: argparse.Namespace) -> int:
    client = create_langsmith_client()
    try:
        return _run_langsmith_command(client, args)
    finally:
        client.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "prepare": run_prepare,
        "ask": run_ask,
        "eval": run_eval,
        "langsmith": run_langsmith,
    }
    try:
        return handlers[args.command](args)
    except (
        FileNotFoundError,
        FileExistsError,
        LangSmithError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
