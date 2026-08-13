"""Command-line entry point for the EnterpriseRAG-Bench baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Sequence

from agent import create_ollama_model, create_rag_agent, run_agent
from dataset import IndexBuildConfig, build_index, load_questions, validate_index
from eval import EvaluationConfig, run_evaluation
from tools import FaissRetriever, create_retrieval_tool


DEFAULT_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "data/index"))
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBEDDING = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
DEFAULT_REVISION = os.getenv("DATASET_REVISION", "main")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    index_parser = subcommands.add_parser("index", help="Download, chunk, and index documents")
    index_parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    corpus = index_parser.add_mutually_exclusive_group()
    corpus.add_argument("--full", action="store_true", help="Index the full benchmark corpus")
    corpus.add_argument("--limit-documents", type=int, default=1_000)
    lifecycle = index_parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--resume", action="store_true")
    lifecycle.add_argument("--rebuild", action="store_true")
    index_parser.add_argument("--seed", type=int, default=42)
    index_parser.add_argument("--chunk-size", type=int, default=512)
    index_parser.add_argument("--chunk-overlap", type=int, default=64)
    index_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    index_parser.add_argument("--embedding-batch-size", type=int, default=32)
    index_parser.add_argument("--checkpoint-every", type=int, default=1_000)
    index_parser.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    index_parser.add_argument("--cache-dir", type=Path)

    ask_parser = subcommands.add_parser("ask", help="Ask one question")
    ask_parser.add_argument("question")
    _add_agent_arguments(ask_parser)
    ask_parser.add_argument("--json", action="store_true", dest="as_json")

    eval_parser = subcommands.add_parser("eval", help="Run resumable benchmark questions")
    _add_agent_arguments(eval_parser)
    eval_scope = eval_parser.add_mutually_exclusive_group()
    eval_scope.add_argument("--limit-questions", type=int, default=10)
    eval_scope.add_argument("--all-questions", action="store_true")
    eval_parser.add_argument("--question-type", action="append", dest="question_types")
    eval_parser.add_argument("--output-dir", type=Path)
    eval_parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    eval_parser.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    eval_parser.add_argument("--cache-dir", type=Path)
    return parser


def _add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent", choices=("simple", "deep"), default="simple")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)


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
    normalized_installed = {name if ":" in name else f"{name}:latest" for name in installed}
    if requested not in normalized_installed:
        raise RuntimeError(f"Ollama model {model_name!r} is missing. Run: ollama pull {model_name}")


def run_index(args: argparse.Namespace) -> int:
    config = IndexBuildConfig(
        index_dir=args.index_dir,
        dataset_revision=args.dataset_revision,
        full_corpus=args.full,
        document_limit=args.limit_documents,
        seed=args.seed,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        checkpoint_every=args.checkpoint_every,
        cache_dir=args.cache_dir,
    )
    manifest = build_index(config, rebuild=args.rebuild, resume=args.resume)
    print(json.dumps(manifest.__dict__, indent=2, sort_keys=True))
    return 0


def _load_agent_stack(args: argparse.Namespace):
    validate_ollama(args.ollama_url, args.model)
    retriever = FaissRetriever.load(args.index_dir)
    retrieval_tool = create_retrieval_tool(retriever, default_top_k=args.top_k)
    model = create_ollama_model(args.model, args.ollama_url)
    rag_agent = create_rag_agent(args.agent, model, retrieval_tool)
    return retriever, rag_agent


def run_ask(args: argparse.Namespace) -> int:
    retriever, rag_agent = _load_agent_stack(args)
    try:
        result = run_agent(rag_agent, args.question, mode=args.agent, model_name=args.model)
    finally:
        retriever.close()
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
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return runs_dir / f"{timestamp}-{args.agent}"


def run_eval(args: argparse.Namespace) -> int:
    manifest = validate_index(args.index_dir)
    if args.dataset_revision != manifest.dataset_revision:
        raise ValueError(
            "Question dataset revision does not match the index revision "
            f"({args.dataset_revision!r} != {manifest.dataset_revision!r})"
        )
    questions = load_questions(args.dataset_revision, args.cache_dir)
    retriever, rag_agent = _load_agent_stack(args)
    try:
        config = EvaluationConfig(
            agent_mode=args.agent,
            output_dir=_evaluation_output_dir(args),
            question_limit=None if args.all_questions else args.limit_questions,
            question_types=args.question_types,
            resume=args.resume,
            model_name=args.model,
        )
        summary = run_evaluation(config, rag_agent, questions, manifest)
    finally:
        retriever.close()
    print(summary.model_dump_json(indent=2))
    return 0 if summary.failed == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            return run_index(args)
        if args.command == "ask":
            return run_ask(args)
        if args.command == "eval":
            return run_eval(args)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
