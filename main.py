"""Run the three ExRAG experiment pipelines from the command line."""

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
from typing import Any, Iterator, Literal, Sequence

from agent import DEFAULT_OLLAMA_MODEL, create_ollama_model, create_rag_agent
from data import (
    ArtifactLayout,
    DownloadConfig,
    EmbeddingConfig,
    IndexConfig,
    ProcessingConfig,
    build_faiss_index,
    chunk_data,
    download_dataset,
    embed_chunks,
    load_frozen_questions,
    review_download,
    validate_index,
)
from data.artifacts import load_manifest
from eval import EvaluationConfig, run_evaluation
from tools import FaissRetriever, create_retrieval_tool


DatasetMode = Literal["sample", "full"]

DEFAULT_ARTIFACT_ROOT = Path(os.getenv("RAG_ARTIFACT_ROOT", "artifacts"))
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBEDDING = os.getenv(
    "EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free"
)
DEFAULT_EMBEDDING_ENGINE = os.getenv(
    "EMBEDDING_ENGINE", "openrouter"
)
DEFAULT_TOKENIZER = os.getenv("TOKENIZER_MODEL", "BAAI/bge-base-en-v1.5")
DEFAULT_REVISION = os.getenv("DATASET_REVISION", "main")
SAMPLE_DOCUMENT_LIMIT = 1_000
SAMPLE_QUESTION_LIMIT = 10


def _add_mode_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "dataset_mode",
        choices=("sample", "full"),
        help="Use the 1,000-document sample or the complete benchmark dataset",
    )


def _add_artifact_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Exact artifact directory (default: artifacts/<dataset_mode>)",
    )


def _add_lifecycle_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume an interrupted pipeline (default: enabled)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild this pipeline and invalidate downstream artifacts",
    )


def _add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent", choices=("simple", "deep"), default="simple")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download and freeze the dataset")
    _add_mode_argument(download)
    _add_artifact_argument(download)
    _add_lifecycle_arguments(download)
    download.add_argument("--sample-size", type=int, default=SAMPLE_DOCUMENT_LIMIT)
    download.add_argument("--seed", type=int, default=42)
    download.add_argument("--shard-size", type=int, default=1_000)
    download.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    download.add_argument("--cache-dir", type=Path)

    init_vectordb = commands.add_parser(
        "init_vectordb",
        help="Chunk documents, create embeddings, and initialize FAISS",
    )
    _add_mode_argument(init_vectordb)
    _add_artifact_argument(init_vectordb)
    _add_lifecycle_arguments(init_vectordb)
    init_vectordb.add_argument("--chunk-size", type=int, default=512)
    init_vectordb.add_argument("--chunk-overlap", type=int, default=64)
    init_vectordb.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER)
    init_vectordb.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    init_vectordb.add_argument(
        "--embedding-engine",
        choices=("sentence-transformers", "mlx", "openrouter"),
        default=DEFAULT_EMBEDDING_ENGINE,
        help="Embedding inference engine (default: EMBEDDING_ENGINE or openrouter)",
    )
    init_vectordb.add_argument("--embedding-revision")
    init_vectordb.add_argument("--embedding-batch-size", type=int, default=32)
    init_vectordb.add_argument("--index-batch-size", type=int, default=10_000)

    run_exper = commands.add_parser(
        "run_exper",
        help="Run 10 sample questions or the full benchmark experiment",
    )
    _add_mode_argument(run_exper)
    _add_artifact_argument(run_exper)
    _add_agent_arguments(run_exper)
    run_exper.add_argument("--output-dir", type=Path)
    run_exper.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _artifact_root(args: argparse.Namespace) -> Path:
    return args.artifact_root or DEFAULT_ARTIFACT_ROOT / args.dataset_mode


def _download_config(args: argparse.Namespace) -> DownloadConfig:
    is_full = args.dataset_mode == "full"
    return DownloadConfig(
        artifact_root=_artifact_root(args),
        dataset_revision=args.dataset_revision,
        full_corpus=is_full,
        document_limit=None if is_full else args.sample_size,
        sample_question_limit=None if is_full else SAMPLE_QUESTION_LIMIT,
        seed=args.seed,
        shard_size=args.shard_size,
        cache_dir=args.cache_dir,
    )


def _processing_config(args: argparse.Namespace) -> ProcessingConfig:
    return ProcessingConfig(
        artifact_root=_artifact_root(args),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        tokenizer_model=args.tokenizer_model,
    )


def _embedding_config(args: argparse.Namespace) -> EmbeddingConfig:
    return EmbeddingConfig(
        artifact_root=_artifact_root(args),
        engine=args.embedding_engine,
        model_name=args.embedding_model,
        model_revision=args.embedding_revision,
        batch_size=args.embedding_batch_size,
    )


def _index_config(args: argparse.Namespace) -> IndexConfig:
    return IndexConfig(
        artifact_root=_artifact_root(args),
        batch_size=args.index_batch_size,
    )


def _print_manifests(manifests: dict[str, Any]) -> None:
    payload = {
        stage: manifest.model_dump(mode="json")
        for stage, manifest in manifests.items()
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _require_dataset_mode(artifact_root: Path, expected: DatasetMode) -> None:
    source = load_manifest(ArtifactLayout(artifact_root), "download")
    actual = source.metadata.get("corpus_mode")
    if actual != expected:
        raise ValueError(
            f"Artifacts contain the {actual!r} dataset, not {expected!r}. "
            "Use the matching dataset mode or a different --artifact-root."
        )


def run_download(args: argparse.Namespace) -> int:
    manifest = download_dataset(
        _download_config(args),
        resume=args.resume,
        rebuild=args.rebuild,
    )
    _print_manifests({"download": manifest})
    return 0


def run_init_vectordb(args: argparse.Namespace) -> int:
    artifact_root = _artifact_root(args)
    _require_dataset_mode(artifact_root, args.dataset_mode)
    review_download(artifact_root)
    processed = chunk_data(
        _processing_config(args),
        resume=args.resume,
        rebuild=args.rebuild,
    )
    embeddings = embed_chunks(
        _embedding_config(args),
        resume=args.resume,
        rebuild=args.rebuild,
    )
    index = build_faiss_index(
        _index_config(args),
        resume=args.resume,
        rebuild=args.rebuild,
    )
    _print_manifests(
        {"process": processed, "embed": embeddings, "index": index}
    )
    return 0


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
    requested = (model_name if ":" in model_name else f"{model_name}:latest").casefold()
    normalized = {
        (name if ":" in name else f"{name}:latest").casefold() for name in installed
    }
    if requested not in normalized:
        raise RuntimeError(
            f"Ollama model {model_name!r} is missing. Run: ollama pull {model_name}"
        )


@contextmanager
def _agent_stack(args: argparse.Namespace) -> Iterator[Any]:
    validate_ollama(args.ollama_url, args.model)
    with FaissRetriever.load(_artifact_root(args)) as retriever:
        retrieval_tool = create_retrieval_tool(
            retriever,
            default_top_k=args.top_k,
            include_filters=False,
            use_full_user_question=True,
        )
        model = create_ollama_model(args.model, args.ollama_url)
        yield create_rag_agent(args.agent, model, retrieval_tool)


def _evaluation_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir
    runs_dir = Path("runs") / args.dataset_mode
    if args.resume and runs_dir.exists():
        matches = sorted(runs_dir.glob(f"*-{args.agent}"))
        if matches:
            return matches[-1]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return runs_dir / f"{timestamp}-{args.agent}"


def run_experiment(args: argparse.Namespace) -> int:
    artifact_root = _artifact_root(args)
    _require_dataset_mode(artifact_root, args.dataset_mode)
    manifest = validate_index(artifact_root)
    source = load_manifest(ArtifactLayout(artifact_root), "download")
    if manifest.metadata.get("source_fingerprint") != source.output_fingerprint:
        raise ValueError(
            "Index and frozen question artifacts do not share the same source lineage"
        )
    questions = load_frozen_questions(artifact_root)
    with _agent_stack(args) as rag_agent:
        config = EvaluationConfig(
            agent_mode=args.agent,
            output_dir=_evaluation_output_dir(args),
            question_limit=(
                SAMPLE_QUESTION_LIMIT if args.dataset_mode == "sample" else None
            ),
            resume=args.resume,
            model_name=args.model,
        )
        summary = run_evaluation(config, rag_agent, questions, manifest)
    print(summary.model_dump_json(indent=2))
    return 0 if summary.failed == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "download": run_download,
        "init_vectordb": run_init_vectordb,
        "run_exper": run_experiment,
    }
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
