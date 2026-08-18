from __future__ import annotations

import pytest

from main import (
    DEFAULT_EMBEDDING,
    DEFAULT_EMBEDDING_ENGINE,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    SAMPLE_DOCUMENT_LIMIT,
    _artifact_root,
    _download_config,
    _embedding_config,
    build_parser,
    validate_ollama,
)


def test_cli_exposes_only_the_three_experiment_pipelines():
    parser = build_parser()

    assert parser.parse_args(["download", "sample"]).command == "download"
    assert (
        parser.parse_args(["init_vectordb", "sample"]).command
        == "init_vectordb"
    )
    assert parser.parse_args(["run_exper", "sample"]).command == "run_exper"

    for removed_command in ("prepare", "ask", "eval", "langsmith"):
        with pytest.raises(SystemExit):
            parser.parse_args([removed_command])


def test_sample_and_full_modes_use_separate_default_artifacts():
    parser = build_parser()
    sample = parser.parse_args(["download", "sample"])
    full = parser.parse_args(["download", "full"])

    assert _artifact_root(sample).as_posix() == "artifacts/sample"
    assert _artifact_root(full).as_posix() == "artifacts/full"
    assert _download_config(sample).document_limit == SAMPLE_DOCUMENT_LIMIT
    assert _download_config(sample).sample_question_limit == 10
    assert _download_config(sample).full_corpus is False
    assert _download_config(full).document_limit is None
    assert _download_config(full).sample_question_limit is None
    assert _download_config(full).full_corpus is True


def test_explicit_artifact_root_is_used_without_appending_mode(tmp_path):
    args = build_parser().parse_args(
        ["init_vectordb", "full", "--artifact-root", str(tmp_path)]
    )

    assert _artifact_root(args) == tmp_path


def test_init_vectordb_accepts_mlx_embedding_engine_and_model():
    args = build_parser().parse_args(
        [
            "init_vectordb",
            "sample",
            "--embedding-engine",
            "mlx",
            "--embedding-model",
            "mlx-community/bge-small-en-v1.5-bf16",
        ]
    )

    config = _embedding_config(args)
    assert config.engine == "mlx"
    assert config.model_name == "mlx-community/bge-small-en-v1.5-bf16"


def test_init_vectordb_defaults_to_openrouter_nemotron_embeddings():
    args = build_parser().parse_args(["init_vectordb", "sample"])

    assert args.embedding_engine == DEFAULT_EMBEDDING_ENGINE == "openrouter"
    assert (
        args.embedding_model
        == DEFAULT_EMBEDDING
        == "nvidia/nemotron-3-embed-1b:free"
    )
    assert args.tokenizer_model == DEFAULT_TOKENIZER == "BAAI/bge-base-en-v1.5"
    assert _embedding_config(args).engine == "openrouter"


def test_run_experiment_defaults_to_simple_agent():
    args = build_parser().parse_args(["run_exper", "sample"])

    assert args.dataset_mode == "sample"
    assert args.agent == "simple"
    assert args.model == "hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M"


def test_dataset_mode_is_required_and_validated():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["download"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run_exper", "tiny"])


def test_ollama_validation_accepts_normalized_hugging_face_model_name(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return (
                b'{"models": [{"name": '
                b'"hf.co/liquidai/lfm2.5-2.6b-gguf:q4_k_m"}]}'
            )

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())

    validate_ollama("http://localhost:11434", DEFAULT_MODEL)
