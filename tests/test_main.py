from __future__ import annotations

import pytest

from main import DEFAULT_MODEL, _download_config, build_parser, validate_ollama


def test_cli_defaults_to_safe_sample_and_ten_questions():
    parser = build_parser()
    download_args = parser.parse_args(["prepare", "download"])
    assert download_args.limit_documents == 1_000
    assert download_args.full is False
    assert str(download_args.artifact_root) == "artifacts"

    eval_args = parser.parse_args(["eval"])
    assert eval_args.limit_questions == 10
    assert eval_args.agent == "simple"
    assert eval_args.model == "hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M"
    assert str(eval_args.artifact_root) == "artifacts"


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


def test_cli_exposes_all_pipeline_stages():
    parser = build_parser()
    for stage in ("download", "process", "embed", "index", "all", "status"):
        args = parser.parse_args(["prepare", stage])
        assert args.prepare_stage == stage


def test_cli_exposes_langsmith_commands_without_breaking_local_eval():
    parser = build_parser()
    sync_args = parser.parse_args(["langsmith", "sync"])
    run_args = parser.parse_args(["langsmith", "run", "--all-questions"])
    compare_args = parser.parse_args(["langsmith", "compare", "simple", "deep"])
    local_args = parser.parse_args(["eval", "--limit-questions", "2"])
    assert sync_args.langsmith_command == "sync"
    assert run_args.langsmith_command == "run"
    assert run_args.all_questions is True
    assert compare_args.experiment_a == "simple"
    assert local_args.command == "eval"


def test_old_top_level_index_command_is_removed():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["index"])


def test_full_download_config_does_not_record_sample_limit():
    args = build_parser().parse_args(["prepare", "download", "--full"])
    assert _download_config(args).document_limit is None
