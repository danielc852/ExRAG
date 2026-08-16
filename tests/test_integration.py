from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_ENTERPRISE_RAG_INTEGRATION") != "1",
    reason="requires Hugging Face downloads and a running Ollama server",
)
def test_live_integration_is_opt_in():
    # The full live flow is intentionally driven through the public CLI so it tests
    # the same wiring users run. Setting the environment flag makes this marker a
    # visible reminder without causing network/model downloads in the unit suite.
    from main import build_parser

    args = build_parser().parse_args(
        ["prepare", "download", "--limit-documents", "1"]
    )
    assert args.command == "prepare"
    assert args.prepare_stage == "download"


@pytest.mark.skipif(
    os.getenv("RUN_LANGSMITH_INTEGRATION") != "1",
    reason="requires LangSmith credentials, prepared artifacts, and a running Ollama server",
)
def test_live_langsmith_integration_is_opt_in(tmp_path):
    from main import main

    assert main(["langsmith", "sync"]) == 0
    assert (
        main(
            [
                "langsmith",
                "run",
                "--limit-questions",
                "1",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert list(tmp_path.glob("*/experiment.json"))
    assert list(tmp_path.glob("*/answers.jsonl"))
