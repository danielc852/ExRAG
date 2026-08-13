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

    assert build_parser().parse_args(["index", "--limit-documents", "1"]).command == "index"
