from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_ENTERPRISE_RAG_INTEGRATION") != "1",
    reason="requires Hugging Face downloads and a running Ollama server",
)
def test_live_integration_is_opt_in(tmp_path):
    # The full live flow is intentionally driven through the public CLI so it tests
    # the same wiring users run. Setting the environment flag makes this marker a
    # visible reminder without causing network/model downloads in the unit suite.
    from main import main

    artifacts = tmp_path / "artifacts"
    output = tmp_path / "run"
    common = ["sample", "--artifact-root", str(artifacts)]

    assert main(["download", *common, "--sample-size", "1"]) == 0
    assert main(["init_vectordb", *common]) == 0
    assert main(["run_exper", *common, "--output-dir", str(output)]) == 0
    assert (output / "answers.jsonl").exists()
