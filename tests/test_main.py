from __future__ import annotations

import pytest

from main import build_parser


def test_cli_defaults_to_safe_sample_and_ten_questions():
    parser = build_parser()
    index_args = parser.parse_args(["index"])
    assert index_args.limit_documents == 1_000
    assert index_args.full is False

    eval_args = parser.parse_args(["eval"])
    assert eval_args.limit_questions == 10
    assert eval_args.agent == "simple"


def test_cli_rejects_conflicting_index_lifecycle_flags():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["index", "--resume", "--rebuild"])
