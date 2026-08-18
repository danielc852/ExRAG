from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace

import numpy as np
import pytest

import data.processing.embed_model as embedding_module
from data.processing.embed_model import create_embedding_model
from data.processing.embeddings.openrouter import (
    OpenRouterAPIError,
    OpenRouterClient,
    OpenRouterEmbedder,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def close(self):
        return None


def test_openrouter_client_sends_typed_batch_and_orders_response():
    captured = {}

    def open_request(request, *, timeout):
        captured.update(
            url=request.full_url,
            timeout=timeout,
            authorization=request.get_header("Authorization"),
            payload=json.loads(request.data),
        )
        return FakeResponse(
            {
                "data": [
                    {"index": 1, "embedding": [0, 2]},
                    {"index": 0, "embedding": [3, 4]},
                ]
            }
        )

    client = OpenRouterClient(
        "secret",
        base_url="https://openrouter.test/api/v1/",
        timeout_seconds=7,
        opener=open_request,
    )

    vectors = client.embed(
        model="nvidia/nemotron-3-embed-1b:free",
        texts=["alpha", "beta"],
        input_type="document",
    )

    assert vectors == [[3.0, 4.0], [0.0, 2.0]]
    assert captured == {
        "url": "https://openrouter.test/api/v1/embeddings",
        "timeout": 7,
        "authorization": "Bearer secret",
        "payload": {
            "model": "nvidia/nemotron-3-embed-1b:free",
            "input": ["alpha", "beta"],
            "input_type": "document",
            "encoding_format": "float",
        },
    }


def test_openrouter_client_reports_provider_error_without_api_key():
    error = urllib.error.HTTPError(
        "https://openrouter.test/embeddings",
        429,
        "Too Many Requests",
        {},
        FakeResponse({"error": {"message": "rate limited"}}),
    )

    def fail_request(*_args, **_kwargs):
        raise error

    client = OpenRouterClient("secret", opener=fail_request)

    with pytest.raises(OpenRouterAPIError, match="HTTP 429: rate limited") as caught:
        client.embed(model="model", texts=["text"], input_type="query")

    assert "secret" not in str(caught.value)


def test_openrouter_embedder_batches_and_normalizes_float32_vectors():
    calls = []

    class Client:
        def embed(self, **kwargs):
            calls.append(kwargs)
            return [
                [3.0, 4.0] if "alpha" in text else [0.0, 2.0]
                for text in kwargs["texts"]
            ]

    embedder = OpenRouterEmbedder("test/model", client=Client())
    vectors = embedder.encode(
        ["alpha", "beta", "alpha again"],
        batch_size=2,
        input_type="query",
    )

    assert [call["texts"] for call in calls] == [
        ["alpha", "beta"],
        ["alpha again"],
    ]
    assert all(call["input_type"] == "query" for call in calls)
    assert vectors.shape == (3, 2)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert embedder.get_sentence_embedding_dimension() == 2


def test_openrouter_factory_uses_environment_client(monkeypatch):
    client = SimpleNamespace()
    monkeypatch.setattr(
        embedding_module.OpenRouterClient,
        "from_env",
        classmethod(lambda _cls: client),
    )

    embedder = create_embedding_model(
        "nvidia/nemotron-3-embed-1b:free", engine="openrouter"
    )

    assert isinstance(embedder, OpenRouterEmbedder)
    assert embedder.client is client


def test_openrouter_factory_rejects_revision(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused")

    with pytest.raises(ValueError, match="do not accept revisions"):
        create_embedding_model("model", revision="main", engine="openrouter")


def test_openrouter_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterClient.from_env()
