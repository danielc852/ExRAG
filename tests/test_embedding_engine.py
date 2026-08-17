from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import data.processing.embed_model as embedding_module
from data.processing.embed_model import MlxEmbeddingModel, create_embedding_model


class FakeMlx:
    def __init__(self):
        self.evaluated = []

    def eval(self, value):
        self.evaluated.append(value)


def test_mlx_adapter_batches_and_returns_normalized_float32_vectors():
    mlx = FakeMlx()
    generated_batches = []

    def generate(_model, _tokenizer, *, texts, **kwargs):
        generated_batches.append((texts, kwargs))
        values = np.asarray(
            [[3.0, 4.0] if "alpha" in text else [0.0, 2.0] for text in texts],
            dtype=np.float16,
        )
        values = values / np.linalg.norm(values, axis=1, keepdims=True)
        return SimpleNamespace(text_embeds=values)

    model = SimpleNamespace(config=SimpleNamespace(hidden_size=2))
    adapter = MlxEmbeddingModel(
        model,
        object(),
        mlx=mlx,
        generate=generate,
    )

    vectors = adapter.encode(
        ["alpha", "beta", "alpha again"],
        batch_size=2,
        normalize_embeddings=True,
    )

    assert [batch[0] for batch in generated_batches] == [
        ["alpha", "beta"],
        ["alpha again"],
    ]
    assert all(batch[1]["max_length"] == 512 for batch in generated_batches)
    assert len(mlx.evaluated) == 2
    assert vectors.shape == (3, 2)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)


def test_mlx_adapter_rejects_outputs_without_text_embeddings():
    adapter = MlxEmbeddingModel(
        SimpleNamespace(config=SimpleNamespace(hidden_size=2)),
        object(),
        mlx=FakeMlx(),
        generate=lambda *_args, **_kwargs: SimpleNamespace(text_embeds=None),
    )

    with pytest.raises(ValueError, match="pooled text embeddings"):
        adapter.encode(["query"])


def test_mlx_engine_rejects_non_apple_silicon_before_importing_dependencies(
    monkeypatch,
):
    monkeypatch.setattr(embedding_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(embedding_module.platform, "machine", lambda: "x86_64")

    with pytest.raises(RuntimeError, match="Apple Silicon"):
        create_embedding_model("mlx-model", engine="mlx")


def test_unknown_embedding_engine_is_rejected():
    with pytest.raises(ValueError, match="Unsupported embedding engine"):
        create_embedding_model("model", engine="unknown")
