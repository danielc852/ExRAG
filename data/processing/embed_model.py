"""Create interchangeable embedding engines and encode float32 text batches."""

from __future__ import annotations

import platform
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np

from .embeddings.openrouter import OpenRouterClient, OpenRouterEmbedder


EmbeddingEngine = Literal["sentence-transformers", "mlx", "openrouter"]


class MlxEmbeddingModel:
    """SentenceTransformer-shaped adapter around ``mlx-embeddings``."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        mlx: Any,
        generate: Callable[..., Any],
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._mlx = mlx
        self._generate = generate

    def get_sentence_embedding_dimension(self) -> int:
        config = getattr(self._model, "config", None)
        dimension = getattr(config, "hidden_size", None)
        if dimension is None:
            text_config = getattr(config, "text_config", None)
            dimension = getattr(text_config, "hidden_size", None)
        if not dimension:
            raise ValueError("MLX model config does not identify its embedding dimension")
        return int(dimension)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del show_progress_bar
        if batch_size < 1:
            raise ValueError("embedding batch size must be at least 1")
        if not convert_to_numpy:
            raise ValueError("The ExRAG MLX adapter returns NumPy embedding artifacts")
        if not normalize_embeddings:
            raise ValueError("The ExRAG MLX pipeline requires normalized embeddings")
        if not texts:
            return np.empty(
                (0, self.get_sentence_embedding_dimension()), dtype=np.float32
            )

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            outputs = self._generate(
                self._model,
                self._tokenizer,
                texts=list(texts[start : start + batch_size]),
                max_length=512,
                padding=True,
                truncation=True,
            )
            embeddings = getattr(outputs, "text_embeds", None)
            if embeddings is None:
                raise ValueError(
                    "MLX model did not return pooled text embeddings; use a supported "
                    "mlx-embeddings text checkpoint"
                )
            self._mlx.eval(embeddings)
            batch = np.asarray(embeddings, dtype=np.float32)
            if batch.ndim != 2:
                raise ValueError("MLX model returned a non-matrix embedding output")
            batches.append(batch)
        return np.concatenate(batches, axis=0)


def _create_mlx_embedding_model(
    model_name: str, revision: str | None
) -> MlxEmbeddingModel:
    if platform.system() != "Darwin" or platform.machine().lower() not in {
        "arm64",
        "aarch64",
    }:
        raise RuntimeError(
            "The MLX embedding engine requires Apple Silicon and macOS 14 or newer"
        )
    try:
        import mlx.core as mx
        from mlx_embeddings import generate
        from mlx_embeddings.utils import get_model_path, load
    except ImportError as exc:
        raise RuntimeError(
            "MLX embedding dependencies are missing; install them with "
            "`uv sync --extra mlx` on an Apple Silicon Mac"
        ) from exc

    try:
        model_path = get_model_path(model_name, revision=revision)
        model, tokenizer = load(str(model_path))
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot load {model_name!r} with mlx-embeddings. Use a converted MLX "
            "text-embedding checkpoint whose architecture is supported by "
            "mlx-embeddings."
        ) from exc
    return MlxEmbeddingModel(
        model,
        tokenizer,
        mlx=mx,
        generate=generate,
    )


def create_embedding_model(
    model_name: str,
    revision: str | None = None,
    *,
    engine: EmbeddingEngine = "sentence-transformers",
):
    if engine == "mlx":
        return _create_mlx_embedding_model(model_name, revision)
    if engine == "openrouter":
        if revision:
            raise ValueError("OpenRouter embedding models do not accept revisions")
        return OpenRouterEmbedder(model_name, client=OpenRouterClient.from_env())
    if engine != "sentence-transformers":
        raise ValueError(f"Unsupported embedding engine: {engine!r}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Sentence Transformers dependencies are missing; install them with "
            "`uv sync --extra local`"
        ) from exc

    kwargs = {"revision": revision} if revision else {}
    return SentenceTransformer(model_name, **kwargs)


def encode_chunk_shard(
    model: Any,
    texts: list[str],
    *,
    batch_size: int,
    normalize: bool,
    input_type: Literal["document", "query"] = "document",
) -> np.ndarray:
    if not texts:
        dimension = int(model.get_sentence_embedding_dimension())
        return np.empty((0, dimension), dtype=np.float32)
    kwargs = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": normalize,
        "show_progress_bar": False,
    }
    if getattr(model, "supports_input_type", False):
        kwargs["input_type"] = input_type
    vectors = model.encode(texts, **kwargs)
    vectors = np.asarray(vectors, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        np.divide(vectors, norms, out=vectors, where=norms > 0)
    return vectors
