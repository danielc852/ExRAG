"""SentenceTransformer-shaped adapter around the OpenRouter API client."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .client import OpenRouterClient


DEFAULT_OPENROUTER_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"
KNOWN_MODEL_DIMENSIONS = {DEFAULT_OPENROUTER_EMBEDDING_MODEL: 2048}


class OpenRouterEmbedder:
    """Expose remote OpenRouter embeddings through the local encode contract."""

    supports_input_type = True

    def __init__(
        self,
        model_name: str,
        *,
        client: OpenRouterClient,
    ) -> None:
        if not model_name.strip():
            raise ValueError("OpenRouter embedding model must not be empty")
        self.model_name = model_name
        self.client = client
        self._dimension = KNOWN_MODEL_DIMENSIONS.get(model_name)

    def get_sentence_embedding_dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError(
                "OpenRouter embedding dimension is unknown before the first request"
            )
        return self._dimension

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        input_type: str = "document",
    ) -> np.ndarray:
        del show_progress_bar
        if batch_size < 1:
            raise ValueError("embedding batch size must be at least 1")
        if not convert_to_numpy:
            raise ValueError("The OpenRouter embedder returns NumPy arrays")
        if input_type not in {"query", "document"}:
            raise ValueError("embedding input_type must be 'query' or 'document'")
        if not texts:
            return np.empty(
                (0, self.get_sentence_embedding_dimension()), dtype=np.float32
            )

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            values = self.client.embed(
                model=self.model_name,
                texts=texts[start : start + batch_size],
                input_type=input_type,
            )
            batch = np.asarray(values, dtype=np.float32)
            if batch.ndim != 2 or batch.shape[0] != len(
                texts[start : start + batch_size]
            ):
                raise ValueError("OpenRouter returned an invalid embedding matrix")
            if self._dimension is not None and batch.shape[1] != self._dimension:
                raise ValueError("OpenRouter embedding dimension changed")
            self._dimension = int(batch.shape[1])
            if normalize_embeddings:
                norms = np.linalg.norm(batch, axis=1, keepdims=True)
                np.divide(batch, norms, out=batch, where=norms > 0)
            batches.append(batch)
        return np.concatenate(batches, axis=0)
