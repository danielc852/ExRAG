"""Create the embedding model and encode text batches."""

from __future__ import annotations

from typing import Any

import numpy as np


def create_embedding_model(model_name: str, revision: str | None = None):
    from sentence_transformers import SentenceTransformer

    kwargs = {"revision": revision} if revision else {}
    return SentenceTransformer(model_name, **kwargs)


def encode_chunk_shard(
    model: Any,
    texts: list[str],
    *,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    if not texts:
        dimension = int(model.get_sentence_embedding_dimension())
        return np.empty((0, dimension), dtype=np.float32)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        np.divide(vectors, norms, out=vectors, where=norms > 0)
    return vectors
