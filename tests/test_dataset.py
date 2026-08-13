from __future__ import annotations

import dataset as dataset_module
from dataset import DocumentRecord, IndexBuildConfig, build_index, chunk_document, normalize_text, select_documents

import numpy as np


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def shuffle(self, seed):
        import random

        rows = self.rows.copy()
        random.Random(seed).shuffle(rows)
        return FakeDataset(rows)

    def select(self, indices):
        return FakeDataset(self.rows[index] for index in indices)


class FakeSplitter:
    def split_text(self, _text):
        return ["Alpha  ", "Alpha", "Beta\r\nline", "   "]


class FakeEmbedding:
    def __init__(self, _model_name):
        pass

    def encode(self, texts, **_kwargs):
        vectors = [[float(index + 1), 1.0] for index, _text in enumerate(texts)]
        values = np.asarray(vectors, dtype="float32")
        return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_normalize_text_preserves_paragraphs_and_normalizes_unicode():
    value = "ＡＢＣ  \r\n\r\n\r\n\r\nNext  "
    assert normalize_text(value) == "ABC\n\n\nNext"


def test_sample_is_deterministic_and_bounded():
    dataset = FakeDataset(range(20))
    first = list(select_documents(dataset, full=False, limit=5, seed=42))
    second = list(select_documents(dataset, full=False, limit=5, seed=42))
    assert first == second
    assert len(first) == 5
    assert list(select_documents(dataset, full=True, limit=1, seed=42)) == list(range(20))


def test_chunk_document_deduplicates_only_within_document():
    document = DocumentRecord("doc-1", "slack", "A title", "ignored")
    chunks = list(chunk_document(document, FakeSplitter()))
    assert [chunk.chunk_id for chunk in chunks] == ["doc-1::000000", "doc-1::000001"]
    assert [chunk.content for chunk in chunks] == ["Alpha", "Beta\nline"]


def test_build_index_with_injected_local_components(tmp_path, monkeypatch):
    rows = FakeDataset(
        [
            {"doc_id": "d1", "source_type": "slack", "title": "One", "content": "Alpha"},
            {"doc_id": "d2", "source_type": "gmail", "title": "Two", "content": "Beta"},
        ]
    )
    monkeypatch.setattr(dataset_module, "load_documents", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(dataset_module, "create_text_splitter", lambda *_args: FakeSplitter())
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeEmbedding)
    config = IndexBuildConfig(
        index_dir=tmp_path / "index",
        document_limit=2,
        checkpoint_every=1,
        embedding_model="fake-model",
    )
    manifest = build_index(config)
    assert manifest.status == "complete"
    assert manifest.processed_documents == 2
    assert manifest.chunk_count == 4
