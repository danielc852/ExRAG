from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import data.preprocessing.data_cleaning as cleaning_module
import data.preprocessing.download as download_module
import data.processing.process as embedding_module
from data import (
    ArtifactLayout,
    DownloadConfig,
    EmbeddingConfig,
    IndexConfig,
    ProcessingConfig,
    build_faiss_index,
    clean_data,
    download_dataset,
    embed_chunks,
    get_status,
    load_frozen_questions,
    pre_data,
    pre_store,
    review_download,
    run_process,
    validate_index,
)
from data.artifacts import fingerprint, load_manifest, write_manifest_atomic
from data.preprocessing.data_cleaning import normalize_text
from data.processing.embed import encode_chunk_shard
from tools import FaissRetriever


class FakeDataset:
    def __init__(self, rows, fingerprint_value):
        self.rows = list(rows)
        self._fingerprint = fingerprint_value

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class WholeTextSplitter:
    def split_text(self, text):
        return [text]


class FakeEmbedding:
    def __init__(self, *_args, **_kwargs):
        pass

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0, 0.0] if "alpha" in lowered else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def fake_sources(monkeypatch):
    documents = FakeDataset(
        [
            {"doc_id": "d1", "source_type": "slack", "title": "One", "content": "Alpha policy"},
            {"doc_id": "d2", "source_type": "gmail", "title": "Two", "content": "Beta policy"},
        ],
        "documents-fingerprint",
    )
    questions = FakeDataset(
        [
            {
                "question_id": "q1",
                "question_type": "basic",
                "source_types": ["slack"],
                "question": "What is the alpha policy?",
                "expected_doc_ids": ["d1"],
                "gold_answer": "Alpha policy",
                "answer_facts": ["Alpha"],
            }
        ],
        "questions-fingerprint",
    )
    monkeypatch.setattr(
        download_module,
        "load_hf_dataset",
        lambda name, **_kwargs: documents if name == "documents" else questions,
    )
    monkeypatch.setattr(
        cleaning_module, "create_text_splitter", lambda *_args: WholeTextSplitter()
    )
    monkeypatch.setattr(
        embedding_module, "create_embedding_model", lambda *_args: FakeEmbedding()
    )
    return documents, questions


def configs(root: Path):
    return (
        DownloadConfig(artifact_root=root, document_limit=2, shard_size=1),
        ProcessingConfig(artifact_root=root, tokenizer_model="fake"),
        EmbeddingConfig(artifact_root=root, model_name="fake", batch_size=2),
        IndexConfig(artifact_root=root, batch_size=1),
    )


def run_fake_pipeline(root: Path):
    download, processing, embedding, index = configs(root)
    source_manifest = download_dataset(download)
    processed_manifest = clean_data(processing)
    embedding_manifest = embed_chunks(embedding)
    index_manifest = build_faiss_index(index)
    return source_manifest, processed_manifest, embedding_manifest, index_manifest


def test_simple_pipeline_entry_points(tmp_path, fake_sources):
    root = tmp_path / "artifacts"
    download, processing, embedding, index = configs(root)

    data_manifests = pre_data(download, processing)
    assert list(data_manifests) == ["download", "process"]

    store_manifests = pre_store(embedding, index)
    assert list(store_manifests) == ["embed", "index"]
    assert all(stage["status"] == "complete" for stage in get_status(root).values())

    second_root = tmp_path / "all-artifacts"
    manifests = run_process(*configs(second_root))
    assert list(manifests) == ["download", "process", "embed", "index"]


def test_pipeline_builds_sharded_artifacts_and_retrieves(tmp_path, fake_sources, monkeypatch):
    manifests = run_fake_pipeline(tmp_path / "artifacts")
    source, processed, embeddings, index = manifests
    assert source.stats["document_shard_count"] == 2
    assert processed.stats["chunk_count"] == 2
    assert embeddings.stats["vector_count"] == 2
    assert index.stats["chunk_count"] == 2
    assert index.metadata["source_fingerprint"] == source.output_fingerprint
    assert len(load_frozen_questions(tmp_path / "artifacts")) == 1
    assert validate_index(tmp_path / "artifacts").status == "complete"

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeEmbedding)
    retriever = FaissRetriever.load(tmp_path / "artifacts")
    try:
        result = retriever.search("alpha", top_k=1)
    finally:
        retriever.close()
    assert result.chunks[0].document_id == "d1"


def test_completed_stage_is_idempotent_and_config_change_requires_rebuild(
    tmp_path, fake_sources
):
    root = tmp_path / "artifacts"
    download, *_rest = configs(root)
    first = download_dataset(download)
    second = download_dataset(download)
    assert second.output_fingerprint == first.output_fingerprint
    changed = DownloadConfig(artifact_root=root, document_limit=1, shard_size=1)
    with pytest.raises(ValueError, match="Use --rebuild"):
        download_dataset(changed)


def test_download_resumes_only_unfinished_shards_and_cleans_temp_files(
    tmp_path, fake_sources, monkeypatch
):
    root = tmp_path / "artifacts"
    download, *_rest = configs(root)
    original_writer = download_module._write_parquet_atomic
    failed = False

    def flaky_writer(table, destination):
        nonlocal failed
        if destination.name == "part-000001.parquet" and not failed:
            failed = True
            raise RuntimeError("simulated interruption")
        return original_writer(table, destination)

    monkeypatch.setattr(download_module, "_write_parquet_atomic", flaky_writer)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        download_dataset(download)
    layout = ArtifactLayout(root)
    partial = load_manifest(layout, "download")
    assert partial.status == "building"
    assert partial.completed_units == ["questions", "part-000000"]
    orphan = layout.source / "documents" / "orphan.tmp"
    orphan.write_text("partial", encoding="utf-8")

    with pytest.raises(ValueError, match="enable --resume"):
        download_dataset(download, resume=False)
    monkeypatch.setattr(download_module, "_write_parquet_atomic", original_writer)
    completed = download_dataset(download, resume=True)
    assert completed.status == "complete"
    assert completed.completed_units == ["questions", "part-000000", "part-000001"]
    assert not orphan.exists()


def test_completed_download_detects_corrupted_shard(tmp_path, fake_sources):
    root = tmp_path / "artifacts"
    download, *_rest = configs(root)
    manifest = download_dataset(download)
    layout = ArtifactLayout(root)
    document_shard = next(shard for shard in manifest.shards if shard.kind == "documents")
    with (layout.source / document_shard.path).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_dataset(download)


def test_download_review_rejects_manifest_count_mismatch(tmp_path, fake_sources):
    root = tmp_path / "artifacts"
    download, *_rest = configs(root)
    manifest = download_dataset(download)
    manifest.stats["document_count"] += 1
    write_manifest_atomic(ArtifactLayout(root), manifest)

    with pytest.raises(ValueError, match="documents count"):
        review_download(root)


def test_process_rebuild_invalidates_downstream(tmp_path, fake_sources):
    root = tmp_path / "artifacts"
    _download, processing, _embedding, _index = configs(root)
    run_fake_pipeline(root)
    clean_data(processing, rebuild=True)
    layout = ArtifactLayout(root)
    assert layout.processed.exists()
    assert not layout.embeddings.exists()
    assert not layout.index.exists()
    status = get_status(root)
    assert status["process"]["status"] == "complete"
    assert status["embed"]["status"] == "missing"


def test_missing_upstream_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare download"):
        clean_data(ProcessingConfig(artifact_root=tmp_path / "artifacts"))


def test_schema_v1_and_unsafe_artifact_roots_are_rejected(tmp_path):
    layout = ArtifactLayout(tmp_path / "artifacts")
    layout.source.mkdir(parents=True)
    layout.manifest_path("download").write_text(
        '{"schema_version":1,"stage":"download","config":{},"config_hash":"x"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema v1"):
        load_manifest(layout, "download")
    with pytest.raises(ValueError, match="dedicated directory"):
        ArtifactLayout(Path.home())


def test_normalize_and_fingerprint_are_deterministic():
    assert normalize_text("ＡＢＣ  \r\n\r\n\r\n\r\nNext  ") == "ABC\n\n\nNext"
    assert fingerprint({"b": 2, "a": 1}) == fingerprint({"a": 1, "b": 2})


def test_encode_chunk_shard_enforces_float32_unit_vectors():
    class UnnormalizedEmbedding(FakeEmbedding):
        def encode(self, texts, **_kwargs):
            return np.asarray([[3.0, 4.0] for _text in texts], dtype=np.float64)

    vectors = encode_chunk_shard(
        UnnormalizedEmbedding(), ["one", "two"], batch_size=2, normalize=True
    )
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2))


def test_index_validation_rejects_broken_source_lineage(tmp_path, fake_sources):
    root = tmp_path / "artifacts"
    run_fake_pipeline(root)
    layout = ArtifactLayout(root)
    source = load_manifest(layout, "download")
    source.output_fingerprint = "different-source"
    write_manifest_atomic(layout, source)
    with pytest.raises(ValueError, match="pipeline lineage"):
        validate_index(root)
