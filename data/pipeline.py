"""Simple entry points for preparing source data and the vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import ArtifactLayout, STAGE_ORDER, StageManifest, load_manifest
from .models import DownloadConfig, EmbeddingConfig, IndexConfig, ProcessingConfig
from .preprocessing import clean_data, download_dataset, review_download
from .processing import build_faiss_index, embed_chunks


PipelineConfig = DownloadConfig | ProcessingConfig | EmbeddingConfig | IndexConfig


def _require_shared_root(*configs: PipelineConfig) -> None:
    roots = {config.artifact_root.resolve(strict=False) for config in configs}
    if len(roots) != 1:
        raise ValueError("All pipeline stages must use the same artifact root")


def pre_data(
    download_config: DownloadConfig,
    processing_config: ProcessingConfig,
    *,
    resume: bool = True,
    rebuild: bool = False,
) -> dict[str, StageManifest]:
    """Download, review, clean, and chunk the source dataset."""
    _require_shared_root(download_config, processing_config)
    source = download_dataset(download_config, resume=resume, rebuild=rebuild)
    review_download(download_config.artifact_root)
    processed = clean_data(processing_config, resume=resume)
    return {"download": source, "process": processed}


def pre_store(
    embedding_config: EmbeddingConfig,
    index_config: IndexConfig,
    *,
    resume: bool = True,
    rebuild: bool = False,
) -> dict[str, StageManifest]:
    """Embed cleaned chunks and prepare the searchable vector store."""
    _require_shared_root(embedding_config, index_config)
    embeddings = embed_chunks(embedding_config, resume=resume, rebuild=rebuild)
    index = build_faiss_index(index_config, resume=resume)
    return {"embed": embeddings, "index": index}


def run_process(
    download_config: DownloadConfig,
    processing_config: ProcessingConfig,
    embedding_config: EmbeddingConfig,
    index_config: IndexConfig,
    *,
    resume: bool = True,
    rebuild: bool = False,
) -> dict[str, StageManifest]:
    """Run source preprocessing followed by vector-store preparation."""
    _require_shared_root(
        download_config,
        processing_config,
        embedding_config,
        index_config,
    )
    manifests = pre_data(
        download_config,
        processing_config,
        resume=resume,
        rebuild=rebuild,
    )
    manifests.update(
        pre_store(
            embedding_config,
            index_config,
            resume=resume,
        )
    )
    return manifests


def get_status(artifact_root: Path | str) -> dict[str, dict[str, Any]]:
    """Return the lifecycle status and summary for every preparation stage."""
    layout = ArtifactLayout(artifact_root)
    status: dict[str, dict[str, Any]] = {}
    for stage in STAGE_ORDER:
        if not layout.manifest_path(stage).exists():
            status[stage] = {"status": "missing"}
            continue
        try:
            manifest = load_manifest(layout, stage)
            status[stage] = {
                "status": manifest.status,
                "output_fingerprint": manifest.output_fingerprint or None,
                "stats": manifest.stats,
            }
        except (ValueError, OSError) as exc:
            status[stage] = {"status": "invalid", "error": str(exc)}
    return status
