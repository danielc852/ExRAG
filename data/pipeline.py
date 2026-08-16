"""Orchestration helpers for individual and complete preparation runs."""

from __future__ import annotations

from typing import Any

from .artifacts import ArtifactLayout, STAGE_ORDER, StageManifest, StageName, load_manifest
from .download import download_dataset
from .embedding import embed_chunks
from .indexing import build_faiss_index
from .models import DownloadConfig, EmbeddingConfig, IndexConfig, ProcessingConfig
from .processing import process_documents


StageConfig = DownloadConfig | ProcessingConfig | EmbeddingConfig | IndexConfig


def run_stage(
    stage: StageName,
    config: StageConfig,
    *,
    resume: bool = True,
    rebuild: bool = False,
) -> StageManifest:
    runners: dict[str, tuple[type, Any]] = {
        "download": (DownloadConfig, download_dataset),
        "process": (ProcessingConfig, process_documents),
        "embed": (EmbeddingConfig, embed_chunks),
        "index": (IndexConfig, build_faiss_index),
    }
    expected_type, runner = runners[stage]
    if not isinstance(config, expected_type):
        raise TypeError(f"Stage {stage!r} requires {expected_type.__name__}")
    return runner(config, resume=resume, rebuild=rebuild)


def run_all(
    download_config: DownloadConfig,
    processing_config: ProcessingConfig,
    embedding_config: EmbeddingConfig,
    index_config: IndexConfig,
    *,
    resume: bool = True,
    rebuild: bool = False,
) -> dict[str, StageManifest]:
    roots = {
        config.artifact_root.resolve(strict=False)
        for config in (
            download_config,
            processing_config,
            embedding_config,
            index_config,
        )
    }
    if len(roots) != 1:
        raise ValueError("All pipeline stages must use the same artifact root")
    manifests = {
        "download": download_dataset(download_config, resume=resume, rebuild=rebuild)
    }
    manifests["process"] = process_documents(processing_config, resume=resume)
    manifests["embed"] = embed_chunks(embedding_config, resume=resume)
    manifests["index"] = build_faiss_index(index_config, resume=resume)
    return manifests


def get_pipeline_status(artifact_root) -> dict[str, dict[str, Any]]:
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
