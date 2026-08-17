"""Artifact layout, integrity metadata, and stage lifecycle helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


SCHEMA_VERSION = 2
StageName = Literal["download", "process", "embed", "index"]
STAGE_ORDER: tuple[StageName, ...] = ("download", "process", "embed", "index")
STAGE_DIRECTORIES: dict[StageName, str] = {
    "download": "source",
    "process": "processed",
    "embed": "embeddings",
    "index": "index",
}


class ShardInfo(BaseModel):
    kind: str
    path: str
    row_count: int = Field(ge=0)
    sha256: str
    min_id: int | None = None
    max_id: int | None = None


class StageManifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    stage: StageName
    status: Literal["building", "complete"] = "building"
    config: dict[str, Any]
    config_hash: str
    upstream_fingerprint: str | None = None
    output_fingerprint: str = ""
    shards: list[ShardInfo] = Field(default_factory=list)
    completed_units: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactLayout:
    """Resolve only the fixed stage directories under an artifact root."""

    def __init__(self, root: Path | str = Path("artifacts")) -> None:
        root_path = Path(root).expanduser()
        resolved = root_path.resolve(strict=False)
        if (
            str(root_path).strip() in {"", "/", "~"}
            or resolved == Path("/")
            or resolved == Path.home().resolve()
        ):
            raise ValueError("artifact root must be a dedicated directory")
        self.root = root_path

    def stage_dir(self, stage: StageName) -> Path:
        if stage not in STAGE_DIRECTORIES:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        return self.root / STAGE_DIRECTORIES[stage]

    def manifest_path(self, stage: StageName) -> Path:
        return self.stage_dir(stage) / "manifest.json"

    @property
    def source(self) -> Path:
        return self.stage_dir("download")

    @property
    def processed(self) -> Path:
        return self.stage_dir("process")

    @property
    def embeddings(self) -> Path:
        return self.stage_dir("embed")

    @property
    def index(self) -> Path:
        return self.stage_dir("index")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def config_dict(config: Any) -> dict[str, Any]:
    payload = _jsonable(config)
    if not isinstance(payload, dict):
        raise TypeError("stage config must serialize to a mapping")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_info(
    *,
    kind: str,
    path: Path,
    base_dir: Path,
    row_count: int,
    min_id: int | None = None,
    max_id: int | None = None,
) -> ShardInfo:
    return ShardInfo(
        kind=kind,
        path=str(path.relative_to(base_dir)),
        row_count=row_count,
        sha256=file_sha256(path),
        min_id=min_id,
        max_id=max_id,
    )


def load_manifest(layout: ArtifactLayout, stage: StageName) -> StageManifest:
    path = layout.manifest_path(stage)
    if not path.exists():
        command = "download" if stage == "download" else "init_vectordb"
        raise FileNotFoundError(
            f"Missing {stage} artifacts. "
            f"Run `python main.py {command} sample|full` first."
        )
    manifest = StageManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Artifact schema v{manifest.schema_version} is unsupported; rebuild pipeline schema v{SCHEMA_VERSION}"
        )
    if manifest.stage != stage:
        raise ValueError(f"Manifest at {path} belongs to stage {manifest.stage!r}")
    return manifest


def write_manifest_atomic(layout: ArtifactLayout, manifest: StageManifest) -> None:
    stage_dir = layout.stage_dir(manifest.stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    destination = layout.manifest_path(manifest.stage)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(destination)


def validate_upstream(layout: ArtifactLayout, stage: StageName) -> StageManifest | None:
    position = STAGE_ORDER.index(stage)
    if position == 0:
        return None
    upstream_stage = STAGE_ORDER[position - 1]
    upstream = load_manifest(layout, upstream_stage)
    if upstream.status != "complete" or not upstream.output_fingerprint:
        command = "download" if upstream_stage == "download" else "init_vectordb"
        raise ValueError(
            f"Upstream stage {upstream_stage!r} is incomplete; "
            f"run `python main.py {command} sample|full`"
        )
    verify_manifest_files(layout, upstream)
    return upstream


def _safe_remove_stage(layout: ArtifactLayout, stage: StageName) -> None:
    target = layout.stage_dir(stage)
    if target.parent != layout.root or target.name != STAGE_DIRECTORIES[stage]:
        raise ValueError(f"Refusing to remove unsafe artifact path: {target}")
    if target.exists():
        shutil.rmtree(target)


def invalidate_from_stage(layout: ArtifactLayout, stage: StageName) -> None:
    start = STAGE_ORDER.index(stage)
    for downstream in STAGE_ORDER[start:]:
        _safe_remove_stage(layout, downstream)


def cleanup_temporary_files(stage_dir: Path) -> None:
    if not stage_dir.exists():
        return
    for path in stage_dir.rglob("*.tmp"):
        if path.is_file():
            path.unlink()


def begin_stage(
    layout: ArtifactLayout,
    stage: StageName,
    config: Any,
    *,
    upstream: StageManifest | None,
    resume: bool,
    rebuild: bool,
) -> StageManifest:
    if rebuild:
        invalidate_from_stage(layout, stage)
    stage_dir = layout.stage_dir(stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    cleanup_temporary_files(stage_dir)
    serialized_config = config_dict(config)
    serialized_config.pop("artifact_root", None)
    current_config_hash = fingerprint(serialized_config)
    upstream_fingerprint = upstream.output_fingerprint if upstream else None
    path = layout.manifest_path(stage)
    if not path.exists():
        manifest = StageManifest(
            stage=stage,
            config=serialized_config,
            config_hash=current_config_hash,
            upstream_fingerprint=upstream_fingerprint,
        )
        write_manifest_atomic(layout, manifest)
        return manifest

    manifest = load_manifest(layout, stage)
    verify_manifest_files(layout, manifest)
    mismatches = []
    if manifest.config_hash != current_config_hash:
        mismatches.append("config")
    if manifest.upstream_fingerprint != upstream_fingerprint:
        mismatches.append("upstream fingerprint")
    if mismatches:
        raise ValueError(
            f"Cannot resume {stage}: {', '.join(mismatches)} changed. Use --rebuild."
        )
    if manifest.status != "complete" and not resume:
        raise ValueError(f"Partial {stage} artifacts exist; enable --resume or use --rebuild")
    return manifest


def finalize_manifest(layout: ArtifactLayout, manifest: StageManifest) -> StageManifest:
    manifest.status = "complete"
    manifest.output_fingerprint = fingerprint(
        {
            "stage": manifest.stage,
            "config_hash": manifest.config_hash,
            "upstream_fingerprint": manifest.upstream_fingerprint,
            "shards": [shard.model_dump(mode="json") for shard in manifest.shards],
            "metadata": manifest.metadata,
        }
    )
    write_manifest_atomic(layout, manifest)
    return manifest


def verify_manifest_files(layout: ArtifactLayout, manifest: StageManifest) -> None:
    stage_dir = layout.stage_dir(manifest.stage)
    for shard in manifest.shards:
        path = stage_dir / shard.path
        if not path.exists():
            raise FileNotFoundError(f"Manifest artifact is missing: {path}")
        if file_sha256(path) != shard.sha256:
            raise ValueError(f"Artifact checksum mismatch: {path}")
