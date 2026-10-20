"""DATA_COLLECTION phase: get competition data on disk, cached across runs."""

from __future__ import annotations

import shutil
import zipfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel

from slayer.kaggle import CompetitionFileInfo
from slayer.loop.engine import RunContext
from slayer.workspace import EventType


class DataCollectionError(Exception):
    """Raised when competition data cannot be collected safely."""

    pass


class FileEntry(BaseModel):
    """One extracted data file's name, size, and content hash."""

    name: str
    size_bytes: int
    sha256: str


class DataManifest(BaseModel):
    """Manifest of the run's extracted data files."""

    competition: str
    total_bytes: int
    files: list[FileEntry]
    created_at: str


class CacheManifest(BaseModel):
    """Cache validity record: remote file list + zip content hash."""

    competition: str
    files: list[CompetitionFileInfo]
    zip_sha256: str
    updated_at: str


def safe_extract(zip_path: Path, target: Path) -> list[str]:
    """Extract a zip into target, rejecting members that escape the target dir."""
    target_resolved = target.resolve()
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            destination = (target / info.filename).resolve()
            if not destination.is_relative_to(target_resolved):
                raise DataCollectionError(
                    f"unsafe zip member escaped target dir: {info.filename}"
                )
            zf.extract(info, target)
            extracted.append(info.filename)
    if not extracted:
        raise DataCollectionError(f"archive is empty: {zip_path}")
    return extracted


def build_manifest(
    competition: str, data_dir: Path, files: list[str], created_at: str
) -> DataManifest:
    """Build the run's data manifest with per-file sizes and sha256 hashes."""
    entries: list[FileEntry] = []
    total = 0
    for name in sorted(files):
        path = data_dir / name
        digest = sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        entries.append(FileEntry(name=name, size_bytes=size, sha256=digest))
        total += size
    return DataManifest(
        competition=competition,
        total_bytes=total,
        files=entries,
        created_at=created_at,
    )


def _cached_zip(ctx: RunContext, remote_files: list[CompetitionFileInfo]) -> Path | None:
    """Return the cached zip if it exists and matches the current remote file list."""
    cache_dir = ctx.store.cache_dir(ctx.run.competition)
    cache_zip = cache_dir / f"{ctx.run.competition}.zip"
    manifest_path = cache_dir / "cache.json"
    if not cache_zip.exists() or not manifest_path.exists():
        return None
    try:
        manifest = CacheManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception:
        return None
    if sha256(cache_zip.read_bytes()).hexdigest() != manifest.zip_sha256:
        return None
    current = sorted((f.name, f.total_bytes) for f in remote_files)
    cached = sorted((f.name, f.total_bytes) for f in manifest.files)
    if current != cached:
        return None
    return cache_zip


def _store_in_cache(
    ctx: RunContext, zip_path: Path, remote_files: list[CompetitionFileInfo]
) -> None:
    """Copy the downloaded zip into the cache with a validity manifest."""
    cache_dir = ctx.store.cache_dir(ctx.run.competition)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, cache_dir / f"{ctx.run.competition}.zip")
    files = [
        CompetitionFileInfo(name=f.name, total_bytes=getattr(f, "total_bytes", None))
        for f in remote_files
    ]
    manifest = CacheManifest(
        competition=ctx.run.competition,
        files=files,
        zip_sha256=sha256(zip_path.read_bytes()).hexdigest(),
        updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    (cache_dir / "cache.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )


def _resolve_zip(
    ctx: RunContext, data_dir: Path
) -> tuple[Path, bool]:
    """Get the competition zip from cache or Kaggle; returns (zip path, from_cache)."""
    remote_files = ctx.kaggle.list_files(ctx.run.competition)
    if not ctx.run.refresh_data:
        cached = _cached_zip(ctx, remote_files)
        if cached is not None:
            return cached, True
    zip_path = ctx.kaggle.download_files(ctx.run.competition, data_dir)
    _store_in_cache(ctx, zip_path, remote_files)
    return zip_path, False


def data_collection_phase(ctx: RunContext) -> None:
    """Collect competition data (cached across runs) and write the run manifest."""
    data_dir = ctx.run.workspace("data")
    zip_path, from_cache = _resolve_zip(ctx, data_dir)
    origin = "reusing cached" if from_cache else "downloaded"
    ctx.say(f"  {origin} {zip_path.name}")
    extracted = safe_extract(zip_path, data_dir)
    manifest = build_manifest(
        ctx.run.competition,
        data_dir,
        extracted,
        datetime.now(UTC).isoformat(timespec="seconds"),
    )
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    ctx.journal.append(
        agent="loop",
        type=EventType.DATA_COLLECTED,
        data={"files": len(extracted), "total_bytes": manifest.total_bytes,
              "from_cache": from_cache},
        run=ctx.run.run_id,
    )
    ctx.say(
        f"  extracted {len(extracted)} file(s), {manifest.total_bytes} bytes — manifest written"
    )
