from __future__ import annotations

import zipfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel

from slayer.loop.engine import RunContext
from slayer.workspace import EventType


class DataCollectionError(Exception):
    pass


class FileEntry(BaseModel):
    name: str
    size_bytes: int
    sha256: str


class DataManifest(BaseModel):
    competition: str
    total_bytes: int
    files: list[FileEntry]
    created_at: str


def safe_extract(zip_path: Path, target: Path) -> list[str]:
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


def data_collection_phase(ctx: RunContext) -> None:
    data_dir = ctx.run.workspace("data")
    zip_path = ctx.kaggle.download_files(ctx.run.competition, data_dir)
    ctx.say(f"  downloaded {zip_path.name}")
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
        data={"files": len(extracted), "total_bytes": manifest.total_bytes},
        run=ctx.run.run_id,
    )
    ctx.say(
        f"  extracted {len(extracted)} file(s), {manifest.total_bytes} bytes — manifest written"
    )
