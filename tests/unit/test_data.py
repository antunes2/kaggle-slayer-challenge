import hashlib
import zipfile

import pytest

from slayer.loop.data import (
    DataCollectionError,
    build_manifest,
    safe_extract,
)


def _make_zip(path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)


def test_safe_extract_rejects_zip_slip(tmp_path) -> None:
    zip_path = tmp_path / "evil.zip"
    _make_zip(zip_path, {"../evil.txt": "boom"})
    with pytest.raises(DataCollectionError, match="unsafe zip member"):
        safe_extract(zip_path, tmp_path / "data")


def test_safe_extract_allows_nested_paths(tmp_path) -> None:
    zip_path = tmp_path / "ok.zip"
    _make_zip(zip_path, {"train.csv": "a,b\n", "sub/test.csv": "c\n"})
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    extracted = safe_extract(zip_path, data_dir)
    assert sorted(extracted) == ["sub/test.csv", "train.csv"]
    assert (data_dir / "sub" / "test.csv").read_text() == "c\n"


def test_safe_extract_rejects_empty_archive(tmp_path) -> None:
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w"):
        pass
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with pytest.raises(DataCollectionError, match="empty"):
        safe_extract(zip_path, data_dir)


def test_build_manifest_hashes_and_sizes(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_bytes(b"1,2,3\n")
    (data_dir / "test.csv").write_bytes(b"4,5\n")

    manifest = build_manifest("comp-x", data_dir, ["train.csv", "test.csv"], "2026-10-19")
    assert [f.name for f in manifest.files] == ["test.csv", "train.csv"]
    assert manifest.total_bytes == len(b"4,5\n") + len(b"1,2,3\n")
    train_entry = next(f for f in manifest.files if f.name == "train.csv")
    assert train_entry.sha256 == hashlib.sha256(b"1,2,3\n").hexdigest()
