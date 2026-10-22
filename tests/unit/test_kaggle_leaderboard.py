import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from slayer.kaggle import KaggleClient, KaggleError

CSV_CONTENT = (
    "\ufeffRank,TeamId,TeamName,LastSubmissionDate,Score,SubmissionCount,TeamMemberUserNames\n"
    "1,16913247,Team Alicia,2026-09-30 23:11:44,0.94945,139,\"krno1legend,lowasitgoes,uradkr\"\n"
    "2545,16811711,feng,2026-09-11 13:55:50,0.94149,1,guofungwong\n"
    "2546,16964693,Gabriel A,2026-09-26 06:42:57,0.94149,1,gantunes2\n"
)


def _fake_api(tmp_path: Path) -> SimpleNamespace:
    """Fake SDK api that writes a realistic leaderboard zip on download."""

    def download(competition: str, path: str, quiet: bool = True) -> None:
        zip_path = Path(path) / f"{competition}-publicleaderboard.csv.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("publicleaderboard.csv", CSV_CONTENT)

    return SimpleNamespace(competition_leaderboard_download=download)


def test_full_leaderboard_parses_rows_and_usernames(tmp_path) -> None:
    """The full CSV download parses ranks, scores, and member usernames."""
    client = KaggleClient(_fake_api(tmp_path))
    rows = client.full_leaderboard("comp-x")

    assert len(rows) == 3
    assert rows[0].rank == 1
    assert rows[0].team_name == "Team Alicia"
    assert rows[0].score == 0.94945
    assert rows[0].member_usernames == ["krno1legend", "lowasitgoes", "uradkr"]
    ours = rows[2]
    assert ours.member_usernames == ["gantunes2"]
    assert ours.score == 0.94149


def test_full_leaderboard_keeps_csv_artifact_when_target_dir_given(tmp_path) -> None:
    """With target_dir, the extracted leaderboard.csv persists as a run artifact."""
    client = KaggleClient(_fake_api(tmp_path))
    keep = tmp_path / "reports"
    client.full_leaderboard("comp-x", target_dir=keep)

    artifact = keep / "leaderboard.csv"
    assert artifact.is_file()
    content = artifact.read_text(encoding="utf-8-sig")
    assert "Team Alicia" in content


def test_download_leaderboard_failure_wrapped(tmp_path) -> None:
    """SDK download failures surface as KaggleError."""

    def broken_download(competition, path, quiet=True):
        raise RuntimeError("nope")

    client = KaggleClient(SimpleNamespace(competition_leaderboard_download=broken_download))
    with pytest.raises(KaggleError, match="leaderboard download failed"):
        client.download_leaderboard("comp-x", tmp_path)


def test_download_leaderboard_missing_zip_raises(tmp_path) -> None:
    """A download that produces no zip fails clearly."""

    def silent_download(competition, path, quiet=True):
        return None

    client = KaggleClient(SimpleNamespace(competition_leaderboard_download=silent_download))
    with pytest.raises(KaggleError, match="not found"):
        client.download_leaderboard("comp-x", tmp_path)
