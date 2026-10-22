import enum
import os
from types import SimpleNamespace

import pytest

import slayer.kaggle.client as kaggle_client_module
from slayer.kaggle import KaggleClient, KaggleError


class _Status(enum.Enum):
    COMPLETE = "complete"


def test_ping_returns_username() -> None:
    client = KaggleClient(SimpleNamespace(config_values={"username": "gabriel"}))
    assert client.ping() == "gabriel"


def test_ping_without_username_raises() -> None:
    client = KaggleClient(SimpleNamespace(config_values={}))
    with pytest.raises(KaggleError, match="no username"):
        client.ping()


def test_from_token_with_empty_token_raises() -> None:
    with pytest.raises(KaggleError, match="token is empty"):
        KaggleClient.from_token("")


def test_from_token_injects_env_and_authenticates(monkeypatch) -> None:
    seen = {}

    class FakeApi:
        def authenticate(self) -> None:
            seen["token"] = os.environ.get("KAGGLE_API_TOKEN")

    monkeypatch.setattr(kaggle_client_module, "_default_api_class", lambda: FakeApi)
    KaggleClient.from_token("KGAT-test-token")
    assert seen["token"] == "KGAT-test-token"


def test_from_token_wraps_auth_failure(monkeypatch) -> None:
    class FakeApi:
        def authenticate(self) -> None:
            raise SystemExit(1)

    monkeypatch.setattr(kaggle_client_module, "_default_api_class", lambda: FakeApi)
    with pytest.raises(KaggleError, match="authentication failed"):
        KaggleClient.from_token("KGAT-test-token")


def test_list_competitions_maps_fields() -> None:
    competition = SimpleNamespace(
        ref="playground-series-s5e10",
        title="S5E10",
        category="playground",
        reward=SimpleNamespace(quantity=50000, id=SimpleNamespace(name="USD")),
        deadline="2026-11-01T23:59:00Z",
        evaluation_metric="AUC",
        user_has_entered=False,
        max_daily_submissions=5,
    )
    api = SimpleNamespace(
        competitions_list=lambda **kw: SimpleNamespace(
            competitions=[competition], next_page_token=""
        )
    )
    result = KaggleClient(api).list_competitions(search="playground")
    assert result[0].ref == "playground-series-s5e10"
    assert result[0].title == "S5E10"
    assert result[0].reward == "50000 USD"
    assert result[0].deadline == "2026-11-01T23:59:00Z"
    assert result[0].evaluation_metric == "AUC"
    assert result[0].max_daily_submissions == 5


def test_list_files() -> None:
    api = SimpleNamespace(
        competition_list_files=lambda comp: SimpleNamespace(
            files=[SimpleNamespace(name="train.csv", total_bytes=123), None]
        )
    )
    files = KaggleClient(api).list_files("comp")
    assert len(files) == 1
    assert files[0].name == "train.csv"
    assert files[0].total_bytes == 123


def test_download_files_returns_zip(tmp_path) -> None:
    def fake_download(competition, path=None, force=False, quiet=True) -> None:
        import pathlib

        pathlib.Path(path, f"{competition}.zip").write_bytes(b"zip")

    api = SimpleNamespace(competition_download_files=fake_download)
    zip_path = KaggleClient(api).download_files("comp", tmp_path / "data")
    assert zip_path == tmp_path / "data" / "comp.zip"
    assert zip_path.exists()


def test_download_files_missing_zip_raises(tmp_path) -> None:
    api = SimpleNamespace(competition_download_files=lambda *a, **kw: None)
    with pytest.raises(KaggleError, match="not found"):
        KaggleClient(api).download_files("comp", tmp_path / "data")


def test_submit_returns_ref(tmp_path) -> None:
    file_path = tmp_path / "submission.csv"
    file_path.write_text("id,target\n", encoding="utf-8")
    api = SimpleNamespace(
        competition_submit=lambda fn, msg, comp, quiet=True: SimpleNamespace(
            ref=42, message="Submission received"
        )
    )
    ref = KaggleClient(api).submit("comp", file_path, "baseline")
    assert ref.id == "42"
    assert ref.message == "Submission received"


def test_submit_missing_file_raises(tmp_path) -> None:
    api = SimpleNamespace(competition_submit=lambda *a, **kw: None)
    with pytest.raises(KaggleError, match="not found"):
        KaggleClient(api).submit("comp", tmp_path / "nope.csv")


def test_submissions_maps_status_enum_name() -> None:
    submission = SimpleNamespace(
        ref=42,
        file_name="sub.csv",
        description="baseline",
        status=_Status.COMPLETE,
        public_score="0.84321",
        private_score="0.84000",
        date="2026-10-19",
    )
    api = SimpleNamespace(competition_submissions=lambda comp: [submission, None])
    infos = KaggleClient(api).submissions("comp")
    assert len(infos) == 1
    assert infos[0].status == "COMPLETE"
    assert infos[0].public_score == "0.84321"


def test_leaderboard_enumerates_ranks() -> None:
    api = SimpleNamespace(
        competition_leaderboard_view=lambda comp, page_size=50, page_token=None: [
            SimpleNamespace(team_name="team-a", score="0.9"),
            None,
            SimpleNamespace(team_name="team-b", score="0.8"),
        ]
    )
    entries = KaggleClient(api).leaderboard("comp")
    assert [(e.rank, e.team_name) for e in entries] == [(1, "team-a"), (2, "team-b")]
