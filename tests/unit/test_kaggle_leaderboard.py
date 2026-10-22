from types import SimpleNamespace

from slayer.kaggle import KaggleClient


def _entry(team: str, score: str) -> SimpleNamespace:
    return SimpleNamespace(team_name=team, score=score)


def test_leaderboard_paginates_and_suppresses_sdk_prints(capsys) -> None:
    """Pagination threads the SDK's printed page token; its noise never leaks."""
    pages = [
        ([_entry("team-a", "0.95"), _entry("team-b", "0.94")], "tok-1"),
        ([_entry("team-c", "0.93")], None),
    ]
    calls = []

    def fake_view(competition, page_size=50, page_token=None):
        calls.append(page_token)
        entries, token = pages[len(calls) - 1]
        if token:
            print(f"Next Page Token = {token}")
        return entries

    client = KaggleClient(SimpleNamespace(competition_leaderboard_view=fake_view))
    entries = client.leaderboard("comp-x", page_size=2, max_pages=5)

    assert [e.team_name for e in entries] == ["team-a", "team-b", "team-c"]
    assert [e.rank for e in entries] == [1, 2, 3]
    assert calls == [None, "tok-1"]
    assert "Next Page Token" not in capsys.readouterr().out


def test_leaderboard_stops_without_token() -> None:
    """A single page with no token ends pagination immediately."""
    client = KaggleClient(
        SimpleNamespace(
            competition_leaderboard_view=lambda c, page_size=50, page_token=None: [
                _entry("solo", "0.9")
            ]
        )
    )
    entries = client.leaderboard("comp-x")
    assert [e.team_name for e in entries] == ["solo"]


def test_leaderboard_respects_max_pages() -> None:
    """Pagination is bounded by max_pages even if tokens keep coming."""
    client = KaggleClient(
        SimpleNamespace(
            competition_leaderboard_view=lambda c, page_size=50, page_token=None: (
                print("Next Page Token = forever") or [_entry("x", "0.9")]
            )
        )
    )
    entries = client.leaderboard("comp-x", page_size=1, max_pages=3)
    assert len(entries) == 3
