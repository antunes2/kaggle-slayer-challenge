from types import SimpleNamespace

import pytest

from slayer.kaggle import KaggleClient, KaggleError


def _competition(ref: str) -> SimpleNamespace:
    return SimpleNamespace(
        ref=f"https://www.kaggle.com/competitions/{ref}",
        title=ref,
        category=None,
        reward=None,
        deadline=None,
        evaluation_metric=None,
        user_has_entered=False,
        max_daily_submissions=None,
    )


def test_find_competition_paginates_and_normalizes_url_refs() -> None:
    pages = [
        SimpleNamespace(competitions=[_competition("other-a")], next_page_token="tok-1"),
        SimpleNamespace(competitions=[_competition("other-b")], next_page_token="tok-2"),
        SimpleNamespace(competitions=[_competition("comp-x")], next_page_token=""),
    ]
    calls = []

    def fake_list(**kwargs) -> SimpleNamespace:
        calls.append(kwargs)
        return pages[len(calls) - 1]

    client = KaggleClient(SimpleNamespace(competitions_list=fake_list))
    info = client.find_competition("comp-x")
    assert info.ref == "comp-x"
    assert calls[0]["page_token"] is None
    assert calls[1]["page_token"] == "tok-1"
    assert calls[2]["page_token"] == "tok-2"


def test_find_competition_falls_back_to_ref_search() -> None:
    def fake_list(**kwargs) -> SimpleNamespace:
        if kwargs.get("search") == "comp-x":
            return SimpleNamespace(competitions=[_competition("comp-x")], next_page_token="")
        return SimpleNamespace(competitions=[_competition("other")], next_page_token="")

    client = KaggleClient(SimpleNamespace(competitions_list=fake_list))
    info = client.find_competition("comp-x")
    assert info.ref == "comp-x"


def test_find_competition_stops_at_max_pages() -> None:
    page = SimpleNamespace(competitions=[_competition("other")], next_page_token="always")
    client = KaggleClient(SimpleNamespace(competitions_list=lambda **kw: page))
    with pytest.raises(KaggleError, match="not found"):
        client.find_competition("comp-x", max_pages=3)


def test_find_competition_not_found_raises_actionable_error() -> None:
    page = SimpleNamespace(competitions=[_competition("other")], next_page_token="")

    def fake_list(**kwargs) -> SimpleNamespace:
        return page

    client = KaggleClient(SimpleNamespace(competitions_list=fake_list))
    with pytest.raises(KaggleError, match="check the ref"):
        client.find_competition("comp-x")
