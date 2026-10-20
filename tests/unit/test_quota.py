from slayer.kaggle import QuotaManager
from slayer.workspace import EventType, Journal, JournalEvent


def _write_old_submission(path) -> None:
    old = JournalEvent(
        ts="2020-01-01T10:00:00+00:00",
        agent="system",
        type=EventType.SUBMISSION_ACCEPTED,
        run="run-x",
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(old.model_dump_json() + "\n")


def test_used_today_counts_only_todays_events(tmp_path) -> None:
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append("system", EventType.SUBMISSION_ACCEPTED, {"submission_id": "s1"})
    _write_old_submission(journal.path)

    quota = QuotaManager(journal, max_per_day=2)
    assert quota.used_today() == 1
    assert quota.remaining() == 1
    assert quota.can_submit()


def test_ignores_other_event_types(tmp_path) -> None:
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append("system", EventType.CANDIDATE_LB, {})
    journal.append("system", EventType.CANDIDATE_ADDED, {})

    quota = QuotaManager(journal, max_per_day=2)
    assert quota.used_today() == 0


def test_blocks_at_limit(tmp_path) -> None:
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append("system", EventType.SUBMISSION_ACCEPTED, {})

    quota = QuotaManager(journal, max_per_day=1)
    assert not quota.can_submit()
    assert quota.remaining() == 0


def test_used_today_respects_injected_now(tmp_path) -> None:
    from datetime import UTC, datetime

    journal = Journal(tmp_path / "journal.jsonl")
    _write_old_submission(journal.path)

    quota = QuotaManager(journal, max_per_day=1)
    assert quota.used_today() == 0
    assert quota.used_today(now=datetime(2020, 1, 1, 12, tzinfo=UTC)) == 1
