from __future__ import annotations

from datetime import UTC, datetime

from slayer.workspace.journal import EventType, Journal


class QuotaManager:
    def __init__(self, journal: Journal, max_per_day: int) -> None:
        self._journal = journal
        self._max_per_day = max_per_day

    def used_today(self, now: datetime | None = None) -> int:
        now = now or datetime.now(UTC)
        today = now.date().isoformat()
        return sum(
            1
            for event in self._journal.read(type=EventType.SUBMISSION_ACCEPTED)
            if event.ts.startswith(today)
        )

    def remaining(self, now: datetime | None = None) -> int:
        return max(0, self._max_per_day - self.used_today(now))

    def can_submit(self, now: datetime | None = None) -> bool:
        return self.remaining(now) > 0
