from __future__ import annotations

from datetime import UTC, datetime

from slayer.workspace.journal import EventType, Journal


class QuotaManager:
    """Counts daily real submissions from the journal to enforce the quota."""

    def __init__(self, journal: Journal, max_per_day: int) -> None:
        """Bind the manager to a run journal and a daily submission cap."""
        self._journal = journal
        self._max_per_day = max_per_day

    def used_today(self, now: datetime | None = None) -> int:
        """Count real (Kaggle) submissions recorded with today's date."""
        now = now or datetime.now(UTC)
        today = now.date().isoformat()
        return sum(
            1
            for event in self._journal.read(type=EventType.SUBMISSION_ACCEPTED)
            if event.ts.startswith(today)
        )

    def remaining(self, now: datetime | None = None) -> int:
        """Submissions left today under the cap."""
        return max(0, self._max_per_day - self.used_today(now))

    def can_submit(self, now: datetime | None = None) -> bool:
        """Whether another real submission is allowed today."""
        return self.remaining(now) > 0
