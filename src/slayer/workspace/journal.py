import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EventType:
    """Known journal event type constants (free-form strings are allowed too)."""

    RUN_CREATED = "run.created"
    RUN_STATUS = "run.status_changed"
    PHASE_ENTER = "phase.enter"
    LOOP_PAUSED = "loop.paused"
    COMPETITION_CONTEXT = "competition.context"
    DATA_COLLECTED = "data.collected"
    AGENT_CALL = "agent.call"
    AGENT_OUTPUT = "agent.output"
    CANDIDATE_ADDED = "candidate.added"
    CANDIDATE_STATUS = "candidate.status_changed"
    CANDIDATE_CV = "candidate.scored"
    CANDIDATE_LB = "candidate.lb_scored"
    SUBMISSION_ACCEPTED = "submission.accepted"
    SUBMISSION_MOCKED = "submission.mocked"
    ITERATE_SKIPPED = "iterate.skipped"
    MONITOR_POSITION = "monitor.position"
    USER_GUIDANCE = "user.guidance"
    BUDGET_WARNING = "budget.warning"


class JournalEvent(BaseModel):
    """One append-only event: timestamp, agent, type, and payload."""

    ts: str
    run: str | None = None
    agent: str
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class Journal:
    """Append-only JSONL event log — the run's source of truth."""

    def __init__(self, path: Path) -> None:
        """Open (and create if needed) a journal file."""
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)

    @property
    def path(self) -> Path:
        """Journal file location."""
        return self._path

    def append(
        self,
        agent: str,
        type: str,
        data: dict[str, Any] | None = None,
        *,
        run: str | None = None,
    ) -> JournalEvent:
        """Append one event (timestamped now) and return it."""
        event = JournalEvent(
            ts=datetime.now(UTC).isoformat(timespec="seconds"),
            run=run,
            agent=agent,
            type=type,
            data=data or {},
        )
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")
        return event

    def read(
        self,
        *,
        type: str | None = None,
        agent: str | None = None,
        limit: int | None = None,
    ) -> list[JournalEvent]:
        """Read events (optionally filtered by type/agent), newest last."""
        events: list[JournalEvent] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                event = JournalEvent.model_validate_json(line)
                if type is not None and event.type != type:
                    continue
                if agent is not None and event.agent != agent:
                    continue
                events.append(event)
        if limit is not None:
            events = events[-limit:]
        return events

    def tail(self, n: int = 20) -> list[JournalEvent]:
        """Return the last n events."""
        return self.read(limit=n)

    @staticmethod
    def render(events: list[JournalEvent]) -> str:
        """Render events as pretty JSON (for display)."""
        return json.dumps([e.model_dump() for e in events], indent=2)
