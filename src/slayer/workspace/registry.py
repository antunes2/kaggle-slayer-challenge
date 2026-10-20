from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from slayer.workspace.journal import EventType, Journal


class CandidateStatus:
    """Candidate lifecycle states."""

    PROPOSED = "proposed"
    SCORED = "scored"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUBMITTED = "submitted"
    RETIRED = "retired"


class ModelCandidate(BaseModel):
    """Input for registering a new candidate pipeline."""

    model_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    feature_code_ref: str | None = None
    parent_id: str | None = None
    notes: str | None = None


class CandidateRecord(BaseModel):
    """Full candidate row: identity, lineage, lifecycle, and scores."""

    id: str
    run_id: str
    parent_id: str | None = None
    created_at: str
    status: str
    model_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    feature_code_ref: str | None = None
    cv_score: float | None = None
    cv_std: float | None = None
    lb_score: float | None = None
    submission_id: str | None = None
    notes: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    parent_id TEXT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    model_type TEXT NOT NULL,
    params_json TEXT NOT NULL DEFAULT '{}',
    feature_code_ref TEXT,
    cv_score REAL,
    cv_std REAL,
    lb_score REAL,
    submission_id TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);
CREATE INDEX IF NOT EXISTS idx_candidates_cv ON candidates(cv_score);
"""


class CandidateRegistry:
    def __init__(
        self, db_path: Path | str, *, run_id: str, journal: Journal | None = None
    ) -> None:
        """Open (or create) the SQLite registry in WAL mode."""
        self._path = Path(db_path)
        self._run_id = run_id
        self._journal = journal
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def db_path(self) -> Path:
        """SQLite database location."""
        return self._path

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def add(self, candidate: ModelCandidate) -> CandidateRecord:
        """Register a new candidate (status=proposed) and journal it."""
        record = CandidateRecord(
            id=self._next_id(),
            run_id=self._run_id,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            status=CandidateStatus.PROPOSED,
            **candidate.model_dump(),
        )
        self._insert(record)
        self._emit(EventType.CANDIDATE_ADDED, record.model_dump())
        return record

    def get(self, candidate_id: str) -> CandidateRecord:
        """Fetch one candidate by id, raising KeyError if missing."""
        row = self._conn.execute(
            "SELECT * FROM candidates WHERE id=?", (candidate_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"candidate not found: {candidate_id}")
        return self._to_record(row)

    def list(self, *, status: str | None = None) -> list[CandidateRecord]:
        """List candidates (optionally filtered by status), oldest first."""
        if status is None:
            rows = self._conn.execute(
                "SELECT * FROM candidates ORDER BY created_at, id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM candidates WHERE status=? ORDER BY created_at, id",
                (status,),
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def set_status(self, candidate_id: str, status: str) -> CandidateRecord:
        """Move a candidate to a new lifecycle status."""
        record = self.get(candidate_id)
        self._apply_status(candidate_id, status)
        self._emit(
            EventType.CANDIDATE_STATUS,
            {"id": candidate_id, "from": record.status, "to": status},
        )
        return self.get(candidate_id)

    def record_cv(
        self, candidate_id: str, score: float, std: float | None = None
    ) -> CandidateRecord:
        """Record a CV score for a candidate (auto-advancing it to scored)."""
        record = self.get(candidate_id)
        auto_scored = self._apply_cv(candidate_id, score, std)
        self._emit(
            EventType.CANDIDATE_CV,
            {"id": candidate_id, "cv_score": score, "cv_std": std},
        )
        if auto_scored:
            self._emit(
                EventType.CANDIDATE_STATUS,
                {"id": candidate_id, "from": record.status, "to": CandidateStatus.SCORED},
            )
        return self.get(candidate_id)

    def record_lb(
        self, candidate_id: str, lb_score: float, submission_id: str
    ) -> CandidateRecord:
        """Record a leaderboard score and mark the candidate submitted."""
        record = self.get(candidate_id)
        self._apply_lb(candidate_id, lb_score, submission_id)
        self._emit(
            EventType.CANDIDATE_LB,
            {"id": candidate_id, "lb_score": lb_score, "submission_id": submission_id},
        )
        if record.status != CandidateStatus.SUBMITTED:
            self._emit(
                EventType.CANDIDATE_STATUS,
                {"id": candidate_id, "from": record.status, "to": CandidateStatus.SUBMITTED},
            )
        return self.get(candidate_id)

    def best(
        self,
        n: int = 1,
        *,
        higher_is_better: bool = True,
        submitted_only: bool = False,
        unsubmitted_only: bool = False,
    ) -> list[CandidateRecord]:
        """Return the top n candidates by CV score, honoring metric direction."""
        order = "DESC" if higher_is_better else "ASC"
        where = "cv_score IS NOT NULL"
        if submitted_only:
            where += f" AND status = '{CandidateStatus.SUBMITTED}'"
        if unsubmitted_only:
            where += f" AND status != '{CandidateStatus.SUBMITTED}'"
        rows = self._conn.execute(
            f"SELECT * FROM candidates WHERE {where} "
            f"ORDER BY cv_score {order}, created_at ASC LIMIT ?",
            (n,),
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def summary_for_agent(self, n: int = 10) -> str:
        """Render the latest candidates as a compact markdown table for LLM context."""
        records = self.list()
        if not records:
            return "No candidates registered yet."
        lines = ["id | status | model | cv | lb | notes"]
        for record in records[-n:]:
            cv = f"{record.cv_score:.5f}" if record.cv_score is not None else "-"
            lb = f"{record.lb_score:.5f}" if record.lb_score is not None else "-"
            notes = (record.notes or "")[:40]
            lines.append(
                f"{record.id} | {record.status} | {record.model_type} | {cv} | {lb} | {notes}"
            )
        return "\n".join(lines)

    @classmethod
    def rebuild(
        cls, db_path: Path | str, *, run_id: str, journal: Journal
    ) -> CandidateRegistry:
        """Rebuild a fresh registry by replaying candidate.* journal events."""
        path = Path(db_path)
        if path.exists():
            path.unlink()
        registry = cls(path, run_id=run_id, journal=None)
        for event in journal.read():
            data = event.data
            if event.type == EventType.CANDIDATE_ADDED:
                registry._insert(CandidateRecord.model_validate(data))
            elif event.type == EventType.CANDIDATE_CV:
                registry._apply_cv(data["id"], data["cv_score"], data.get("cv_std"))
            elif event.type == EventType.CANDIDATE_STATUS:
                registry._apply_status(data["id"], data["to"])
            elif event.type == EventType.CANDIDATE_LB:
                registry._apply_lb(data["id"], data["lb_score"], data["submission_id"])
        return registry

    def _next_id(self) -> str:
        """Generate the next sequential candidate id (cand-0001)."""
        rows = self._conn.execute("SELECT id FROM candidates").fetchall()
        nums = [int(row["id"].rsplit("-", 1)[1]) for row in rows]
        return f"cand-{max(nums, default=0) + 1:04d}"

    def _insert(self, record: CandidateRecord) -> None:
        """Insert a full candidate row."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO candidates (id, run_id, parent_id, created_at, status, "
                "model_type, params_json, feature_code_ref, cv_score, cv_std, "
                "lb_score, submission_id, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.id,
                    record.run_id,
                    record.parent_id,
                    record.created_at,
                    record.status,
                    record.model_type,
                    json.dumps(record.params),
                    record.feature_code_ref,
                    record.cv_score,
                    record.cv_std,
                    record.lb_score,
                    record.submission_id,
                    record.notes,
                ),
            )

    def _apply_cv(self, candidate_id: str, score: float, std: float | None) -> bool:
        """Store a CV score; auto-advance proposed candidates to scored."""
        record = self.get(candidate_id)
        with self._conn:
            self._conn.execute(
                "UPDATE candidates SET cv_score=?, cv_std=? WHERE id=?",
                (score, std, candidate_id),
            )
            if record.status == CandidateStatus.PROPOSED:
                self._conn.execute(
                    "UPDATE candidates SET status=? WHERE id=?",
                    (CandidateStatus.SCORED, candidate_id),
                )
        return record.status == CandidateStatus.PROPOSED

    def _apply_status(self, candidate_id: str, status: str) -> None:
        """Set a candidate's status (no journal event)."""
        with self._conn:
            self._conn.execute(
                "UPDATE candidates SET status=? WHERE id=?", (status, candidate_id)
            )

    def _apply_lb(self, candidate_id: str, lb_score: float, submission_id: str) -> None:
        """Store a leaderboard score and mark the candidate submitted."""
        with self._conn:
            self._conn.execute(
                "UPDATE candidates SET lb_score=?, submission_id=?, status=? WHERE id=?",
                (lb_score, submission_id, CandidateStatus.SUBMITTED, candidate_id),
            )

    def _to_record(self, row: sqlite3.Row) -> CandidateRecord:
        """Convert a sqlite row into a CandidateRecord."""
        data = dict(row)
        data["params"] = json.loads(data.pop("params_json") or "{}")
        return CandidateRecord.model_validate(data)

    def _emit(self, type: str, data: dict[str, Any]) -> None:
        """Journal a registry mutation (when a journal is attached)."""
        if self._journal is not None:
            self._journal.append(agent="system", type=type, data=data, run=self._run_id)
