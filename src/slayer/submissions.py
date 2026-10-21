"""Submission sinks: mocked (default MVP) and real Kaggle submission transports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256

from pydantic import BaseModel

from slayer.workspace import CandidateRecord, EventType, Journal, Run


class SubmissionRecord(BaseModel):
    """Outcome of lodging one submission."""

    submission_id: str
    candidate_id: str
    message: str = ""
    simulated: bool = False
    lb_score: float | None = None


class SubmissionSink(ABC):
    """Transport + recording for competition submissions."""

    @abstractmethod
    def submit(
        self,
        run: Run,
        registry,
        journal: Journal,
        record: CandidateRecord,
        message: str = "",
    ) -> SubmissionRecord:
        """Lodge a submission and record it in the registry and journal."""
        raise NotImplementedError


class MockSink(SubmissionSink):
    """Simulates submissions end-to-end without touching Kaggle (D0002 MVP)."""

    def submit(
        self,
        run: Run,
        registry,
        journal: Journal,
        record: CandidateRecord,
        message: str = "",
    ) -> SubmissionRecord:
        """Record a simulated submission with a deterministic pseudo LB score."""
        noise = (int(sha256(record.id.encode()).hexdigest()[:8], 16) % 2001 - 1000) / 1e6
        lb_score = round((record.cv_score or 0.0) + noise, 6)
        registry.record_lb(record.id, lb_score, f"mock-{record.id}")
        journal.append(
            agent="submission",
            type=EventType.SUBMISSION_MOCKED,
            data={
                "candidate_id": record.id,
                "submission_id": f"mock-{record.id}",
                "lb_score": lb_score,
                "message": message,
            },
            run=run.run_id,
        )
        return SubmissionRecord(
            submission_id=f"mock-{record.id}",
            candidate_id=record.id,
            message=message,
            simulated=True,
            lb_score=lb_score,
        )


class KaggleSink(SubmissionSink):
    """Real Kaggle submission transport; LB scores arrive later via MONITOR."""

    def __init__(self, kaggle) -> None:
        """Bind the sink to a KaggleClient."""
        self._kaggle = kaggle

    def submit(
        self,
        run: Run,
        registry,
        journal: Journal,
        record: CandidateRecord,
        message: str = "",
    ) -> SubmissionRecord:
        """Submit the candidate's file to Kaggle and record the acceptance."""
        submission_file = run.workspace("models") / record.id / "submission.csv"
        ref = self._kaggle.submit(run.competition, submission_file, message)
        registry.set_status(record.id, "submitted")
        journal.append(
            agent="submission",
            type=EventType.SUBMISSION_ACCEPTED,
            data={
                "candidate_id": record.id,
                "submission_id": ref.id,
                "message": message,
            },
            run=run.run_id,
        )
        return SubmissionRecord(
            submission_id=ref.id,
            candidate_id=record.id,
            message=message,
            simulated=False,
            lb_score=None,
        )
