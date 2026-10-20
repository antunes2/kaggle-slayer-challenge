import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from slayer.config import DefaultsConfig
from slayer.workspace.journal import EventType, Journal
from slayer.workspace.registry import CandidateRegistry

WORKSPACE_DIRS = ("data", "eda", "features", "models", "submissions", "reports", "logs")


class RunStatus:
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    FINISHED = "finished"
    FAILED = "failed"


class Run(BaseModel):
    run_id: str
    competition: str
    created_at: str
    status: str = RunStatus.CREATED
    target_percentile: int
    budget_usd: float
    submissions_per_day: int
    autonomy: str
    dry_run: bool
    root: Path

    def workspace(self, name: str) -> Path:
        return self.root / "workspace" / name

    @property
    def journal_path(self) -> Path:
        return self.root / "journal.jsonl"

    @property
    def registry_path(self) -> Path:
        return self.root / "experiments.db"


class RunStore:
    def __init__(self, runs_dir: Path | str = Path("runs")) -> None:
        self._runs_dir = Path(runs_dir)
        self._runs_dir.mkdir(parents=True, exist_ok=True)

    def create(self, competition: str, defaults: DefaultsConfig) -> Run:
        slug = re.sub(r"[^a-z0-9]+", "-", competition.lower()).strip("-") or "run"
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{slug}"
        root = self._runs_dir / run_id
        suffix = 2
        while root.exists():
            run_id = f"{stamp}-{slug}-{suffix}"
            root = self._runs_dir / run_id
            suffix += 1
        root.mkdir(parents=True)
        for name in WORKSPACE_DIRS:
            (root / "workspace" / name).mkdir(parents=True)
        run = Run(
            run_id=run_id,
            competition=competition,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            target_percentile=defaults.target_percentile,
            budget_usd=defaults.budget_usd,
            submissions_per_day=defaults.submissions_per_day,
            autonomy=defaults.autonomy,
            dry_run=defaults.dry_run,
            root=root,
        )
        self._save(run)
        self.journal(run).append(
            agent="system",
            type=EventType.RUN_CREATED,
            data={"competition": competition, "defaults": defaults.model_dump()},
            run=run_id,
        )
        return run

    def get(self, run_id: str) -> Run:
        path = self._runs_dir / run_id / "run.json"
        if not path.exists():
            raise KeyError(f"run not found: {run_id}")
        return Run.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[Run]:
        runs: list[Run] = []
        for child in sorted(self._runs_dir.iterdir()):
            path = child / "run.json"
            if path.is_file():
                runs.append(Run.model_validate_json(path.read_text(encoding="utf-8")))
        return runs

    def update_status(self, run: Run, status: str) -> Run:
        updated = run.model_copy(update={"status": status})
        self._save(updated)
        self.journal(updated).append(
            agent="system",
            type=EventType.RUN_STATUS,
            data={"from": run.status, "to": status},
            run=run.run_id,
        )
        return updated

    def journal(self, run: Run) -> Journal:
        return Journal(run.journal_path)

    def registry(self, run: Run, journal: Journal | None = None) -> CandidateRegistry:
        return CandidateRegistry(
            run.registry_path, run_id=run.run_id, journal=journal or self.journal(run)
        )

    def _save(self, run: Run) -> None:
        tmp = run.root / "run.json.tmp"
        tmp.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(run.root / "run.json")
