from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class KaggleError(Exception):
    """Raised for any Kaggle API failure surfaced to the loop."""

    pass


class CompetitionInfo(BaseModel):
    ref: str
    title: str | None = None
    category: str | None = None
    reward: str | None = None
    deadline: str | None = None
    evaluation_metric: str | None = None
    user_has_entered: bool | None = None
    max_daily_submissions: int | None = None


class CompetitionFileInfo(BaseModel):
    name: str
    total_bytes: int | None = None


class SubmissionRef(BaseModel):
    id: str
    message: str | None = None


class SubmissionInfo(BaseModel):
    ref: str
    file_name: str | None = None
    description: str | None = None
    status: str | None = None
    public_score: str | None = None
    private_score: str | None = None
    date: str | None = None


class FullLeaderboardRow(BaseModel):
    """One row of the full public leaderboard CSV (includes member usernames)."""

    rank: int
    team_id: str | None = None
    team_name: str | None = None
    score: float | None = None
    member_usernames: list[str] = Field(default_factory=list)


class NotebookInfo(BaseModel):
    """A public notebook (kernel) attached to a competition."""

    ref: str
    title: str | None = None
    votes: int | None = None


def _code_from_source(source: str) -> str:
    """Extract code cells from an .ipynb JSON source; plain scripts pass through."""
    try:
        notebook = json.loads(source)
    except (json.JSONDecodeError, TypeError):
        return source or ""
    if not isinstance(notebook, dict) or "cells" not in notebook:
        return source or ""
    cells = []
    for cell in notebook.get("cells") or []:
        if cell.get("cell_type") != "code":
            continue
        body = cell.get("source") or ""
        cells.append("".join(body) if isinstance(body, list) else str(body))
    return "\n".join(cells)


def _text(value: Any) -> str | None:
    """Stringify a value, keeping None as None."""
    if value is None:
        return None
    return str(value)


def _reward_text(reward: Any) -> str | None:
    """Render a kagglesdk Reward object as text like '50000 USD'."""
    if reward is None:
        return None
    quantity = getattr(reward, "quantity", None)
    reward_id = getattr(reward, "id", None)
    id_name = getattr(reward_id, "name", None)
    parts = [str(part) for part in (quantity, id_name) if part]
    return " ".join(parts) or None


def _default_api_class() -> Any:
    """Lazily import the kaggle package API class (avoids import side effects)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    return KaggleApi


def _build_default_api(token: str) -> Any:
    """Authenticate a real kaggle API instance with the given access token."""
    os.environ["KAGGLE_API_TOKEN"] = token
    api_class = _default_api_class()
    api = api_class()
    try:
        api.authenticate()
    except (Exception, SystemExit) as exc:
        raise KaggleError(f"kaggle authentication failed: {exc}") from exc
    return api


class KaggleClient:
    """Thin, defensively-converted wrapper over the kaggle package API."""

    def __init__(self, api: Any) -> None:
        """Wrap an (injected or real) kaggle API instance."""
        self._api = api

    @classmethod
    def from_token(cls, token: str) -> KaggleClient:
        """Build an authenticated client from a Kaggle access token."""
        if not token:
            raise KaggleError("Kaggle API token is empty")
        return cls(_build_default_api(token))

    def ping(self) -> str:
        """Return the authenticated username (auth verified at construction)."""
        config = dict(getattr(self._api, "config_values", None) or {})
        username = config.get("username")
        if not username:
            raise KaggleError("authenticated but no username resolved")
        return str(username)

    def list_competitions(
        self,
        *,
        search: str | None = None,
        category: str | None = None,
        sort_by: str | None = None,
        page_size: int = 20,
    ) -> list[CompetitionInfo]:
        """List competitions matching search/category/sort filters."""
        response = self._api.competitions_list(
            search=search, category=category, sort_by=sort_by, page_size=page_size
        )
        competitions = list(getattr(response, "competitions", None) or [])
        return [self._to_competition(c) for c in competitions if c is not None]

    @staticmethod
    def _normalize_ref(raw: Any) -> str:
        """Reduce a ref or full competition URL to the bare ref name."""
        text = str(raw or "").rstrip("/")
        if "/" in text:
            return text.rsplit("/", 1)[-1]
        return text

    def find_competition(
        self,
        ref: str,
        *,
        search: str | None = None,
        max_pages: int = 5,
        page_size: int = 100,
    ) -> CompetitionInfo:
        """Find one competition by exact ref, paginating with a search fallback."""
        page_token: str | None = None
        for _ in range(max_pages):
            response = self._api.competitions_list(
                search=search, page_size=page_size, page_token=page_token
            )
            competitions = list(getattr(response, "competitions", None) or [])
            for competition in competitions:
                if competition is not None and self._normalize_ref(
                    getattr(competition, "ref", "")
                ) == ref:
                    return self._to_competition(competition)
            page_token = getattr(response, "next_page_token", "") or None
            if not page_token:
                break
        if search is None:
            response = self._api.competitions_list(search=ref, page_size=page_size)
            competitions = list(getattr(response, "competitions", None) or [])
            for competition in competitions:
                if competition is not None and self._normalize_ref(
                    getattr(competition, "ref", "")
                ) == ref:
                    return self._to_competition(competition)
        raise KaggleError(
            f"competition not found on Kaggle: '{ref}' — check the ref, e.g. "
            "'playground-series-s5e10'"
        )

    def _to_competition(self, competition: Any) -> CompetitionInfo:
        """Convert a raw sdk competition object into our pydantic model."""
        return CompetitionInfo(
            ref=self._normalize_ref(getattr(competition, "ref", "")),
            title=_text(getattr(competition, "title", None)),
            category=_text(getattr(competition, "category", None)),
            reward=_reward_text(getattr(competition, "reward", None)),
            deadline=_text(getattr(competition, "deadline", None)),
            evaluation_metric=_text(getattr(competition, "evaluation_metric", None)),
            user_has_entered=getattr(competition, "user_has_entered", None),
            max_daily_submissions=getattr(competition, "max_daily_submissions", None),
        )

    def list_files(self, competition: str) -> list[CompetitionFileInfo]:
        """List a competition's data files (also used as a rules probe)."""
        response = self._api.competition_list_files(competition)
        files = list(getattr(response, "files", None) or [])
        return [
            CompetitionFileInfo(
                name=str(getattr(f, "name", "")),
                total_bytes=getattr(f, "total_bytes", None),
            )
            for f in files
            if f is not None
        ]

    def download_files(self, competition: str, target_dir: Path, *, force: bool = False) -> Path:
        """Download all competition data into target_dir; returns the zip path."""
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._api.competition_download_files(
                competition, path=str(target_dir), force=force, quiet=True
            )
        except Exception as exc:
            message = f"data download failed: {exc}"
            if "403" in str(exc) or "401" in str(exc):
                message += (
                    " — you likely need to accept the competition rules first: "
                    f"https://www.kaggle.com/competitions/{competition}/rules"
                )
            raise KaggleError(message) from exc
        zip_path = target_dir / f"{competition}.zip"
        if not zip_path.exists():
            raise KaggleError(f"expected {zip_path} after download, not found")
        return zip_path

    def submit(self, competition: str, file_path: Path, message: str = "") -> SubmissionRef:
        """Submit a predictions file to the competition."""
        if not file_path.exists():
            raise KaggleError(f"submission file not found: {file_path}")
        response = self._api.competition_submit(str(file_path), message, competition, quiet=True)
        ref = getattr(response, "ref", None)
        return SubmissionRef(id=str(ref or ""), message=_text(getattr(response, "message", None)))

    def submissions(self, competition: str) -> list[SubmissionInfo]:
        """List the user's submissions with status and scores."""
        items = self._api.competition_submissions(competition) or []
        infos: list[SubmissionInfo] = []
        for item in items:
            if item is None:
                continue
            status = getattr(item, "status", None)
            status_name = getattr(status, "name", None) if status is not None else None
            infos.append(
                SubmissionInfo(
                    ref=str(getattr(item, "ref", "")),
                    file_name=_text(getattr(item, "file_name", None)),
                    description=_text(getattr(item, "description", None)),
                    status=_text(status_name),
                    public_score=_text(getattr(item, "public_score", None)),
                    private_score=_text(getattr(item, "private_score", None)),
                    date=_text(getattr(item, "date", None)),
                )
            )
        return infos

    def list_notebooks(self, competition: str, n: int = 5) -> list[NotebookInfo]:
        """The competition's most-voted public notebooks (read-only)."""
        try:
            kernels = self._api.kernels_list(
                competition=competition, sort_by="voteCount", page_size=n
            ) or []
        except Exception as exc:
            raise KaggleError(f"notebook listing failed: {exc}") from exc
        infos = []
        for kernel in list(kernels)[:n]:
            if kernel is None or not getattr(kernel, "ref", None):
                continue
            votes = getattr(kernel, "total_votes", None)
            infos.append(
                NotebookInfo(
                    ref=str(kernel.ref),
                    title=_text(getattr(kernel, "title", None)),
                    votes=int(votes) if isinstance(votes, (int, float)) else None,
                )
            )
        return infos

    def notebook_code(self, ref: str) -> str:
        """Fetch a public notebook's source in memory and return its code cells only."""
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

        owner, _, slug = ref.partition("/")
        if not owner or not slug:
            raise KaggleError(f"invalid notebook ref: '{ref}'")
        try:
            with self._api.build_kaggle_client() as kaggle:
                request = ApiGetKernelRequest()
                request.user_name = owner
                request.kernel_slug = slug
                response = kaggle.kernels.kernels_api_client.get_kernel(request)
        except Exception as exc:
            raise KaggleError(f"notebook fetch failed for {ref}: {exc}") from exc
        source = getattr(getattr(response, "blob", None), "source", "") or ""
        return _code_from_source(source)

    def download_leaderboard(self, competition: str, target_dir: Path) -> Path:
        """Download the full public leaderboard zip into target_dir."""
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self._api.competition_leaderboard_download(
                    competition=competition, path=str(target_dir), quiet=True
                )
        except Exception as exc:
            raise KaggleError(f"leaderboard download failed: {exc}") from exc
        zips = sorted(target_dir.glob("*.zip"))
        if not zips:
            raise KaggleError(f"expected leaderboard zip in {target_dir}, not found")
        return zips[-1]

    def full_leaderboard(
        self, competition: str, target_dir: Path | None = None
    ) -> list[FullLeaderboardRow]:
        """Download and parse the full leaderboard CSV (authoritative: all teams).

        When target_dir is given the extracted CSV is kept there as
        leaderboard.csv (run artifact); otherwise a temp dir is used.
        """
        keep_dir = target_dir
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = self.download_leaderboard(competition, Path(keep_dir or tmp))
            with zipfile.ZipFile(zip_path) as zf:
                member = zf.namelist()[0]
                rows: list[FullLeaderboardRow] = []
                with zf.open(member) as fh:
                    reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8-sig"))
                    for raw in reader:
                        score_raw = (raw.get("Score") or "").strip()
                        try:
                            score = float(score_raw) if score_raw else None
                        except ValueError:
                            score = None
                        try:
                            rank = int(str(raw.get("Rank") or "").strip())
                        except ValueError:
                            rank = 0
                        usernames = [
                            u.strip().lower()
                            for u in (raw.get("TeamMemberUserNames") or "").split(",")
                            if u.strip()
                        ]
                        rows.append(
                            FullLeaderboardRow(
                                rank=rank,
                                team_id=(raw.get("TeamId") or "").strip() or None,
                                team_name=(raw.get("TeamName") or "").strip() or None,
                                score=score,
                                member_usernames=usernames,
                            )
                        )
                if keep_dir is not None:
                    with zf.open(member) as src, (Path(keep_dir) / "leaderboard.csv").open(
                        "wb"
                    ) as dst:
                        dst.write(src.read())
        return rows
