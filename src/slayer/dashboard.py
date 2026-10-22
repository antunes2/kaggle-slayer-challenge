"""Dashboard (challenge bonus feature): attempts and scores across all runs.

Read-only: everything is derived from each run's run.json, journal, and registry.
Two renderings: a terminal table and a self-contained HTML report (inline SVG charts,
no external assets or dependencies).
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field

from slayer.cv import build_cv_protocol
from slayer.schemas import StrategyPlan
from slayer.workspace import EventType, RunStore, llm_spend


@dataclass
class CandidatePoint:
    """One candidate's scores for charts and tables."""

    id: str
    model_type: str
    status: str
    cv: float | None
    lb: float | None


@dataclass
class RunSummary:
    """Per-run campaign summary."""

    run_id: str
    competition: str
    status: str
    created_at: str
    live: bool
    rounds: int
    candidates: int
    scored: int
    failed: int
    real_submissions: int
    mock_submissions: int
    best_cv: float | None
    best_lb: float | None
    position: str
    percentile: float | None
    spend_usd: float
    points: list[CandidatePoint] = field(default_factory=list)


def _higher_is_better(run) -> bool:
    """Metric direction from the run's strategy plan (default: higher is better)."""
    reports = run.workspace("reports")
    try:
        plan = StrategyPlan.model_validate_json(
            (reports / "strategy_plan.json").read_text(encoding="utf-8"))
        competition = json.loads((reports / "competition.json").read_text(encoding="utf-8"))
        return build_cv_protocol(competition.get("evaluation_metric") or "", plan.cv
                                 ).higher_is_better
    except Exception:
        return True


def summarize_run(store: RunStore, run) -> RunSummary:
    """Build one run's summary from its artifacts."""
    journal = store.journal(run)
    registry = store.registry(run, journal)
    try:
        records = registry.list()
    finally:
        registry.close()
    higher = _higher_is_better(run)
    pick = max if higher else min
    cvs = [r.cv_score for r in records if r.cv_score is not None]
    real_lb = [r.lb_score for r in records
               if r.lb_score is not None and not (r.submission_id or "").startswith("mock-")]
    positions = [e.data for e in journal.read(type=EventType.MONITOR_POSITION)
                 if e.data.get("percentile") is not None]
    if positions:
        last = positions[-1]
        basis = last.get("basis")
        estimate = f", est. from {basis}" if basis not in (None, "listed") else ""
        position = f"{last.get('rank')}/{last.get('total')} (top {last['percentile']}%{estimate})"
        percentile = last["percentile"]
    else:
        position, percentile = "-", None
    rounds = sum(1 for e in journal.read(type=EventType.ROADMAP_EXTENDED)
                 if e.data.get("steps_added", 0) > 0)
    return RunSummary(
        run_id=run.run_id,
        competition=run.competition,
        status=run.status,
        created_at=run.created_at,
        live=not run.dry_run,
        rounds=rounds,
        candidates=len(records),
        scored=len(cvs),
        failed=sum(1 for r in records if r.status == "failed"),
        real_submissions=len(journal.read(type=EventType.SUBMISSION_ACCEPTED)),
        mock_submissions=len(journal.read(type=EventType.SUBMISSION_MOCKED)),
        best_cv=pick(cvs) if cvs else None,
        best_lb=pick(real_lb) if real_lb else None,
        position=position,
        percentile=percentile,
        spend_usd=llm_spend(journal),
        points=[
            CandidatePoint(
                r.id, r.model_type, r.status, r.cv_score,
                None if (r.submission_id or "").startswith("mock-") else r.lb_score,
            )
            for r in records
        ],
    )


def collect(store: RunStore, competition: str | None = None) -> list[RunSummary]:
    """Summaries for all runs (optionally one competition), oldest first."""
    return [summarize_run(store, run) for run in store.list()
            if competition is None or run.competition == competition]


def _fmt(value: float | None, digits: int = 5) -> str:
    """Format an optional score."""
    return "-" if value is None else f"{value:.{digits}f}"


def render_terminal(summaries: list[RunSummary]) -> str:
    """A fixed-width table, one line per run."""
    header = (f"{'run':<44} {'mode':<5} {'status':<9} {'rnd':>3} {'cand':>5} {'fail':>4} "
              f"{'subs':>9} {'best cv':>8} {'best lb':>8} {'spend':>7}  position")
    lines = [header, "-" * len(header)]
    for s in summaries:
        subs = f"{s.real_submissions}r/{s.mock_submissions}m"
        lines.append(
            f"{s.run_id[:44]:<44} {'live' if s.live else 'dry':<5} {s.status:<9} {s.rounds:>3} "
            f"{s.candidates:>5} {s.failed:>4} {subs:>9} {_fmt(s.best_cv):>8} "
            f"{_fmt(s.best_lb):>8} ${s.spend_usd:>6.2f}  {s.position}"
        )
    return "\n".join(lines)


def _svg_chart(points: list[CandidatePoint], width: int = 640, height: int = 220) -> str:
    """Inline SVG: CV (blue line+dots) and real LB (orange dots) per candidate."""
    scored = [(i, p) for i, p in enumerate(points) if p.cv is not None or p.lb is not None]
    values = [v for _, p in scored for v in (p.cv, p.lb) if v is not None]
    if not values:
        return "<p><em>no scored candidates</em></p>"
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1e-6
    pad = 30
    n = max(len(points) - 1, 1)

    def xy(index: int, value: float) -> tuple[float, float]:
        """Map (candidate index, score) to SVG coordinates."""
        x = pad + index / n * (width - 2 * pad)
        y = height - pad - (value - lo) / span * (height - 2 * pad)
        return round(x, 1), round(y, 1)

    cv_pts = [xy(i, p.cv) for i, p in scored if p.cv is not None]
    parts = [f'<svg width="{width}" height="{height}" role="img">',
             f'<rect width="{width}" height="{height}" fill="#fafafa" stroke="#ddd"/>',
             f'<text x="4" y="14" font-size="11">{hi:.5f}</text>',
             f'<text x="4" y="{height - 6}" font-size="11">{lo:.5f}</text>']
    if len(cv_pts) > 1:
        path = " ".join(f"{x},{y}" for x, y in cv_pts)
        parts.append(f'<polyline points="{path}" fill="none" stroke="#3b6fd8" stroke-width="1.5"/>')
    for i, p in scored:
        if p.cv is not None:
            x, y = xy(i, p.cv)
            parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#3b6fd8">'
                         f'<title>{html.escape(p.id)} {html.escape(p.model_type)} '
                         f'CV {p.cv:.5f}</title></circle>')
        if p.lb is not None:
            x, y = xy(i, p.lb)
            parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#e8822a">'
                         f'<title>{html.escape(p.id)} LB {p.lb:.5f}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(summaries: list[RunSummary], generated_at: str) -> str:
    """Self-contained HTML report: overview table + per-run chart and candidate table."""
    esc = html.escape
    rows = "".join(
        f"<tr><td>{esc(s.run_id)}</td><td>{esc(s.competition)}</td>"
        f"<td>{'live' if s.live else 'dry'}</td><td>{esc(s.status)}</td><td>{s.rounds}</td>"
        f"<td>{s.candidates}</td><td>{s.failed}</td>"
        f"<td>{s.real_submissions} real / {s.mock_submissions} mock</td>"
        f"<td>{_fmt(s.best_cv)}</td><td>{_fmt(s.best_lb)}</td><td>{esc(s.position)}</td>"
        f"<td>${s.spend_usd:.2f}</td></tr>"
        for s in summaries
    )
    sections = []
    for s in summaries:
        cand_rows = "".join(
            f"<tr><td>{esc(p.id)}</td><td>{esc(p.model_type)}</td><td>{esc(p.status)}</td>"
            f"<td>{_fmt(p.cv)}</td><td>{_fmt(p.lb)}</td></tr>"
            for p in s.points
        )
        sections.append(
            f"<h2>{esc(s.run_id)}</h2><p>{esc(s.competition)} · {esc(s.status)} · "
            f"position {esc(s.position)}</p>{_svg_chart(s.points)}"
            "<p class=legend><span style='color:#3b6fd8'>●</span> CV &nbsp; "
            "<span style='color:#e8822a'>●</span> real leaderboard</p>"
            "<details><summary>candidates</summary><table><tr><th>id</th><th>model</th>"
            f"<th>status</th><th>CV</th><th>LB</th></tr>{cand_rows}</table></details>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Slayer dashboard</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#222}"
        "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;"
        "padding:4px 8px;text-align:left}th{background:#f0f0f0}.legend{font-size:12px}"
        "</style></head><body><h1>Slayer — competition attempts &amp; scores</h1>"
        f"<p>Generated {esc(generated_at)} · {len(summaries)} runs</p>"
        "<table><tr><th>run</th><th>competition</th><th>mode</th><th>status</th>"
        "<th>rounds</th><th>candidates</th><th>failed</th><th>submissions</th>"
        "<th>best CV</th><th>best real LB</th><th>last position</th><th>LLM spend</th></tr>"
        f"{rows}</table>{''.join(sections)}</body></html>"
    )
