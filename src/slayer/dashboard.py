"""Dashboard (challenge bonus feature): attempts and scores across all runs.

Read-only: everything is derived from each run's run.json, journal, and registry.
Two renderings: a terminal table and a self-contained dark HTML report (inline CSS,
JS, and SVG charts — no external assets or dependencies).
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
    rank: int | None = None
    total: int | None = None
    target_percentile: int | None = None
    higher_is_better: bool = True
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
        rank, total = last.get("rank"), last.get("total")
    else:
        position, percentile = "-", None
        rank, total = None, None
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
        rank=rank,
        total=total,
        target_percentile=run.target_percentile,
        higher_is_better=higher,
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


_PAGE_CSS = """
:root{
  --bg:#06070c; --panel:rgba(19,22,34,.74); --line:rgba(255,255,255,.08);
  --line-strong:rgba(255,255,255,.16); --text:#e9ebf3; --muted:#9098ac; --dim:#646c80;
  --violet:#8b5cf6; --violet-2:#a78bfa; --cyan:#22d3ee; --amber:#f59e0b;
  --green:#34d399; --red:#f87171;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
html{color-scheme:dark}
body{
  margin:0;min-height:100vh;background:var(--bg);color:var(--text);
  font-family:var(--sans);-webkit-font-smoothing:antialiased;line-height:1.45;
}
body::before{
  content:"";position:fixed;inset:0;z-index:-2;pointer-events:none;
  background:
    radial-gradient(60rem 42rem at 8% -12%,rgba(139,92,246,.24),transparent 62%),
    radial-gradient(52rem 40rem at 98% -4%,rgba(34,211,238,.13),transparent 60%),
    radial-gradient(46rem 34rem at 54% 112%,rgba(245,158,11,.07),transparent 62%);
}
body::after{
  content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;opacity:.5;
  background-image:
    linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),
    linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);
  background-size:56px 56px;
  mask-image:radial-gradient(ellipse at 50% 0%,#000 20%,transparent 75%);
  -webkit-mask-image:radial-gradient(ellipse at 50% 0%,#000 20%,transparent 75%);
}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px}
.hero{padding:54px 0 26px}
.brand{
  display:flex;align-items:center;gap:10px;font-size:12px;font-weight:800;
  letter-spacing:.22em;text-transform:uppercase;color:var(--muted);
}
.brand .mark{
  display:grid;place-items:center;width:27px;height:27px;border-radius:9px;
  background:linear-gradient(135deg,var(--violet),var(--cyan));color:#07080d;
  font-size:14px;box-shadow:0 0 26px rgba(139,92,246,.55);
}
.brand .sub{color:var(--dim);font-weight:600;letter-spacing:.14em}
h1{
  margin:18px 0 12px;font-size:clamp(2rem,4.2vw,3.2rem);line-height:1.04;
  letter-spacing:-.035em;font-weight:800;
}
h1 .grad{
  background:linear-gradient(100deg,#fff 8%,var(--violet-2) 56%,var(--cyan));
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.lede{max-width:700px;margin:0;color:var(--muted);font-size:15px;line-height:1.65}
.meta{margin-top:14px;font-family:var(--mono);font-size:12px;color:var(--dim)}
.stats{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:14px;margin-top:30px;
}
.stat{
  position:relative;overflow:hidden;padding:17px 18px 15px;border:1px solid var(--line);
  border-radius:16px;background:linear-gradient(180deg,rgba(255,255,255,.05),
  rgba(255,255,255,.015));backdrop-filter:blur(10px);
}
.stat::after{
  content:"";position:absolute;right:-44px;bottom:-64px;width:150px;height:150px;
  background:radial-gradient(circle,rgba(139,92,246,.26),transparent 65%);
  pointer-events:none;
}
.stat .k{
  font-size:10.5px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);
}
.stat .v{
  display:block;margin-top:8px;font-size:clamp(1.3rem,2.4vw,1.85rem);font-weight:750;
  letter-spacing:-.02em;
}
.stat .s{
  display:block;margin-top:3px;font-size:11.5px;color:var(--dim);white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;
}
.good{color:var(--green)}
.toolbar{
  position:sticky;top:0;z-index:20;display:flex;flex-wrap:wrap;gap:12px;align-items:center;
  justify-content:space-between;padding:14px 0;margin:8px 0 22px;border-bottom:1px solid
  var(--line);background:linear-gradient(180deg,rgba(6,7,12,.97),rgba(6,7,12,.78));
  backdrop-filter:blur(14px);
}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{
  appearance:none;border:1px solid var(--line);background:rgba(255,255,255,.03);
  color:var(--muted);border-radius:999px;padding:7px 12px;font:inherit;font-size:12.5px;
  font-weight:600;cursor:pointer;transition:color .18s,border-color .18s,background .18s,
  transform .18s;
}
.chip:hover{color:var(--text);border-color:var(--line-strong);background:rgba(255,255,255,.06)}
.chip.is-active{
  color:#0a0b10;background:linear-gradient(135deg,var(--violet-2),var(--cyan));
  border-color:transparent;box-shadow:0 8px 24px -8px rgba(139,92,246,.75);
}
.chip .n{margin-left:4px;font-family:var(--mono);font-size:11px;opacity:.65}
.search{
  display:flex;align-items:center;gap:8px;min-width:270px;padding:8px 12px;
  border:1px solid var(--line);border-radius:999px;background:rgba(255,255,255,.03);
  transition:border-color .18s,box-shadow .18s;
}
.search:focus-within{border-color:rgba(139,92,246,.6);box-shadow:0 0 0 3px rgba(139,92,246,.15)}
.search svg{width:15px;height:15px;flex:none;color:var(--dim)}
.search input{all:unset;width:100%;font:inherit;font-size:13px;color:var(--text)}
.search input::placeholder{color:var(--dim)}
.grid{display:grid;gap:16px}
.run{
  border:1px solid var(--line);border-radius:18px;overflow:hidden;
  background:linear-gradient(165deg,rgba(26,29,44,.8),rgba(12,14,23,.74));
  backdrop-filter:blur(12px);box-shadow:0 18px 50px -32px rgba(0,0,0,.95);
  transition:border-color .2s,box-shadow .2s,transform .2s;
}
.run:hover{border-color:rgba(139,92,246,.35);box-shadow:0 24px 60px -32px rgba(139,92,246,.35)}
.run[open]{border-color:rgba(139,92,246,.45)}
.run[hidden]{display:none}
.run>summary{
  position:relative;display:grid;gap:14px;padding:20px 22px;cursor:pointer;
  list-style:none;outline:none;
}
.run>summary::-webkit-details-marker{display:none}
.run>summary:focus-visible{box-shadow:inset 0 0 0 2px rgba(139,92,246,.65)}
.run-head{
  display:flex;flex-wrap:wrap;gap:14px;justify-content:space-between;
  align-items:flex-start;padding-right:32px;
}
.comp{display:block;font-size:17px;font-weight:700;letter-spacing:-.015em;word-break:break-word}
.rid{
  display:block;margin-top:3px;font-family:var(--mono);font-size:11.5px;color:var(--dim);
  word-break:break-all;
}
.badges{display:flex;flex-wrap:wrap;gap:6px;justify-content:flex-end}
.pill{
  display:inline-flex;align-items:center;gap:6px;padding:4px 9px;border-radius:999px;
  border:1px solid var(--line);background:rgba(255,255,255,.04);color:var(--muted);
  font-size:10.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  white-space:nowrap;
}
.pill::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor;opacity:.9}
.pill-live{color:#a7f3d0;border-color:rgba(52,211,153,.32);background:rgba(52,211,153,.11)}
.pill-live::before{box-shadow:0 0 0 0 rgba(52,211,153,.6);animation:pulse 2.2s infinite}
.pill-dry{color:#aeb6c9;border-color:rgba(174,182,201,.22);background:rgba(174,182,201,.07)}
.pill-finished{color:#c4b5fd;border-color:rgba(167,139,250,.3);background:rgba(139,92,246,.12)}
.pill-active{color:#a5f3fc;border-color:rgba(34,211,238,.3);background:rgba(34,211,238,.1)}
.pill-paused{color:#fcd34d;border-color:rgba(245,158,11,.3);background:rgba(245,158,11,.1)}
.pill-failed{color:#fca5a5;border-color:rgba(248,113,113,.32);background:rgba(248,113,113,.1)}
.pill-created{color:#aeb6c9}
.pill-ghost{color:var(--dim);background:transparent}
@keyframes pulse{
  0%{box-shadow:0 0 0 0 rgba(52,211,153,.55)}
  70%{box-shadow:0 0 0 6px rgba(52,211,153,0)}
  100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}
}
.run-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(108px,1fr));gap:9px}
.metric{
  min-width:0;padding:9px 11px;border:1px solid var(--line);border-radius:11px;
  background:rgba(255,255,255,.022);
}
.metric .k{
  display:block;font-size:9.5px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.metric .v{
  display:block;margin-top:4px;font-family:var(--mono);font-size:14px;font-weight:600;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.rank{display:block}
.rank-head{
  display:flex;flex-wrap:wrap;gap:8px;justify-content:space-between;font-size:11.5px;
  color:var(--dim);
}
.rank-head .strong{color:var(--text);font-family:var(--mono)}
.rank-pct{color:var(--amber);font-weight:700}
.rank.is-hit .rank-pct{color:var(--green)}
.track{
  position:relative;display:block;height:7px;margin-top:7px;border-radius:999px;
  border:1px solid var(--line);background:rgba(255,255,255,.04);
}
.fill{
  position:absolute;left:0;top:0;bottom:0;border-radius:999px;
  background:linear-gradient(90deg,var(--green),var(--amber) 42%,var(--red));
  box-shadow:0 0 16px rgba(245,158,11,.3);
}
.rank.is-hit .fill{
  background:linear-gradient(90deg,var(--green),var(--cyan));
  box-shadow:0 0 16px rgba(34,211,153,.4);
}
.goal{
  position:absolute;top:-4px;bottom:-4px;width:2px;transform:translateX(-1px);
  background:var(--cyan);box-shadow:0 0 10px var(--cyan);
}
.chev{
  position:absolute;right:20px;top:23px;width:18px;height:18px;color:var(--dim);
  transition:transform .25s,color .2s;
}
.run[open] .chev{transform:rotate(180deg);color:var(--violet-2)}
.run-body{
  padding:0 22px 22px;border-top:1px solid var(--line);background:rgba(0,0,0,.14);
}
.chart-head{
  display:flex;flex-wrap:wrap;gap:12px;justify-content:space-between;align-items:center;
  padding:16px 0 10px;
}
.chart-head h3{
  margin:0;font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
}
.detail-actions{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.legend{display:flex;flex-wrap:wrap;gap:6px}
.legend-btn,.copy{
  appearance:none;display:inline-flex;align-items:center;gap:6px;padding:5px 10px;
  border:1px solid var(--line);border-radius:999px;background:rgba(255,255,255,.03);
  color:var(--muted);font:inherit;font-size:11.5px;font-weight:600;cursor:pointer;
  transition:color .18s,border-color .18s,opacity .18s;
}
.legend-btn:hover,.copy:hover{color:var(--text);border-color:var(--line-strong)}
.legend-btn .dotmark{width:8px;height:8px;border-radius:50%}
.legend-btn[data-series="cv"] .dotmark{background:var(--violet-2);box-shadow:0 0 9px var(--violet)}
.legend-btn[data-series="lb"] .dotmark{background:var(--amber);box-shadow:0 0 9px var(--amber)}
.legend-btn.off{opacity:.42;text-decoration:line-through}
.chart{
  position:relative;overflow:hidden;border:1px solid var(--line);border-radius:14px;
  padding:8px;background:radial-gradient(120% 120% at 50% 0%,rgba(139,92,246,.09),
  rgba(255,255,255,.015));
}
.chart svg{display:block;width:100%;height:auto}
.grid-line{stroke:rgba(255,255,255,.07)}
.axis-label{fill:#717a90;font-size:11px;font-family:var(--mono)}
.cv-line{fill:none;stroke:#a78bfa;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot{cursor:pointer;transition:filter .15s,opacity .15s}
.dot:hover{filter:brightness(1.6)}
.dot-cv{fill:#0b0d16;stroke:#a78bfa;stroke-width:2}
.dot-lb{fill:#0b0d16;stroke:#f59e0b;stroke-width:2.5}
.dot.best{stroke-width:3.5}
.series{transition:opacity .2s}
.series.off{opacity:0;pointer-events:none}
.chart-empty{
  display:grid;place-items:center;min-height:150px;border:1px dashed var(--line);
  border-radius:14px;color:var(--dim);font-size:12.5px;
}
.cand-head{
  display:flex;justify-content:space-between;gap:10px;align-items:baseline;
  margin:18px 2px 8px;
}
.cand-head h3{
  margin:0;font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
}
.cand-head span{font-size:11.5px;color:var(--dim)}
.table-wrap{max-height:360px;overflow:auto;border:1px solid var(--line);border-radius:14px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
thead th{
  position:sticky;top:0;z-index:1;padding:10px 12px;text-align:left;white-space:nowrap;
  border-bottom:1px solid var(--line);background:#12141f;color:var(--muted);
  font-size:10px;font-weight:700;letter-spacing:.11em;text-transform:uppercase;
  cursor:pointer;user-select:none;
}
thead th:hover{color:var(--text)}
thead th[aria-sort="ascending"]::after{content:" ▲";color:var(--violet-2)}
thead th[aria-sort="descending"]::after{content:" ▼";color:var(--violet-2)}
thead th.num{text-align:right}
tbody td{padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.045);white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover{background:rgba(139,92,246,.07)}
tbody tr.failed{opacity:.48}
td.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
td code{font-family:var(--mono);font-size:11.5px;color:#c4c9d6}
.st{
  display:inline-flex;align-items:center;padding:2px 7px;border-radius:999px;
  border:1px solid var(--line);background:rgba(255,255,255,.035);font-size:10px;
  font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
}
.st-failed{color:#fca5a5;border-color:rgba(248,113,113,.3);background:rgba(248,113,113,.09)}
.st-submitted{color:#fcd34d;border-color:rgba(245,158,11,.3);background:rgba(245,158,11,.09)}
.st-approved{color:#c4b5fd;border-color:rgba(167,139,250,.28);background:rgba(139,92,246,.1)}
.st-scored{color:#a5f3fc;border-color:rgba(34,211,238,.26);background:rgba(34,211,238,.08)}
.empty-state{display:grid;place-items:center;padding:60px 20px;color:var(--dim);text-align:center}
.empty-state strong{display:block;color:var(--text);margin-bottom:6px;font-size:15px}
.tip{
  position:fixed;z-index:50;pointer-events:none;max-width:320px;padding:8px 11px;
  border:1px solid var(--line-strong);border-radius:10px;background:rgba(10,12,19,.96);
  color:var(--text);font-family:var(--mono);font-size:11.5px;line-height:1.4;
  box-shadow:0 14px 40px -16px rgba(0,0,0,.9);backdrop-filter:blur(8px);
}
.tip[data-series="lb"]{border-color:rgba(245,158,11,.5)}
.tip[data-series="cv"]{border-color:rgba(167,139,250,.5)}
.foot{padding:34px 0 46px;color:var(--dim);font-size:11.5px;font-family:var(--mono)}
@media(max-width:640px){
  .wrap{padding:0 16px}
  .hero{padding-top:38px}
  .toolbar{position:static}
  .search{min-width:0;width:100%}
  .run>summary{padding:17px 16px}
  .run-body{padding:0 16px 18px}
  .comp{font-size:15px}
  .chev{right:15px;top:19px}
  .stats{grid-template-columns:1fr 1fr}
}
@media(prefers-reduced-motion:reduce){
  *{animation-duration:.001ms!important;transition-duration:.001ms!important}
}
"""

_PAGE_JS = """
(function () {
  var chips = Array.prototype.slice.call(document.querySelectorAll('.chip'));
  var search = document.getElementById('search');
  var cards = Array.prototype.slice.call(document.querySelectorAll('.run'));
  var empty = document.getElementById('empty');
  var filter = 'all';

  function applyFilters() {
    var q = (search.value || '').trim().toLowerCase();
    var shown = 0;
    cards.forEach(function (card) {
      var okComp = filter === 'all' || card.dataset.competition === filter;
      var okText = !q || card.dataset.search.indexOf(q) !== -1;
      var ok = okComp && okText;
      card.hidden = !ok;
      if (ok) shown += 1;
    });
    empty.hidden = shown > 0;
  }

  chips.forEach(function (chip) {
    chip.addEventListener('click', function () {
      filter = chip.dataset.filter;
      chips.forEach(function (c) {
        var active = c === chip;
        c.classList.toggle('is-active', active);
        c.setAttribute('aria-pressed', String(active));
      });
      applyFilters();
    });
  });
  search.addEventListener('input', applyFilters);

  document.querySelectorAll('.legend-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var card = btn.closest('.run');
      var series = btn.dataset.series;
      var off = btn.classList.toggle('off');
      btn.setAttribute('aria-pressed', String(!off));
      card.querySelectorAll('.series-' + series).forEach(function (group) {
        group.classList.toggle('off', off);
      });
    });
  });

  document.querySelectorAll('.cand-table').forEach(function (table) {
    var headers = Array.prototype.slice.call(table.querySelectorAll('th'));
    headers.forEach(function (th, index) {
      th.addEventListener('click', function () {
        var dir = th.dataset.dir === 'asc' ? 'desc' : 'asc';
        headers.forEach(function (h) {
          h.removeAttribute('aria-sort');
          delete h.dataset.dir;
        });
        th.dataset.dir = dir;
        th.setAttribute('aria-sort', dir === 'asc' ? 'ascending' : 'descending');
        var numeric = th.dataset.type === 'num';
        Array.prototype.slice.call(table.tBodies[0].rows).sort(function (a, b) {
          var av = a.cells[index].dataset.v;
          var bv = b.cells[index].dataset.v;
          if (av === undefined) av = a.cells[index].textContent.trim();
          if (bv === undefined) bv = b.cells[index].textContent.trim();
          var cmp;
          if (numeric) {
            var an = av === '' ? null : Number(av);
            var bn = bv === '' ? null : Number(bv);
            if (an === null && bn === null) cmp = 0;
            else if (an === null) cmp = 1;
            else if (bn === null) cmp = -1;
            else cmp = an - bn;
          } else {
            cmp = String(av).localeCompare(String(bv), undefined,
                                          {numeric: true, sensitivity: 'base'});
          }
          return dir === 'asc' ? cmp : -cmp;
        }).forEach(function (row) {
          table.tBodies[0].appendChild(row);
        });
      });
    });
  });

  var tip = document.getElementById('tip');
  function placeTip(x, y) {
    var pad = 14;
    var box = tip.getBoundingClientRect();
    var left = x + pad;
    var top = y + pad;
    if (left + box.width > window.innerWidth - 8) left = x - box.width - pad;
    if (top + box.height > window.innerHeight - 8) top = y - box.height - pad;
    tip.style.left = Math.max(8, left) + 'px';
    tip.style.top = Math.max(8, top) + 'px';
  }
  document.addEventListener('pointerover', function (event) {
    var dot = event.target.closest('.dot');
    if (!dot) return;
    tip.textContent = dot.dataset.label + '  ·  ' + dot.dataset.series + ' '
      + dot.dataset.score;
    tip.dataset.series = dot.dataset.series.toLowerCase();
    tip.hidden = false;
    placeTip(event.clientX, event.clientY);
  });
  document.addEventListener('pointermove', function (event) {
    if (!tip.hidden) placeTip(event.clientX, event.clientY);
  });
  document.addEventListener('pointerout', function (event) {
    if (event.target.closest('.dot')) tip.hidden = true;
  });

  document.querySelectorAll('.copy').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var text = btn.dataset.id;
      var done = function (ok) {
        btn.textContent = ok ? 'Copied' : 'Copy failed';
        setTimeout(function () { btn.textContent = 'Copy id'; }, 1200);
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(function () { done(true); },
                                                   function () { done(false); });
        return;
      }
      var area = document.createElement('textarea');
      area.value = text;
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.appendChild(area);
      area.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (err) { ok = false; }
      document.body.removeChild(area);
      done(ok);
    });
  });
})();
"""


def _status_class(status: str) -> str:
    """CSS class suffix for a run status pill."""
    known = {"finished", "active", "paused", "failed", "created"}
    return status if status in known else "created"


def _candidate_status(status: str) -> str:
    """CSS class suffix for a candidate status tag."""
    known = {"failed", "submitted", "approved", "scored"}
    return status if status in known else "other"


def _svg_chart(points: list[CandidatePoint], chart_id: str, higher: bool = True) -> str:
    """Interactive inline SVG: CV trajectory and real LB points per candidate."""
    esc = html.escape
    scored = [(i, p) for i, p in enumerate(points) if p.cv is not None or p.lb is not None]
    values = [v for _, p in scored for v in (p.cv, p.lb) if v is not None]
    if not values:
        return '<div class="chart-empty">No scored candidates yet.</div>'

    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        lo_pad, hi_pad = lo - 1e-6, hi + 1e-6
    else:
        margin = (hi - lo) * 0.16
        lo_pad, hi_pad = lo - margin, hi + margin

    width, height = 760, 258
    left, right, top, bottom = 62, 18, 16, 30
    span = hi_pad - lo_pad
    n = max(len(points) - 1, 1)
    cid = "".join(ch for ch in chart_id if ch.isalnum()) or "chart"
    best_of = max if higher else min
    best_cv = best_of((p.cv for _, p in scored if p.cv is not None), default=None)
    best_lb = best_of((p.lb for _, p in scored if p.lb is not None), default=None)

    def xy(index: int, value: float) -> tuple[float, float]:
        """Map (candidate index, score) to SVG coordinates."""
        x = left + index / n * (width - left - right)
        y = top + (hi_pad - value) / span * (height - top - bottom)
        return round(x, 1), round(y, 1)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="CV and leaderboard scores per candidate">',
        "<defs>",
        f'<linearGradient id="fill{cid}" x1="0" y1="0" x2="0" y2="1">',
        '<stop offset="0%" stop-color="#8b5cf6" stop-opacity=".34"/>',
        '<stop offset="100%" stop-color="#8b5cf6" stop-opacity="0"/>',
        "</linearGradient>",
        f'<filter id="glow{cid}" x="-50%" y="-50%" width="200%" height="200%">',
        '<feGaussianBlur stdDeviation="3" result="blur"/>',
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>',
        "</filter>",
        "</defs>",
    ]

    for step in range(5):
        value = lo_pad + span * step / 4
        _, y = xy(0, value)
        parts.append(f'<line class="grid-line" x1="{left}" y1="{y}" '
                     f'x2="{width - right}" y2="{y}"/>')
        parts.append(f'<text class="axis-label" x="{left - 10}" y="{y + 4}" '
                     f'text-anchor="end">{value:.4f}</text>')

    tick_count = min(6, len(points))
    if tick_count > 1:
        for step in range(tick_count):
            index = round(step * (len(points) - 1) / (tick_count - 1))
            x, _ = xy(index, lo_pad)
            parts.append(f'<text class="axis-label" x="{x}" y="{height - 8}" '
                         f'text-anchor="middle">#{index + 1}</text>')

    cv_pts = [xy(i, p.cv) for i, p in scored if p.cv is not None]
    if len(cv_pts) > 1:
        line = " ".join(f"{x},{y}" for x, y in cv_pts)
        base = height - bottom
        area = (f"M{cv_pts[0][0]},{base} L" + " L".join(f"{x},{y}" for x, y in cv_pts)
                + f" L{cv_pts[-1][0]},{base} Z")
        parts.append(f'<path d="{area}" fill="url(#fill{cid})"/>')
        parts.append(f'<polyline class="cv-line" points="{line}" filter="url(#glow{cid})"/>')

    parts.append('<g class="series series-cv">')
    for index, point in scored:
        if point.cv is None:
            continue
        x, y = xy(index, point.cv)
        best = " best" if point.cv == best_cv else ""
        label = f"{point.id} · {point.model_type}"
        parts.append(
            f'<circle class="dot dot-cv{best}" cx="{x}" cy="{y}" r="4" '
            f'data-label="{esc(label)}" data-series="CV" data-score="{point.cv:.6f}">'
            f'<title>{esc(label)} CV {point.cv:.5f}</title></circle>'
        )
    parts.append("</g>")

    parts.append('<g class="series series-lb">')
    for index, point in scored:
        if point.lb is None:
            continue
        x, y = xy(index, point.lb)
        best = " best" if point.lb == best_lb else ""
        label = f"{point.id} · {point.model_type}"
        parts.append(
            f'<circle class="dot dot-lb{best}" cx="{x}" cy="{y}" r="5" '
            f'data-label="{esc(label)}" data-series="LB" data-score="{point.lb:.6f}">'
            f'<title>{esc(label)} LB {point.lb:.5f}</title></circle>'
        )
    parts.append("</g></svg>")
    return "".join(parts)


def _stat_tile(label: str, value: str, sub: str = "", extra: str = "") -> str:
    """One glassy aggregate stat tile for the hero."""
    esc = html.escape
    return (f'<div class="stat"><span class="k">{esc(label)}</span>'
            f'<span class="v {extra}">{esc(value)}</span>'
            f'<span class="s">{esc(sub)}</span></div>')


def _run_card(summary: RunSummary, index: int, open_card: bool = False) -> str:
    """One expandable campaign card with metrics, rank bar, chart, and table."""
    esc = html.escape
    s = summary
    target = s.target_percentile or 20
    has_rank = s.percentile is not None and s.rank is not None and s.total is not None
    hit = has_rank and s.percentile <= target
    pct = max(0.0, min(100.0, s.percentile)) if s.percentile is not None else 0.0
    if has_rank:
        rank_head = (
            f'<span><span class="strong">{s.rank}/{s.total}</span>'
            f' · top <span class="rank-pct">{s.percentile:.2f}%</span></span>'
            f'<span>target top {target}%</span>'
        )
    else:
        rank_head = ('<span><span class="strong">Position not recorded</span></span>'
                     f'<span>target top {target}%</span>')
    rank = (
        f'<span class="rank{" is-hit" if hit else ""}">'
        f'<span class="rank-head">{rank_head}</span>'
        '<span class="track">'
        f'<span class="fill" style="width:{pct if has_rank else 0:.2f}%"></span>'
        f'<span class="goal" style="left:{target}%" title="Goal: top {target}%"></span>'
        '</span></span>'
    )

    status_cls = _status_class(s.status)
    round_word = "round" if s.rounds == 1 else "rounds"
    sub_value = f"{s.real_submissions}/{s.mock_submissions}"
    metrics = "".join([
        f'<span class="metric"><span class="k">Best CV</span>'
        f'<span class="v">{_fmt(s.best_cv)}</span></span>',
        f'<span class="metric"><span class="k">Best LB</span>'
        f'<span class="v">{_fmt(s.best_lb)}</span></span>',
        f'<span class="metric"><span class="k">Candidates</span>'
        f'<span class="v">{s.scored}/{s.candidates}</span></span>',
        f'<span class="metric"><span class="k">Failed</span>'
        f'<span class="v">{s.failed}</span></span>',
        f'<span class="metric"><span class="k">Subs r/m</span>'
        f'<span class="v" title="{s.real_submissions} real, {s.mock_submissions} mock">'
        f'{sub_value}</span></span>',
        f'<span class="metric"><span class="k">LLM spend</span>'
        f'<span class="v">${s.spend_usd:.2f}</span></span>',
    ])

    rows = []
    for point in s.points:
        cv = "" if point.cv is None else f"{point.cv:.6f}"
        lb = "" if point.lb is None else f"{point.lb:.6f}"
        failed = " failed" if point.status == "failed" else ""
        rows.append(
            f'<tr class="{failed.strip()}">'
            f'<td><code>{esc(point.id)}</code></td>'
            f'<td>{esc(point.model_type)}</td>'
            f'<td><span class="st st-{_candidate_status(point.status)}">'
            f'{esc(point.status)}</span></td>'
            f'<td class="num" data-v="{cv}">{_fmt(point.cv)}</td>'
            f'<td class="num" data-v="{lb}">{_fmt(point.lb)}</td></tr>'
        )
    tbody = "".join(rows) or '<tr><td colspan="5">no candidates</td></tr>'
    table = (
        '<div class="cand-head"><h3>Candidates</h3>'
        f'<span>{s.scored} scored · {s.failed} failed · {s.candidates} total</span></div>'
        '<div class="table-wrap"><table class="cand-table"><thead><tr>'
        '<th>id</th><th>model</th><th>status</th>'
        '<th class="num" data-type="num">CV</th>'
        '<th class="num" data-type="num">LB</th>'
        f'</tr></thead><tbody>{tbody}</tbody></table></div>'
    )

    search_blob = " ".join([
        s.run_id, s.competition, s.status, "live" if s.live else "dry",
        " ".join(sorted({p.model_type for p in s.points})),
    ]).lower()
    chart = _svg_chart(s.points, f"r{index}", s.higher_is_better)
    body = (
        '<div class="run-body">'
        '<div class="chart-head"><h3>Score trajectory</h3>'
        '<div class="detail-actions">'
        '<div class="legend">'
        '<button type="button" class="legend-btn" data-series="cv" aria-pressed="true">'
        '<span class="dotmark"></span>CV</button>'
        '<button type="button" class="legend-btn" data-series="lb" aria-pressed="true">'
        '<span class="dotmark"></span>Real LB</button>'
        '</div>'
        f'<button type="button" class="copy" data-id="{esc(s.run_id)}">Copy id</button>'
        '</div></div>'
        f'<div class="chart">{chart}</div>{table}</div>'
    )
    return (
        f'<details class="run{" open" if open_card else ""}" '
        f'data-competition="{esc(s.competition)}" data-search="{esc(search_blob)}">'
        '<summary>'
        '<svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M6 9l6 6 6-6"/></svg>'
        '<span class="run-head"><span>'
        f'<span class="comp">{esc(s.competition)}</span>'
        f'<span class="rid">{esc(s.run_id)}</span></span>'
        '<span class="badges">'
        f'<span class="pill pill-{"live" if s.live else "dry"}">'
        f'{"live" if s.live else "dry"}</span>'
        f'<span class="pill pill-{status_cls}">{esc(s.status)}</span>'
        f'<span class="pill pill-ghost">{s.rounds} {round_word}</span>'
        '</span></span>'
        f'<span class="run-metrics">{metrics}</span>'
        f'{rank}'
        '</summary>'
        f'{body}</details>'
    )


def render_html(summaries: list[RunSummary], generated_at: str) -> str:
    """Self-contained dark report: aggregates, filters, run cards, charts, tables."""
    esc = html.escape
    competitions: dict[str, int] = {}
    for s in summaries:
        competitions[s.competition] = competitions.get(s.competition, 0) + 1
    chips = [f'<button type="button" class="chip is-active" data-filter="all" '
             f'aria-pressed="true">All <span class="n">{len(summaries)}</span></button>']
    for competition, count in competitions.items():
        chips.append(
            f'<button type="button" class="chip" data-filter="{esc(competition)}" '
            f'aria-pressed="false">{esc(competition)} <span class="n">{count}</span></button>'
        )

    ranked = [s for s in summaries if s.percentile is not None]
    best = min(ranked, key=lambda s: s.percentile) if ranked else None
    total_spend = sum(s.spend_usd for s in summaries)
    real_subs = sum(s.real_submissions for s in summaries)
    mock_subs = sum(s.mock_submissions for s in summaries)
    scored = sum(s.scored for s in summaries)
    candidates = sum(s.candidates for s in summaries)

    stats = "".join([
        _stat_tile("Campaigns", str(len(summaries)),
                   f"{len(competitions)} competition{'s' if len(competitions) != 1 else ''}"),
        _stat_tile("Candidates scored", str(scored), f"of {candidates} proposed"),
        _stat_tile("Real submissions", str(real_subs), f"{mock_subs} mock (dry runs)"),
        _stat_tile("Best finish", f"top {best.percentile:.1f}%" if best else "—",
                   f"{best.competition}" if best else "no leaderboard position yet",
                   "good" if best and best.percentile <= (best.target_percentile or 20) else ""),
        _stat_tile("LLM spend", f"${total_spend:.2f}", "bounded, journal-summed"),
    ])

    featured = None
    if ranked:
        featured = summaries.index(best) if best is not None else None
    elif summaries:
        featured = max(range(len(summaries)), key=lambda i: summaries[i].candidates)
    cards = "".join(_run_card(s, i, i == featured) for i, s in enumerate(summaries))
    head = (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Slayer — competition attempts &amp; scores</title>"
        f"<style>{_PAGE_CSS}</style></head><body>"
        '<div class="wrap"><header class="hero">'
        '<div class="brand"><span class="mark">◆</span>Slayer'
        '<span class="sub">/ campaign report</span></div>'
        '<h1>Competition attempts <span class="grad">&amp; scores</span></h1>'
        '<p class="lede">Autonomous Kaggle campaigns from the Slayer agent team. '
        'Every score below comes from the deterministic CV harness and real leaderboard '
        'reads — expand a campaign to inspect its trajectory and candidates.</p>'
        f'<div class="meta">generated {esc(generated_at)} · '
        f'{len(summaries)} campaign{"s" if len(summaries) != 1 else ""}</div>'
        f'<div class="stats">{stats}</div></header>'
        '<div class="toolbar">'
        '<div class="chips" role="group" aria-label="Filter by competition">'
        f'{"".join(chips)}</div>'
        '<label class="search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" aria-hidden="true">'
        '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.2-3.2"/></svg>'
        '<input id="search" type="search" placeholder="Search runs, competitions, models…" '
        'aria-label="Search campaigns"></label>'
        '</div><main>'
    )
    tail = (
        f'<div class="grid" id="runs">{cards}</div>'
        '<div class="empty-state" id="empty" hidden><div><strong>No matching campaigns'
        '</strong>Try a different competition filter or search term.</div></div>'
        '</main><footer class="foot">Slayer · LLM judgment on deterministic scaffolding · '
        'self-contained report, no external assets</footer></div>'
        '<div class="tip" id="tip" role="tooltip" hidden></div>'
        f"<script>{_PAGE_JS}</script></body></html>"
    )
    return head + tail
