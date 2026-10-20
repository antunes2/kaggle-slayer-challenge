# 03 — Kaggle Client & Quota Manager

- **Phase:** 0 (step 0.4) · **Status:** implemented · **Date:** 2026-10-19
- **ADR:** [D0002](../decisions/D0002-submission-autonomy-mock-sink.md) (submissions sink lands in Phase 2; this is the transport)

## Goal

A thin, defensively-converted wrapper over the installed `kaggle` package (v2.2.4,
kagglesdk-based), plus a journal-backed daily submission quota manager. The loop and
sinks never touch `kaggle.api` directly.

## Findings driving the design (kaggle 2.2.4)

- **Auth**: the package reads the `KAGGLE_API_TOKEN` env var (new-style `KGAT...` access
  token — matching our `.env`), introspects it, and resolves the username. Legacy
  username/key and OAuth exist as fallbacks; we use the token path only.
- **Quirk**: `api.authenticate()` calls `exit(1)` on failure — the wrapper must catch
  `SystemExit` and raise `KaggleError` instead of letting the process die.
- **Import side effect**: importing `kaggle` authenticates a module-level instance.
  We therefore import it lazily (inside the factory), so merely importing
  `slayer.kaggle` does no network work.
- **Env injection**: secrets live in `.env` (loaded by pydantic-settings, not exported);
  the kaggle package reads `os.environ` — so `from_token()` injects the token into the
  process env before building the API.
- Response objects are kagglesdk `KaggleObject`s; everything is converted to our
  pydantic models via defensive `getattr` so sdk field changes can't crash the loop.

## Interface

```python
KaggleClient(api)                     # api injected (fakes in tests)
KaggleClient.from_token(token)        # lazy kaggle import + env inject + authenticate

.ping() -> str                        # authenticated username (introspection already
                                      # happened in authenticate)
.list_competitions(search, category, sort_by, page_size) -> list[CompetitionInfo]
.list_files(competition) -> list[CompetitionFileInfo]
.download_files(competition, target_dir, force=False) -> Path   # returns the zip path
.submit(competition, file_path, message="") -> SubmissionRef
.submissions(competition) -> list[SubmissionInfo]               # poll scores/status
.leaderboard(competition, page_size=50) -> list[LeaderboardEntry]

QuotaManager(journal, max_per_day)
.used_today(now=None) -> int          # counts today's `submission.accepted` events
.remaining(now=None) / .can_submit(now=None)
```

Models: `CompetitionInfo` (ref, title, category, reward, deadline, evaluation_metric,
user_has_entered, max_daily_submissions), `CompetitionFileInfo`, `SubmissionRef`,
`SubmissionInfo` (status as enum-name string, scores as strings — Kaggle sends strings),
`LeaderboardEntry` (rank by enumeration).

## Decisions

- **Quota counts real submissions only**: `submission.accepted` is emitted by the
  *Kaggle* sink (Phase 2). MockSink submissions will emit their own event type and never
  consume quota (per D0002).
- **Quota scope is per-run journal** (v1): one run = one competition. Multi-run quota
  aggregation across runs on the same competition is a known limitation, revisited if
  we ever run parallel runs on the same comp.
- **`download_files` returns the zip** — unzipping + manifest writing is Phase 1
  (DATA_COLLECTION) business, not transport.
- **Score fields are strings** — Kaggle formats leaderboards/submission scores as
  strings; parsing to float is the metric-aware consumer's job (planner CV harness).

## Tests

- Unit (injected fakes): ping, missing token, env injection + lazy import factory,
  field mapping for competitions (incl. reward), files, download (happy + missing-zip
  error), submit (ref mapping), submissions (status enum-name), leaderboard ranking.
- Quota: counts today's events only (old ts lines crafted directly), ignores other
  event types, blocks at limit.
- CLI: `_kaggle_ping` monkeypatched like `_llm_ping`; real ping exercised via
  `slayer init` in real usage (live introspection, no quota cost).

## Retro

Implemented as designed. `slayer init` now shows a real `kaggle: ok (user=...)` line.
Note: importing `kaggle` performs one extra introspection at import time (module-level
`api.authenticate()` in the package itself) — harmless, already authenticated with the
same token.
