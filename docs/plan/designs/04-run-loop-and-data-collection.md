# 04 — Run Loop Engine, INIT & DATA_COLLECTION

- **Phase:** 1 (step 1.1) · **Status:** implemented · **Date:** 2026-10-19
- **Change:** [C0002](../changes/C0002-loop-phase-reorder.md) — phase order (PLANNING after ANALYSIS)

## Goal

The deterministic loop skeleton (D0003) with its first two phases, exposed via the main
user command: `slayer run <competition>`.

## Interface

```python
Phase (str Enum): init | data_collection | analysis | planning | baseline | iterate | review | submit | monitor
PHASE_ORDER: tuple[Phase, ...]

RunContext(run, journal, registry, kaggle, store, reporter=None)   # dataclass; .say() streams to reporter

LoopEngine(ctx)
  .start() -> Run    # runs phases in order; active → (paused | finished | failed)

slayer run <competition> [ --budget-usd | --autonomy | --dry-run/--live | --submissions-per-day
                          --config-dir | --env-file ]
```

## Phase semantics

- **Enter**: every phase appends a `phase.enter` journal event and reports to the CLI.
- **Unimplemented phase** → run pauses (`run.status_changed` + `loop.paused` with
  reason). Phase 1 stops after `data_collection`; step 1.2 adds `analysis`, etc. No
  stubs, no skips — the pause point is always the first unimplemented phase.
- **Handler exception** → run status `failed`, journal updated, exception re-raised
  (CLI exits 1).
- **All phases done** → `finished` (reached only once ITERATE/MONITOR cycling exists).

## INIT

- `KaggleClient.find_competition(ref)` — paginated scan of the active competition list
  (search-based listing can't be trusted for custom refs like `playground-series-s5e10`),
  up to 5 pages × 100, matching exact `ref`.
- Writes `workspace/reports/competition.json` (the `CompetitionInfo` model — metric,
  deadline, quota — ready for the planner) + `competition.context` journal event.
- Not found → `KaggleError` → run fails with actionable message.

## DATA_COLLECTION

- `KaggleClient.download_files` → `<competition>.zip` in `workspace/data/` (zip kept).
- **Safe unzip**: every member's resolved destination must stay inside the data dir —
  zip-slip (`../evil.csv`) aborts the phase. Empty archives are an error.
- **Manifest** (`workspace/data/manifest.json`): per-file `size_bytes` + `sha256`,
  totals, timestamp — the integrity baseline for later phases.
- `data.collected` journal event with file count / total bytes.

## CLI (`slayer run`)

- Flags override `config/slayer.yaml` defaults (re-validated through `DefaultsConfig`).
- Requires `KAGGLE_API_TOKEN`; builds `KaggleClient.from_token` (real) — injectable in
  tests via `cli_main.KaggleClient`.
- Streams phase activity line-by-line; final status echoed with the run id + workspace
  path so the user can inspect `journal.jsonl` / `reports/` afterwards.

## Tests

- Engine: INIT+DATA_COLLECTION happy path (status paused, files on disk, manifest,
  phase events, pause reason names `analysis`); failure path (kaggle error → `failed`
  status persisted, exception propagates).
- Data: zip-slip rejected, nested paths allowed, empty archive error, manifest hashes
  match recomputation.
- Kaggle wrapper: `find_competition` pagination (page-token threading) + not-found.
- CLI: end-to-end `slayer run` with fake kaggle client + tmp runs dir → exit 0, run
  created, journal on disk.

## Retro

Implemented as designed. Live-run findings (against `playground-series-s6e9`):

- **Refs are full URLs** in kaggle 2.x (`https://www.kaggle.com/competitions/<name>`) —
  `find_competition` normalizes refs to the bare name and `CompetitionInfo.ref` always
  carries the canonical bare form.
- **Closed comps are invisible to the default listing** (active-only) — `find_competition`
  falls back to a `search=<ref>` pass, which matches via the description (searching the
  bare name finds the comp even when it's closed).
- **Data download is rules-gated**: unaccepted competition rules → HTTP 403. The error
  is wrapped with an actionable "accept the rules at .../rules" hint. Rule acceptance is
  a manual per-competition user step (the Kaggle API doesn't expose it) — INIT should
  surface it before the loop starts (later milestone).
- `RunContext` is deliberately hand-rolled (not pydantic): it carries live handles
  (journal, registry, client), not serialized state.
- CLI `run` catches unexpected phase errors without dumping tracebacks; the journal +
  `failed` run status carry the details.
