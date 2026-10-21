# 09 — REVIEW & SUBMIT: Reviewer Gate + Submission Sinks (Mock-first)

- **Phase:** 2 (step 2.4) · **Status:** implemented · **Date:** 2026-10-20
- **Decisions:** [D0002](../decisions/D0002-submission-autonomy-mock-sink.md) (approve-first + MockSink MVP)

## Goal

Close the MVP loop: an adversarial reviewer gates every candidate, and approved
candidates flow through a `SubmissionSink` — mocked by default (per D0002), real
Kaggle at Phase 2b. This completes Phase 2's acceptance: end-to-end pipeline with a
submission recorded via MockSink, zero real quota spent.

## REVIEW

- Candidate selection: best scored-but-unreviewed candidate (registry: `scored`,
  not yet approved/rejected/submitted), ranked by CV with the metric's direction.
- **ReviewerAgent** + `prompts/reviewer.md`: receives the candidate's `pipeline.py`
  code, its CV results, the strategy guidance, and the analyst report →
  `ReviewVerdict(approve | reject | revise, reasoning, risk_flags[])`.
- Verdicts: approve → registry `approved`; reject → `rejected`; revise → stays
  `scored` with the feedback journaled (Phase 3's iteration engine consumes it).
- The reviewer's checklist hunts: target leakage, fit-on-test, id misuse, CV protocol
  violations, metric handling errors, **intent-vs-implementation gaps** (e.g. the
  scale_pos_weight "placeholder" found in the live baseline), quota value.

## SUBMIT

Order of gates, each journaled:

1. **CV gate** (D0002: only-if-CV-improved): first submission always allowed; later
   ones need `cv - best_submitted_cv` (direction-aware) ≥ `plan.min_cv_improvement`.
2. **Quota gate**: real submissions only — `QuotaManager` counts journal
   `submission.accepted` events; MockSink never emits them (mock submissions are free).
3. **Human approval gate** (`approve-first`, the default): the CLI's approver
   (typer.confirm) must confirm; `--autonomy agentic` skips it.
4. **Sink**: `SubmissionSink.submit(...)` → registry status `submitted` + journal.

```python
SubmissionRecord(submission_id, candidate_id, message, simulated, lb_score)

MockSink:   simulated LB = cv + deterministic tiny noise (hash-seeded);
            journals `submission.mocked`; consumes no quota.
KaggleSink:  kaggle.submit() → journals `submission.accepted` (quota!);
            LB score filled later by MONITOR (Phase 4) — none recorded yet.
```

The sink is wired in `slayer run` from `run.dry_run` (config/CLI flag); the engine
never talks to Kaggle transport directly.

## Loop plumbing

- `iterate_phase` is an honest MVP passthrough (journaled `iterate.skipped`): the real
  iteration engine is Phase 3 — so the pipeline flows BASELINE → REVIEW → SUBMIT and
  pauses at MONITOR.
- `RunContext` gains `sink` and `approver` (a `Callable[[str], bool]`); the CLI injects
  a typer.confirm-based approver, tests inject lambdas.

## Tests

- Verdict schema literals; reviewer agent prompt contains code + guidance.
- Review phase: approve/reject/revise transitions in the registry + journaling.
- Submit phase: mock first submission (registry submitted, `submission.mocked`, no
  quota consumption), CV gate blocks insufficient improvement, approver refusal skips,
  KaggleSink path with fake kaggle client emits `submission.accepted` and the quota
  manager then blocks.
- CLI end-to-end: `dry_run + agentic` config → full pipeline to a mocked submission.

## Retro

Live-validated twice; both failure *and* success paths taught us something:

- **First live run**: the researcher generated code that fit a `OneHotEncoder` inside
  `make_features` and converted the sparse matrix to a DataFrame — pathologically slow
  on 668k rows, killed by the 600s harness timeout. The failure path worked exactly as
  designed: candidate `failed`, run failed with a clear error, nothing else corrupted.
  Fix: the researcher contract now *explicitly* forbids fitting inside `make_features`
  and requires dense DataFrames (violations were already contract-breaking; now the
  prompt says so in executable terms).
- **Second live run**: clean pipeline — researcher code scored **0.941585 CV**, and the
  **reviewer returned REVISE** with the risk flag `scale_pos_weight not dynamically
  calculated per fold` (the code hardcoded the global 82.5/17.5 ratio instead of
  fold-derived weights). SUBMIT correctly refused to submit an unapproved candidate.
  The gate works: an LLM reviewing another LLM caught a real statistical subtlety.
- Phase 2 acceptance (submission recorded via MockSink, approvals + quota logic
  exercised, zero real quota) is met: the CLI end-to-end test runs the full loop
  including the mocked submission; the live runs demonstrate the honest gating path.
- Note: a REVISE verdict currently parks feedback for the (unimplemented) iteration
  engine — Phase 3 closes that loop.
