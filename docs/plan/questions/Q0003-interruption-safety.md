# Q0003 — What happens when the CLI is stopped mid-run? (interruption safety)

- **Status:** settled for v1 (options 1 + 2 implemented; hard-kill residual risk documented)
- **Raised:** 2026-10-20
- **Blocks:** nothing — current behavior is safe-but-untidy for tokens, with one orphan-process edge case.

## Findings (as of step 3.2)

| Component | On Ctrl+C (SIGINT) | Risk |
|-----------|--------------------|------|
| Agent LLM calls | In-process, synchronous — die with the CLI; OpenRouter stops generating on the dropped request | **No eternal token burn.** Worst case: pay for the one in-flight call's partial output (bounded by max_tokens) |
| Harness subprocess | Deliberately detached (`start_new_session=True`) — **survives** the CLI's death, finishes the CV, writes results, exits | No tokens (pure CPU), **but the parent was the only timeout enforcer**: a pathological orphaned candidate has no kill switch → potential unkillable orphan process |
| Run state | `KeyboardInterrupt` isn't caught — run status stays `active` (the plan's §6.2 "graceful Ctrl+C pause" marking is **not implemented**) | Misleading status; recovery works anyway via `slayer resume` (skip logic) |
| Journal/registry | All writes are append-only/atomic up to the interrupt point | Safe — resume picks up cleanly |

## Options

1. **Harness self-enforced deadline** — pass `budget_s` in `cv_config.json`; the harness
   sets its own `signal.alarm`/deadline so an orphan can never outlive its budget even
   without a parent. (Step 3.3 passes the budget into the config already; the
   self-enforcement is the small extra.)
2. **Graceful SIGINT handler in the CLI** — catch KeyboardInterrupt at the engine
   boundary, mark the run `paused` with a `run.interrupted` journal event, exit 0.
   Makes the plan's §6.2 promise true.
3. **Adoption of orphan harnesses on resume** — scan for running harness processes of
   the run's candidates and wait/kill on resume. Most complex; only needed if orphans
   become common.

## Recommendation

**(1) + (2)** — both are small, close the only real risks (orphan process, lying run
status), and make Ctrl+C semantics match the plan. **(3)** deferred until proven
necessary. Suggested slot: fold (1) into step 3.3 (it shares the budget plumbing);
(2) into Phase 4 ops.

**Update 2026-10-20:** option (1) implemented (step 3.3: SIGALRM self-deadline).
**Update 2026-10-21 — live incident confirming the caveat:** a user-stopped run left
an orphan CatBoost harness running at ~600% CPU for 84 minutes — 3× past its budget —
because SIGALRM cannot interrupt a long C fit and the parent (the authoritative
killer) was already dead. Process killed manually. The bounded-retry cap (design 10
addendum) prevents this specific step from re-spawning, but the general orphan risk
remains; a detached watchdog (runner-spawned killer process) is the candidate fix if
incidents recur. Option (2) (graceful SIGINT → paused status) remains open for Phase 4.

**Update 2026-10-22 — option (2) implemented:** Ctrl+C (KeyboardInterrupt) anywhere in
the loop → run `paused` with a journaled `loop.paused` reason; Ctrl+C at the y/n
submission prompt is treated the same; the runner kills the detached harness process
group on KeyboardInterrupt (closing the Ctrl+C orphan path). Remaining: a *hard* kill
(`kill -9`, power loss) can still orphan a harness — option (3)/a watchdog stays
deferred. Status: **settled for v1** (residual risk in `docs/limitations.md`).
