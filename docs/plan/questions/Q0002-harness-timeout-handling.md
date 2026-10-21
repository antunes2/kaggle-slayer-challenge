# Q0002 — How should the harness handle slow or pathological candidates?

- **Status:** open (plan agreed; implementation = Phase 3 step 3.3)
- **Raised:** 2026-10-20
- **Blocks:** nothing — current fixed 600s timeout + clean failure path work today.

## Question

The harness currently uses a **fixed 600s timeout** for every candidate. Live runs
showed this is both too blunt and too opaque:

- Real candidates vary 162s → 472s for the *same* dataset, depending on the
  researcher's preprocessing choices (a heavy-but-valid candidate can be killed).
- A pathological candidate (sparse→DataFrame densify on 668k rows) burns the **full
  600s** before failing — the failure message is just "timeout", giving the researcher
  nothing actionable for repair.
- The researcher has **no idea the time budget exists** — nothing in its prompt
  calibrates `n_estimators`/preprocessing cost to it.

## Options considered

1. **Configurable budget** — `--harness-timeout-s` flag + config default. Trivial,
   immediate relief for legitimately-heavy candidates.
2. **Budget-aware researcher** — state the time budget in the researcher prompt
   ("CV must finish in ~N minutes; calibrate n_estimators/preprocessing") so it
   costs what it has.
3. **Probe fold / early abort** — run fold 0 alone with a fraction of the budget;
   if it exceeds a per-fold cap, abort immediately with per-fold timing instead of
   burning the full budget. Pathological candidates fail in ~1/5 of the time.
4. **Incremental fold reporting** — write each fold's score to `cv_results.json` as
   it completes, so a timeout still yields partial results + timing data for the
   repair prompt ("fold 1 took >120s with your preprocessing").
5. **Planner-estimated protocol cost** — planner picks folds/params from an estimated
   runtime model. Too speculative for now; the harness doesn't know model costs.

## Recommendation (agreed with the user)

Implement in Phase 3 step 3.3, in this order: **(1) + (2)** immediately (cheap, most
of the value — calibration beats detection), then **(3) + (4)** as part of the repair
loop (they turn timeouts into *actionable* repair feedback, which the iteration
engine consumes). **(5)** rejected for v1.

The clean existing failure path (candidate `failed`, run fails, nothing corrupted)
stays as the backstop in all cases.
