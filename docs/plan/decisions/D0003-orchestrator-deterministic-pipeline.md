# D0003 — Orchestrator pattern: deterministic pipeline with LLM decision points

- **Status:** accepted
- **Date:** 2026-10-19

## Context

Two viable patterns for the orchestration loop: a code-controlled phase machine, or a
fully agentic orchestrator (an LLM with tools deciding every step dynamically).

## Options considered

1. **Deterministic pipeline with LLM decision points** — code controls phase
   transitions (INIT → PLANNING → … → MONITOR); LLM agents decide the *content* within
   each phase and make defined decisions (submit vs iterate, which candidate).
2. **Fully agentic orchestrator** — the orchestrator is an LLM agent with tools
   (spawn agents, submit, replan) choosing every next step. More adaptive, but harder
   to test, costlier, and prone to loops/dead ends.

## Decision

v1 implements the **deterministic pipeline with LLM decision points**. The orchestrator
agent cannot invent new phases; it operates inside the state machine. The agentic
upgrade (dynamic replanning, free-form tool use) is an explicit Phase 5 milestone with
the state machine as fallback.

## Consequences

- Predictable cost and control flow; phase transitions are unit-testable.
- Degraded flexibility for genuinely novel situations — accepted for v1; the playbook
  (known-good phase sequence) matches how strong Kaggle teams actually work.
- Journal + blackboard design already gives the Phase 5 upgrade a clean substrate.
