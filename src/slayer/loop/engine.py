from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from slayer.config import AgentsConfig
from slayer.kaggle import KaggleClient
from slayer.llm import LLMClient
from slayer.submissions import SubmissionSink
from slayer.workspace import CandidateRegistry, EventType, Journal, Run, RunStatus, RunStore

Handler = Callable[["RunContext"], None]


class Phase(StrEnum):
    """Loop phases in execution order (post-C0002)."""

    INIT = "init"
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    BASELINE = "baseline"
    ITERATE = "iterate"
    REVIEW = "review"
    SUBMIT = "submit"
    MONITOR = "monitor"
    REPLAN = "replan"


PHASE_ORDER: tuple[Phase, ...] = (
    Phase.INIT,
    Phase.DATA_COLLECTION,
    Phase.ANALYSIS,
    Phase.PLANNING,
    Phase.BASELINE,
    Phase.ITERATE,
    Phase.REVIEW,
    Phase.SUBMIT,
    Phase.MONITOR,
)

# One re-entry round after a target shortfall (D0007).
ROUND_ORDER: tuple[Phase, ...] = (
    Phase.REPLAN,
    Phase.ITERATE,
    Phase.REVIEW,
    Phase.SUBMIT,
    Phase.MONITOR,
)


class CampaignStop(Exception):
    """Raised by REPLAN when the planner declares the campaign exhausted."""

    pass


@dataclass
class ReentryDecision:
    """Outcome of the deterministic re-entry gate after MONITOR."""

    reenter: bool
    reason: str
    round: int = 0


Decider = Callable[["RunContext"], ReentryDecision]


@dataclass
class RunContext:
    """Live handles for one loop execution: run, journal, registry, clients."""

    run: Run
    journal: Journal
    registry: CandidateRegistry
    kaggle: KaggleClient
    store: RunStore
    reporter: Callable[[str], None] | None = None
    llm: LLMClient | None = None
    agent_configs: AgentsConfig | None = None
    sink: SubmissionSink | None = None
    approver: Callable[[str], bool] | None = None

    def say(self, message: str) -> None:
        """Stream a progress line to the reporter (if attached)."""
        if self.reporter is not None:
            self.reporter(message)


def _default_handlers() -> dict[Phase, Handler]:
    """Resolve the default phase handlers (lazy import to avoid cycles)."""
    from slayer.loop.campaign import replan_phase
    from slayer.loop.phases import HANDLERS

    return {**HANDLERS, Phase.REPLAN: replan_phase}


def _default_decider() -> Decider:
    """Resolve the default re-entry gate (lazy import to avoid cycles)."""
    from slayer.loop.campaign import decide_reentry

    return decide_reentry


class LoopEngine:
    def __init__(
        self,
        ctx: RunContext,
        handlers: Mapping[Phase, Handler] | None = None,
        decider: Decider | None = None,
    ) -> None:
        """Wire the engine to a run context, handlers, and (optionally) a re-entry gate.

        With the default handler set the default re-entry gate is used; a custom
        handler map runs re-entry rounds only when a decider is passed explicitly.
        """
        self._ctx = ctx
        if handlers is None:
            self._handlers: dict[Phase, Handler] = _default_handlers()
            self._decider = decider or _default_decider()
        else:
            self._handlers = dict(handlers)
            self._decider = decider

    def start(self) -> Run:
        """Execute the linear pass, then bounded re-entry rounds, until done/pause/failure.

        Ctrl+C (KeyboardInterrupt) pauses the run cleanly — status `paused`, journaled —
        so `slayer resume` continues it (Q0003).
        """
        self._ctx.run = self._ctx.store.update_status(self._ctx.run, RunStatus.ACTIVE)
        try:
            for phase in PHASE_ORDER:
                handler = self._handlers.get(phase)
                if handler is None:
                    return self._pause(f"'{phase.value}' phase arrives with the next milestone")
                self._run_phase(phase, handler)
            if self._decider is not None:
                self._run_rounds()
        except KeyboardInterrupt:
            return self._pause("interrupted by user (Ctrl+C) — resume to continue")
        self._ctx.say("[done] all phases completed")
        return self._ctx.store.update_status(self._ctx.run, RunStatus.FINISHED)

    def _run_phase(self, phase: Phase, handler: Handler, round_number: int | None = None) -> None:
        """Announce, journal, and execute one phase; mark the run failed on errors."""
        label = phase.value if round_number is None else f"{phase.value} (round {round_number})"
        self._ctx.say(f"[phase] {label}")
        data: dict = {"phase": phase.value}
        if round_number is not None:
            data["round"] = round_number
        self._ctx.journal.append(
            agent="loop", type=EventType.PHASE_ENTER, data=data, run=self._ctx.run.run_id
        )
        try:
            handler(self._ctx)
        except CampaignStop:
            raise
        except Exception as exc:
            self._fail(exc)
            raise

    def _run_rounds(self) -> None:
        """Re-enter REPLAN → … → MONITOR while the gate allows (hard-capped by max_rounds)."""
        executed = 0
        while True:
            decision = self._decider(self._ctx)
            if not decision.reenter:
                self._ctx.say(f"[campaign] no re-entry: {decision.reason}")
                return
            if executed >= self._ctx.run.max_rounds:
                self._ctx.say("[campaign] round cap reached in this invocation")
                return
            executed += 1
            self._ctx.say(f"[round {decision.round}] re-entering: {decision.reason}")
            try:
                for phase in ROUND_ORDER:
                    handler = self._handlers.get(phase)
                    if handler is None:
                        self._ctx.say(f"[campaign] no '{phase.value}' handler — rounds disabled")
                        return
                    self._run_phase(phase, handler, decision.round)
            except CampaignStop as stop:
                self._ctx.say(f"[campaign] planner stopped the campaign: {stop}")
                return

    def _pause(self, reason: str) -> Run:
        """Pause the run at the first unimplemented phase."""
        self._ctx.say(f"[pause] {reason}")
        self._ctx.journal.append(
            agent="loop",
            type=EventType.LOOP_PAUSED,
            data={"reason": reason},
            run=self._ctx.run.run_id,
        )
        return self._ctx.store.update_status(self._ctx.run, RunStatus.PAUSED)

    def _fail(self, exc: Exception) -> None:
        """Mark the run failed after a phase handler exception."""
        self._ctx.say(f"[fail] {exc}")
        self._ctx.run = self._ctx.store.update_status(self._ctx.run, RunStatus.FAILED)
