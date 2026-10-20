from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from slayer.config import AgentsConfig
from slayer.kaggle import KaggleClient
from slayer.llm import LLMClient
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

    def say(self, message: str) -> None:
        """Stream a progress line to the reporter (if attached)."""
        if self.reporter is not None:
            self.reporter(message)


def _default_handlers() -> dict[Phase, Handler]:
    """Resolve the default phase handlers (lazy import to avoid cycles)."""
    from slayer.loop.phases import HANDLERS

    return dict(HANDLERS)


class LoopEngine:
    def __init__(
        self, ctx: RunContext, handlers: Mapping[Phase, Handler] | None = None
    ) -> None:
        """Wire the engine to a run context and (optionally) custom handlers."""
        self._ctx = ctx
        self._handlers: dict[Phase, Handler] = (
            dict(handlers) if handlers is not None else _default_handlers()
        )

    def start(self) -> Run:
        """Execute phases in order until pause, failure, or completion."""
        self._ctx.run = self._ctx.store.update_status(self._ctx.run, RunStatus.ACTIVE)
        for phase in PHASE_ORDER:
            handler = self._handlers.get(phase)
            if handler is None:
                return self._pause(f"'{phase.value}' phase arrives with the next milestone")
            self._ctx.say(f"[phase] {phase.value}")
            self._ctx.journal.append(
                agent="loop",
                type=EventType.PHASE_ENTER,
                data={"phase": phase.value},
                run=self._ctx.run.run_id,
            )
            try:
                handler(self._ctx)
            except Exception as exc:
                self._fail(exc)
                raise
        self._ctx.say("[done] all phases completed")
        return self._ctx.store.update_status(self._ctx.run, RunStatus.FINISHED)

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
