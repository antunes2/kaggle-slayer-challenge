from __future__ import annotations

from slayer.loop.data import data_collection_phase
from slayer.loop.engine import Handler, Phase, RunContext
from slayer.workspace import EventType


def init_phase(ctx: RunContext) -> None:
    info = ctx.kaggle.find_competition(ctx.run.competition)
    report_path = ctx.run.workspace("reports") / "competition.json"
    report_path.write_text(info.model_dump_json(indent=2), encoding="utf-8")
    ctx.journal.append(
        agent="loop",
        type=EventType.COMPETITION_CONTEXT,
        data={
            "ref": info.ref,
            "title": info.title,
            "deadline": info.deadline,
            "metric": info.evaluation_metric,
            "max_daily_submissions": info.max_daily_submissions,
        },
        run=ctx.run.run_id,
    )
    ctx.say(
        f"  competition: {info.title} "
        f"(metric={info.evaluation_metric}, quota={info.max_daily_submissions}/day)"
    )


HANDLERS: dict[Phase, Handler] = {
    Phase.INIT: init_phase,
    Phase.DATA_COLLECTION: data_collection_phase,
}
