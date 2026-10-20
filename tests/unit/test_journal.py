from slayer.workspace import EventType, Journal


def test_append_read_roundtrip(tmp_path) -> None:
    journal = Journal(tmp_path / "run" / "journal.jsonl")
    e1 = journal.append("planner", EventType.AGENT_OUTPUT, {"plan": "baseline"})
    e2 = journal.append("reviewer", EventType.AGENT_OUTPUT, {"verdict": "approve"})

    events = journal.read()
    assert [e.agent for e in events] == ["planner", "reviewer"]
    assert events[0].data == {"plan": "baseline"}
    assert events[1].type == EventType.AGENT_OUTPUT
    assert events[0].ts == e1.ts
    assert events[1].ts == e2.ts


def test_append_with_run_field(tmp_path) -> None:
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append("system", EventType.RUN_CREATED, {"competition": "x"}, run="run-42")
    events = journal.read()
    assert events[0].run == "run-42"


def test_read_filters_by_type_and_agent(tmp_path) -> None:
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append("planner", "agent.output", {})
    journal.append("reviewer", "agent.output", {})
    journal.append("system", "phase.enter", {"phase": "PLANNING"})
    journal.append("planner", "budget.warning", {})

    assert len(journal.read(type="agent.output")) == 2
    assert len(journal.read(agent="planner")) == 2
    assert len(journal.read(type="agent.output", agent="reviewer")) == 1
    assert journal.read(type="phase.enter")[0].data == {"phase": "PLANNING"}


def test_limit_and_tail(tmp_path) -> None:
    journal = Journal(tmp_path / "journal.jsonl")
    for i in range(10):
        journal.append("system", "phase.enter", {"i": i})

    assert len(journal.read()) == 10
    assert len(journal.read(limit=3)) == 3
    assert journal.read(limit=3)[0].data == {"i": 7}
    assert journal.tail(2)[-1].data == {"i": 9}


def test_blank_lines_tolerated(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = Journal(path)
    journal.append("system", "phase.enter", {})
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n\n")
    journal.append("system", "phase.enter", {"after": True})
    assert len(journal.read()) == 2
