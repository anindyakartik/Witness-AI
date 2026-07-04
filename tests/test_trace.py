"""Tests for the trace layer: round-trip serialization and event ordering."""

from __future__ import annotations

from pathlib import Path

from witness.core.trace import (
    EventType,
    TraceEvent,
    TraceRun,
    TraceStore,
    drift_alert_payload,
    llm_call_payload,
    tool_call_payload,
)


def test_trace_event_round_trip() -> None:
    event = TraceEvent.new(
        run_id="run_abc",
        agent_name="ticket_filer",
        event_type=EventType.TOOL_CALL,
        payload=tool_call_payload(
            tool_name="create_ticket",
            args={"subject": "billing issue"},
            result={"ticket_id": 4471},
            ok=True,
        ),
        cost_usd=0.0001,
        latency_ms=12.5,
        parent_id="evt_parent",
    )

    restored = TraceEvent.from_dict(event.to_dict())

    assert restored == event
    assert restored.event_type is EventType.TOOL_CALL
    assert restored.payload["result"]["ticket_id"] == 4471


def test_trace_run_round_trip_via_store(tmp_path: Path) -> None:
    store = TraceStore(base_dir=tmp_path)
    run = TraceRun.start(agent_name="ticket_filer", seed=1729, scenario="clean_run")
    store.create_run(run)

    e1 = TraceEvent.new(
        run_id=run.run_id,
        agent_name=run.agent_name,
        event_type=EventType.LLM_CALL,
        payload=llm_call_payload(
            model="gemini-2.5-flash-lite",
            system="You are a ticket filer.",
            messages=[{"role": "user", "content": "file a ticket"}],
            response_text=None,
            function_call={"name": "create_ticket", "args": {"subject": "x"}},
            prompt_tokens=42,
            completion_tokens=10,
            replayed=True,
        ),
        cost_usd=0.00002,
        latency_ms=340.0,
    )
    store.append_event(run, e1)

    e2 = TraceEvent.new(
        run_id=run.run_id,
        agent_name=run.agent_name,
        event_type=EventType.TOOL_CALL,
        payload=tool_call_payload(
            tool_name="create_ticket",
            args={"subject": "x"},
            result={"ticket_id": 100},
            ok=True,
        ),
        cost_usd=0.0,
        latency_ms=1.0,
        parent_id=e1.id,
    )
    store.append_event(run, e2)
    store.finish_run(run, outcome="success")

    loaded = store.load_run(run.run_id)

    assert loaded.run_id == run.run_id
    assert loaded.agent_name == "ticket_filer"
    assert loaded.seed == 1729
    assert loaded.scenario == "clean_run"
    assert loaded.outcome == "success"
    assert loaded.ended_at is not None

    # Ordering must be preserved exactly as appended.
    assert [e.id for e in loaded.events] == [e1.id, e2.id]
    assert loaded.events[0].event_type is EventType.LLM_CALL
    assert loaded.events[1].event_type is EventType.TOOL_CALL
    assert loaded.events[1].parent_id == e1.id

    # Cost aggregation.
    assert loaded.total_cost_usd == e1.cost_usd + e2.cost_usd


def test_trace_store_list_and_load_all_runs(tmp_path: Path) -> None:
    store = TraceStore(base_dir=tmp_path)

    run_a = TraceRun.start(agent_name="summarizer", seed=1)
    store.create_run(run_a)
    store.finish_run(run_a, outcome="success")

    run_b = TraceRun.start(agent_name="data_lookup", seed=2)
    store.create_run(run_b)
    store.finish_run(run_b, outcome="success")

    run_ids = store.list_run_ids()
    assert set(run_ids) == {run_a.run_id, run_b.run_id}

    all_runs = store.load_all_runs()
    assert {r.agent_name for r in all_runs} == {"summarizer", "data_lookup"}


def test_events_of_type_filters_correctly() -> None:
    run = TraceRun.start(agent_name="data_lookup", seed=1)

    tool_event = TraceEvent.new(
        run_id=run.run_id,
        agent_name=run.agent_name,
        event_type=EventType.TOOL_CALL,
        payload=tool_call_payload(tool_name="search_customer", args={}, result=[], ok=True),
    )
    drift_event = TraceEvent.new(
        run_id=run.run_id,
        agent_name=run.agent_name,
        event_type=EventType.DRIFT_ALERT,
        payload=drift_alert_payload(
            agent_name="data_lookup", distance=0.62, reason="new tool used", details={}
        ),
    )
    run.add_event(tool_event)
    run.add_event(drift_event)

    assert run.events_of_type(EventType.TOOL_CALL) == [tool_event]
    assert run.events_of_type(EventType.DRIFT_ALERT) == [drift_event]
    assert run.events_of_type(EventType.CLAIM) == []
