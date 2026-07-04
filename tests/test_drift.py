"""Tests for the DriftDetector: silent on stable behavior, alerts on injected drift."""

from __future__ import annotations

from pathlib import Path

from witness.core.trace import EventType, TraceEvent, TraceRun, TraceStore, tool_call_payload
from witness.governance.drift import DriftDetector, build_fingerprint


def _synthetic_run(
    agent_name: str,
    *,
    tool_names: list[str],
    total_cost_usd: float,
    llm_calls: int,
) -> TraceRun:
    """Build a run with the given tool calls and cost, without needing the LLM
    client or real mocks -- drift only cares about the shape of the trace."""
    run = TraceRun.start(agent_name=agent_name, seed=1)
    per_call_cost = total_cost_usd / llm_calls if llm_calls else 0.0
    for _ in range(llm_calls):
        run.add_event(
            TraceEvent.new(
                run_id=run.run_id,
                agent_name=agent_name,
                event_type=EventType.LLM_CALL,
                payload={
                    "model": "test",
                    "system": "",
                    "messages": [],
                    "response_text": None,
                    "function_call": None,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "replayed": True,
                },
                cost_usd=per_call_cost,
            )
        )
    for name in tool_names:
        run.add_event(
            TraceEvent.new(
                run_id=run.run_id,
                agent_name=agent_name,
                event_type=EventType.TOOL_CALL,
                payload=tool_call_payload(tool_name=name, args={}, result={"ok": True}, ok=True),
            )
        )
    return run


def _stable_baseline(agent_name: str, n: int = 10) -> list[TraceRun]:
    """N clean data_lookup-shaped runs: 1 search + 1 record lookup, with the small
    natural cost variance real runs have (different task inputs, token counts)."""
    costs = [0.00038, 0.00040, 0.00041, 0.00039, 0.00042, 0.00040, 0.00038, 0.00041, 0.00039, 0.00040]
    return [
        _synthetic_run(
            agent_name,
            tool_names=["search_customer", "get_customer_record"],
            total_cost_usd=costs[i % len(costs)],
            llm_calls=3,
        )
        for i in range(n)
    ]


def test_no_alert_on_stable_behavior() -> None:
    baseline = _stable_baseline("data_lookup")
    fingerprint = build_fingerprint("data_lookup", baseline)
    detector = DriftDetector(fingerprint)

    new_run = _synthetic_run(
        "data_lookup",
        tool_names=["search_customer", "get_customer_record"],
        total_cost_usd=0.00042,  # tiny natural variation
        llm_calls=3,
    )
    result = detector.check(new_run)

    assert result.is_drift is False
    assert result.reason is None
    assert result.new_tools == ()


def test_alert_on_new_tool_absent_from_baseline() -> None:
    baseline = _stable_baseline("data_lookup")
    fingerprint = build_fingerprint("data_lookup", baseline)
    detector = DriftDetector(fingerprint)

    drifted_run = _synthetic_run(
        "data_lookup",
        tool_names=["search_customer", "get_customer_record", "send_email"],
        total_cost_usd=0.0004,
        llm_calls=3,
    )
    result = detector.check(drifted_run)

    assert result.is_drift is True
    assert "send_email" in result.new_tools
    assert result.reason is not None
    assert "send_email" in result.reason
    assert str(fingerprint.num_baseline_runs) in result.reason


def test_alert_on_cost_spike_via_z_score() -> None:
    baseline = _stable_baseline("data_lookup")
    fingerprint = build_fingerprint("data_lookup", baseline)
    detector = DriftDetector(fingerprint)

    # Same tools, but cost is wildly higher than baseline -- e.g. 50x.
    expensive_run = _synthetic_run(
        "data_lookup",
        tool_names=["search_customer", "get_customer_record"],
        total_cost_usd=0.02,
        llm_calls=3,
    )
    result = detector.check(expensive_run)

    assert result.is_drift is True
    assert abs(result.z_scores["cost_usd"]) > 3.0
    assert "cost_usd" in result.reason


def test_zero_variance_baseline_does_not_crash_on_identical_run() -> None:
    """All baseline runs have identical cost (zero std); DRIFT_STD_FLOOR must
    prevent a division by zero, and an identical new run must not falsely alert."""
    baseline = [
        _synthetic_run("summarizer", tool_names=["summarize"], total_cost_usd=0.0001, llm_calls=2)
        for _ in range(5)
    ]
    fingerprint = build_fingerprint("summarizer", baseline)
    assert fingerprint.metric_stds["cost_usd"] == 0.0

    detector = DriftDetector(fingerprint)
    identical_run = _synthetic_run(
        "summarizer", tool_names=["summarize"], total_cost_usd=0.0001, llm_calls=2
    )
    result = detector.check(identical_run)

    assert result.is_drift is False


def _persist_via_store(store: TraceStore, run: TraceRun) -> TraceRun:
    """Re-persist a synthetic run's events through the real TraceStore API, so
    check_and_record's TraceEvent emission round-trips through disk realistically."""
    persisted = TraceRun.start(agent_name=run.agent_name, seed=run.seed, scenario="test")
    store.create_run(persisted)
    for e in run.events:
        store.append_event(persisted, e)
    return persisted


def test_check_and_record_appends_drift_alert_only_when_flagged(tmp_path: Path) -> None:
    baseline = _stable_baseline("data_lookup")
    fingerprint = build_fingerprint("data_lookup", baseline)
    detector = DriftDetector(fingerprint)

    stable_store = TraceStore(base_dir=tmp_path / "stable")
    stable_run = _persist_via_store(
        stable_store,
        _synthetic_run(
            "data_lookup",
            tool_names=["search_customer", "get_customer_record"],
            total_cost_usd=0.0004,
            llm_calls=3,
        ),
    )
    stable_result = detector.check_and_record(stable_store, stable_run)
    stable_store.finish_run(stable_run, outcome="success")

    assert stable_result.is_drift is False
    loaded_stable = stable_store.load_run(stable_run.run_id)
    assert loaded_stable.events_of_type(EventType.DRIFT_ALERT) == []

    drifted_store = TraceStore(base_dir=tmp_path / "drifted")
    drifted_run = _persist_via_store(
        drifted_store,
        _synthetic_run(
            "data_lookup",
            tool_names=["search_customer", "get_customer_record", "send_email"],
            total_cost_usd=0.0004,
            llm_calls=3,
        ),
    )
    drifted_result = detector.check_and_record(drifted_store, drifted_run)
    drifted_store.finish_run(drifted_run, outcome="success")

    assert drifted_result.is_drift is True
    loaded_drifted = drifted_store.load_run(drifted_run.run_id)
    alerts = loaded_drifted.events_of_type(EventType.DRIFT_ALERT)
    assert len(alerts) == 1
    assert "send_email" in alerts[0].payload["reason"]
    assert alerts[0].payload["distance"] == drifted_result.distance
