"""Tests for audit report aggregation and the Governance Readiness Score rubric."""

from __future__ import annotations

from pathlib import Path

import config
from witness.audit.report import build_audit_report, compute_readiness_score, save_report
from witness.core.trace import (
    EventType,
    TraceEvent,
    TraceRun,
    drift_alert_payload,
    grounding_result_payload,
    policy_violation_payload,
    tool_call_payload,
)


def test_compute_readiness_score_clean_is_100() -> None:
    score = compute_readiness_score(ungrounded=0, contradicted=0, policy_violations=0, drift_alerts=0)
    assert score == 100


def test_compute_readiness_score_deducts_per_class() -> None:
    score = compute_readiness_score(ungrounded=1, contradicted=1, policy_violations=1, drift_alerts=1)
    expected = 100 - sum(config.SCORE_DEDUCTIONS.values())
    assert score == expected


def test_compute_readiness_score_floors_at_zero() -> None:
    score = compute_readiness_score(ungrounded=100, contradicted=0, policy_violations=0, drift_alerts=0)
    assert score == 0


def _run_with(agent_name: str, *, violations=(), grounding=(), drift=()) -> TraceRun:
    run = TraceRun.start(agent_name=agent_name, seed=1)
    for rule_name in violations:
        run.add_event(
            TraceEvent.new(
                run_id=run.run_id,
                agent_name=agent_name,
                event_type=EventType.POLICY_VIOLATION,
                payload=policy_violation_payload(
                    rule_name=rule_name, severity="high", description="x", offending_event_id=None
                ),
            )
        )
    for classification in grounding:
        run.add_event(
            TraceEvent.new(
                run_id=run.run_id,
                agent_name=agent_name,
                event_type=EventType.GROUNDING_RESULT,
                payload=grounding_result_payload(
                    claim_text="Filed ticket #1 for: x.",
                    claim_type="ticket_filed",
                    classification=classification,
                    trace_evidence=None,
                    system_evidence=None,
                    evidence_gap="no evidence" if classification != "GROUNDED" else None,
                ),
            )
        )
    for _ in drift:
        run.add_event(
            TraceEvent.new(
                run_id=run.run_id,
                agent_name=agent_name,
                event_type=EventType.DRIFT_ALERT,
                payload=drift_alert_payload(
                    agent_name=agent_name, distance=0.5, reason="new tool used", details={}
                ),
            )
        )
    run.finish(outcome="success")
    return run


def test_build_audit_report_clean_fleet_scores_100() -> None:
    runs = [_run_with("summarizer", grounding=["GROUNDED"]), _run_with("data_lookup", grounding=["GROUNDED"])]
    report = build_audit_report(runs)

    assert report.readiness_score == 100
    assert report.total_runs == 2
    assert len(report.agent_summaries) == 2
    for agent in report.agent_summaries:
        assert agent.readiness_score == 100
        assert agent.policy_violations == []
        assert agent.claim_issues == []


def test_build_audit_report_aggregates_violations_and_claim_issues() -> None:
    runs = [
        _run_with(
            "ticket_filer",
            violations=["approval_gate"],
            grounding=["UNGROUNDED"],
        ),
        _run_with("ticket_filer", grounding=["GROUNDED", "GROUNDED"]),
        _run_with("data_lookup", drift=[None]),
    ]
    report = build_audit_report(runs)

    ticket_filer = next(a for a in report.agent_summaries if a.agent_name == "ticket_filer")
    data_lookup = next(a for a in report.agent_summaries if a.agent_name == "data_lookup")

    assert ticket_filer.run_count == 2
    assert ticket_filer.grounded_claim_count == 2
    assert len(ticket_filer.claim_issues) == 1
    assert ticket_filer.claim_issues[0].classification == "UNGROUNDED"
    assert len(ticket_filer.policy_violations) == 1
    assert ticket_filer.policy_violations[0].rule_name == "approval_gate"
    expected_ticket_filer_score = 100 - config.SCORE_DEDUCTIONS["ungrounded_claim"] - config.SCORE_DEDUCTIONS["policy_violation"]
    assert ticket_filer.readiness_score == expected_ticket_filer_score

    assert data_lookup.run_count == 1
    assert len(data_lookup.drift_alerts) == 1
    assert data_lookup.readiness_score == 100 - config.SCORE_DEDUCTIONS["drift_alert"]

    # Fleet score reflects the sum of all violation classes across every agent.
    expected_fleet_score = (
        100
        - config.SCORE_DEDUCTIONS["ungrounded_claim"]
        - config.SCORE_DEDUCTIONS["policy_violation"]
        - config.SCORE_DEDUCTIONS["drift_alert"]
    )
    assert report.readiness_score == expected_fleet_score


def test_policy_violation_entry_includes_offending_event_excerpt() -> None:
    run = TraceRun.start(agent_name="report_generator", seed=1)
    tool_event = TraceEvent.new(
        run_id=run.run_id,
        agent_name="report_generator",
        event_type=EventType.TOOL_CALL,
        payload=tool_call_payload(
            tool_name="send_email", args={"to": "x@example.com"}, result={"ok": True}, ok=True
        ),
    )
    run.add_event(tool_event)
    run.add_event(
        TraceEvent.new(
            run_id=run.run_id,
            agent_name="report_generator",
            event_type=EventType.POLICY_VIOLATION,
            payload=policy_violation_payload(
                rule_name="approval_gate",
                severity="high",
                description="missing approval",
                offending_event_id=tool_event.id,
            ),
        )
    )

    report = build_audit_report([run])
    violation = report.agent_summaries[0].policy_violations[0]
    assert violation.offending_event_excerpt is not None
    assert violation.offending_event_excerpt["payload"]["tool_name"] == "send_email"


def test_to_markdown_and_to_dict_do_not_crash_and_contain_key_figures() -> None:
    runs = [_run_with("ticket_filer", violations=["pii_leak"], grounding=["CONTRADICTED"])]
    report = build_audit_report(runs)

    md = report.to_markdown()
    assert "Governance Readiness Score" in md
    assert "ticket_filer" in md
    assert "pii_leak" in md
    assert "CONTRADICTED" in md

    d = report.to_dict()
    assert d["readiness_score"] == report.readiness_score
    assert d["agent_summaries"][0]["agent_name"] == "ticket_filer"


def test_save_report_writes_both_files(tmp_path: Path) -> None:
    report = build_audit_report([_run_with("summarizer", grounding=["GROUNDED"])])
    json_path, md_path = save_report(report, directory=tmp_path)

    assert json_path.exists()
    assert md_path.exists()
    assert json_path.read_text(encoding="utf-8").strip().startswith("{")
    assert md_path.read_text(encoding="utf-8").startswith("# Witness Audit Report")


def test_empty_runs_list_yields_perfect_score_and_no_agents() -> None:
    report = build_audit_report([])
    assert report.readiness_score == 100
    assert report.total_runs == 0
    assert report.agent_summaries == []
