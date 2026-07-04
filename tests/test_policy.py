"""Tests for the policy engine and its rules: each rule fires on a crafted
violating trace and stays silent on a clean one (false-positive checks included)."""

from __future__ import annotations

from pathlib import Path

import config
from witness.core.trace import (
    EventType,
    TraceEvent,
    TraceRun,
    TraceStore,
    tool_call_payload,
)
from witness.governance.policy import PolicyEngine, build_default_policy_engine
from witness.governance.rules import (
    ApprovalGateRule,
    CostCapRule,
    PIILeakRule,
    ToolAllowlistRule,
)


def _tool_event(run: TraceRun, tool_name: str, args: dict, *, ok: bool = True, cost_usd: float = 0.0) -> TraceEvent:
    return TraceEvent.new(
        run_id=run.run_id,
        agent_name=run.agent_name,
        event_type=EventType.TOOL_CALL,
        payload=tool_call_payload(tool_name=tool_name, args=args, result={"ok": ok}, ok=ok),
        cost_usd=cost_usd,
    )


# --------------------------------------------------------------------------------
# PIILeakRule
# --------------------------------------------------------------------------------


def test_pii_leak_rule_fires_on_ssn_in_email_body() -> None:
    run = TraceRun.start(agent_name="report_generator", seed=1)
    event = _tool_event(
        run,
        "send_email",
        {"to": "x@example.com", "subject": "Your info", "body": "Your SSN is 123-45-6789."},
    )
    violation = PIILeakRule().check(event, [], run)
    assert violation is not None
    assert violation.rule_name == "pii_leak"
    assert violation.offending_event_id == event.id


def test_pii_leak_rule_fires_on_credit_card_in_email_body() -> None:
    run = TraceRun.start(agent_name="report_generator", seed=1)
    event = _tool_event(
        run,
        "send_email",
        {"to": "x@example.com", "subject": "Receipt", "body": "Card charged: 4111 1111 1111 1111"},
    )
    violation = PIILeakRule().check(event, [], run)
    assert violation is not None


def test_pii_leak_rule_silent_on_clean_email() -> None:
    run = TraceRun.start(agent_name="report_generator", seed=1)
    event = _tool_event(
        run,
        "send_email",
        {"to": "ravi.shah@example.com", "subject": "Your Account Report", "body": "Here is your summary."},
    )
    assert PIILeakRule().check(event, [], run) is None


def test_pii_leak_rule_silent_on_recipient_email_address_in_body() -> None:
    """A recipient's own email/phone appearing in their message is not a leak."""
    run = TraceRun.start(agent_name="report_generator", seed=1)
    event = _tool_event(
        run,
        "send_email",
        {
            "to": "ravi.shah@example.com",
            "subject": "Confirmation",
            "body": "We've sent this confirmation to ravi.shah@example.com and 555-0101.",
        },
    )
    assert PIILeakRule().check(event, [], run) is None


def test_pii_leak_rule_ignores_non_send_email_tools() -> None:
    run = TraceRun.start(agent_name="report_generator", seed=1)
    event = _tool_event(run, "get_customer_record", {"customer_id": 1})
    assert PIILeakRule().check(event, [], run) is None


# --------------------------------------------------------------------------------
# ApprovalGateRule
# --------------------------------------------------------------------------------


def test_approval_gate_rule_fires_without_prior_approval() -> None:
    run = TraceRun.start(agent_name="report_generator", seed=1)
    send_event = _tool_event(run, "send_email", {"to": "x@example.com", "subject": "s", "body": "b"})
    violation = ApprovalGateRule().check(send_event, [], run)
    assert violation is not None
    assert violation.rule_name == "approval_gate"


def test_approval_gate_rule_silent_when_approval_precedes() -> None:
    run = TraceRun.start(agent_name="report_generator", seed=1)
    approval_event = _tool_event(run, "request_approval", {"action": "send email"})
    send_event = _tool_event(run, "send_email", {"to": "x@example.com", "subject": "s", "body": "b"})
    assert ApprovalGateRule().check(send_event, [approval_event], run) is None


def test_approval_gate_rule_silent_on_failed_approval_followed_by_no_send() -> None:
    run = TraceRun.start(agent_name="report_generator", seed=1)
    failed_approval = _tool_event(run, "request_approval", {"action": "send email"}, ok=False)
    send_event = _tool_event(run, "send_email", {"to": "x@example.com", "subject": "s", "body": "b"})
    # A failed approval should NOT count as satisfying the gate.
    violation = ApprovalGateRule().check(send_event, [failed_approval], run)
    assert violation is not None


def test_approval_gate_rule_silent_on_failed_send() -> None:
    run = TraceRun.start(agent_name="report_generator", seed=1)
    send_event = _tool_event(run, "send_email", {"to": "x@example.com", "subject": "s", "body": "b"}, ok=False)
    assert ApprovalGateRule().check(send_event, [], run) is None


def test_approval_gate_rule_ignores_non_sensitive_tools() -> None:
    run = TraceRun.start(agent_name="data_lookup", seed=1)
    event = _tool_event(run, "search_customer", {"query": "Ravi"})
    assert ApprovalGateRule().check(event, [], run) is None


# --------------------------------------------------------------------------------
# CostCapRule
# --------------------------------------------------------------------------------


def test_cost_cap_rule_fires_once_when_exceeded() -> None:
    run = TraceRun.start(agent_name="ticket_filer", seed=1)
    under_cap = _tool_event(run, "create_ticket", {}, cost_usd=config.COST_CAP_USD * 0.5)
    over_cap = _tool_event(run, "create_ticket", {}, cost_usd=config.COST_CAP_USD * 0.6)

    # First event alone: under the cap, no violation.
    assert CostCapRule().check(under_cap, [], run) is None

    # Second event pushes cumulative cost over the cap: violation fires here.
    violation = CostCapRule().check(over_cap, [under_cap], run)
    assert violation is not None
    assert violation.rule_name == "cost_cap"


def test_cost_cap_rule_does_not_refire_once_already_over() -> None:
    run = TraceRun.start(agent_name="ticket_filer", seed=1)
    already_over = _tool_event(run, "create_ticket", {}, cost_usd=config.COST_CAP_USD * 1.5)
    another_event = _tool_event(run, "create_ticket", {}, cost_usd=0.0001)

    # prior_events already exceed the cap -> this rule should not fire again.
    violation = CostCapRule().check(another_event, [already_over], run)
    assert violation is None


def test_cost_cap_rule_silent_under_cap() -> None:
    run = TraceRun.start(agent_name="ticket_filer", seed=1)
    event = _tool_event(run, "create_ticket", {}, cost_usd=config.COST_CAP_USD * 0.1)
    assert CostCapRule().check(event, [], run) is None


# --------------------------------------------------------------------------------
# ToolAllowlistRule
# --------------------------------------------------------------------------------


def test_tool_allowlist_rule_fires_on_disallowed_tool() -> None:
    rule = ToolAllowlistRule({"data_lookup": ("search_customer", "get_customer_record")})
    run = TraceRun.start(agent_name="data_lookup", seed=1)
    event = _tool_event(run, "send_email", {"to": "x@example.com", "subject": "s", "body": "b"})
    violation = rule.check(event, [], run)
    assert violation is not None
    assert violation.rule_name == "tool_allowlist"


def test_tool_allowlist_rule_silent_on_allowed_tool() -> None:
    rule = ToolAllowlistRule({"data_lookup": ("search_customer", "get_customer_record")})
    run = TraceRun.start(agent_name="data_lookup", seed=1)
    event = _tool_event(run, "search_customer", {"query": "Ravi"})
    assert rule.check(event, [], run) is None


def test_tool_allowlist_rule_silent_for_unknown_agent() -> None:
    rule = ToolAllowlistRule({"data_lookup": ("search_customer",)})
    run = TraceRun.start(agent_name="some_future_agent", seed=1)
    event = _tool_event(run, "anything", {})
    assert rule.check(event, [], run) is None


# --------------------------------------------------------------------------------
# PolicyEngine integration
# --------------------------------------------------------------------------------


def test_policy_engine_silent_on_fully_clean_run(tmp_path: Path) -> None:
    store = TraceStore(base_dir=tmp_path)
    run = TraceRun.start(agent_name="report_generator", seed=1)
    store.create_run(run)
    store.append_event(run, _tool_event(run, "get_customer_record", {"customer_id": 1}))
    store.append_event(run, _tool_event(run, "request_approval", {"action": "send report"}))
    store.append_event(
        run,
        _tool_event(
            run,
            "send_email",
            {"to": "ravi.shah@example.com", "subject": "Your Report", "body": "Summary attached."},
        ),
    )
    store.finish_run(run, outcome="success")

    engine = build_default_policy_engine()
    violations = engine.evaluate_and_record(store, run)

    assert violations == []
    loaded = store.load_run(run.run_id)
    assert loaded.events_of_type(EventType.POLICY_VIOLATION) == []


def test_policy_engine_catches_pii_leak_and_missing_approval(tmp_path: Path) -> None:
    store = TraceStore(base_dir=tmp_path)
    run = TraceRun.start(agent_name="report_generator", seed=1)
    store.create_run(run)
    store.append_event(run, _tool_event(run, "get_customer_record", {"customer_id": 1}))
    # No request_approval call, and the body leaks an SSN: two independent violations.
    store.append_event(
        run,
        _tool_event(
            run,
            "send_email",
            {
                "to": "ravi.shah@example.com",
                "subject": "Your Report",
                "body": "For verification, your SSN is 123-45-6701.",
            },
        ),
    )
    store.finish_run(run, outcome="success")

    engine = build_default_policy_engine()
    violations = engine.evaluate_and_record(store, run)

    rule_names = {v.rule_name for v in violations}
    assert rule_names == {"pii_leak", "approval_gate"}

    loaded = store.load_run(run.run_id)
    recorded = loaded.events_of_type(EventType.POLICY_VIOLATION)
    assert len(recorded) == 2
    assert {e.payload["rule_name"] for e in recorded} == {"pii_leak", "approval_gate"}


def test_policy_engine_catches_tool_allowlist_violation(tmp_path: Path) -> None:
    store = TraceStore(base_dir=tmp_path)
    run = TraceRun.start(agent_name="data_lookup", seed=1)
    store.create_run(run)
    store.append_event(run, _tool_event(run, "search_customer", {"query": "Ravi"}))
    store.append_event(
        run, _tool_event(run, "send_email", {"to": "x@example.com", "subject": "s", "body": "b"})
    )
    store.finish_run(run, outcome="success")

    engine = build_default_policy_engine()
    violations = engine.evaluate_and_record(store, run)

    assert any(v.rule_name == "tool_allowlist" for v in violations)


def test_policy_engine_with_empty_rules_never_flags_anything(tmp_path: Path) -> None:
    store = TraceStore(base_dir=tmp_path)
    run = TraceRun.start(agent_name="ticket_filer", seed=1)
    store.create_run(run)
    store.append_event(
        run,
        _tool_event(
            run,
            "send_email",
            {"to": "x@example.com", "subject": "s", "body": "SSN 123-45-6789"},
        ),
    )
    store.finish_run(run, outcome="success")

    engine = PolicyEngine(rules=[])
    assert engine.evaluate(run) == []
