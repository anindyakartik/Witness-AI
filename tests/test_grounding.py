"""Tests for the GroundingChecker -- the project's core.

Proves the central claim: a truthful run yields all GROUNDED claims, and the
manufactured hallucination (a tool that reports success for a write it silently
dropped) yields an UNGROUNDED claim with the specific claim text and evidence gap
captured. Also covers CONTRADICTED (real evidence exists but disagrees with the
claim) across each claim type, since that's the classification most easily
mishandled if system state isn't treated as authoritative.
"""

from __future__ import annotations

from pathlib import Path

from witness.core.trace import EventType, TraceRun, TraceStore, tool_call_payload
from witness.core.trace import TraceEvent as _TraceEvent
from witness.governance.grounding import (
    CONTRADICTED,
    GROUNDED,
    UNGROUNDED,
    GroundingChecker,
    MockSystems,
    extract_claims,
)
from witness.mocks.database import CustomerDatabase
from witness.mocks.outbox import EmailOutbox
from witness.mocks.ticketing import TicketingSystem


def _tool_event(run: TraceRun, tool_name: str, args: dict, result: dict) -> _TraceEvent:
    ok = bool(result.get("ok", True))
    return _TraceEvent.new(
        run_id=run.run_id,
        agent_name=run.agent_name,
        event_type=EventType.TOOL_CALL,
        payload=tool_call_payload(tool_name=tool_name, args=args, result=result, ok=ok),
    )


def _checker() -> tuple[GroundingChecker, TicketingSystem, CustomerDatabase, EmailOutbox]:
    ticketing = TicketingSystem(start_id=4470, degraded=False)
    database = CustomerDatabase()
    outbox = EmailOutbox()
    checker = GroundingChecker(MockSystems(ticketing=ticketing, database=database, outbox=outbox))
    return checker, ticketing, database, outbox


# --------------------------------------------------------------------------------
# Claim extraction
# --------------------------------------------------------------------------------


def test_extract_claims_returns_empty_for_no_message() -> None:
    assert extract_claims(None) == []
    assert extract_claims("") == []


def test_extract_claims_parses_ticket_filed() -> None:
    claims = extract_claims("Filed ticket #4470 for: Billing issue.")
    assert len(claims) == 1
    assert claims[0].claim_type == "ticket_filed"
    assert claims[0].fields == {"ticket_id": 4470}


def test_extract_claims_parses_multiple_types_from_one_message() -> None:
    # Contrived, but proves extraction doesn't stop after the first match type.
    claims = extract_claims("Filed ticket #4470 for: x. Emailed the report to a@example.com.")
    types = {c.claim_type for c in claims}
    assert types == {"ticket_filed", "email_sent"}


# --------------------------------------------------------------------------------
# Ticket filed: GROUNDED / UNGROUNDED (the manufactured hallucination) / CONTRADICTED
# --------------------------------------------------------------------------------


def test_truthful_ticket_run_is_grounded() -> None:
    checker, ticketing, _db, _outbox = _checker()
    run = TraceRun.start(agent_name="ticket_filer", seed=1)
    real_result = ticketing.create_ticket("Billing issue", "Customer overcharged.")
    run.add_event(_tool_event(run, "create_ticket", {"subject": "Billing issue", "body": "..."}, real_result))

    results = checker.check_run(run, f"Filed ticket #{real_result['ticket_id']} for: Billing issue.")

    assert len(results) == 1
    assert results[0].classification == GROUNDED
    assert results[0].evidence_gap is None


def test_hallucination_is_caught_as_ungrounded_with_evidence_gap() -> None:
    """The manufactured hallucination: degraded ticketing reports success for a
    ticket it never persists. The agent's claim, built honestly from what the tool
    told it, must still come back UNGROUNDED."""
    ticketing = TicketingSystem(start_id=4470, degraded=True)
    checker = GroundingChecker(
        MockSystems(ticketing=ticketing, database=CustomerDatabase(), outbox=EmailOutbox())
    )
    run = TraceRun.start(agent_name="ticket_filer", seed=1)
    fake_result = ticketing.create_ticket("Password reset", "User locked out.")
    assert fake_result["ok"] is True  # the tool lied about success
    assert not ticketing.exists(fake_result["ticket_id"])  # ...and never persisted it

    run.add_event(
        _tool_event(run, "create_ticket", {"subject": "Password reset", "body": "..."}, fake_result)
    )
    claim_text = f"Filed ticket #{fake_result['ticket_id']} for: Password reset."

    results = checker.check_run(run, claim_text)

    assert len(results) == 1
    assert results[0].classification == UNGROUNDED
    assert results[0].claim.claim_text == claim_text
    assert results[0].claim.fields == {"ticket_id": fake_result["ticket_id"]}
    assert results[0].evidence_gap is not None
    assert str(fake_result["ticket_id"]) in results[0].evidence_gap


def test_ticket_id_mismatch_is_contradicted_not_ungrounded() -> None:
    """A real ticket was filed, but the agent misreports its id. Since real
    evidence for *a* ticket from this run exists, this must be CONTRADICTED, not
    UNGROUNDED -- the distinction the brief draws between 'no evidence' and
    'evidence disagrees.'"""
    checker, ticketing, _db, _outbox = _checker()
    run = TraceRun.start(agent_name="ticket_filer", seed=1)
    real_result = ticketing.create_ticket("Billing issue", "...")  # really creates #4470
    run.add_event(_tool_event(run, "create_ticket", {"subject": "Billing issue", "body": "..."}, real_result))

    misreported_id = real_result["ticket_id"] + 1
    results = checker.check_run(run, f"Filed ticket #{misreported_id} for: Billing issue.")

    assert results[0].classification == CONTRADICTED
    assert str(real_result["ticket_id"]) in results[0].evidence_gap


def test_ticket_claim_with_no_matching_call_at_all_is_ungrounded() -> None:
    checker, _ticketing, _db, _outbox = _checker()
    run = TraceRun.start(agent_name="ticket_filer", seed=1)
    # No tool calls at all in this run.
    results = checker.check_run(run, "Filed ticket #9999 for: Made up issue.")
    assert results[0].classification == UNGROUNDED


# --------------------------------------------------------------------------------
# Email sent
# --------------------------------------------------------------------------------


def test_truthful_email_run_is_grounded() -> None:
    checker, _ticketing, _db, outbox = _checker()
    run = TraceRun.start(agent_name="report_generator", seed=1)
    result = outbox.send_email("ravi.shah@example.com", "Your Report", "Summary attached.")
    run.add_event(
        _tool_event(
            run,
            "send_email",
            {"to": "ravi.shah@example.com", "subject": "Your Report", "body": "Summary attached."},
            result,
        )
    )
    results = checker.check_run(run, "Emailed the report to ravi.shah@example.com.")
    assert results[0].classification == GROUNDED


def test_email_never_sent_is_ungrounded() -> None:
    checker, _ticketing, _db, _outbox = _checker()
    run = TraceRun.start(agent_name="report_generator", seed=1)
    results = checker.check_run(run, "Emailed the report to nobody@example.com.")
    assert results[0].classification == UNGROUNDED


def test_email_wrong_recipient_is_contradicted() -> None:
    checker, _ticketing, _db, outbox = _checker()
    run = TraceRun.start(agent_name="report_generator", seed=1)
    result = outbox.send_email("ravi.shah@example.com", "Your Report", "Summary attached.")
    run.add_event(
        _tool_event(
            run,
            "send_email",
            {"to": "ravi.shah@example.com", "subject": "Your Report", "body": "Summary attached."},
            result,
        )
    )
    results = checker.check_run(run, "Emailed the report to wrong-person@example.com.")
    assert results[0].classification == CONTRADICTED
    assert "ravi.shah@example.com" in results[0].evidence_gap


# --------------------------------------------------------------------------------
# Customer lookups
# --------------------------------------------------------------------------------


def test_customers_found_count_grounded() -> None:
    checker, _ticketing, db, _outbox = _checker()
    run = TraceRun.start(agent_name="data_lookup", seed=1)
    live = db.search_customer("Ravi")
    run.add_event(_tool_event(run, "search_customer", {"query": "Ravi"}, live))
    results = checker.check_run(run, "Found 1 matching customers: Ravi Shah.")
    assert results[0].classification == GROUNDED


def test_customers_found_wrong_count_is_contradicted() -> None:
    checker, _ticketing, db, _outbox = _checker()
    run = TraceRun.start(agent_name="data_lookup", seed=1)
    live = db.search_customer("Ravi")
    run.add_event(_tool_event(run, "search_customer", {"query": "Ravi"}, live))
    results = checker.check_run(run, "Found 3 matching customers: Ravi Shah.")
    assert results[0].classification == CONTRADICTED


def test_customer_by_id_grounded() -> None:
    checker, _ticketing, db, _outbox = _checker()
    run = TraceRun.start(agent_name="data_lookup", seed=1)
    live = db.get_customer_record(1)
    run.add_event(_tool_event(run, "get_customer_record", {"customer_id": 1}, live))
    results = checker.check_run(run, "Customer 1 is Ravi Shah.")
    assert results[0].classification == GROUNDED


def test_customer_by_id_wrong_name_is_contradicted() -> None:
    checker, _ticketing, db, _outbox = _checker()
    run = TraceRun.start(agent_name="data_lookup", seed=1)
    live = db.get_customer_record(1)
    run.add_event(_tool_event(run, "get_customer_record", {"customer_id": 1}, live))
    results = checker.check_run(run, "Customer 1 is Someone Else.")
    assert results[0].classification == CONTRADICTED


def test_customer_by_id_nonexistent_is_ungrounded() -> None:
    checker, _ticketing, db, _outbox = _checker()
    run = TraceRun.start(agent_name="data_lookup", seed=1)
    live = db.get_customer_record(999)
    run.add_event(_tool_event(run, "get_customer_record", {"customer_id": 999}, live))
    results = checker.check_run(run, "Customer 999 is Nobody Real.")
    assert results[0].classification == UNGROUNDED


def test_customers_found_with_no_lookup_call_is_ungrounded() -> None:
    checker, *_ = _checker()
    run = TraceRun.start(agent_name="data_lookup", seed=1)
    results = checker.check_run(run, "Found 5 matching customers: Nobody, Nowhere.")
    assert results[0].classification == UNGROUNDED


# --------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------


def test_summary_matching_tool_output_is_grounded() -> None:
    checker, *_ = _checker()
    run = TraceRun.start(agent_name="summarizer", seed=1)
    real_result = {"ok": True, "summary": "Revenue increased.", "original_sentence_count": 1}
    run.add_event(_tool_event(run, "summarize", {"text": "Revenue increased."}, real_result))
    results = checker.check_run(run, "Summary: Revenue increased.")
    assert results[0].classification == GROUNDED


def test_summary_not_matching_tool_output_is_contradicted() -> None:
    checker, *_ = _checker()
    run = TraceRun.start(agent_name="summarizer", seed=1)
    real_result = {"ok": True, "summary": "Revenue increased.", "original_sentence_count": 1}
    run.add_event(_tool_event(run, "summarize", {"text": "Revenue increased."}, real_result))
    results = checker.check_run(run, "Summary: Revenue skyrocketed and profits tripled.")
    assert results[0].classification == CONTRADICTED


# --------------------------------------------------------------------------------
# check_and_record: TraceEvent emission and linkage
# --------------------------------------------------------------------------------


def test_check_and_record_emits_linked_claim_and_grounding_events(tmp_path: Path) -> None:
    ticketing = TicketingSystem(start_id=4470, degraded=True)
    checker = GroundingChecker(
        MockSystems(ticketing=ticketing, database=CustomerDatabase(), outbox=EmailOutbox())
    )
    store = TraceStore(base_dir=tmp_path)
    run = TraceRun.start(agent_name="ticket_filer", seed=1)
    store.create_run(run)

    fake_result = ticketing.create_ticket("Password reset", "...")
    store.append_event(
        run, _tool_event(run, "create_ticket", {"subject": "Password reset", "body": "..."}, fake_result)
    )

    results = checker.check_and_record(
        store, run, f"Filed ticket #{fake_result['ticket_id']} for: Password reset."
    )
    store.finish_run(run, outcome="success")

    assert len(results) == 1
    assert results[0].classification == UNGROUNDED

    loaded = store.load_run(run.run_id)
    claim_events = loaded.events_of_type(EventType.CLAIM)
    grounding_events = loaded.events_of_type(EventType.GROUNDING_RESULT)
    assert len(claim_events) == 1
    assert len(grounding_events) == 1
    assert grounding_events[0].parent_id == claim_events[0].id
    assert grounding_events[0].payload["classification"] == UNGROUNDED
    assert grounding_events[0].payload["evidence_gap"] is not None


def test_check_and_record_no_claims_emits_no_events(tmp_path: Path) -> None:
    checker, *_ = _checker()
    store = TraceStore(base_dir=tmp_path)
    run = TraceRun.start(agent_name="ticket_filer", seed=1)
    store.create_run(run)

    results = checker.check_and_record(store, run, "I could not complete this task.")
    store.finish_run(run, outcome="success")

    assert results == []
    loaded = store.load_run(run.run_id)
    assert loaded.events_of_type(EventType.CLAIM) == []
    assert loaded.events_of_type(EventType.GROUNDING_RESULT) == []
