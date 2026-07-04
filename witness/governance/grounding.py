"""GroundingChecker: verifies an agent's claims against independent ground truth.

This is the project's core. An agent's final message contains natural-language
claims about what it did ("filed ticket #4471", "emailed the report to X"). Those
claims are extracted deterministically (rule-based regex, no LLM), then checked
against two independent sources: the execution trace (what tools were actually
called, with what results) and the true state of the mock systems acted on.

System state is authoritative. If the mock system doesn't confirm the claimed
effect exists, the claim is UNGROUNDED -- regardless of what the trace shows the
tool itself reported (a tool can report success for a write that was silently
dropped; that gap is exactly the failure mode this project exists to catch).
CONTRADICTED is reserved for the narrower case where the run demonstrably produced
some confirmed effect, just not the one claimed (e.g., ticket #4470 was really
created, but the agent's final message says #4471).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from witness.core.trace import (
    EventType,
    TraceEvent,
    TraceRun,
    TraceStore,
    claim_payload,
    grounding_result_payload,
)
from witness.mocks.database import CustomerDatabase
from witness.mocks.outbox import EmailOutbox
from witness.mocks.ticketing import TicketingSystem

GROUNDED = "GROUNDED"
UNGROUNDED = "UNGROUNDED"
CONTRADICTED = "CONTRADICTED"


@dataclass(frozen=True)
class Claim:
    """One discrete, checkable claim extracted from an agent's final message."""

    claim_type: str
    claim_text: str
    fields: dict[str, Any]


@dataclass(frozen=True)
class GroundingResult:
    """The verdict on one claim, with the evidence that produced it."""

    claim: Claim
    classification: str
    trace_evidence: dict[str, Any] | None
    system_evidence: dict[str, Any] | None
    evidence_gap: str | None


@dataclass(frozen=True)
class MockSystems:
    """The exact mock instances a run acted on -- grounding must check against
    these, not fresh ones, since they hold the run's real resulting state."""

    ticketing: TicketingSystem
    database: CustomerDatabase
    outbox: EmailOutbox


# ------------------------------------------------------------------------------------
# Claim extraction: rule-based, deterministic, tied to each agent's required final-
# message format (see witness/agents/*.py system prompts).
# ------------------------------------------------------------------------------------

_TICKET_FILED_RE = re.compile(r"filed ticket #(\d+)", re.IGNORECASE)
_EMAIL_SENT_RE = re.compile(r"emailed .*? to ([\w.+-]+@[\w.-]+\.\w+)", re.IGNORECASE)
_CUSTOMERS_FOUND_RE = re.compile(r"found (\d+) matching customers?", re.IGNORECASE)
_CUSTOMER_BY_ID_RE = re.compile(r"customer (\d+) is ([^.]+)\.", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"^summary:\s*(.+)$", re.IGNORECASE | re.DOTALL)


def extract_claims(final_message: str | None) -> list[Claim]:
    """Parse an agent's final message into zero or more discrete, checkable claims."""
    if not final_message:
        return []
    text = final_message.strip()
    claims: list[Claim] = []

    if m := _TICKET_FILED_RE.search(text):
        claims.append(Claim("ticket_filed", text, {"ticket_id": int(m.group(1))}))

    if m := _EMAIL_SENT_RE.search(text):
        claims.append(Claim("email_sent", text, {"email": m.group(1)}))

    if m := _CUSTOMERS_FOUND_RE.search(text):
        claims.append(Claim("customers_found", text, {"count": int(m.group(1))}))

    if m := _CUSTOMER_BY_ID_RE.search(text):
        claims.append(
            Claim(
                "customer_by_id", text, {"customer_id": int(m.group(1)), "name": m.group(2).strip()}
            )
        )

    if m := _SUMMARY_RE.match(text):
        claims.append(Claim("summary_given", text, {"summary": m.group(1).strip()}))

    return claims


def _tool_calls(run: TraceRun, tool_name: str) -> list[TraceEvent]:
    return [
        e
        for e in run.events
        if e.event_type is EventType.TOOL_CALL and e.payload.get("tool_name") == tool_name
    ]


class GroundingChecker:
    """Verifies a run's claims against its trace and the true state of the mock
    systems it acted on."""

    def __init__(self, mocks: MockSystems) -> None:
        self.mocks = mocks

    def check_run(self, run: TraceRun, final_message: str | None) -> list[GroundingResult]:
        """Verify all claims in `final_message` without touching the trace store."""
        return [self._verify(claim, run) for claim in extract_claims(final_message)]

    def check_and_record(
        self, store: TraceStore, run: TraceRun, final_message: str | None
    ) -> list[GroundingResult]:
        """Verify all claims, recording a `claim` event and a `grounding_result`
        event (linked via parent_id) for each."""
        results: list[GroundingResult] = []
        for claim in extract_claims(final_message):
            claim_event = TraceEvent.new(
                run_id=run.run_id,
                agent_name=run.agent_name,
                event_type=EventType.CLAIM,
                payload=claim_payload(
                    claim_text=claim.claim_text, claim_type=claim.claim_type, fields=claim.fields
                ),
            )
            store.append_event(run, claim_event)

            result = self._verify(claim, run)
            results.append(result)

            grounding_event = TraceEvent.new(
                run_id=run.run_id,
                agent_name=run.agent_name,
                event_type=EventType.GROUNDING_RESULT,
                payload=grounding_result_payload(
                    claim_text=result.claim.claim_text,
                    claim_type=result.claim.claim_type,
                    classification=result.classification,
                    trace_evidence=result.trace_evidence,
                    system_evidence=result.system_evidence,
                    evidence_gap=result.evidence_gap,
                ),
                parent_id=claim_event.id,
            )
            store.append_event(run, grounding_event)
        return results

    def _verify(self, claim: Claim, run: TraceRun) -> GroundingResult:
        if claim.claim_type == "ticket_filed":
            return self._verify_ticket_filed(claim, run)
        if claim.claim_type == "email_sent":
            return self._verify_email_sent(claim, run)
        if claim.claim_type in ("customers_found", "customer_by_id"):
            return self._verify_customer_lookup(claim, run)
        if claim.claim_type == "summary_given":
            return self._verify_summary(claim, run)
        raise ValueError(f"Unknown claim type: {claim.claim_type}")

    def _verify_ticket_filed(self, claim: Claim, run: TraceRun) -> GroundingResult:
        claimed_id = claim.fields["ticket_id"]
        ticket_calls = _tool_calls(run, "create_ticket")
        trace_evidence = {
            "create_ticket_calls": [
                {"args": e.payload["args"], "result": e.payload["result"]} for e in ticket_calls
            ]
        }

        if self.mocks.ticketing.exists(claimed_id):
            return GroundingResult(
                claim=claim,
                classification=GROUNDED,
                trace_evidence=trace_evidence,
                system_evidence={"ticket": self.mocks.ticketing.get_ticket(claimed_id)},
                evidence_gap=None,
            )

        for e in ticket_calls:
            real_id = e.payload["result"].get("ticket_id")
            if (
                real_id is not None
                and real_id != claimed_id
                and self.mocks.ticketing.exists(real_id)
            ):
                return GroundingResult(
                    claim=claim,
                    classification=CONTRADICTED,
                    trace_evidence=trace_evidence,
                    system_evidence={"ticket": self.mocks.ticketing.get_ticket(real_id)},
                    evidence_gap=(
                        f"Claimed ticket #{claimed_id}, but the ticket actually created by "
                        f"this run is #{real_id}."
                    ),
                )

        return GroundingResult(
            claim=claim,
            classification=UNGROUNDED,
            trace_evidence=trace_evidence,
            system_evidence={"ticket": self.mocks.ticketing.get_ticket(claimed_id)},
            evidence_gap=f"No ticket #{claimed_id} exists in the ticketing system.",
        )

    def _verify_email_sent(self, claim: Claim, run: TraceRun) -> GroundingResult:
        claimed_email = claim.fields["email"]
        send_calls = _tool_calls(run, "send_email")
        trace_evidence = {
            "send_email_calls": [
                {"args": e.payload["args"], "result": e.payload["result"]} for e in send_calls
            ]
        }

        if self.mocks.outbox.exists(to=claimed_email):
            sent = [asdict(s) for s in self.mocks.outbox.find(to=claimed_email)]
            return GroundingResult(
                claim=claim,
                classification=GROUNDED,
                trace_evidence=trace_evidence,
                system_evidence={"sent_emails": sent},
                evidence_gap=None,
            )

        for e in send_calls:
            real_to = e.payload["args"].get("to")
            if real_to and real_to != claimed_email and self.mocks.outbox.exists(to=real_to):
                sent = [asdict(s) for s in self.mocks.outbox.find(to=real_to)]
                return GroundingResult(
                    claim=claim,
                    classification=CONTRADICTED,
                    trace_evidence=trace_evidence,
                    system_evidence={"sent_emails": sent},
                    evidence_gap=(
                        f"Claimed email to {claimed_email}, but the email actually sent by "
                        f"this run went to {real_to}."
                    ),
                )

        return GroundingResult(
            claim=claim,
            classification=UNGROUNDED,
            trace_evidence=trace_evidence,
            system_evidence={"sent_emails": []},
            evidence_gap=f"No email to {claimed_email} exists in the outbox.",
        )

    def _verify_customer_lookup(self, claim: Claim, run: TraceRun) -> GroundingResult:
        lookup_calls = [
            e
            for e in run.events
            if e.event_type is EventType.TOOL_CALL
            and e.payload.get("tool_name") in ("search_customer", "get_customer_record")
        ]
        trace_evidence = {
            "lookup_calls": [
                {
                    "tool_name": e.payload["tool_name"],
                    "args": e.payload["args"],
                    "result": e.payload["result"],
                }
                for e in lookup_calls
            ]
        }

        if not lookup_calls:
            return GroundingResult(
                claim=claim,
                classification=UNGROUNDED,
                trace_evidence=trace_evidence,
                system_evidence=None,
                evidence_gap="No customer lookup tool call occurred in this run.",
            )

        if claim.claim_type == "customers_found":
            return self._verify_customers_found(claim, lookup_calls, trace_evidence)
        return self._verify_customer_by_id(claim, trace_evidence)

    def _verify_customers_found(
        self, claim: Claim, lookup_calls: list[TraceEvent], trace_evidence: dict[str, Any]
    ) -> GroundingResult:
        claimed_count = claim.fields["count"]
        search_calls = [e for e in lookup_calls if e.payload["tool_name"] == "search_customer"]
        if not search_calls:
            return GroundingResult(
                claim=claim,
                classification=UNGROUNDED,
                trace_evidence=trace_evidence,
                system_evidence=None,
                evidence_gap="No search_customer call to verify the claimed count against.",
            )

        # Independently re-run the same query live rather than trusting the tool's
        # own cached result -- this is the second, independent evidence source.
        query = search_calls[-1].payload["args"].get("query", "")
        live = self.mocks.database.search_customer(query)
        actual_count = len(live.get("matches", []))

        if actual_count == claimed_count:
            return GroundingResult(
                claim=claim,
                classification=GROUNDED,
                trace_evidence=trace_evidence,
                system_evidence={"actual_count": actual_count, "query": query},
                evidence_gap=None,
            )
        return GroundingResult(
            claim=claim,
            classification=CONTRADICTED,
            trace_evidence=trace_evidence,
            system_evidence={"actual_count": actual_count, "query": query},
            evidence_gap=(
                f"Claimed {claimed_count} matching customers, but the database actually "
                f"has {actual_count} for query '{query}'."
            ),
        )

    def _verify_customer_by_id(
        self, claim: Claim, trace_evidence: dict[str, Any]
    ) -> GroundingResult:
        customer_id = claim.fields["customer_id"]
        claimed_name = claim.fields["name"]
        live = self.mocks.database.get_customer_record(customer_id)

        if not live.get("ok"):
            return GroundingResult(
                claim=claim,
                classification=UNGROUNDED,
                trace_evidence=trace_evidence,
                system_evidence=live,
                evidence_gap=f"No customer with id {customer_id} exists.",
            )

        real_name = live.get("name")
        if real_name == claimed_name:
            return GroundingResult(
                claim=claim,
                classification=GROUNDED,
                trace_evidence=trace_evidence,
                system_evidence=live,
                evidence_gap=None,
            )
        return GroundingResult(
            claim=claim,
            classification=CONTRADICTED,
            trace_evidence=trace_evidence,
            system_evidence=live,
            evidence_gap=f"Claimed customer {customer_id} is '{claimed_name}', but records show '{real_name}'.",
        )

    def _verify_summary(self, claim: Claim, run: TraceRun) -> GroundingResult:
        """Summaries have no persistent mock state; the deterministic `summarize`
        tool's own recorded output is the authoritative ground truth here."""
        claimed_summary = claim.fields["summary"]
        summarize_calls = _tool_calls(run, "summarize")
        trace_evidence = {
            "summarize_calls": [
                {"args": e.payload["args"], "result": e.payload["result"]} for e in summarize_calls
            ]
        }

        if not summarize_calls:
            return GroundingResult(
                claim=claim,
                classification=UNGROUNDED,
                trace_evidence=trace_evidence,
                system_evidence=None,
                evidence_gap="No summarize tool call occurred in this run.",
            )

        real_summary = summarize_calls[-1].payload["result"].get("summary")
        if real_summary == claimed_summary:
            return GroundingResult(
                claim=claim,
                classification=GROUNDED,
                trace_evidence=trace_evidence,
                system_evidence={"summary": real_summary},
                evidence_gap=None,
            )
        return GroundingResult(
            claim=claim,
            classification=CONTRADICTED,
            trace_evidence=trace_evidence,
            system_evidence={"summary": real_summary},
            evidence_gap="Claimed summary text does not match the tool's actual output.",
        )
