"""Deterministic, in-memory support ticket system: ground truth for grounding checks.

Ticket ids are assigned like a real autoincrement primary key, and existence can be
queried directly (`exists`, `get_ticket`) -- this is what the GroundingChecker
compares agent claims against, independent of anything the agent believes happened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Ticket:
    id: int
    subject: str
    body: str
    status: str
    created_at: str


class TicketingSystem:
    """Deterministic mock ticketing system with an optional degraded mode.

    In `degraded` mode, `create_ticket` allocates an id and returns a plausible
    success payload for it, but never persists the ticket -- simulating a backend
    that acknowledges a write it silently dropped. The id counter still advances
    (as a real autoincrement PK would even on a rolled-back insert), so the id in
    the fake acknowledgment looks exactly like a real one. This is how a genuine,
    reproducible agent hallucination is manufactured: the agent is truthfully told
    (by the tool) that ticket N was created, but ticket N never exists in the system.
    """

    def __init__(self, *, start_id: int = 4470, degraded: bool = False) -> None:
        self._next_id = start_id
        self._tickets: dict[int, Ticket] = {}
        self.degraded = degraded

    def create_ticket(self, subject: str, body: str) -> dict[str, Any]:
        ticket_id = self._next_id
        self._next_id += 1

        if self.degraded:
            return {"ok": True, "ticket_id": ticket_id, "status": "queued"}

        ticket = Ticket(
            id=ticket_id,
            subject=subject,
            body=body,
            status="open",
            created_at=datetime.now(UTC).isoformat(),
        )
        self._tickets[ticket_id] = ticket
        return {"ok": True, "ticket_id": ticket_id, "status": ticket.status}

    def get_ticket(self, ticket_id: int) -> dict[str, Any]:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            return {"ok": False, "error": f"no ticket with id {ticket_id}"}
        return {"ok": True, **asdict(ticket)}

    def exists(self, ticket_id: int) -> bool:
        return ticket_id in self._tickets

    def all_tickets(self) -> list[Ticket]:
        return list(self._tickets.values())
