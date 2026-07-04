"""Deterministic, in-memory email outbox: the real record of what was actually sent.

Grounding checks an agent's "I emailed X" claim against this outbox, not against
the agent's own say-so.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class SentEmail:
    id: int
    to: str
    subject: str
    body: str
    sent_at: str


class EmailOutbox:
    """Deterministic mock email outbox."""

    def __init__(self) -> None:
        self._next_id = 1
        self._sent: list[SentEmail] = []

    def send_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        email = SentEmail(
            id=self._next_id,
            to=to,
            subject=subject,
            body=body,
            sent_at=datetime.now(UTC).isoformat(),
        )
        self._next_id += 1
        self._sent.append(email)
        return {"ok": True, "email_id": email.id}

    def exists(self, *, to: str | None = None, subject: str | None = None) -> bool:
        return any(
            (to is None or e.to == to) and (subject is None or e.subject == subject)
            for e in self._sent
        )

    def find(self, *, to: str | None = None, subject: str | None = None) -> list[SentEmail]:
        return [
            e
            for e in self._sent
            if (to is None or e.to == to) and (subject is None or e.subject == subject)
        ]

    def all_sent(self) -> list[SentEmail]:
        return list(self._sent)
