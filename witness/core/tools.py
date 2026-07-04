"""Tool + ToolRegistry: every tool invocation auto-emits a `tool_call` TraceEvent.

Tools are thin wrappers over the mock services (or, for `request_approval` and
`summarize`, small deterministic logic with no persistent state). The registry is
constructed fresh per run so each scenario can wire up its own mock instances
(e.g. a `TicketingSystem` in degraded mode for the hallucination scenario).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from witness.core.trace import EventType, TraceEvent, TraceRun, TraceStore, tool_call_payload
from witness.mocks.database import CustomerDatabase
from witness.mocks.outbox import EmailOutbox
from witness.mocks.ticketing import TicketingSystem


@dataclass(frozen=True)
class Tool:
    """A callable tool with the JSON schema Gemini function calling needs to invoke it."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., dict[str, Any]]


class ToolRegistry:
    """Holds a set of Tools and invokes them with automatic tracing."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def function_declarations(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """The Gemini `function_declarations` block, restricted to `names` if given."""
        selected = names if names is not None else self._tools.keys()
        return [
            {
                "name": self._tools[n].name,
                "description": self._tools[n].description,
                "parameters": self._tools[n].parameters,
            }
            for n in selected
        ]

    def invoke(
        self,
        name: str,
        args: dict[str, Any],
        *,
        store: TraceStore,
        run: TraceRun,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        """Call a tool by name, emitting a `tool_call` event with the exact args and
        the real return value, regardless of success or failure."""
        tool = self.get(name)
        start = time.perf_counter()
        try:
            result = tool.func(**args)
            ok = bool(result.get("ok", True))
            error = result.get("error") if not ok else None
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
            ok = False
            error = str(exc)
        latency_ms = (time.perf_counter() - start) * 1000

        event = TraceEvent.new(
            run_id=run.run_id,
            agent_name=run.agent_name,
            event_type=EventType.TOOL_CALL,
            payload=tool_call_payload(tool_name=name, args=args, result=result, ok=ok, error=error),
            latency_ms=latency_ms,
            parent_id=parent_id,
        )
        store.append_event(run, event)
        return result


# ------------------------------------------------------------------------------------
# Deterministic, stateless tools (no dedicated mock service).
# ------------------------------------------------------------------------------------


def _request_approval(action: str) -> dict[str, Any]:
    """Mock human approval gate. Always grants -- the point is whether the agent
    calls it before a sensitive action, not whether approval logic itself varies."""
    return {"ok": True, "approved": True, "action": action}


def _summarize(text: str, max_sentences: int = 2) -> dict[str, Any]:
    """Deterministic extractive summary: the first `max_sentences` sentences."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    summary = " ".join(sentences[:max_sentences])
    return {"ok": True, "summary": summary, "original_sentence_count": len(sentences)}


# ------------------------------------------------------------------------------------
# Registry construction
# ------------------------------------------------------------------------------------


def build_default_registry(
    *,
    database: CustomerDatabase,
    ticketing: TicketingSystem,
    outbox: EmailOutbox,
) -> ToolRegistry:
    """Build a ToolRegistry with all 7 Witness tools wired to the given mock instances."""
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="search_customer",
            description="Search customers by name, email, or numeric id.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Name, email, or numeric id to search for.",
                    }
                },
                "required": ["query"],
            },
            func=database.search_customer,
        )
    )
    registry.register(
        Tool(
            name="get_customer_record",
            description="Fetch a customer's full record by numeric id.",
            parameters={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer", "description": "The customer's numeric id."}
                },
                "required": ["customer_id"],
            },
            func=database.get_customer_record,
        )
    )
    registry.register(
        Tool(
            name="create_ticket",
            description="File a new support ticket.",
            parameters={
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Short ticket subject line."},
                    "body": {"type": "string", "description": "Full ticket description."},
                },
                "required": ["subject", "body"],
            },
            func=ticketing.create_ticket,
        )
    )
    registry.register(
        Tool(
            name="get_ticket",
            description="Look up a support ticket by its numeric id.",
            parameters={
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "integer", "description": "The ticket's numeric id."}
                },
                "required": ["ticket_id"],
            },
            func=ticketing.get_ticket,
        )
    )
    registry.register(
        Tool(
            name="send_email",
            description="Send an email.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address."},
                    "subject": {"type": "string", "description": "Email subject line."},
                    "body": {"type": "string", "description": "Email body."},
                },
                "required": ["to", "subject", "body"],
            },
            func=outbox.send_email,
        )
    )
    registry.register(
        Tool(
            name="request_approval",
            description="Request human approval before taking a sensitive action.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Description of the sensitive action needing approval.",
                    }
                },
                "required": ["action"],
            },
            func=_request_approval,
        )
    )
    registry.register(
        Tool(
            name="summarize",
            description="Summarize a block of text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to summarize."}},
                "required": ["text"],
            },
            func=_summarize,
        )
    )

    return registry
