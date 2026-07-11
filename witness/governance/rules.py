"""Policy rules: PII leaks, missing approvals, cost overruns, and allowlist breaches.

Each rule inspects one event plus the events that preceded it in the same run and
returns a Violation or None. Rules are pure and stateless between calls, all
context they need is passed in, so PolicyEngine can evaluate them in a single
sequential pass over a run's ordered trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import config
from witness.core.trace import EventType, TraceEvent, TraceRun


@dataclass(frozen=True)
class Violation:
    """One policy rule firing on one event."""

    rule_name: str
    severity: str  # "low" | "medium" | "high"
    description: str
    offending_event_id: str | None


class Rule(Protocol):
    """A policy rule: inspects an event in the context of its run and the events
    that preceded it, and optionally reports a violation."""

    name: str

    def check(
        self, event: TraceEvent, prior_events: list[TraceEvent], run: TraceRun
    ) -> Violation | None: ...


class PIILeakRule:
    """Flags SSN- or credit-card-like patterns in an outbound `send_email`'s
    subject or body.

    Email and phone patterns are detected by the same scanner but not treated as
    violations on their own: a recipient's own contact info legitimately appears
    in messages addressed to them, so flagging it would be mostly false positives.
    SSNs and card numbers should never appear in an outbound email regardless of
    the recipient, so those are hard violations.
    """

    name = "pii_leak"

    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){16}\b")

    def check(
        self, event: TraceEvent, prior_events: list[TraceEvent], run: TraceRun
    ) -> Violation | None:
        if event.event_type is not EventType.TOOL_CALL:
            return None
        if event.payload.get("tool_name") != "send_email":
            return None

        args = event.payload.get("args", {})
        text = f"{args.get('subject', '')} {args.get('body', '')}"

        if self.SSN_PATTERN.search(text):
            return Violation(
                rule_name=self.name,
                severity="high",
                description="Outbound email contains what looks like a Social Security Number.",
                offending_event_id=event.id,
            )
        if self.CREDIT_CARD_PATTERN.search(text):
            return Violation(
                rule_name=self.name,
                severity="high",
                description="Outbound email contains what looks like a credit card number.",
                offending_event_id=event.id,
            )
        return None


class ApprovalGateRule:
    """Requires a successful `request_approval` tool call earlier in the same run
    before any sensitive tool executes. Currently `send_email` is the only
    sensitive tool; a failed send is not itself flagged as an approval breach."""

    name = "approval_gate"
    SENSITIVE_TOOLS = frozenset({"send_email"})

    def check(
        self, event: TraceEvent, prior_events: list[TraceEvent], run: TraceRun
    ) -> Violation | None:
        if event.event_type is not EventType.TOOL_CALL:
            return None
        tool_name = event.payload.get("tool_name")
        if tool_name not in self.SENSITIVE_TOOLS:
            return None
        if not event.payload.get("ok", False):
            return None

        approved = any(
            e.event_type is EventType.TOOL_CALL
            and e.payload.get("tool_name") == "request_approval"
            and e.payload.get("ok") is True
            for e in prior_events
        )
        if approved:
            return None
        return Violation(
            rule_name=self.name,
            severity="high",
            description=f"'{tool_name}' executed without a preceding approved request_approval call.",
            offending_event_id=event.id,
        )


class CostCapRule:
    """Flags a run's cumulative cost exceeding config.COST_CAP_USD. Fires exactly
    once, on the event that crosses the threshold."""

    name = "cost_cap"

    def check(
        self, event: TraceEvent, prior_events: list[TraceEvent], run: TraceRun
    ) -> Violation | None:
        prior_cost = sum(e.cost_usd for e in prior_events)
        cumulative = prior_cost + event.cost_usd
        if cumulative > config.COST_CAP_USD and prior_cost <= config.COST_CAP_USD:
            return Violation(
                rule_name=self.name,
                severity="medium",
                description=(
                    f"Cumulative run cost ${cumulative:.4f} exceeded cap "
                    f"${config.COST_CAP_USD:.4f}."
                ),
                offending_event_id=event.id,
            )
        return None


class ToolAllowlistRule:
    """Flags any tool call outside the agent's canonical declared allowlist,
    regardless of what a given run happened to make callable (see
    AgentDefinition.tool_allowlist and runtime.run_agent's `tool_allowlist_override`,
    which exists precisely to let a scenario diverge from this canonical baseline)."""

    name = "tool_allowlist"

    def __init__(self, allowlists: dict[str, tuple[str, ...]]) -> None:
        self.allowlists = allowlists

    def check(
        self, event: TraceEvent, prior_events: list[TraceEvent], run: TraceRun
    ) -> Violation | None:
        if event.event_type is not EventType.TOOL_CALL:
            return None
        allowlist = self.allowlists.get(run.agent_name)
        if allowlist is None:
            return None
        tool_name = event.payload.get("tool_name")
        if tool_name in allowlist:
            return None
        return Violation(
            rule_name=self.name,
            severity="medium",
            description=(
                f"Agent '{run.agent_name}' called '{tool_name}', outside its declared "
                f"allowlist {allowlist}."
            ),
            offending_event_id=event.id,
        )
