"""Shared helpers for scenario scripts: fresh mock environments and result bundling."""

from __future__ import annotations

from dataclasses import dataclass

from witness.core.runtime import RunResult
from witness.core.tools import ToolRegistry, build_default_registry
from witness.mocks.database import CustomerDatabase
from witness.mocks.outbox import EmailOutbox
from witness.mocks.ticketing import TicketingSystem


@dataclass(frozen=True)
class ScenarioEnvironment:
    """A fresh, self-contained set of mocks plus the tool registry wired to them."""

    registry: ToolRegistry
    database: CustomerDatabase
    ticketing: TicketingSystem
    outbox: EmailOutbox


def build_environment(*, ticketing_degraded: bool = False) -> ScenarioEnvironment:
    """Construct a fresh mock environment. `ticketing_degraded` enables the mode
    that manufactures the hallucination (see mocks.ticketing.TicketingSystem)."""
    database = CustomerDatabase()
    ticketing = TicketingSystem(degraded=ticketing_degraded)
    outbox = EmailOutbox()
    registry = build_default_registry(database=database, ticketing=ticketing, outbox=outbox)
    return ScenarioEnvironment(registry=registry, database=database, ticketing=ticketing, outbox=outbox)


@dataclass(frozen=True)
class ScenarioOutcome:
    """A scenario's produced runs plus the mocks they acted on. Grounding checks
    must verify against these exact instances, not fresh ones, since they hold
    the run's real resulting state."""

    scenario_name: str
    results: list[RunResult]
    environment: ScenarioEnvironment
