"""The trace layer: TraceEvent, TraceRun, TraceStore.

Every LLM call, tool call, and governance decision in Witness is recorded as a
TraceEvent. A TraceRun is the ordered sequence of events for one agent execution.
TraceStore persists runs to append-only JSONL files under `runs/` and loads them
back. Every other component (policy, grounding, drift, audit) reads from this
layer and nothing else, so its schema and round-trip fidelity matter more than
any other module in the system.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import config


class EventType(str, Enum):
    """The kinds of events that can appear in a trace."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    CLAIM = "claim"
    POLICY_VIOLATION = "policy_violation"
    GROUNDING_RESULT = "grounding_result"
    DRIFT_ALERT = "drift_alert"


def _new_id(prefix: str) -> str:
    """Generate a short unique id. Not seeded: ids are identifiers, not trace content."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class TraceEvent:
    """A single structured record in the trace.

    `payload` is a JSON-serializable dict whose shape depends on `event_type`; use
    the `*_payload` builder functions below to construct it correctly for each type.
    """

    id: str
    run_id: str
    agent_name: str
    timestamp: str
    event_type: EventType
    payload: dict[str, Any]
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    parent_id: str | None = None

    @classmethod
    def new(
        cls,
        *,
        run_id: str,
        agent_name: str,
        event_type: EventType,
        payload: dict[str, Any],
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        parent_id: str | None = None,
    ) -> TraceEvent:
        """Construct a new event stamped with a fresh id and the current time."""
        return cls(
            id=_new_id("evt"),
            run_id=run_id,
            agent_name=agent_name,
            timestamp=_utc_now_iso(),
            event_type=event_type,
            payload=payload,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            parent_id=parent_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEvent:
        """Reconstruct from a dict produced by `to_dict`."""
        return cls(
            id=data["id"],
            run_id=data["run_id"],
            agent_name=data["agent_name"],
            timestamp=data["timestamp"],
            event_type=EventType(data["event_type"]),
            payload=data["payload"],
            cost_usd=data.get("cost_usd", 0.0),
            latency_ms=data.get("latency_ms", 0.0),
            parent_id=data.get("parent_id"),
        )


# ------------------------------------------------------------------------------------
# Typed payload builders, one per event_type. Keeping these as plain functions with
# typed keyword arguments (rather than a dataclass per payload) avoids a parallel
# class hierarchy while still documenting and validating each event's shape.
# ------------------------------------------------------------------------------------


def llm_call_payload(
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    response_text: str | None,
    function_call: dict[str, Any] | None,
    prompt_tokens: int,
    completion_tokens: int,
    replayed: bool,
) -> dict[str, Any]:
    """Payload for an `llm_call` event."""
    return {
        "model": model,
        "system": system,
        "messages": messages,
        "response_text": response_text,
        "function_call": function_call,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "replayed": replayed,
    }


def tool_call_payload(
    *,
    tool_name: str,
    args: dict[str, Any],
    result: Any,
    ok: bool,
    error: str | None = None,
) -> dict[str, Any]:
    """Payload for a `tool_call` event. `result` is the real return value from the mock."""
    return {"tool_name": tool_name, "args": args, "result": result, "ok": ok, "error": error}


def claim_payload(
    *,
    claim_text: str,
    claim_type: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    """Payload for a `claim` event: one discrete, checkable claim extracted from an
    agent's final message."""
    return {"claim_text": claim_text, "claim_type": claim_type, "fields": fields}


def policy_violation_payload(
    *,
    rule_name: str,
    severity: str,
    description: str,
    offending_event_id: str | None,
) -> dict[str, Any]:
    """Payload for a `policy_violation` event."""
    return {
        "rule_name": rule_name,
        "severity": severity,
        "description": description,
        "offending_event_id": offending_event_id,
    }


def grounding_result_payload(
    *,
    claim_text: str,
    claim_type: str,
    classification: str,
    trace_evidence: dict[str, Any] | None,
    system_evidence: dict[str, Any] | None,
    evidence_gap: str | None,
) -> dict[str, Any]:
    """Payload for a `grounding_result` event: the verdict on one claim."""
    return {
        "claim_text": claim_text,
        "claim_type": claim_type,
        "classification": classification,
        "trace_evidence": trace_evidence,
        "system_evidence": system_evidence,
        "evidence_gap": evidence_gap,
    }


def drift_alert_payload(
    *,
    agent_name: str,
    distance: float,
    reason: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    """Payload for a `drift_alert` event."""
    return {"agent_name": agent_name, "distance": distance, "reason": reason, "details": details}


@dataclass
class TraceRun:
    """The ordered sequence of events for one agent execution, plus run-level metadata."""

    run_id: str
    agent_name: str
    seed: int
    started_at: str
    scenario: str | None = None
    ended_at: str | None = None
    outcome: str | None = None
    events: list[TraceEvent] = field(default_factory=list)

    @classmethod
    def start(cls, *, agent_name: str, seed: int, scenario: str | None = None) -> TraceRun:
        """Begin a new run."""
        return cls(
            run_id=_new_id("run"),
            agent_name=agent_name,
            seed=seed,
            started_at=_utc_now_iso(),
            scenario=scenario,
        )

    @property
    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.events)

    def add_event(self, event: TraceEvent) -> None:
        self.events.append(event)

    def finish(self, outcome: str) -> None:
        self.outcome = outcome
        self.ended_at = _utc_now_iso()

    def events_of_type(self, event_type: EventType) -> list[TraceEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def _meta(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "seed": self.seed,
            "started_at": self.started_at,
            "scenario": self.scenario,
            "ended_at": self.ended_at,
            "outcome": self.outcome,
        }


class TraceStore:
    """Persists TraceRuns as append-only JSONL files under `runs/` and loads them back.

    Each run is one file, `<run_id>.jsonl`. The first line is a `meta` record; each
    subsequent line is one `event` record, written as it is appended, so a run's file
    is a durable, ordered, streaming log even before the run finishes.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else config.RUNS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, run_id: str) -> Path:
        return self.base_dir / f"{run_id}.jsonl"

    def create_run(self, run: TraceRun) -> None:
        """Write the initial meta line for a new run, truncating any prior file."""
        with self._path_for(run.run_id).open("w", encoding="utf-8") as f:
            f.write(json.dumps({"record_type": "meta", **run._meta()}) + "\n")

    def append_event(self, run: TraceRun, event: TraceEvent) -> None:
        """Add an event to the in-memory run and durably append it to disk."""
        run.add_event(event)
        with self._path_for(run.run_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps({"record_type": "event", **event.to_dict()}) + "\n")

    def finish_run(self, run: TraceRun, outcome: str) -> None:
        """Mark the run finished in memory and record the outcome on disk."""
        run.finish(outcome)
        with self._path_for(run.run_id).open("a", encoding="utf-8") as f:
            record = {"record_type": "meta_update", "ended_at": run.ended_at, "outcome": run.outcome}
            f.write(json.dumps(record) + "\n")

    def load_run(self, run_id: str) -> TraceRun:
        """Reconstruct a TraceRun from its JSONL file, preserving event order."""
        meta: dict[str, Any] = {}
        events: list[TraceEvent] = []
        with self._path_for(run_id).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_type = record.pop("record_type")
                if record_type == "meta":
                    meta = record
                elif record_type == "event":
                    events.append(TraceEvent.from_dict(record))
                elif record_type == "meta_update":
                    meta.update(record)
        run = TraceRun(
            run_id=meta["run_id"],
            agent_name=meta["agent_name"],
            seed=meta["seed"],
            started_at=meta["started_at"],
            scenario=meta.get("scenario"),
            ended_at=meta.get("ended_at"),
            outcome=meta.get("outcome"),
        )
        run.events = events
        return run

    def list_run_ids(self) -> list[str]:
        """All run ids currently persisted, in filename order."""
        return sorted(p.stem for p in self.base_dir.glob("*.jsonl"))

    def load_all_runs(self) -> list[TraceRun]:
        return [self.load_run(rid) for rid in self.list_run_ids()]
