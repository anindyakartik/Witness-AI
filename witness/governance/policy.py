"""PolicyEngine: evaluates a fixed list of Rules against a run's event stream.

Evaluation is a single sequential pass over the run's already-recorded, ordered
trace, giving each rule exactly the events that preceded the one it's judging --
the same information a live, streaming evaluator would have at that point. This
keeps governance fully decoupled from core/runtime.py (which never imports this
module), at the cost of enforcement happening after a run completes rather than
blocking it live; see the README's production section for the streaming variant.
"""

from __future__ import annotations

from witness.core.trace import (
    EventType,
    TraceEvent,
    TraceRun,
    TraceStore,
    policy_violation_payload,
)
from witness.governance.rules import (
    ApprovalGateRule,
    CostCapRule,
    PIILeakRule,
    Rule,
    ToolAllowlistRule,
    Violation,
)


class PolicyEngine:
    """Evaluates a list of Rules against a TraceRun's events, in emission order."""

    def __init__(self, rules: list[Rule]) -> None:
        self.rules = rules

    def evaluate(self, run: TraceRun) -> list[Violation]:
        """Evaluate all rules against `run`'s events; does not mutate the trace."""
        violations: list[Violation] = []
        for i, event in enumerate(run.events):
            prior_events = run.events[:i]
            for rule in self.rules:
                violation = rule.check(event, prior_events, run)
                if violation is not None:
                    violations.append(violation)
        return violations

    def evaluate_and_record(self, store: TraceStore, run: TraceRun) -> list[Violation]:
        """Evaluate, then append a `policy_violation` TraceEvent per violation found."""
        violations = self.evaluate(run)
        for v in violations:
            event = TraceEvent.new(
                run_id=run.run_id,
                agent_name=run.agent_name,
                event_type=EventType.POLICY_VIOLATION,
                payload=policy_violation_payload(
                    rule_name=v.rule_name,
                    severity=v.severity,
                    description=v.description,
                    offending_event_id=v.offending_event_id,
                ),
                parent_id=v.offending_event_id,
            )
            store.append_event(run, event)
        return violations


def build_default_policy_engine() -> PolicyEngine:
    """Construct the standard PolicyEngine with all 4 rules, including the
    canonical per-agent allowlists sourced from the agents package."""
    from witness.agents.data_lookup import DATA_LOOKUP
    from witness.agents.report_generator import REPORT_GENERATOR
    from witness.agents.summarizer import SUMMARIZER
    from witness.agents.ticket_filer import TICKET_FILER

    allowlists = {
        a.name: a.tool_allowlist for a in (SUMMARIZER, DATA_LOOKUP, TICKET_FILER, REPORT_GENERATOR)
    }
    return PolicyEngine(
        rules=[
            PIILeakRule(),
            ApprovalGateRule(),
            CostCapRule(),
            ToolAllowlistRule(allowlists),
        ]
    )
