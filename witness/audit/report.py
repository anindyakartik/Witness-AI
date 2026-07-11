"""AuditReport: aggregates a set of TraceRuns into a compliance-style report with
a transparent, documented Governance Readiness Score.

Reads only from the trace layer (policy_violation, grounding_result, and
drift_alert events already recorded by the governance modules). This module
performs no verification of its own, only aggregation and scoring.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import config
from witness.core.trace import EventType, TraceRun


def compute_readiness_score(
    *, ungrounded: int, contradicted: int, policy_violations: int, drift_alerts: int
) -> int:
    """Governance Readiness Score: start at config.SCORE_START, deduct weighted
    points per violation class (config.SCORE_DEDUCTIONS), floor at 0. A pure
    function of the raw counts, so the number is reproducible and defensible."""
    score = config.SCORE_START
    score -= ungrounded * config.SCORE_DEDUCTIONS["ungrounded_claim"]
    score -= contradicted * config.SCORE_DEDUCTIONS["contradicted_claim"]
    score -= policy_violations * config.SCORE_DEDUCTIONS["policy_violation"]
    score -= drift_alerts * config.SCORE_DEDUCTIONS["drift_alert"]
    return max(0, score)


@dataclass(frozen=True)
class PolicyViolationEntry:
    rule_name: str
    severity: str
    description: str
    run_id: str
    agent_name: str
    offending_event_id: str | None
    offending_event_excerpt: dict[str, Any] | None


@dataclass(frozen=True)
class ClaimIssueEntry:
    """An ungrounded or contradicted claim, with its evidence gap. GROUNDED
    claims are counted but not itemized here, there's nothing to audit."""

    claim_text: str
    claim_type: str
    classification: str
    evidence_gap: str | None
    run_id: str
    agent_name: str


@dataclass(frozen=True)
class DriftAlertEntry:
    reason: str
    distance: float
    run_id: str
    agent_name: str


@dataclass(frozen=True)
class AgentSummary:
    agent_name: str
    run_count: int
    total_cost_usd: float
    grounded_claim_count: int
    policy_violations: list[PolicyViolationEntry] = field(default_factory=list)
    claim_issues: list[ClaimIssueEntry] = field(default_factory=list)
    drift_alerts: list[DriftAlertEntry] = field(default_factory=list)
    readiness_score: int = config.SCORE_START


@dataclass(frozen=True)
class AuditReport:
    generated_at: str
    total_runs: int
    total_cost_usd: float
    readiness_score: int
    agent_summaries: list[AgentSummary]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# Witness Audit Report",
            "",
            f"Generated: {self.generated_at}",
            f"Total runs analyzed: {self.total_runs}",
            f"Total cost: ${self.total_cost_usd:.4f}",
            "",
            f"## Governance Readiness Score: {self.readiness_score}/100",
            "",
            "Score starts at 100 and deducts weighted points per violation class:",
        ]
        for name, weight in config.SCORE_DEDUCTIONS.items():
            lines.append(f"- {name.replace('_', ' ')}: -{weight} each")

        for agent in self.agent_summaries:
            ungrounded = [c for c in agent.claim_issues if c.classification == "UNGROUNDED"]
            contradicted = [c for c in agent.claim_issues if c.classification == "CONTRADICTED"]

            lines += [
                "",
                f"## Agent: {agent.agent_name}",
                f"- Runs: {agent.run_count}",
                f"- Total cost: ${agent.total_cost_usd:.4f}",
                f"- Readiness score: {agent.readiness_score}/100",
                f"- Grounded claims: {agent.grounded_claim_count}",
                f"- Ungrounded claims: {len(ungrounded)}",
                f"- Contradicted claims: {len(contradicted)}",
                f"- Policy violations: {len(agent.policy_violations)}",
                f"- Drift alerts: {len(agent.drift_alerts)}",
            ]

            if agent.claim_issues:
                lines += ["", "### Claim Issues"]
                lines += [
                    f'- **{c.classification}** ({c.claim_type}): "{c.claim_text}". {c.evidence_gap}'
                    for c in agent.claim_issues
                ]

            if agent.policy_violations:
                lines += ["", "### Policy Violations"]
                lines += [
                    f"- **{v.rule_name}** ({v.severity}): {v.description}"
                    for v in agent.policy_violations
                ]

            if agent.drift_alerts:
                lines += ["", "### Drift Alerts"]
                lines += [f"- distance={d.distance:.2f}: {d.reason}" for d in agent.drift_alerts]

        return "\n".join(lines) + "\n"


def _find_event(run: TraceRun, event_id: str | None) -> dict[str, Any] | None:
    if event_id is None:
        return None
    for e in run.events:
        if e.id == event_id:
            return {"event_type": e.event_type.value, "payload": e.payload}
    return None


def build_audit_report(runs: list[TraceRun]) -> AuditReport:
    """Aggregate `runs` (typically all persisted runs from a demo) into a report."""
    by_agent: dict[str, list[TraceRun]] = {}
    for run in runs:
        by_agent.setdefault(run.agent_name, []).append(run)

    agent_summaries: list[AgentSummary] = []
    for agent_name in sorted(by_agent):
        agent_runs = by_agent[agent_name]
        policy_violations: list[PolicyViolationEntry] = []
        claim_issues: list[ClaimIssueEntry] = []
        drift_alerts: list[DriftAlertEntry] = []
        grounded_count = 0

        for run in agent_runs:
            for e in run.events_of_type(EventType.POLICY_VIOLATION):
                policy_violations.append(
                    PolicyViolationEntry(
                        rule_name=e.payload["rule_name"],
                        severity=e.payload["severity"],
                        description=e.payload["description"],
                        run_id=run.run_id,
                        agent_name=agent_name,
                        offending_event_id=e.payload["offending_event_id"],
                        offending_event_excerpt=_find_event(run, e.payload["offending_event_id"]),
                    )
                )
            for e in run.events_of_type(EventType.GROUNDING_RESULT):
                classification = e.payload["classification"]
                if classification == "GROUNDED":
                    grounded_count += 1
                else:
                    claim_issues.append(
                        ClaimIssueEntry(
                            claim_text=e.payload["claim_text"],
                            claim_type=e.payload["claim_type"],
                            classification=classification,
                            evidence_gap=e.payload["evidence_gap"],
                            run_id=run.run_id,
                            agent_name=agent_name,
                        )
                    )
            for e in run.events_of_type(EventType.DRIFT_ALERT):
                drift_alerts.append(
                    DriftAlertEntry(
                        reason=e.payload["reason"],
                        distance=e.payload["distance"],
                        run_id=run.run_id,
                        agent_name=agent_name,
                    )
                )

        ungrounded_count = sum(1 for c in claim_issues if c.classification == "UNGROUNDED")
        contradicted_count = sum(1 for c in claim_issues if c.classification == "CONTRADICTED")

        agent_summaries.append(
            AgentSummary(
                agent_name=agent_name,
                run_count=len(agent_runs),
                total_cost_usd=sum(r.total_cost_usd for r in agent_runs),
                grounded_claim_count=grounded_count,
                policy_violations=policy_violations,
                claim_issues=claim_issues,
                drift_alerts=drift_alerts,
                readiness_score=compute_readiness_score(
                    ungrounded=ungrounded_count,
                    contradicted=contradicted_count,
                    policy_violations=len(policy_violations),
                    drift_alerts=len(drift_alerts),
                ),
            )
        )

    total_ungrounded = sum(
        sum(1 for c in a.claim_issues if c.classification == "UNGROUNDED") for a in agent_summaries
    )
    total_contradicted = sum(
        sum(1 for c in a.claim_issues if c.classification == "CONTRADICTED")
        for a in agent_summaries
    )
    total_policy_violations = sum(len(a.policy_violations) for a in agent_summaries)
    total_drift_alerts = sum(len(a.drift_alerts) for a in agent_summaries)

    return AuditReport(
        generated_at=datetime.now(UTC).isoformat(),
        total_runs=len(runs),
        total_cost_usd=sum(r.total_cost_usd for r in runs),
        readiness_score=compute_readiness_score(
            ungrounded=total_ungrounded,
            contradicted=total_contradicted,
            policy_violations=total_policy_violations,
            drift_alerts=total_drift_alerts,
        ),
        agent_summaries=agent_summaries,
    )


def save_report(report: AuditReport, directory: Path | None = None) -> tuple[Path, Path]:
    """Persist `report` as both JSON and Markdown under `directory` (default
    config.AUDIT_DIR). Returns (json_path, markdown_path)."""
    out_dir = directory if directory is not None else config.AUDIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "audit_report.json"
    md_path = out_dir / "audit_report.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return json_path, md_path
