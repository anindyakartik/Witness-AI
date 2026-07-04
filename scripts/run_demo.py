"""One-command demo: runs all scenarios, persists traces, builds the audit
report, and prints the three headline results.

Usage: python scripts/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from scenarios import clean_run, drift, hallucination, policy_violation
from scenarios.common import ScenarioOutcome
from witness.audit.report import AuditReport, build_audit_report, save_report
from witness.core.trace import TraceRun, TraceStore
from witness.governance.drift import DriftCheckResult, DriftDetector, build_fingerprint
from witness.governance.grounding import GroundingChecker, GroundingResult, MockSystems
from witness.governance.policy import PolicyEngine, build_default_policy_engine
from witness.governance.rules import Violation


def _clear_prior_demo_output() -> None:
    """Each demo run is self-contained: clear previously persisted runs and
    audit output so the report and dashboard reflect exactly this invocation."""
    for f in config.RUNS_DIR.glob("*.jsonl"):
        f.unlink()
    if config.AUDIT_DIR.exists():
        for f in config.AUDIT_DIR.glob("*"):
            f.unlink()


def _process_outcome(
    store: TraceStore, policy_engine: PolicyEngine, outcome: ScenarioOutcome
) -> tuple[list[list[GroundingResult]], list[list[Violation]]]:
    """Run grounding checks and policy evaluation for every run in a scenario
    outcome. Returns per-run lists so callers can inspect a specific scenario's
    results without re-deriving them from the full trace store."""
    checker = GroundingChecker(
        MockSystems(
            ticketing=outcome.environment.ticketing,
            database=outcome.environment.database,
            outbox=outcome.environment.outbox,
        )
    )
    grounding_per_run = []
    violations_per_run = []
    for result in outcome.results:
        grounding_per_run.append(checker.check_and_record(store, result.run, result.final_message))
        violations_per_run.append(policy_engine.evaluate_and_record(store, result.run))
    return grounding_per_run, violations_per_run


def main() -> int:
    _clear_prior_demo_output()
    store = TraceStore(base_dir=config.RUNS_DIR)
    policy_engine = build_default_policy_engine()
    all_produced_runs: list[TraceRun] = []

    print("Witness demo: running scenarios...")

    print("  - clean control (0-violations baseline across all 4 agents)")
    control_outcome = clean_run.run_control(store)
    _process_outcome(store, policy_engine, control_outcome)
    all_produced_runs += [r.run for r in control_outcome.results]

    print(f"  - drift baseline ({config.DRIFT_BASELINE_RUNS} data_lookup runs)")
    baseline_outcome = clean_run.run_drift_baseline(store)
    _process_outcome(store, policy_engine, baseline_outcome)
    all_produced_runs += [r.run for r in baseline_outcome.results]
    fingerprint = build_fingerprint("data_lookup", [r.run for r in baseline_outcome.results])
    drift_detector = DriftDetector(fingerprint)

    print("  - hallucination scenario")
    hallucination_outcome = hallucination.run(store)
    hallucination_grounding, _ = _process_outcome(store, policy_engine, hallucination_outcome)
    all_produced_runs += [r.run for r in hallucination_outcome.results]

    print("  - policy violation scenario")
    policy_outcome = policy_violation.run(store)
    _, policy_violations_per_run = _process_outcome(store, policy_engine, policy_outcome)
    all_produced_runs += [r.run for r in policy_outcome.results]

    print("  - drift scenario")
    drift_outcome = drift.run(store)
    _process_outcome(store, policy_engine, drift_outcome)
    all_produced_runs += [r.run for r in drift_outcome.results]
    drift_check_result = drift_detector.check_and_record(store, drift_outcome.results[0].run)

    report = build_audit_report(all_produced_runs)
    json_path, md_path = save_report(report)

    print()
    print("=" * 70)
    print("WITNESS DEMO RESULTS")
    print("=" * 70)

    ok = True
    ok &= _print_hallucination_headline(hallucination_outcome, hallucination_grounding)
    ok &= _print_policy_headline(policy_violations_per_run)
    ok &= _print_drift_headline(drift_outcome, drift_check_result, fingerprint.num_baseline_runs)

    print()
    _print_score_breakdown(report)
    print(f"Governance Readiness Score: {report.readiness_score}/100")
    print(f"Full report: {json_path}")
    print(f"          and {md_path}")
    print("View the dashboard: streamlit run witness/dashboard/app.py")

    return 0 if ok else 1


def _print_score_breakdown(report: AuditReport) -> None:
    """Print the fleet-wide violation counts underlying the readiness score.

    The three headlines above each describe one scenario's own designed-for
    failure, but any run -- including ones not called out by name, like the
    drift scenario's send_email call also independently leaking PII -- can
    trigger additional flags that still count toward the score. This line makes
    the score's arithmetic checkable from console output alone.
    """
    total_ungrounded = 0
    total_contradicted = 0
    total_policy = 0
    total_drift = 0
    for a in report.agent_summaries:
        total_ungrounded += sum(1 for c in a.claim_issues if c.classification == "UNGROUNDED")
        total_contradicted += sum(1 for c in a.claim_issues if c.classification == "CONTRADICTED")
        total_policy += len(a.policy_violations)
        total_drift += len(a.drift_alerts)

    print(
        f"Fleet-wide across all {report.total_runs} runs: {total_ungrounded} ungrounded, "
        f"{total_contradicted} contradicted, {total_policy} policy violations, "
        f"{total_drift} drift alert(s) -- see the full report for anything beyond "
        "the 3 scenarios above."
    )


def _print_hallucination_headline(
    outcome: ScenarioOutcome, grounding_per_run: list[list[GroundingResult]]
) -> bool:
    ungrounded = [
        r for results in grounding_per_run for r in results if r.classification == "UNGROUNDED"
    ]
    if not ungrounded:
        print("x Hallucination scenario did not produce an UNGROUNDED claim (unexpected).")
        return False
    agent_name = outcome.results[0].run.agent_name
    claim = ungrounded[0]
    print(
        f"✓ Caught {len(ungrounded)} UNGROUNDED claim: {agent_name} reported "
        f'"{claim.claim.claim_text}" -- {claim.evidence_gap}'
    )
    return True


def _print_policy_headline(violations_per_run: list[list[Violation]]) -> bool:
    violations = [v for run_violations in violations_per_run for v in run_violations]
    if not violations:
        print("x Policy violation scenario did not trigger any violations (unexpected).")
        return False
    descriptions = "; ".join(v.description for v in violations)
    print(f"✓ Caught {len(violations)} policy violations: {descriptions}")
    return True


def _print_drift_headline(
    outcome: ScenarioOutcome, result: DriftCheckResult, num_baseline_runs: int
) -> bool:
    if not result.is_drift:
        print("x Drift scenario did not trigger a drift alert (unexpected).")
        return False
    agent_name = outcome.results[0].run.agent_name
    print(
        f"✓ Drift alert: {agent_name} tool-usage diverged {result.distance:.2f} from "
        f"{num_baseline_runs}-run baseline ({result.reason})"
    )
    return True


if __name__ == "__main__":
    sys.exit(main())
