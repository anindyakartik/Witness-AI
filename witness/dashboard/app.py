"""Streamlit dashboard over persisted Witness runs.

Three views: a fleet overview (readiness score + counts), a per-agent breakdown
table, and a per-run inspector with an event timeline and the grounding panel --
claim vs. reality, side by side -- which is the project's money shot.

Run with: streamlit run witness/dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from witness.audit.report import AuditReport, build_audit_report
from witness.core.trace import EventType, TraceRun, TraceStore

_CLASSIFICATION_COLOR = {
    "GROUNDED": "green",
    "CONTRADICTED": "orange",
    "UNGROUNDED": "red",
}
_EVENT_ICON = {
    EventType.LLM_CALL: "🧠",
    EventType.TOOL_CALL: "🔧",
    EventType.CLAIM: "💬",
    EventType.POLICY_VIOLATION: "🚨",
    EventType.GROUNDING_RESULT: "🔍",
    EventType.DRIFT_ALERT: "📈",
}


@st.cache_data(show_spinner=False)
def _load_runs(_runs_dir_marker: str) -> list[TraceRun]:
    """Load all persisted runs. Cached on the runs directory path so a fresh
    demo run (which changes the directory's contents) invalidates the cache
    when the dashboard is reloaded with a different marker."""
    store = TraceStore(base_dir=config.RUNS_DIR)
    return store.load_all_runs()


def render_fleet_overview(report: AuditReport) -> None:
    st.header("Fleet Overview")
    cols = st.columns(4)
    cols[0].metric("Governance Readiness Score", f"{report.readiness_score}/100")
    cols[1].metric("Total Runs", report.total_runs)
    cols[2].metric("Total Cost", f"${report.total_cost_usd:.4f}")

    total_issues = sum(len(a.claim_issues) for a in report.agent_summaries)
    total_violations = sum(len(a.policy_violations) for a in report.agent_summaries)
    total_drift = sum(len(a.drift_alerts) for a in report.agent_summaries)
    cols[3].metric("Flags Raised", total_issues + total_violations + total_drift)

    with st.expander("How the score is calculated"):
        st.write(f"Starts at {config.SCORE_START}, deducts weighted points per violation class:")
        for name, weight in config.SCORE_DEDUCTIONS.items():
            st.write(f"- **{name.replace('_', ' ')}**: -{weight} each")
        st.write("Floored at 0.")


def render_agent_table(report: AuditReport) -> None:
    st.header("Agent Breakdown")
    if not report.agent_summaries:
        st.info("No agents to show yet.")
        return

    rows = []
    for a in report.agent_summaries:
        ungrounded = sum(1 for c in a.claim_issues if c.classification == "UNGROUNDED")
        contradicted = sum(1 for c in a.claim_issues if c.classification == "CONTRADICTED")
        rows.append(
            {
                "Agent": a.agent_name,
                "Runs": a.run_count,
                "Cost ($)": round(a.total_cost_usd, 5),
                "Readiness Score": a.readiness_score,
                "Grounded": a.grounded_claim_count,
                "Ungrounded": ungrounded,
                "Contradicted": contradicted,
                "Policy Violations": len(a.policy_violations),
                "Drift Alerts": len(a.drift_alerts),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_run_timeline(run: TraceRun) -> None:
    st.subheader("Event Timeline")
    for e in run.events:
        icon = _EVENT_ICON.get(e.event_type, "•")
        label = e.event_type.value

        if e.event_type is EventType.TOOL_CALL:
            ok = e.payload.get("ok")
            detail = f"`{e.payload.get('tool_name')}` args={e.payload.get('args')}"
            if not ok:
                st.error(f"{icon} **{label}** {detail} -- FAILED: {e.payload.get('error')}")
            else:
                st.write(f"{icon} **{label}** {detail} -> {e.payload.get('result')}")
        elif e.event_type is EventType.POLICY_VIOLATION:
            st.error(
                f"{icon} **{e.payload['rule_name']}** ({e.payload['severity']}): {e.payload['description']}"
            )
        elif e.event_type is EventType.GROUNDING_RESULT:
            classification = e.payload["classification"]
            color = _CLASSIFICATION_COLOR.get(classification, "gray")
            st.markdown(
                f"{icon} **{label}** -- :{color}[{classification}] \"{e.payload['claim_text']}\""
            )
        elif e.event_type is EventType.DRIFT_ALERT:
            st.warning(
                f"{icon} **{label}**: {e.payload['reason']} (distance={e.payload['distance']:.2f})"
            )
        elif e.event_type is EventType.LLM_CALL:
            summary = e.payload.get("response_text") or (
                f"function_call: {e.payload.get('function_call', {}).get('name')}"
            )
            replayed = " (replayed)" if e.payload.get("replayed") else ""
            st.write(f"{icon} **{label}**{replayed}: {summary}")
        elif e.event_type is EventType.CLAIM:
            st.write(
                f"{icon} **{label}** ({e.payload['claim_type']}): \"{e.payload['claim_text']}\""
            )
        else:
            st.write(f"{icon} **{label}**: {e.payload}")


def render_grounding_panel(run: TraceRun) -> None:
    """The money shot: what the agent claimed vs. what actually happened, side by side."""
    st.subheader("Grounding: Claim vs. Reality")
    grounding_events = run.events_of_type(EventType.GROUNDING_RESULT)
    if not grounding_events:
        st.info("No claims were extracted from this run's final message.")
        return

    for e in grounding_events:
        classification = e.payload["classification"]
        color = _CLASSIFICATION_COLOR.get(classification, "gray")
        st.markdown(f"### :{color}[{classification}]")
        left, right = st.columns(2)
        with left:
            st.markdown("**What the agent claimed**")
            st.info(e.payload["claim_text"])
        with right:
            st.markdown("**What actually happened**")
            if e.payload["evidence_gap"]:
                st.error(e.payload["evidence_gap"])
            else:
                st.success("Confirmed by trace and system state.")
            with st.expander("Evidence detail"):
                st.write("Trace evidence:", e.payload["trace_evidence"])
                st.write("System evidence:", e.payload["system_evidence"])
        st.divider()


def render_run_inspector(runs: list[TraceRun]) -> None:
    st.header("Run Inspector")
    if not runs:
        st.info("No runs to inspect yet.")
        return

    options = {
        f"{r.run_id} — {r.agent_name} ({r.scenario or 'no scenario'}) — {r.outcome}": r.run_id
        for r in runs
    }
    selected_label = st.selectbox("Choose a run", list(options.keys()))
    selected_run = next(r for r in runs if r.run_id == options[selected_label])

    st.caption(
        f"Agent: **{selected_run.agent_name}** | Outcome: **{selected_run.outcome}** | "
        f"Cost: ${selected_run.total_cost_usd:.5f} | Events: {len(selected_run.events)}"
    )

    tab_grounding, tab_timeline = st.tabs(["Grounding Panel", "Event Timeline"])
    with tab_grounding:
        render_grounding_panel(selected_run)
    with tab_timeline:
        render_run_timeline(selected_run)


def main() -> None:
    st.set_page_config(page_title="Witness — Agent Governance Dashboard", layout="wide")
    st.title("Witness")
    st.caption("Runtime governance and observability for multi-agent AI systems.")

    runs_dir_marker = str(sorted(config.RUNS_DIR.glob("*.jsonl")))
    runs = _load_runs(runs_dir_marker)

    if not runs:
        st.warning(
            "No runs found. Run `python scripts/run_demo.py` first to generate a "
            "demo trace, then reload this page."
        )
        return

    report = build_audit_report(runs)
    render_fleet_overview(report)
    render_agent_table(report)
    st.divider()
    render_run_inspector(runs)


if __name__ == "__main__":
    main()
