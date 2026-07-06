"""Streamlit dashboard over persisted Witness runs.

Story-first layout. A first-time visitor lands on a guided, step-by-step
*simulation* of the flagship failure -- an agent reporting success on a ticket
that was never saved -- built from the real trace data, so the "claim vs.
reality" moment lands before any table does. Deeper data views (the three
catches, the fleet dashboard, and a raw run inspector) sit behind a sidebar nav
for anyone who wants to dig in.

Run with: streamlit run witness/dashboard/app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Streamlit adds this file's own directory to sys.path, not the repo root, so
# `streamlit run witness/dashboard/app.py` needs this to resolve `import config`
# and the `witness` package regardless of the caller's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

import config
from witness.audit.report import AuditReport, build_audit_report
from witness.core.trace import EventType, TraceRun, TraceStore

# --------------------------------------------------------------------------- #
# Look & feel
# --------------------------------------------------------------------------- #

_VERDICT = {
    "GROUNDED": {"color": "#1a9e5f", "bg": "rgba(26,158,95,0.12)", "icon": "✅",
                 "label": "GROUNDED", "gloss": "trace and reality agree"},
    "CONTRADICTED": {"color": "#d98a00", "bg": "rgba(217,138,0,0.12)", "icon": "⚠️",
                     "label": "CONTRADICTED", "gloss": "it happened, but not the way claimed"},
    "UNGROUNDED": {"color": "#e0413a", "bg": "rgba(224,65,58,0.12)", "icon": "❌",
                   "label": "UNGROUNDED", "gloss": "claimed, but no evidence it happened"},
}
_EVENT_ICON = {
    EventType.LLM_CALL: "🧠",
    EventType.TOOL_CALL: "🔧",
    EventType.CLAIM: "💬",
    EventType.POLICY_VIOLATION: "🚨",
    EventType.GROUNDING_RESULT: "🔍",
    EventType.DRIFT_ALERT: "📈",
}

_CSS = """
<style>
:root {
  --wit-line: rgba(128,128,128,0.28);
  --wit-soft: rgba(128,128,128,0.08);
}
.wit-hero {
  border: 1px solid var(--wit-line);
  border-radius: 14px;
  padding: 1.4rem 1.6rem;
  background: linear-gradient(135deg, rgba(99,102,241,0.14), rgba(14,165,233,0.06));
  margin-bottom: 1.1rem;
}
.wit-hero h1 { font-size: 2.0rem; margin: 0 0 .35rem 0; letter-spacing: -0.5px; }
.wit-hero p  { font-size: 1.02rem; margin: 0; opacity: .85; line-height: 1.5; }
.wit-hero .tag { font-size: .78rem; text-transform: uppercase; letter-spacing: 2px;
  opacity: .6; font-weight: 700; }

.wit-step {
  border: 1px solid var(--wit-line);
  border-left: 5px solid var(--accent, #6366f1);
  border-radius: 12px;
  padding: .9rem 1.1rem;
  margin: .55rem 0;
  background: var(--wit-soft);
}
.wit-step .head { display: flex; align-items: center; gap: .6rem; margin-bottom: .35rem; }
.wit-step .num {
  flex: 0 0 auto; width: 26px; height: 26px; border-radius: 50%;
  background: var(--accent, #6366f1); color: #fff; font-weight: 700; font-size: .82rem;
  display: flex; align-items: center; justify-content: center;
}
.wit-step .title { font-weight: 700; font-size: 1.02rem; }
.wit-step .body { opacity: .9; line-height: 1.55; }
.wit-step .body b { opacity: 1; }
.wit-mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .86rem; background: rgba(128,128,128,0.13);
  padding: .5rem .7rem; border-radius: 8px; display: block; margin-top: .5rem;
  white-space: pre-wrap; word-break: break-word;
}
.wit-note { font-size: .85rem; opacity: .62; margin-top: .5rem; font-style: italic; }

.wit-verdict {
  border-radius: 12px; padding: 1rem 1.2rem; margin: .3rem 0 .9rem 0;
  border: 1px solid var(--vc); background: var(--vbg);
  display: flex; align-items: center; gap: .8rem;
}
.wit-verdict .big { font-size: 1.6rem; font-weight: 800; color: var(--vc); letter-spacing: .5px; }
.wit-verdict .gloss { opacity: .8; font-size: .95rem; }

.wit-score {
  text-align: center; border: 1px solid var(--vc); border-radius: 16px;
  padding: 1.2rem; background: var(--vbg);
}
.wit-score .n { font-size: 3.4rem; font-weight: 800; color: var(--vc); line-height: 1; }
.wit-score .l { text-transform: uppercase; letter-spacing: 2px; font-size: .72rem;
  opacity: .7; font-weight: 700; margin-top: .4rem; }

.wit-pillrow { display: flex; gap: .5rem; flex-wrap: wrap; margin: .3rem 0 1rem 0; }
.wit-pill { border: 1px solid var(--wit-line); border-radius: 999px;
  padding: .28rem .8rem; font-size: .84rem; opacity: .9; }

.wit-tri { border: 1px solid var(--wit-line); border-radius: 12px; padding: 1rem 1.1rem;
  height: 100%; background: var(--wit-soft); }
.wit-tri .k { font-weight: 700; font-size: 1rem; margin-bottom: .3rem; }
.wit-tri .v { opacity: .82; font-size: .92rem; line-height: 1.5; }
</style>
"""


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def _load_runs(_runs_dir_marker: str) -> list[TraceRun]:
    """Load all persisted runs. Cached on the runs directory path so a fresh
    demo run (which changes the directory's contents) invalidates the cache
    when the dashboard is reloaded with a different marker."""
    store = TraceStore(base_dir=config.RUNS_DIR)
    return store.load_all_runs()


def _run_for_scenario(runs: list[TraceRun], scenario: str) -> TraceRun | None:
    return next((r for r in runs if r.scenario == scenario), None)


def _first(run: TraceRun, event_type: EventType):
    return next((e for e in run.events_of_type(event_type)), None)


def _card(*, accent: str, num: str, icon: str, title: str, body_html: str,
          mono: str | None = None, note: str | None = None) -> str:
    mono_html = f'<span class="wit-mono">{html.escape(mono)}</span>' if mono else ""
    note_html = f'<div class="wit-note">{html.escape(note)}</div>' if note else ""
    return (
        f'<div class="wit-step" style="--accent:{accent}">'
        f'<div class="head"><div class="num">{num}</div>'
        f'<div class="title">{icon}&nbsp;{html.escape(title)}</div></div>'
        f'<div class="body">{body_html}{mono_html}{note_html}</div></div>'
    )


# --------------------------------------------------------------------------- #
# 1. The simulation -- the front door
# --------------------------------------------------------------------------- #

def _sim_steps(run: TraceRun) -> list[dict]:
    """Build the guided walkthrough from the real hallucination trace, so every
    id, payload, and verdict shown is the genuine recorded data."""
    tool = _first(run, EventType.TOOL_CALL)
    grounding = _first(run, EventType.GROUNDING_RESULT)
    claim = _first(run, EventType.CLAIM)

    ticket_id = (tool.payload.get("result") or {}).get("ticket_id", "?") if tool else "?"
    tool_result = (tool.payload.get("result") if tool else {}) or {}
    subject = (tool.payload.get("args") or {}).get("subject", "") if tool else ""
    claim_text = claim.payload["claim_text"] if claim else ""
    gap = grounding.payload.get("evidence_gap") if grounding else ""
    sys_ev = grounding.payload.get("system_evidence") if grounding else {}

    indigo, teal, red, green = "#6366f1", "#0ea5e9", "#e0413a", "#1a9e5f"

    return [
        _card(
            accent=indigo, num="1", icon="🎫", title="The task",
            body_html=(
                "A support agent named <b>ticket_filer</b> is asked to do one simple thing:"
                "<br><br><i>&ldquo;File a ticket: the customer is locked out of their account "
                "after a password reset.&rdquo;</i><br><br>"
                "Nothing exotic. This is the kind of job a fleet runs thousands of times a day."
            ),
        ),
        _card(
            accent=teal, num="2", icon="🧠", title="The agent decides to act",
            body_html=(
                "The model reads the task and does exactly what it should: it calls its "
                "<b>create_ticket</b> tool. Witness records the real call and its real arguments."
            ),
            mono=f"create_ticket(subject={subject!r})",
        ),
        _card(
            accent=green, num="3", icon="🔧", title="The tool reports success",
            body_html=(
                f"The ticketing backend answers with a clean, real-looking success and a "
                f"genuine-looking ticket id <b>#{ticket_id}</b>."
            ),
            mono=f"→ {tool_result}",
            note="Under the hood the backend acknowledged a write it silently dropped — a real, "
                 "mundane failure mode. The agent has no way to see this from the inside.",
        ),
        _card(
            accent=green, num="4", icon="💬", title="The agent reports the job done",
            body_html=(
                "Trusting its tool — as any honest agent would — it tells everyone the ticket "
                "is filed:"
                f"<br><br><b>&ldquo;{html.escape(claim_text)}&rdquo;</b><br><br>"
                "The customer is reassured. The dashboard is green. <b>Every observer sees "
                "success.</b> This is exactly where the story normally ends — and where the "
                "ticket quietly never gets worked."
            ),
        ),
        _card(
            accent=red, num="5", icon="🔍", title="Witness asks the one question nobody else did",
            body_html=(
                f"Instead of trusting the agent's word or the tool's self-report, Witness queries "
                f"the ticketing system's <b>actual state</b>:<br><br>"
                f"<i>Does ticket #{ticket_id} really exist, right now?</i>"
            ),
            mono=f"ticketing.exists({ticket_id})  →  {sys_ev.get('ticket', sys_ev)}",
            note="No LLM in this path. Just a hard, deterministic fact check against ground truth.",
        ),
        {"verdict": grounding.payload if grounding else None, "gap": gap, "claim": claim_text},
    ]


def render_simulation(runs: list[TraceRun]) -> None:
    st.markdown(
        '<div class="wit-hero"><div class="tag">Live walkthrough · real trace data</div>'
        "<h1>Watch Witness catch a lie</h1>"
        "<p>An AI agent can be perfectly honest and still completely wrong — because it "
        "trusted a tool that lied to it. Step through a real run and watch the moment "
        "everyone else misses.</p></div>",
        unsafe_allow_html=True,
    )

    run = _run_for_scenario(runs, "hallucination")
    if run is None:
        st.info("No hallucination run found in the trace data yet.")
        return

    steps = _sim_steps(run)
    last = len(steps) - 1

    if "sim_step" not in st.session_state:
        st.session_state.sim_step = 0

    def _advance() -> None:
        st.session_state.sim_step = min(st.session_state.sim_step + 1, last)

    def _reset() -> None:
        st.session_state.sim_step = 0

    cur = st.session_state.sim_step

    # Controls
    c1, c2, c3 = st.columns([1.1, 1.1, 3])
    if cur < last:
        label = "▶  Start the run" if cur == 0 else "Next step  →"
        c1.button(label, type="primary", use_container_width=True, on_click=_advance)
    else:
        c1.button("↻  Replay", use_container_width=True, on_click=_reset)
    if 0 < cur < last:
        c2.button("↻  Replay", use_container_width=True, on_click=_reset)
    c3.progress((cur) / last, text=f"Step {cur} of {last}")

    st.write("")

    # Reveal every step up to the current one, so the story visibly stacks up.
    for i in range(cur + 1):
        step = steps[i]
        if isinstance(step, str):
            st.markdown(step, unsafe_allow_html=True)
        elif step.get("verdict"):
            _render_verdict_panel(step["verdict"])

    if cur == 0:
        st.caption("Press **Start the run** and follow along — six steps to the catch.")


def _render_verdict_panel(payload: dict) -> None:
    """The money shot: a big verdict badge over claim-vs-reality, side by side."""
    v = _VERDICT.get(payload["classification"], _VERDICT["UNGROUNDED"])
    st.markdown(
        f'<div class="wit-verdict" style="--vc:{v["color"]};--vbg:{v["bg"]}">'
        f'<div class="big">{v["icon"]} {v["label"]}</div>'
        f'<div class="gloss">— {v["gloss"]}</div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**What the agent claimed**")
        st.info(payload["claim_text"])
    with right:
        st.markdown("**What actually happened**")
        if payload.get("evidence_gap"):
            st.error(payload["evidence_gap"])
        else:
            st.success("Confirmed by the trace *and* the system's real state.")
    with st.expander("Show the evidence Witness checked"):
        st.write("**Trace evidence** — what the tools reported:")
        st.json(payload.get("trace_evidence"))
        st.write("**System evidence** — what the real system state says:")
        st.json(payload.get("system_evidence"))

    if payload["classification"] == "UNGROUNDED":
        st.markdown(
            "> **This is the whole thesis.** The agent was honest. The tool lied. Only a direct "
            "query against the system's real state — not the agent's word, not another AI's "
            "opinion — reveals the gap. Every verdict comes with the evidence trail above, ready "
            "for an auditor."
        )


# --------------------------------------------------------------------------- #
# 2. The three catches
# --------------------------------------------------------------------------- #

def render_three_catches(runs: list[TraceRun]) -> None:
    st.markdown(
        '<div class="wit-hero"><div class="tag">The three failure modes</div>'
        "<h1>Three ways an agent fleet goes wrong</h1>"
        "<p>Each scenario below is engineered to trip exactly one governance signal — and every "
        "catch is genuine, not staged.</p></div>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs([
        "❌  Hallucination", "🚨  Policy violation", "📈  Behavioral drift",
    ])

    with tab1:
        st.subheader("A claim with no reality behind it")
        st.caption("`ticket_filer` reports a ticket that the backend silently never saved.")
        run = _run_for_scenario(runs, "hallucination")
        if run:
            for e in run.events_of_type(EventType.GROUNDING_RESULT):
                _render_verdict_panel(e.payload)

    with tab2:
        st.subheader("A rule broken, flagged with the offending evidence")
        st.caption(
            "A misconfigured `report_generator` — its prompt simply never mentioned the rules — "
            "emails a customer their full profile, SSN and all, skipping approval."
        )
        run = _run_for_scenario(runs, "policy_violation")
        if run:
            viols = run.events_of_type(EventType.POLICY_VIOLATION)
            if not viols:
                st.info("No policy violations recorded for this run.")
            for e in viols:
                st.error(
                    f"**{e.payload['rule_name']}** · _{e.payload['severity']}_ — "
                    f"{e.payload['description']}"
                )
            st.markdown(
                "These fired **after the fact, from the trace** — the independent policy layer "
                "catches the violation no matter how carefully any single agent's prompt was "
                "written."
            )

    with tab3:
        st.subheader("Behavior that diverged from its own baseline")
        st.caption(
            "`data_lookup` — which only ever searched the database across 20 baseline runs — "
            "suddenly starts sending email."
        )
        run = _run_for_scenario(runs, "drift")
        if run:
            alerts = run.events_of_type(EventType.DRIFT_ALERT)
            if not alerts:
                st.info("No drift alert recorded for this run.")
            for e in alerts:
                st.warning(f"**Drift alert** — {e.payload['reason']}")
                st.caption(
                    f"Tool-usage distance from baseline: {e.payload['distance']:.2f} "
                    f"(threshold {config.DRIFT_DISTANCE_THRESHOLD}). "
                    "Under the distance threshold — the z-score channel on cost and call counts "
                    "is what caught it. Two independent channels; either alone would have missed."
                )
            extra = run.events_of_type(EventType.POLICY_VIOLATION)
            if extra:
                st.markdown("**The same rogue action also tripped policy rules nobody scripted:**")
                for e in extra:
                    st.error(f"**{e.payload['rule_name']}** — {e.payload['description']}")
                st.caption("Drift and policy failures compound in one incident. That's the whole "
                           "argument for running every check together.")


# --------------------------------------------------------------------------- #
# 3. The fleet dashboard
# --------------------------------------------------------------------------- #

def _score_style(score: int) -> dict:
    if score < 40:
        return _VERDICT["UNGROUNDED"]
    if score < 70:
        return _VERDICT["CONTRADICTED"]
    return _VERDICT["GROUNDED"]


def render_fleet_dashboard(report: AuditReport, runs: list[TraceRun]) -> None:
    st.markdown(
        '<div class="wit-hero"><div class="tag">Fleet governance readiness</div>'
        "<h1>The whole fleet, scored</h1>"
        "<p>Every catch rolls up into one transparent number you can argue with — because a "
        "score you can question is worth more than one you must take on faith.</p></div>",
        unsafe_allow_html=True,
    )

    total_issues = sum(len(a.claim_issues) for a in report.agent_summaries)
    total_violations = sum(len(a.policy_violations) for a in report.agent_summaries)
    total_drift = sum(len(a.drift_alerts) for a in report.agent_summaries)

    sc = _score_style(report.readiness_score)
    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f'<div class="wit-score" style="--vc:{sc["color"]};--vbg:{sc["bg"]}">'
            f'<div class="n">{report.readiness_score}</div>'
            f'<div class="l">Readiness / 100</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        m = st.columns(2)
        m[0].metric("Runs analyzed", report.total_runs)
        m[1].metric("Notional cost", f"${report.total_cost_usd:.4f}")
        m2 = st.columns(3)
        m2[0].metric("Claim issues", total_issues)
        m2[1].metric("Policy violations", total_violations)
        m2[2].metric("Drift alerts", total_drift)

    with st.expander("How the score is calculated"):
        st.write(f"Starts at **{config.SCORE_START}**, deducts weighted points per violation class:")
        for name, weight in config.SCORE_DEDUCTIONS.items():
            st.write(f"- **{name.replace('_', ' ')}**: −{weight} each")
        st.write("Floored at 0. Weights live in `config.SCORE_DEDUCTIONS` — change one number to "
                 "disagree.")

    st.subheader("Per-agent breakdown")
    if not report.agent_summaries:
        st.info("No agents to show yet.")
        return
    rows = []
    for a in report.agent_summaries:
        ungrounded = sum(1 for c in a.claim_issues if c.classification == "UNGROUNDED")
        contradicted = sum(1 for c in a.claim_issues if c.classification == "CONTRADICTED")
        rows.append({
            "Agent": a.agent_name,
            "Runs": a.run_count,
            "Cost ($)": round(a.total_cost_usd, 5),
            "Score": a.readiness_score,
            "Grounded": a.grounded_claim_count,
            "Ungrounded": ungrounded,
            "Contradicted": contradicted,
            "Policy": len(a.policy_violations),
            "Drift": len(a.drift_alerts),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# 4. Raw run inspector
# --------------------------------------------------------------------------- #

def render_run_timeline(run: TraceRun) -> None:
    for e in run.events:
        icon = _EVENT_ICON.get(e.event_type, "•")
        label = e.event_type.value
        if e.event_type is EventType.TOOL_CALL:
            detail = f"`{e.payload.get('tool_name')}` args={e.payload.get('args')}"
            if not e.payload.get("ok"):
                st.error(f"{icon} **{label}** {detail} — FAILED: {e.payload.get('error')}")
            else:
                st.write(f"{icon} **{label}** {detail} → {e.payload.get('result')}")
        elif e.event_type is EventType.POLICY_VIOLATION:
            st.error(f"{icon} **{e.payload['rule_name']}** ({e.payload['severity']}): "
                     f"{e.payload['description']}")
        elif e.event_type is EventType.GROUNDING_RESULT:
            v = _VERDICT.get(e.payload["classification"], _VERDICT["UNGROUNDED"])
            st.markdown(f"{icon} **{label}** — :{('red' if v['label']=='UNGROUNDED' else 'orange' if v['label']=='CONTRADICTED' else 'green')}[{e.payload['classification']}] "
                        f"\"{e.payload['claim_text']}\"")
        elif e.event_type is EventType.DRIFT_ALERT:
            st.warning(f"{icon} **{label}**: {e.payload['reason']} "
                       f"(distance={e.payload['distance']:.2f})")
        elif e.event_type is EventType.LLM_CALL:
            summary = e.payload.get("response_text") or (
                f"function_call: {e.payload.get('function_call', {}).get('name')}")
            replayed = " (replayed)" if e.payload.get("replayed") else ""
            st.write(f"{icon} **{label}**{replayed}: {summary}")
        elif e.event_type is EventType.CLAIM:
            st.write(f"{icon} **{label}** ({e.payload['claim_type']}): "
                     f"\"{e.payload['claim_text']}\"")
        else:
            st.write(f"{icon} **{label}**: {e.payload}")


def render_run_inspector(runs: list[TraceRun]) -> None:
    st.markdown(
        '<div class="wit-hero"><div class="tag">Raw trace</div>'
        "<h1>Run inspector</h1>"
        "<p>Every run is a durable, append-only trace. Pick one and read the black box "
        "event by event.</p></div>",
        unsafe_allow_html=True,
    )
    if not runs:
        st.info("No runs to inspect yet.")
        return

    # Put the interesting scenarios first; the 20 baseline runs are near-identical.
    def _rank(r: TraceRun) -> tuple[int, str]:
        priority = {"hallucination": 0, "policy_violation": 1, "drift": 2, "clean_run": 3}
        return (priority.get(r.scenario or "", 4), r.run_id)

    ordered = sorted(runs, key=_rank)
    options = {
        f"{r.scenario or 'no scenario'} — {r.agent_name} — {r.outcome} ({r.run_id[:10]})": r.run_id
        for r in ordered
    }
    selected_label = st.selectbox("Choose a run", list(options.keys()))
    selected_run = next(r for r in runs if r.run_id == options[selected_label])

    st.markdown(
        f'<div class="wit-pillrow">'
        f'<span class="wit-pill">agent · {html.escape(selected_run.agent_name)}</span>'
        f'<span class="wit-pill">outcome · {html.escape(selected_run.outcome or "?")}</span>'
        f'<span class="wit-pill">cost · ${selected_run.total_cost_usd:.5f}</span>'
        f'<span class="wit-pill">events · {len(selected_run.events)}</span></div>',
        unsafe_allow_html=True,
    )
    tab_grounding, tab_timeline = st.tabs(["Grounding panel", "Event timeline"])
    with tab_grounding:
        grounding = selected_run.events_of_type(EventType.GROUNDING_RESULT)
        if not grounding:
            st.info("No claims were extracted from this run's final message.")
        for e in grounding:
            _render_verdict_panel(e.payload)
    with tab_timeline:
        render_run_timeline(selected_run)


# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #

def render_thesis() -> None:
    cols = st.columns(3)
    cards = [
        ("🎥 Records everything", "Every LLM call and tool call becomes an immutable event in "
         "an append-only trace — a flight recorder for the agent."),
        ("🔎 Verifies against reality", "It checks each of the agent's claims against the real "
         "state of the systems it touched — not the agent's word, not the tool's self-report."),
        ("⚖️ No AI judging AI", "Claims are pulled out with deterministic pattern-matching and "
         "checked against hard facts. Every verdict ships with an auditable evidence trail."),
    ]
    for col, (k, v) in zip(cols, cards, strict=False):
        col.markdown(f'<div class="wit-tri"><div class="k">{k}</div>'
                     f'<div class="v">{v}</div></div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Witness — see what your agents actually did",
                       page_icon="🔍", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    runs_dir_marker = str(sorted(config.RUNS_DIR.glob("*.jsonl")))
    runs = _load_runs(runs_dir_marker)

    if not runs:
        with st.spinner(
            "First load — generating the demo from committed cassettes "
            "(offline, no API key needed). Takes a few seconds…"
        ):
            from scripts.run_demo import generate_demo

            generate_demo(verbose=False)
        _load_runs.clear()
        runs = _load_runs(str(sorted(config.RUNS_DIR.glob("*.jsonl"))))

    report = build_audit_report(runs)

    with st.sidebar:
        st.markdown("## 🔍 Witness")
        st.caption("Runtime governance for AI agent fleets.")
        page = st.radio(
            "Navigate",
            [
                "▶  Watch it catch a lie",
                "The three catches",
                "Fleet dashboard",
                "Run inspector (raw)",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.metric("Readiness score", f"{report.readiness_score}/100")
        st.caption(
            "New here? Start with **Watch it catch a lie** — a 6-step walkthrough of the "
            "flagship failure, built from real trace data."
        )

    if page.startswith("▶"):
        render_simulation(runs)
        st.divider()
        st.markdown("#### What Witness is doing underneath")
        render_thesis()
    elif page == "The three catches":
        render_three_catches(runs)
    elif page == "Fleet dashboard":
        render_fleet_dashboard(report, runs)
    else:
        render_run_inspector(runs)


if __name__ == "__main__":
    main()
