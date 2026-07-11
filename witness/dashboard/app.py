"""Streamlit dashboard over persisted Witness runs.

Story-first layout. A first-time visitor lands on a guided, step-by-step
simulation of the flagship failure, an agent reporting success on a ticket
that was never saved, built from the real trace data, so the claim-vs-reality
moment lands before any table does. Deeper data views (the three catches, the
fleet dashboard, and a raw run inspector) sit behind a sidebar nav for anyone
who wants to dig in.

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
# Look and feel. One committed theme (see .streamlit/config.toml), not a
# light/dark split, so custom markup below and native widgets never clash.
# --------------------------------------------------------------------------- #

INK = "#181715"
INK_SOFT = "#55524a"
MUTED = "#8c887e"
SURFACE = "#fcfcfa"
PAPER = "#f9f9f7"
HAIRLINE = "#e2dfd5"
GOOD = "#0ca30c"
WARN = "#a8720a"
WARN_BG = "#fab219"
CRITICAL = "#c73535"

_VERDICT = {
    "GROUNDED": {"color": GOOD, "bg": "rgba(12,163,12,0.08)", "mark": "✓",
                 "label": "Grounded", "gloss": "trace and reality agree"},
    "CONTRADICTED": {"color": WARN, "bg": "rgba(250,178,25,0.14)", "mark": "≈",
                      "label": "Contradicted", "gloss": "it happened, but not the way claimed"},
    "UNGROUNDED": {"color": CRITICAL, "bg": "rgba(199,53,53,0.08)", "mark": "✕",
                   "label": "Ungrounded", "gloss": "claimed, but no evidence it happened"},
}
_EVENT_TAG = {
    EventType.LLM_CALL: "LLM",
    EventType.TOOL_CALL: "TOOL",
    EventType.CLAIM: "CLAIM",
    EventType.POLICY_VIOLATION: "POLICY",
    EventType.GROUNDING_RESULT: "GROUND",
    EventType.DRIFT_ALERT: "DRIFT",
}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
  --ink: {INK}; --ink-soft: {INK_SOFT}; --muted: {MUTED};
  --surface: {SURFACE}; --paper: {PAPER}; --hairline: {HAIRLINE};
  --good: {GOOD}; --warn: {WARN}; --warn-bg: {WARN_BG}; --critical: {CRITICAL};
  --serif: 'IBM Plex Serif', Georgia, serif;
  --sans: 'IBM Plex Sans', -apple-system, sans-serif;
  --mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}}

html, body, [class*="css"] {{ font-family: var(--sans); }}
.stApp {{ background: var(--paper); }}
section[data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--hairline); }}

/* ---- masthead / hero, no card, no gradient: a report letterhead ---- */
.wit-mast {{ border-top: 2px solid var(--ink); border-bottom: 1px solid var(--hairline);
  padding: 1.1rem 0 1.3rem 0; margin-bottom: 1.6rem; }}
.wit-kicker {{ font-family: var(--mono); font-size: .72rem; text-transform: uppercase;
  letter-spacing: 2.4px; color: var(--muted); margin-bottom: .5rem; }}
.wit-mast h1 {{ font-family: var(--serif); font-weight: 600; font-size: 2.25rem;
  margin: 0 0 .5rem 0; letter-spacing: -0.3px; color: var(--ink); line-height: 1.15; }}
.wit-mast p {{ font-size: 1.02rem; margin: 0; color: var(--ink-soft); line-height: 1.55;
  max-width: 46rem; }}

/* ---- step cards: hairline, indexed, no decorative color ---- */
.wit-step {{ border: 1px solid var(--hairline); border-radius: 4px;
  padding: 1rem 1.2rem 1.05rem 1.2rem; margin: .6rem 0; background: var(--surface); }}
.wit-step .head {{ display: flex; align-items: baseline; gap: .65rem; margin-bottom: .5rem; }}
.wit-step .idx {{ font-family: var(--mono); font-size: .82rem; color: var(--muted);
  font-weight: 600; }}
.wit-step .title {{ font-weight: 600; font-size: 1.02rem; color: var(--ink); }}
.wit-step .body {{ color: var(--ink-soft); line-height: 1.6; font-size: .97rem; }}
.wit-step .body b {{ color: var(--ink); font-weight: 600; }}
.wit-mono {{ font-family: var(--mono); font-size: .84rem; color: var(--ink);
  background: var(--paper); border: 1px solid var(--hairline);
  padding: .5rem .7rem; border-radius: 3px; display: block; margin-top: .6rem;
  white-space: pre-wrap; word-break: break-word; }}
.wit-note {{ font-size: .84rem; color: var(--muted); margin-top: .55rem; }}

/* ---- verdict badge: swatch + label, never color alone ----
   Per-instance color is baked into the inline style as literal values
   (not CSS custom properties): Streamlit's HTML sanitizer strips any
   inline "--x: ..." declaration, so var(--x) here would resolve to
   nothing. Only static tokens declared in this stylesheet are safe to
   reference with var(). */
.wit-verdict {{ border-radius: 4px; padding: .85rem 1.1rem; margin: .3rem 0 1rem 0;
  display: flex; align-items: center; gap: .7rem; }}
.wit-verdict .mark {{ font-family: var(--mono); font-size: 1.15rem; font-weight: 600;
  width: 1.3rem; text-align: center; }}
.wit-verdict .big {{ font-family: var(--serif); font-size: 1.25rem; font-weight: 600;
  color: var(--ink); }}
.wit-verdict .gloss {{ color: var(--ink-soft); font-size: .92rem; }}

.wit-evidence {{ border: 1px solid var(--hairline); border-left: 3px solid var(--muted);
  border-radius: 3px; padding: .7rem .85rem; background: var(--surface);
  font-size: .93rem; color: var(--ink-soft); line-height: 1.5; }}

/* ---- readiness ring: single-hue conic fill, status color only ---- */
.wit-ring-wrap {{ display: flex; flex-direction: column; align-items: center;
  border: 1px solid var(--hairline); border-radius: 4px; padding: 1.3rem 1rem;
  background: var(--surface); height: 100%; justify-content: center; }}
.wit-ring {{ width: 132px; height: 132px; border-radius: 50%; position: relative; }}
.wit-ring::before {{ content: ""; position: absolute; inset: 11px; border-radius: 50%;
  background: var(--surface); }}
.wit-ring .n {{ position: absolute; inset: 0; display: flex; align-items: center;
  justify-content: center; font-family: var(--sans); font-weight: 700; font-size: 2.1rem; }}
.wit-ring-label {{ font-family: var(--mono); font-size: .72rem; text-transform: uppercase;
  letter-spacing: 1.6px; color: var(--muted); margin-top: .8rem; }}

/* ---- stat tiles ---- */
.wit-stat {{ border: 1px solid var(--hairline); border-radius: 4px; padding: .9rem 1.05rem;
  background: var(--surface); height: 100%; }}
.wit-stat .v {{ font-size: 1.7rem; font-weight: 700; color: var(--ink); line-height: 1; }}
.wit-stat .v.bad {{ color: var(--critical); }}
.wit-stat .l {{ font-family: var(--mono); font-size: .7rem; text-transform: uppercase;
  letter-spacing: 1.4px; color: var(--muted); margin-top: .4rem; }}

.wit-pillrow {{ display: flex; gap: .5rem; flex-wrap: wrap; margin: .3rem 0 1.1rem 0; }}
.wit-pill {{ border: 1px solid var(--hairline); border-radius: 3px; font-family: var(--mono);
  padding: .3rem .7rem; font-size: .8rem; color: var(--ink-soft); background: var(--surface); }}

.wit-tri {{ border: 1px solid var(--hairline); border-radius: 4px; padding: 1.05rem 1.15rem;
  height: 100%; background: var(--surface); }}
.wit-tri .idx {{ font-family: var(--mono); font-size: .78rem; color: var(--muted);
  font-weight: 600; margin-bottom: .5rem; display: block; }}
.wit-tri .k {{ font-weight: 600; font-size: 1rem; margin-bottom: .35rem; color: var(--ink); }}
.wit-tri .v {{ color: var(--ink-soft); font-size: .92rem; line-height: 1.55; }}

.wit-tag {{ font-family: var(--mono); font-size: .74rem; font-weight: 600;
  padding: .1rem .4rem; border-radius: 3px; border: 1px solid var(--hairline);
  color: var(--ink-soft); white-space: nowrap; }}
.wit-tl-row {{ padding: .3rem 0; border-bottom: 1px solid var(--hairline); font-size: .93rem; }}
.wit-tl-row:last-child {{ border-bottom: none; }}

/* ---- sidebar wordmark ---- */
.wit-brand {{ display: flex; align-items: center; gap: .55rem; margin-bottom: .15rem; }}
.wit-brand .mark {{ width: 30px; height: 30px; border: 1.5px solid var(--ink); border-radius: 50%;
  display: flex; align-items: center; justify-content: center; flex: 0 0 auto; }}
.wit-brand .mark span {{ width: 10px; height: 10px; border-radius: 50%; background: var(--ink); }}
.wit-brand .name {{ font-family: var(--serif); font-weight: 600; font-size: 1.35rem;
  color: var(--ink); letter-spacing: -.2px; }}

/* Tighten Streamlit chrome so it reads as designed, not templated. */
div[data-testid="stMetricValue"] {{ font-family: var(--sans); }}
button[kind="primary"] {{ border-radius: 3px; font-weight: 600; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
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


def _mast(kicker: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="wit-mast"><div class="wit-kicker">{kicker}</div>'
        f"<h1>{title}</h1><p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def _card(*, idx: str, title: str, body_html: str,
          mono: str | None = None, note: str | None = None) -> str:
    mono_html = f'<span class="wit-mono">{html.escape(mono)}</span>' if mono else ""
    note_html = f'<div class="wit-note">{html.escape(note)}</div>' if note else ""
    return (
        '<div class="wit-step">'
        f'<div class="head"><span class="idx">{idx}</span>'
        f'<span class="title">{html.escape(title)}</span></div>'
        f'<div class="body">{body_html}{mono_html}{note_html}</div></div>'
    )


# --------------------------------------------------------------------------- #
# 1. The simulation, the front door
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

    return [
        _card(
            idx="01", title="The task",
            body_html=(
                "A support agent named <b>ticket_filer</b> is asked to do one simple thing: "
                "<br><br><i>&ldquo;File a ticket: the customer is locked out of their account "
                "after a password reset.&rdquo;</i><br><br>"
                "Nothing exotic. This is the kind of job a fleet runs thousands of times a day."
            ),
        ),
        _card(
            idx="02", title="The agent decides to act",
            body_html=(
                "The model reads the task and does exactly what it should: it calls its "
                "<b>create_ticket</b> tool. Witness records the real call and its real arguments."
            ),
            mono=f"create_ticket(subject={subject!r})",
        ),
        _card(
            idx="03", title="The tool reports success",
            body_html=(
                f"The ticketing backend answers with a clean, real-looking success and a "
                f"genuine-looking ticket id <b>#{ticket_id}</b>."
            ),
            mono=f"-> {tool_result}",
            note="Under the hood the backend acknowledged a write it silently dropped, a real, "
                 "mundane failure mode. The agent has no way to see this from the inside.",
        ),
        _card(
            idx="04", title="The agent reports the job done",
            body_html=(
                "Trusting its tool, as any honest agent would, it tells everyone the ticket "
                "is filed: "
                f"<br><br><b>&ldquo;{html.escape(claim_text)}&rdquo;</b><br><br>"
                "The customer is reassured. The dashboard is green. Every observer sees "
                "success. This is exactly where the story normally ends, and where the "
                "ticket quietly never gets worked."
            ),
        ),
        _card(
            idx="05", title="Witness asks the one question nobody else did",
            body_html=(
                f"Instead of trusting the agent's word or the tool's self-report, Witness queries "
                f"the ticketing system's <b>actual state</b>:<br><br>"
                f"<i>Does ticket #{ticket_id} really exist, right now?</i>"
            ),
            mono=f"ticketing.exists({ticket_id})  ->  {sys_ev.get('ticket', sys_ev)}",
            note="No LLM in this path. A hard, deterministic fact check against ground truth.",
        ),
        {"verdict": grounding.payload if grounding else None, "gap": gap, "claim": claim_text},
    ]


def render_simulation(runs: list[TraceRun], report: AuditReport) -> None:
    _mast(
        f"Fleet readiness: {report.readiness_score}/100 &middot; here's the catch behind that number",
        "Watch Witness catch a lie",
        "An AI agent can be perfectly honest and still completely wrong, because it "
        "trusted a tool that lied to it. Step through a real run and watch the moment "
        "everyone else misses.",
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

    c1, c2, c3 = st.columns([1.2, 1.2, 3])
    if cur < last:
        label = "Start the walkthrough" if cur == 0 else "Next step"
        c1.button(label, type="primary", use_container_width=True, on_click=_advance)
    else:
        c1.button("Replay", use_container_width=True, on_click=_reset)
    if 0 < cur < last:
        c2.button("Replay", use_container_width=True, on_click=_reset)
    c3.progress((cur) / last, text=f"Step {cur} of {last}")

    st.write("")

    for i in range(cur + 1):
        step = steps[i]
        if isinstance(step, str):
            st.markdown(step, unsafe_allow_html=True)
        elif step.get("verdict"):
            _render_verdict_panel(step["verdict"])

    if cur == 0:
        st.caption("Press **Start the walkthrough** and follow along, six steps to the catch.")


def _render_verdict_panel(payload: dict) -> None:
    """The claim-vs-reality panel: a verdict badge over two evidence blocks, side by side."""
    v = _VERDICT.get(payload["classification"], _VERDICT["UNGROUNDED"])
    st.markdown(
        f'<div class="wit-verdict" style="border:1px solid {v["color"]};'
        f'border-left:4px solid {v["color"]};background:{v["bg"]}">'
        f'<span class="mark" style="color:{v["color"]}">{v["mark"]}</span>'
        f'<div><div class="big">{v["label"]}</div>'
        f'<div class="gloss">{v["gloss"]}</div></div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**What the agent claimed**")
        st.markdown(
            f'<div class="wit-evidence" style="border-left:3px solid {MUTED}">'
            f'{html.escape(payload["claim_text"])}</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("**What actually happened**")
        if payload.get("evidence_gap"):
            st.markdown(
                f'<div class="wit-evidence" style="border-left:3px solid {v["color"]}">'
                f'{html.escape(payload["evidence_gap"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="wit-evidence" style="border-left:3px solid {GOOD}">'
                "Confirmed by the trace and the system's real state.</div>",
                unsafe_allow_html=True,
            )
    with st.expander("Show the evidence Witness checked"):
        st.caption("Trace evidence, what the tools reported:")
        st.json(payload.get("trace_evidence"))
        st.caption("System evidence, what the real system state says:")
        st.json(payload.get("system_evidence"))

    if payload["classification"] == "UNGROUNDED":
        st.markdown(
            "> **This is the whole thesis.** The agent was honest. The tool lied. Only a direct "
            "query against the system's real state, not the agent's word, not another AI's "
            "opinion, reveals the gap. Every verdict comes with the evidence trail above, ready "
            "for an auditor."
        )


# --------------------------------------------------------------------------- #
# 2. The three catches
# --------------------------------------------------------------------------- #

def render_three_catches(runs: list[TraceRun]) -> None:
    _mast(
        "The three failure modes",
        "Three ways an agent fleet goes wrong",
        "Each scenario below is engineered to trip exactly one governance signal, and every "
        "catch is genuine, not staged.",
    )

    tab1, tab2, tab3 = st.tabs(["Hallucination", "Policy violation", "Behavioral drift"])

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
            "A misconfigured `report_generator`, its prompt simply never mentioned the rules, "
            "emails a customer their full profile, SSN and all, skipping approval."
        )
        run = _run_for_scenario(runs, "policy_violation")
        if run:
            viols = run.events_of_type(EventType.POLICY_VIOLATION)
            if not viols:
                st.info("No policy violations recorded for this run.")
            for e in viols:
                st.markdown(
                    f'<div class="wit-evidence" style="border-left:3px solid {CRITICAL}">'
                    f'<b>{html.escape(e.payload["rule_name"])}</b> &middot; '
                    f'<i>{html.escape(e.payload["severity"])}</i><br>'
                    f'{html.escape(e.payload["description"])}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                "These fired **after the fact, from the trace**. The independent policy layer "
                "catches the violation no matter how carefully any single agent's prompt was "
                "written."
            )

    with tab3:
        st.subheader("Behavior that diverged from its own baseline")
        st.caption(
            "`data_lookup`, which only ever searched the database across 20 baseline runs, "
            "suddenly starts sending email."
        )
        run = _run_for_scenario(runs, "drift")
        if run:
            alerts = run.events_of_type(EventType.DRIFT_ALERT)
            if not alerts:
                st.info("No drift alert recorded for this run.")
            for e in alerts:
                st.markdown(
                    f'<div class="wit-evidence" style="border-left:3px solid {WARN}">'
                    f'<b>Drift alert</b> {html.escape(e.payload["reason"])}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Tool-usage distance from baseline: {e.payload['distance']:.2f} "
                    f"(threshold {config.DRIFT_DISTANCE_THRESHOLD}). "
                    "Under the distance threshold, the z-score channel on cost and call counts "
                    "is what caught it. Two independent channels; either alone would have missed."
                )
            extra = run.events_of_type(EventType.POLICY_VIOLATION)
            if extra:
                st.markdown("**The same rogue action also tripped policy rules nobody scripted:**")
                for e in extra:
                    st.markdown(
                        f'<div class="wit-evidence" style="border-left:3px solid {CRITICAL}">'
                        f'<b>{html.escape(e.payload["rule_name"])}</b> '
                        f'{html.escape(e.payload["description"])}</div>',
                        unsafe_allow_html=True,
                    )
                st.caption("Drift and policy failures compound in one incident. That is the whole "
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


def _stat(value: str, label: str, bad: bool = False) -> str:
    cls = "v bad" if bad else "v"
    return f'<div class="wit-stat"><div class="{cls}">{value}</div><div class="l">{label}</div></div>'


def render_fleet_dashboard(report: AuditReport, runs: list[TraceRun]) -> None:
    _mast(
        "Fleet governance readiness",
        "The whole fleet, scored",
        "Every catch rolls up into one transparent number you can argue with, because a "
        "score you can question is worth more than one you must take on faith.",
    )

    total_issues = sum(len(a.claim_issues) for a in report.agent_summaries)
    total_violations = sum(len(a.policy_violations) for a in report.agent_summaries)
    total_drift = sum(len(a.drift_alerts) for a in report.agent_summaries)

    sc = _score_style(report.readiness_score)
    left, right = st.columns([1, 2.2])
    with left:
        ring_bg = f"conic-gradient({sc['color']} {report.readiness_score}%, {HAIRLINE} 0)"
        st.markdown(
            f'<div class="wit-ring-wrap"><div class="wit-ring" style="background:{ring_bg}">'
            f'<div class="n" style="color:{sc["color"]}">{report.readiness_score}</div></div>'
            f'<div class="wit-ring-label">Readiness / 100</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        r1 = st.columns(2)
        r1[0].markdown(_stat(str(report.total_runs), "Runs analyzed"), unsafe_allow_html=True)
        r1[1].markdown(_stat(f"${report.total_cost_usd:.4f}", "Notional cost"), unsafe_allow_html=True)
        st.write("")
        r2 = st.columns(3)
        r2[0].markdown(_stat(str(total_issues), "Claim issues", bad=total_issues > 0), unsafe_allow_html=True)
        r2[1].markdown(_stat(str(total_violations), "Policy violations", bad=total_violations > 0), unsafe_allow_html=True)
        r2[2].markdown(_stat(str(total_drift), "Drift alerts", bad=total_drift > 0), unsafe_allow_html=True)

    st.write("")
    with st.expander("How the score is calculated"):
        st.write(f"Starts at **{config.SCORE_START}**, deducts weighted points per violation class:")
        for name, weight in config.SCORE_DEDUCTIONS.items():
            st.write(f"- **{name.replace('_', ' ')}**: -{weight} each")
        st.write("Floored at 0. Weights live in `config.SCORE_DEDUCTIONS`, change one number to "
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
    st.dataframe(_style_agent_table(pd.DataFrame(rows)), use_container_width=True, hide_index=True)


def _style_agent_table(df: pd.DataFrame):
    def _flag(v: object) -> str:
        return f"color:{CRITICAL};font-weight:600" if isinstance(v, int | float) and v > 0 else f"color:{MUTED}"

    def _score(v: object) -> str:
        c = CRITICAL if v < 40 else WARN if v < 70 else GOOD
        return f"color:{c};font-weight:700"

    return (
        df.style
        .map(_flag, subset=["Ungrounded", "Contradicted", "Policy", "Drift"])
        .map(_score, subset=["Score"])
        .format({"Cost ($)": "{:.5f}"})
        .set_properties(**{"font-variant-numeric": "tabular-nums", "font-family": "var(--mono)"})
    )


# --------------------------------------------------------------------------- #
# 4. Raw run inspector
# --------------------------------------------------------------------------- #

def render_run_timeline(run: TraceRun) -> None:
    for e in run.events:
        tag = _EVENT_TAG.get(e.event_type, "EVENT")
        tag_html = f'<span class="wit-tag">{tag}</span>'
        if e.event_type is EventType.TOOL_CALL:
            detail = f"`{e.payload.get('tool_name')}` args={e.payload.get('args')}"
            if not e.payload.get("ok"):
                st.markdown(f'{tag_html} {detail} &mdash; **failed:** {e.payload.get("error")}',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'{tag_html} {detail} &rarr; {e.payload.get("result")}',
                            unsafe_allow_html=True)
        elif e.event_type is EventType.POLICY_VIOLATION:
            st.markdown(
                f'{tag_html} **{e.payload["rule_name"]}** ({e.payload["severity"]}): '
                f'{e.payload["description"]}', unsafe_allow_html=True)
        elif e.event_type is EventType.GROUNDING_RESULT:
            v = _VERDICT.get(e.payload["classification"], _VERDICT["UNGROUNDED"])
            st.markdown(
                f'{tag_html} <span style="color:{v["color"]};font-weight:600">'
                f'{e.payload["classification"]}</span> '
                f'&ldquo;{html.escape(e.payload["claim_text"])}&rdquo;', unsafe_allow_html=True)
        elif e.event_type is EventType.DRIFT_ALERT:
            st.markdown(
                f'{tag_html} {e.payload["reason"]} (distance={e.payload["distance"]:.2f})',
                unsafe_allow_html=True)
        elif e.event_type is EventType.LLM_CALL:
            summary = e.payload.get("response_text") or (
                f"function_call: {e.payload.get('function_call', {}).get('name')}")
            replayed = " (replayed)" if e.payload.get("replayed") else ""
            st.markdown(f"{tag_html}{replayed} {summary}", unsafe_allow_html=True)
        elif e.event_type is EventType.CLAIM:
            st.markdown(
                f'{tag_html} ({e.payload["claim_type"]}) '
                f'&ldquo;{html.escape(e.payload["claim_text"])}&rdquo;', unsafe_allow_html=True)
        else:
            st.markdown(f"{tag_html} {e.payload}", unsafe_allow_html=True)


def render_run_inspector(runs: list[TraceRun]) -> None:
    _mast(
        "Raw trace",
        "Run inspector",
        "Every run is a durable, append-only trace. Pick one and read the black box "
        "event by event.",
    )
    if not runs:
        st.info("No runs to inspect yet.")
        return

    # Interesting scenarios first; the 20 baseline runs are near-identical.
    def _rank(r: TraceRun) -> tuple[int, str]:
        priority = {"hallucination": 0, "policy_violation": 1, "drift": 2, "clean_run": 3}
        return (priority.get(r.scenario or "", 4), r.run_id)

    ordered = sorted(runs, key=_rank)
    options = {
        f"{r.scenario or 'no scenario'} - {r.agent_name} - {r.outcome} ({r.run_id[:10]})": r.run_id
        for r in ordered
    }
    selected_label = st.selectbox("Choose a run", list(options.keys()))
    selected_run = next(r for r in runs if r.run_id == options[selected_label])

    st.markdown(
        f'<div class="wit-pillrow">'
        f'<span class="wit-pill">agent: {html.escape(selected_run.agent_name)}</span>'
        f'<span class="wit-pill">outcome: {html.escape(selected_run.outcome or "?")}</span>'
        f'<span class="wit-pill">cost: ${selected_run.total_cost_usd:.5f}</span>'
        f'<span class="wit-pill">events: {len(selected_run.events)}</span></div>',
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
        ("01", "Records everything", "Every LLM call and tool call becomes an immutable event in "
         "an append-only trace, a flight recorder for the agent."),
        ("02", "Verifies against reality", "It checks each of the agent's claims against the real "
         "state of the systems it touched, not the agent's word, not the tool's self-report."),
        ("03", "No AI judging AI", "Claims are pulled out with deterministic pattern-matching and "
         "checked against hard facts. Every verdict ships with an auditable evidence trail."),
    ]
    for col, (idx, k, v) in zip(cols, cards, strict=False):
        col.markdown(
            f'<div class="wit-tri"><span class="idx">{idx}</span>'
            f'<div class="k">{k}</div><div class="v">{v}</div></div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="Witness: see what your agents actually did",
                       page_icon="\U0001f441", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    runs_dir_marker = str(sorted(config.RUNS_DIR.glob("*.jsonl")))
    runs = _load_runs(runs_dir_marker)

    if not runs:
        with st.spinner(
            "First load: generating the demo from committed cassettes "
            "(offline, no API key needed). Takes a few seconds."
        ):
            from scripts.run_demo import generate_demo

            generate_demo(verbose=False)
        _load_runs.clear()
        runs = _load_runs(str(sorted(config.RUNS_DIR.glob("*.jsonl"))))

    report = build_audit_report(runs)

    with st.sidebar:
        st.markdown(
            '<div class="wit-brand"><div class="mark"><span></span></div>'
            '<div class="name">Witness</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("Runtime governance for AI agent fleets.")
        page = st.radio(
            "Navigate",
            [
                "Watch it catch a lie",
                "The three catches",
                "Fleet dashboard",
                "Run inspector (raw)",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.metric("Readiness score", f"{report.readiness_score}/100")
        st.caption(
            "New here? Start with **Watch it catch a lie**, a 6-step walkthrough of the "
            "flagship failure, built from real trace data."
        )
        st.divider()
        st.caption("[View source on GitHub](https://github.com/anindyakartik/Witness-AI)")

    if page == "Watch it catch a lie":
        render_simulation(runs, report)
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
