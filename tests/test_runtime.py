"""Tests for the agent runtime loop, exercised fully offline via replay cassettes.

Each test pre-seeds a cassette with the exact scripted Gemini responses for a
multi-turn conversation, then runs the real `run_agent` loop against real tools and
mock services in LLM_MODE="replay". This proves the plan -> call tool -> observe ->
repeat control flow, trace completeness, and allowlist enforcement without needing
a live GEMINI_API_KEY, only recording *new* cassettes needs one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from witness.agents.data_lookup import DATA_LOOKUP
from witness.agents.report_generator import REPORT_GENERATOR
from witness.agents.summarizer import SUMMARIZER
from witness.agents.ticket_filer import TICKET_FILER
from witness.core.llm import Cassette, LLMClient, _hash_request
from witness.core.runtime import run_agent
from witness.core.tools import build_default_registry
from witness.core.trace import EventType, TraceStore
from witness.mocks.database import CustomerDatabase
from witness.mocks.outbox import EmailOutbox
from witness.mocks.ticketing import TicketingSystem


def _seed(
    cassette: Cassette,
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tool_names: list[str],
    function_call: dict[str, Any] | None = None,
    text: str | None = None,
) -> None:
    req_hash = _hash_request(
        model=model, system=system, messages=messages, tool_names=tool_names, temperature=0.0
    )
    cassette.put(
        req_hash,
        {
            "text": text,
            "function_call": function_call,
            "prompt_tokens": 20,
            "completion_tokens": 8,
            "cost_usd": 0.0000015,
        },
    )


def _build_registry() -> tuple[Any, CustomerDatabase, TicketingSystem, EmailOutbox]:
    db = CustomerDatabase()
    ticketing = TicketingSystem(start_id=4470, degraded=False)
    outbox = EmailOutbox()
    return (
        build_default_registry(database=db, ticketing=ticketing, outbox=outbox),
        db,
        ticketing,
        outbox,
    )


def test_ticket_filer_clean_run_produces_valid_trace(tmp_path: Path) -> None:
    registry, _db, ticketing, _outbox = _build_registry()
    cassette_dir = tmp_path / "cassettes"
    cassette = Cassette(cassette_dir / "test.json")
    tool_names = list(TICKET_FILER.tool_allowlist)
    task = "File a ticket: customer was double-charged on their invoice."

    turn1_messages = [{"role": "user", "parts": [{"text": task}]}]
    function_call = {
        "name": "create_ticket",
        "args": {"subject": "Double charge", "body": "Customer was double-charged on invoice."},
    }
    _seed(
        cassette,
        model=LLMClient(cassette_name="x").model_name,
        system=TICKET_FILER.system_prompt,
        messages=turn1_messages,
        tool_names=tool_names,
        function_call=function_call,
    )

    turn2_messages = turn1_messages + [
        {"role": "model", "parts": [{"function_call": function_call}]},
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "create_ticket",
                        "response": {"ok": True, "ticket_id": 4470, "status": "open"},
                    }
                }
            ],
        },
    ]
    _seed(
        cassette,
        model=LLMClient(cassette_name="x").model_name,
        system=TICKET_FILER.system_prompt,
        messages=turn2_messages,
        tool_names=tool_names,
        text="Filed ticket #4470 for: Double charge.",
    )

    store = TraceStore(base_dir=tmp_path / "runs")
    llm = LLMClient(cassette_name="test", mode="replay", cassette_dir=cassette_dir)

    result = run_agent(TICKET_FILER, task, llm=llm, tools=registry, store=store, scenario="test")

    assert result.outcome == "success"
    assert result.final_message == "Filed ticket #4470 for: Double charge."
    assert ticketing.exists(4470)

    llm_events = result.run.events_of_type(EventType.LLM_CALL)
    tool_events = result.run.events_of_type(EventType.TOOL_CALL)
    assert len(llm_events) == 2
    assert len(tool_events) == 1
    assert tool_events[0].payload["tool_name"] == "create_ticket"
    assert tool_events[0].payload["ok"] is True
    assert all(e.payload["replayed"] for e in llm_events)

    # Round-trips from disk correctly too.
    loaded = store.load_run(result.run.run_id)
    assert loaded.outcome == "success"
    assert len(loaded.events) == 3  # 2 llm_call + 1 tool_call


def test_summarizer_clean_run(tmp_path: Path) -> None:
    registry, *_ = _build_registry()
    cassette_dir = tmp_path / "cassettes"
    cassette = Cassette(cassette_dir / "test.json")
    tool_names = list(SUMMARIZER.tool_allowlist)
    text_to_summarize = "Revenue increased by 12 percent. Customer churn decreased slightly."
    task = f"Summarize the following text: {text_to_summarize}"

    turn1_messages = [{"role": "user", "parts": [{"text": task}]}]
    function_call = {"name": "summarize", "args": {"text": text_to_summarize}}
    model_name = LLMClient(cassette_name="x").model_name
    _seed(
        cassette,
        model=model_name,
        system=SUMMARIZER.system_prompt,
        messages=turn1_messages,
        tool_names=tool_names,
        function_call=function_call,
    )

    turn2_messages = turn1_messages + [
        {"role": "model", "parts": [{"function_call": function_call}]},
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "summarize",
                        "response": {
                            "ok": True,
                            "summary": text_to_summarize,
                            "original_sentence_count": 2,
                        },
                    }
                }
            ],
        },
    ]
    _seed(
        cassette,
        model=model_name,
        system=SUMMARIZER.system_prompt,
        messages=turn2_messages,
        tool_names=tool_names,
        text=f"Summary: {text_to_summarize}",
    )

    store = TraceStore(base_dir=tmp_path / "runs")
    llm = LLMClient(cassette_name="test", mode="replay", cassette_dir=cassette_dir)

    result = run_agent(SUMMARIZER, task, llm=llm, tools=registry, store=store, scenario="test")

    assert result.outcome == "success"
    assert result.final_message == f"Summary: {text_to_summarize}"


def test_report_generator_clean_run_multi_step(tmp_path: Path) -> None:
    registry, _db, _ticketing, outbox = _build_registry()
    cassette_dir = tmp_path / "cassettes"
    cassette = Cassette(cassette_dir / "test.json")
    tool_names = list(REPORT_GENERATOR.tool_allowlist)
    task = "Send customer 1 their account report."
    model_name = LLMClient(cassette_name="x").model_name

    messages = [{"role": "user", "parts": [{"text": task}]}]
    _seed(
        cassette,
        model=model_name,
        system=REPORT_GENERATOR.system_prompt,
        messages=messages,
        tool_names=tool_names,
        function_call={"name": "get_customer_record", "args": {"customer_id": 1}},
    )
    messages = messages + [
        {
            "role": "model",
            "parts": [
                {"function_call": {"name": "get_customer_record", "args": {"customer_id": 1}}}
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "get_customer_record",
                        "response": {
                            "ok": True,
                            "id": 1,
                            "name": "Ravi Shah",
                            "email": "ravi.shah@example.com",
                            "phone": "555-0101",
                            "ssn": "123-45-6701",
                            "plan": "enterprise",
                        },
                    }
                }
            ],
        },
    ]
    _seed(
        cassette,
        model=model_name,
        system=REPORT_GENERATOR.system_prompt,
        messages=messages,
        tool_names=tool_names,
        function_call={
            "name": "request_approval",
            "args": {"action": "send customer report email"},
        },
    )
    messages = messages + [
        {
            "role": "model",
            "parts": [
                {
                    "function_call": {
                        "name": "request_approval",
                        "args": {"action": "send customer report email"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "request_approval",
                        "response": {
                            "ok": True,
                            "approved": True,
                            "action": "send customer report email",
                        },
                    }
                }
            ],
        },
    ]
    send_args = {
        "to": "ravi.shah@example.com",
        "subject": "Your Account Report",
        "body": "Here is your account summary.",
    }
    _seed(
        cassette,
        model=model_name,
        system=REPORT_GENERATOR.system_prompt,
        messages=messages,
        tool_names=tool_names,
        function_call={"name": "send_email", "args": send_args},
    )
    messages = messages + [
        {"role": "model", "parts": [{"function_call": {"name": "send_email", "args": send_args}}]},
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "send_email",
                        "response": {"ok": True, "email_id": 1},
                    }
                }
            ],
        },
    ]
    _seed(
        cassette,
        model=model_name,
        system=REPORT_GENERATOR.system_prompt,
        messages=messages,
        tool_names=tool_names,
        text="Emailed the report to ravi.shah@example.com.",
    )

    store = TraceStore(base_dir=tmp_path / "runs")
    llm = LLMClient(cassette_name="test", mode="replay", cassette_dir=cassette_dir)

    result = run_agent(
        REPORT_GENERATOR, task, llm=llm, tools=registry, store=store, scenario="test"
    )

    assert result.outcome == "success"
    assert result.final_message == "Emailed the report to ravi.shah@example.com."
    assert outbox.exists(to="ravi.shah@example.com")
    assert len(result.run.events_of_type(EventType.TOOL_CALL)) == 3
    assert len(result.run.events_of_type(EventType.LLM_CALL)) == 4


def test_tool_outside_allowlist_is_blocked_and_traced_not_executed(tmp_path: Path) -> None:
    registry, _db, _ticketing, outbox = _build_registry()
    cassette_dir = tmp_path / "cassettes"
    cassette = Cassette(cassette_dir / "test.json")
    tool_names = list(DATA_LOOKUP.tool_allowlist)  # does NOT include send_email
    task = "Look up Ravi Shah and email him a copy of his record."
    model_name = LLMClient(cassette_name="x").model_name

    turn1_messages = [{"role": "user", "parts": [{"text": task}]}]
    disallowed_call = {
        "name": "send_email",
        "args": {"to": "ravi.shah@example.com", "subject": "x", "body": "y"},
    }
    _seed(
        cassette,
        model=model_name,
        system=DATA_LOOKUP.system_prompt,
        messages=turn1_messages,
        tool_names=tool_names,
        function_call=disallowed_call,
    )

    turn2_messages = turn1_messages + [
        {"role": "model", "parts": [{"function_call": disallowed_call}]},
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "send_email",
                        "response": {
                            "ok": False,
                            "error": "tool 'send_email' not permitted for this agent",
                        },
                    }
                }
            ],
        },
    ]
    _seed(
        cassette,
        model=model_name,
        system=DATA_LOOKUP.system_prompt,
        messages=turn2_messages,
        tool_names=tool_names,
        text="I cannot send emails; that capability is not available to me.",
    )

    store = TraceStore(base_dir=tmp_path / "runs")
    llm = LLMClient(cassette_name="test", mode="replay", cassette_dir=cassette_dir)

    result = run_agent(DATA_LOOKUP, task, llm=llm, tools=registry, store=store, scenario="test")

    assert result.outcome == "success"
    # The real mock service must show no side effect: the block prevented execution.
    assert outbox.all_sent() == []

    tool_events = result.run.events_of_type(EventType.TOOL_CALL)
    assert len(tool_events) == 1
    assert tool_events[0].payload["tool_name"] == "send_email"
    assert tool_events[0].payload["ok"] is False


def test_tool_allowlist_override_permits_extra_tool_for_scenario_use(tmp_path: Path) -> None:
    """The drift scenario needs an agent to genuinely call a tool outside its
    canonical allowlist. The runtime must let that through when explicitly
    overridden, while governance rules (built later) still judge against the
    agent's real, declared allowlist, not the override."""
    registry, _db, _ticketing, outbox = _build_registry()
    cassette_dir = tmp_path / "cassettes"
    cassette = Cassette(cassette_dir / "test.json")
    override = (*DATA_LOOKUP.tool_allowlist, "send_email")
    task = "Look up Ravi Shah and email him a copy of his record."
    model_name = LLMClient(cassette_name="x").model_name

    turn1_messages = [{"role": "user", "parts": [{"text": task}]}]
    send_args = {"to": "ravi.shah@example.com", "subject": "Your record", "body": "See attached."}
    function_call = {"name": "send_email", "args": send_args}
    _seed(
        cassette,
        model=model_name,
        system=DATA_LOOKUP.system_prompt,
        messages=turn1_messages,
        tool_names=list(override),
        function_call=function_call,
    )
    turn2_messages = turn1_messages + [
        {"role": "model", "parts": [{"function_call": function_call}]},
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "send_email",
                        "response": {"ok": True, "email_id": 1},
                    }
                }
            ],
        },
    ]
    _seed(
        cassette,
        model=model_name,
        system=DATA_LOOKUP.system_prompt,
        messages=turn2_messages,
        tool_names=list(override),
        text="Emailed Ravi Shah his record.",
    )

    store = TraceStore(base_dir=tmp_path / "runs")
    llm = LLMClient(cassette_name="test", mode="replay", cassette_dir=cassette_dir)

    result = run_agent(
        DATA_LOOKUP,
        task,
        llm=llm,
        tools=registry,
        store=store,
        scenario="test",
        tool_allowlist_override=override,
    )

    assert result.outcome == "success"
    # The override genuinely let the side effect happen this run.
    assert outbox.exists(to="ravi.shah@example.com")
    tool_events = result.run.events_of_type(EventType.TOOL_CALL)
    assert tool_events[0].payload["ok"] is True
