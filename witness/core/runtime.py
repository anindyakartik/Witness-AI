"""Agent runtime: the tool-use loop that drives one agent through one task.

Every step is traced: LLMClient.generate() emits `llm_call` events and
ToolRegistry.invoke() emits `tool_call` events. This module's only job is the
plan -> call tool -> observe -> repeat control flow, and capturing the agent's
final natural-language message -- the claims the GroundingChecker later verifies.
It knows nothing about specific agents, policies, or scenarios beyond the generic
AgentDefinition contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import config
from witness.agents.base import AgentDefinition
from witness.core.llm import LLMClient
from witness.core.tools import ToolRegistry
from witness.core.trace import EventType, TraceEvent, TraceRun, TraceStore, tool_call_payload


@dataclass(frozen=True)
class RunResult:
    """The outcome of one agent execution."""

    run: TraceRun
    final_message: str | None
    outcome: str  # "success" | "max_steps" | "error"


def run_agent(
    agent: AgentDefinition,
    task: str,
    *,
    llm: LLMClient,
    tools: ToolRegistry,
    store: TraceStore,
    seed: int = config.SEED,
    scenario: str | None = None,
    max_steps: int = config.MAX_AGENT_STEPS,
    tool_allowlist_override: tuple[str, ...] | None = None,
) -> RunResult:
    """Run `agent` on `task` to completion (or until `max_steps`), fully traced.

    `tool_allowlist_override`, if given, replaces the set of tools actually made
    callable this run -- used by scenarios to simulate an agent's behavior
    expanding beyond its declared allowlist (e.g. the drift scenario). Governance
    rules still judge tool calls against `agent.tool_allowlist`, the canonical
    declaration, regardless of what was made callable in a given run.
    """
    run = TraceRun.start(agent_name=agent.name, seed=seed, scenario=scenario)
    store.create_run(run)

    effective_allowlist = (
        tool_allowlist_override if tool_allowlist_override is not None else agent.tool_allowlist
    )
    tool_declarations = tools.function_declarations(effective_allowlist)
    messages: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": task}]}]

    final_message: str | None = None
    outcome = "max_steps"

    try:
        for _ in range(max_steps):
            response = llm.generate(
                system=agent.system_prompt,
                messages=messages,
                tool_declarations=tool_declarations,
                store=store,
                run=run,
            )

            if response.function_call is None:
                final_message = response.text
                outcome = "success"
                break

            messages.append({"role": "model", "parts": [{"function_call": response.function_call}]})
            name = response.function_call["name"]
            args = response.function_call["args"]

            if name in effective_allowlist and tools.has(name):
                result = tools.invoke(name, args, store=store, run=run, parent_id=response.event_id)
            else:
                result = {"ok": False, "error": f"tool '{name}' not permitted for this agent"}
                event = TraceEvent.new(
                    run_id=run.run_id,
                    agent_name=run.agent_name,
                    event_type=EventType.TOOL_CALL,
                    payload=tool_call_payload(
                        tool_name=name, args=args, result=result, ok=False, error=result["error"]
                    ),
                    parent_id=response.event_id,
                )
                store.append_event(run, event)

            messages.append(
                {"role": "user", "parts": [{"function_response": {"name": name, "response": result}}]}
            )
    except Exception:
        outcome = "error"
        raise
    finally:
        store.finish_run(run, outcome=outcome)

    return RunResult(run=run, final_message=final_message, outcome=outcome)
