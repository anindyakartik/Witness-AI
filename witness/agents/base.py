"""Agent definitions: name, system prompt, and tool allowlist.

An AgentDefinition is pure data. The runtime and governance layers never know about
a specific agent's identity beyond this -- adding a new agent means adding a new
AgentDefinition, not touching core/runtime.py or governance/*.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentDefinition:
    """One agent's identity: its name, its instructions, and the tools it may use.

    `tool_allowlist` is the canonical, declared set of tools this agent is permitted
    to call -- this is what governance.rules.ToolAllowlistRule checks actual tool
    calls against, independent of what a given run happens to make callable.
    """

    name: str
    system_prompt: str
    tool_allowlist: tuple[str, ...]
