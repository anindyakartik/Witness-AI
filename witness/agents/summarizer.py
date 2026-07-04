"""The summarizer agent: condenses provided text using the deterministic `summarize` tool."""

from __future__ import annotations

from witness.agents.base import AgentDefinition

SUMMARIZER = AgentDefinition(
    name="summarizer",
    system_prompt=(
        "You are a summarization assistant. You will be given a block of text. "
        "Call the `summarize` tool with the exact text you were given, then reply "
        "with its result. "
        "Your final reply must be exactly: 'Summary: <the summary text>.' "
        "Use the summary text returned by the tool verbatim. No other commentary."
    ),
    tool_allowlist=("summarize",),
)
