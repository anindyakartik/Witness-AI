"""Gemini client: function calling, rate limiting, backoff, cost accounting, tracing.

Also implements a deterministic record/replay cache ("cassette"): the first live call
for a given request is recorded to a JSON file under `cassettes/`; subsequent runs
with identical content replay that recording instead of calling the API. This is
what makes a 20-run drift baseline and a "one command" demo fast, reproducible, and
runnable offline after the initial recording. See `config.LLM_MODE` for the modes.

The multi-turn message shape (role="user" for prompts and function results, role=
"model" for text/function-call turns) was verified against the installed SDK's own
`ChatSession` implementation before writing this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

import config
from witness.core.trace import EventType, TraceEvent, TraceRun, TraceStore, llm_call_payload

load_dotenv()


@dataclass(frozen=True)
class LLMResponse:
    """A normalized Gemini response: either a function call or a final text answer."""

    text: str | None
    function_call: dict[str, Any] | None  # {"name": str, "args": dict}
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    replayed: bool
    event_id: str


class TokenBucket:
    """Token-bucket rate limiter pacing calls to an average requests-per-minute cap."""

    def __init__(self, rpm: int) -> None:
        self.rate = rpm / 60.0
        self.capacity = float(rpm)
        self._tokens = self.capacity
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def acquire(self) -> None:
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            time.sleep((1.0 - self._tokens) / self.rate)


def _call_with_backoff(fn: Any) -> Any:
    """Call `fn()`, retrying on HTTP 429 per config.BACKOFF_SCHEDULE_S, then re-raising."""
    schedule = list(config.BACKOFF_SCHEDULE_S)
    while True:
        try:
            return fn()
        except ResourceExhausted:
            if not schedule:
                raise
            time.sleep(schedule.pop(0))


def _to_jsonable(value: Any) -> Any:
    """Coerce proto Struct-derived values (MapComposite/RepeatedComposite) to plain
    JSON-safe Python types before caching or serializing into the trace."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "items"):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    return value


class Cassette:
    """A JSON-backed dict of recorded (request_hash -> response) pairs for one scenario.

    Committed to the repo (not git-ignored) so a fresh clone can replay demo runs
    deterministically and offline.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, Any] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> dict[str, Any] | None:
        return self._data.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")


def _hash_request(
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tool_names: list[str],
    temperature: float,
) -> str:
    """Canonical content hash identifying a request, used as the cassette lookup key."""
    canonical = json.dumps(
        {
            "model": model,
            "system": system,
            "messages": messages,
            "tools": tool_names,
            "temperature": temperature,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LLMClient:
    """Thin wrapper over Gemini function calling with rate limiting, backoff, cost
    accounting, deterministic record/replay, and automatic `llm_call` trace emission.

    The rate limiter is shared across all instances in a process, since free-tier
    limits apply to the whole fleet, not per agent.
    """

    _shared_rate_limiter: TokenBucket | None = None

    def __init__(
        self,
        *,
        cassette_name: str,
        model_name: str = config.MODEL_NAME,
        mode: str | None = None,
        cassette_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.mode = mode or config.LLM_MODE
        self.cassette = Cassette((cassette_dir or config.CASSETTES_DIR) / f"{cassette_name}.json")

        if LLMClient._shared_rate_limiter is None:
            LLMClient._shared_rate_limiter = TokenBucket(config.RATE_LIMIT_RPM)
        self._rate_limiter = LLMClient._shared_rate_limiter

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tool_declarations: list[dict[str, Any]],
        store: TraceStore,
        run: TraceRun,
        parent_id: str | None = None,
        temperature: float = config.TEMPERATURE,
    ) -> LLMResponse:
        """Generate the next turn given `messages`, replaying or recording as configured."""
        tool_names = [t["name"] for t in tool_declarations]
        request_hash = _hash_request(
            model=self.model_name,
            system=system,
            messages=messages,
            tool_names=tool_names,
            temperature=temperature,
        )

        cached = self.cassette.get(request_hash) if self.mode in ("auto", "replay") else None

        if cached is not None:
            response_data = cached
            replayed = True
            latency_ms = 0.0
        elif self.mode == "replay":
            raise RuntimeError(
                f"No cassette entry for request hash {request_hash} in replay-only mode "
                f"(cassette: {self.cassette.path})."
            )
        else:
            start = time.perf_counter()
            self._rate_limiter.acquire()
            raw = _call_with_backoff(
                lambda: self._call_live(
                    system=system,
                    messages=messages,
                    tool_declarations=tool_declarations,
                    temperature=temperature,
                )
            )
            latency_ms = (time.perf_counter() - start) * 1000
            response_data = self._parse_response(raw)
            replayed = False
            if self.mode in ("auto", "record"):
                self.cassette.put(request_hash, response_data)

        event = TraceEvent.new(
            run_id=run.run_id,
            agent_name=run.agent_name,
            event_type=EventType.LLM_CALL,
            payload=llm_call_payload(
                model=self.model_name,
                system=system,
                messages=messages,
                response_text=response_data["text"],
                function_call=response_data["function_call"],
                prompt_tokens=response_data["prompt_tokens"],
                completion_tokens=response_data["completion_tokens"],
                replayed=replayed,
            ),
            cost_usd=response_data["cost_usd"],
            latency_ms=latency_ms,
            parent_id=parent_id,
        )
        store.append_event(run, event)

        return LLMResponse(
            text=response_data["text"],
            function_call=response_data["function_call"],
            prompt_tokens=response_data["prompt_tokens"],
            completion_tokens=response_data["completion_tokens"],
            cost_usd=response_data["cost_usd"],
            replayed=replayed,
            event_id=event.id,
        )

    def _ensure_model(
        self, *, system: str, tool_declarations: list[dict[str, Any]]
    ) -> genai.GenerativeModel:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set, and no cassette entry covers this request. "
                "Set GEMINI_API_KEY in .env to record it, or switch LLM_MODE to "
                "'replay' if you only intend to use existing cassettes."
            )
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name=self.model_name,
            tools=[{"function_declarations": tool_declarations}] if tool_declarations else None,
            system_instruction=system,
        )

    def _call_live(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tool_declarations: list[dict[str, Any]],
        temperature: float,
    ) -> Any:
        model = self._ensure_model(system=system, tool_declarations=tool_declarations)
        return model.generate_content(
            contents=messages,
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )

    def _parse_response(self, raw: Any) -> dict[str, Any]:
        """Extract text/function-call + usage from a raw Gemini response into a plain,
        JSON-safe dict (the shape stored in cassettes and TraceEvent payloads)."""
        candidate = raw.candidates[0]
        text: str | None = None
        function_call: dict[str, Any] | None = None

        for part in candidate.content.parts:
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                function_call = {"name": fc.name, "args": _to_jsonable(fc.args)}
                break
            part_text = getattr(part, "text", None)
            if part_text:
                text = part_text

        usage = raw.usage_metadata
        prompt_tokens = usage.prompt_token_count
        completion_tokens = usage.candidates_token_count
        cost_usd = (
            prompt_tokens / 1_000_000 * config.PRICE_PER_1M_INPUT_TOKENS
            + completion_tokens / 1_000_000 * config.PRICE_PER_1M_OUTPUT_TOKENS
        )

        return {
            "text": text,
            "function_call": function_call,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
        }
