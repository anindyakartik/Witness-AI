"""DriftDetector: flags when an agent's behavior structurally diverges from baseline.

A per-agent behavioral fingerprint is built from a set of baseline (clean) runs: a
normalized tool-usage frequency vector over the observed tool vocabulary, plus the
mean and standard deviation of three scalar metrics (tool calls, cost, LLM calls
per run). A new run is compared against this fingerprint via cosine distance on
the tool vector and z-scores on the scalars; exceeding either threshold raises a
drift alert with a human-readable reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import config
from witness.core.trace import EventType, TraceEvent, TraceRun, TraceStore, drift_alert_payload

_METRIC_NAMES = ("tool_calls", "cost_usd", "llm_calls")


def _tool_usage_counts(run: TraceRun) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in run.events_of_type(EventType.TOOL_CALL):
        name = e.payload.get("tool_name", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return counts


def _tool_usage_frequencies(run: TraceRun) -> dict[str, float]:
    counts = _tool_usage_counts(run)
    total = sum(counts.values())
    return {name: (c / total if total else 0.0) for name, c in counts.items()}


def _run_metrics(run: TraceRun) -> dict[str, float]:
    return {
        "tool_calls": float(len(run.events_of_type(EventType.TOOL_CALL))),
        "cost_usd": run.total_cost_usd,
        "llm_calls": float(len(run.events_of_type(EventType.LLM_CALL))),
    }


def _cosine_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """1 - cosine similarity, over the union of both vectors' keys."""
    vocab = sorted(set(a.keys()) | set(b.keys()))
    va = [a.get(name, 0.0) for name in vocab]
    vb = [b.get(name, 0.0) for name in vocab]
    dot = sum(x * y for x, y in zip(va, vb, strict=True))
    norm_a = math.sqrt(sum(x * x for x in va))
    norm_b = math.sqrt(sum(y * y for y in vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0 if norm_a == norm_b else 1.0
    return 1.0 - dot / (norm_a * norm_b)


@dataclass(frozen=True)
class AgentFingerprint:
    """A behavioral baseline for one agent, built from a set of clean runs."""

    agent_name: str
    tool_vocabulary: tuple[str, ...]
    mean_tool_frequencies: dict[str, float]
    metric_means: dict[str, float]
    metric_stds: dict[str, float]
    num_baseline_runs: int


def build_fingerprint(agent_name: str, baseline_runs: list[TraceRun]) -> AgentFingerprint:
    """Build a behavioral fingerprint for `agent_name` from a set of baseline runs."""
    if not baseline_runs:
        raise ValueError("Cannot build a fingerprint from zero baseline runs.")

    per_run_freq = [_tool_usage_frequencies(r) for r in baseline_runs]
    per_run_metrics = [_run_metrics(r) for r in baseline_runs]
    vocabulary = tuple(sorted({name for freq in per_run_freq for name in freq}))

    mean_tool_frequencies = {
        name: sum(freq.get(name, 0.0) for freq in per_run_freq) / len(per_run_freq)
        for name in vocabulary
    }

    metric_means: dict[str, float] = {}
    metric_stds: dict[str, float] = {}
    for m in _METRIC_NAMES:
        values = [pm[m] for pm in per_run_metrics]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        metric_means[m] = mean
        metric_stds[m] = math.sqrt(variance)

    return AgentFingerprint(
        agent_name=agent_name,
        tool_vocabulary=vocabulary,
        mean_tool_frequencies=mean_tool_frequencies,
        metric_means=metric_means,
        metric_stds=metric_stds,
        num_baseline_runs=len(baseline_runs),
    )


@dataclass(frozen=True)
class DriftCheckResult:
    """The outcome of comparing one run against an agent's baseline fingerprint."""

    agent_name: str
    distance: float
    z_scores: dict[str, float]
    new_tools: tuple[str, ...]
    is_drift: bool
    reason: str | None


class DriftDetector:
    """Compares a run's behavior against a pre-built AgentFingerprint."""

    def __init__(self, fingerprint: AgentFingerprint) -> None:
        self.fingerprint = fingerprint

    def check(self, run: TraceRun) -> DriftCheckResult:
        run_freq = _tool_usage_frequencies(run)
        distance = _cosine_distance(self.fingerprint.mean_tool_frequencies, run_freq)
        new_tools = tuple(sorted(set(run_freq) - set(self.fingerprint.tool_vocabulary)))

        metrics = _run_metrics(run)
        z_scores = {
            name: (metrics[name] - self.fingerprint.metric_means.get(name, 0.0))
            / max(self.fingerprint.metric_stds.get(name, 0.0), config.DRIFT_STD_FLOOR)
            for name in _METRIC_NAMES
        }

        distance_exceeded = distance > config.DRIFT_DISTANCE_THRESHOLD
        z_exceeded = {n: z for n, z in z_scores.items() if abs(z) > config.DRIFT_Z_THRESHOLD}
        is_drift = distance_exceeded or bool(z_exceeded)

        reason = None
        if is_drift:
            reason = self._build_reason(distance, distance_exceeded, new_tools, z_exceeded, metrics)

        return DriftCheckResult(
            agent_name=self.fingerprint.agent_name,
            distance=distance,
            z_scores=z_scores,
            new_tools=new_tools,
            is_drift=is_drift,
            reason=reason,
        )

    def _build_reason(
        self,
        distance: float,
        distance_exceeded: bool,
        new_tools: tuple[str, ...],
        z_exceeded: dict[str, float],
        metrics: dict[str, float],
    ) -> str:
        parts: list[str] = []
        if new_tools:
            parts.append(
                f"began calling {', '.join(new_tools)}, absent from its "
                f"{self.fingerprint.num_baseline_runs}-run baseline"
            )
        if distance_exceeded:
            parts.append(f"tool-usage diverged {distance:.2f} from baseline")
        for name, z in z_exceeded.items():
            mean = self.fingerprint.metric_means.get(name, 0.0)
            value = metrics[name]
            if mean:
                pct = (value - mean) / mean * 100
                parts.append(f"{name} {pct:+.0f}% vs baseline (z={z:.1f})")
            else:
                parts.append(f"{name} z-score {z:.1f}")
        return "; ".join(parts) if parts else "behavior diverged from baseline"

    def check_and_record(self, store: TraceStore, run: TraceRun) -> DriftCheckResult:
        """Check `run` against the baseline, recording a `drift_alert` event if flagged."""
        result = self.check(run)
        if result.is_drift:
            event = TraceEvent.new(
                run_id=run.run_id,
                agent_name=run.agent_name,
                event_type=EventType.DRIFT_ALERT,
                payload=drift_alert_payload(
                    agent_name=result.agent_name,
                    distance=result.distance,
                    reason=result.reason or "",
                    details={"z_scores": result.z_scores, "new_tools": list(result.new_tools)},
                ),
            )
            store.append_event(run, event)
        return result
