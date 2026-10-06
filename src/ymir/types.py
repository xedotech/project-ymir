"""Core data types shared by backends, policies, benchmarks and the runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Task:
    """One benchmark item.

    `kind` decides how the answer is extracted and graded: "math" compares a final
    answer, "code" executes hidden tests.
    """

    id: str
    benchmark: str
    kind: str
    question: str
    reference: Any
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Generation:
    """One raw completion call to a backend."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_s: float
    token_logprobs: list[float] | None = None
    # Per generated token: {token: logprob} for the top-k alternatives.
    top_logprobs: list[dict[str, float]] | None = None


@dataclass
class Cost:
    """Compute spent by a policy on one task.

    `prefill_tokens` counts only tokens the model had to newly process. When a policy
    continues a previous call (budget forcing), the server's prefix cache makes the
    repeated prefix nearly free, so it is excluded here but still visible in
    `prompt_tokens`, which is the billed total.
    """

    calls: int = 0
    prompt_tokens: int = 0
    prefill_tokens: int = 0
    generated_tokens: int = 0
    thinking_tokens: int = 0
    latency_s: float = 0.0

    def add(self, other: "Cost") -> "Cost":
        return Cost(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            prefill_tokens=self.prefill_tokens + other.prefill_tokens,
            generated_tokens=self.generated_tokens + other.generated_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
            latency_s=self.latency_s + other.latency_s,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Step:
    """One backend call inside a policy, kept for the trajectory log."""

    name: str
    prompt_tail: str
    text: str
    finish_reason: str
    completion_tokens: int


@dataclass
class Attempt:
    """What a policy returns for one task: its final output, its cost, and the
    signals a router can use to decide whether to spend more."""

    output: str
    cost: Cost
    steps: list[Step] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)
    # For cascades: which stage produced the final output.
    stopped_at: str | None = None
