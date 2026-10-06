"""Gravitas-lite: spend more compute only when cheap signals say it is needed.

Run stage 0 (cheap). Score its signals. If the score clears that stage's threshold,
stop; otherwise escalate to the next, more expensive stage. Every stage's cost is
charged, including the cheap attempts that were thrown away.

Threshold sweeps do not need new model calls: `metrics.simulate_cascade` replays the
same decision rule over logged runs of the individual stages. Live cascade runs then
confirm the chosen operating point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ymir.policies.base import Policy, PolicyContext
from ymir.policies.confidence import Scorer
from ymir.types import Attempt, Cost, Task


@dataclass
class Cascade(Policy):
    id: str
    stages: list[Policy] = field(default_factory=list)
    # scorers[i] and thresholds[i] gate stopping after stage i; the last stage always stops.
    # Each stage gets its own scorer because signals mean different things after
    # a direct answer and after a long think.
    scorers: list[Scorer] = field(default_factory=list)
    thresholds: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = max(len(self.stages) - 1, 0)
        if len(self.thresholds) != n or len(self.scorers) != n:
            raise ValueError(f"{self.id}: need {n} scorers and {n} thresholds for {len(self.stages)} stages")

    def supports(self, task: Task) -> bool:
        return all(s.supports(task) for s in self.stages)

    def run(self, task: Task, ctx: PolicyContext, seed: int) -> Attempt:
        total = Cost()
        steps = []
        stage_scores: dict[str, float] = {}
        attempt: Attempt | None = None
        for i, stage in enumerate(self.stages):
            attempt = stage.run(task, ctx, seed)
            total = total.add(attempt.cost)
            steps.extend(attempt.steps)
            if i == len(self.stages) - 1:
                break
            score = self.scorers[i].score(attempt.signals)
            stage_scores[f"score_{stage.id}"] = score
            if score >= self.thresholds[i]:
                break
        assert attempt is not None
        signals = {**attempt.signals, **stage_scores, "stages_run": float(i + 1)}
        return Attempt(output=attempt.output, cost=total, steps=steps, signals=signals, stopped_at=stage.id)
