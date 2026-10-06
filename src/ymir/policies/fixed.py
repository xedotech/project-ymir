"""Fixed-compute baselines: the conventional strategies Gravitas-lite has to beat.

B0  NoThink(style="direct")    answer only
B1  NoThink(style="cot")       visible step-by-step reasoning, no think block
B2  Think(budget=1k/4k/16k)    think block capped by budget forcing
B3  Think(budget=full)         think until the model stops
B4  SelfConsistency(k)         k samples, majority vote
B5  oracle router              computed offline from logs (see metrics.oracle)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ymir.grading import extract_answer
from ymir.grading.math_answer import normalize
from ymir.policies.base import Policy, PolicyContext
from ymir.policies.confidence import generation_signals, task_signals
from ymir.prompts import user_message
from ymir.types import Attempt, Cost, Step, Task

STEP_LOG_CHARS = 20000


@dataclass
class NoThink(Policy):
    id: str
    style: str = "direct"  # "direct" or "cot"
    max_tokens: int = 1024
    # Qwen3 model-card recommendation for non-thinking mode.
    temperature: float = 0.7
    top_p: float = 0.8

    def run(self, task: Task, ctx: PolicyContext, seed: int) -> Attempt:
        prompt = ctx.fmt.non_thinking(user_message(task, self.style))
        gen = ctx.backend.complete(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=seed,
            logprobs=ctx.logprobs or None,
        )
        cost = Cost(
            calls=1,
            prompt_tokens=gen.prompt_tokens,
            prefill_tokens=gen.prompt_tokens,
            generated_tokens=gen.completion_tokens,
            latency_s=gen.latency_s,
        )
        signals = {**task_signals(task), **generation_signals(gen, task)}
        step = Step("answer", prompt[-200:], gen.text[:STEP_LOG_CHARS], gen.finish_reason, gen.completion_tokens)
        return Attempt(output=gen.text, cost=cost, steps=[step], signals=signals)


@dataclass
class Think(Policy):
    """Thinking mode with budget forcing (as in s1): think for at most `budget`
    tokens, then close the think block and generate the answer."""

    id: str
    budget: int = 4096
    answer_tokens: int = 1024
    # Qwen3 model-card recommendation for thinking mode.
    temperature: float = 0.6
    top_p: float = 0.95

    def run(self, task: Task, ctx: PolicyContext, seed: int) -> Attempt:
        prompt = ctx.fmt.thinking(user_message(task, "think"))
        think = ctx.backend.complete(
            prompt,
            max_tokens=self.budget,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=seed,
            stop=["</think>"],
        )
        thoughts = think.text
        if "</think>" in thoughts:
            # The server did not apply the stop string (e.g. the tag was decoded as a
            # special token and stripped differently); cut at the tag ourselves.
            thoughts = thoughts.split("</think>", 1)[0]
        prompt2 = ctx.fmt.close_thinking(prompt + thoughts)
        ans = ctx.backend.complete(
            prompt2,
            max_tokens=self.answer_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=seed,
            logprobs=ctx.logprobs or None,
        )
        # The second call re-sends the thinking as prompt. With prefix caching the
        # server only processes the few new tokens, so only those count as prefill.
        new_prefill = max(0, ans.prompt_tokens - think.prompt_tokens - think.completion_tokens)
        cost = Cost(
            calls=2,
            prompt_tokens=think.prompt_tokens + ans.prompt_tokens,
            prefill_tokens=think.prompt_tokens + new_prefill,
            generated_tokens=think.completion_tokens + ans.completion_tokens,
            thinking_tokens=think.completion_tokens,
            latency_s=think.latency_s + ans.latency_s,
        )
        signals = {**task_signals(task), **generation_signals(ans, task)}
        signals["think_hit_budget"] = 1.0 if think.finish_reason == "length" else 0.0
        signals["think_tokens_log"] = float(np.log1p(think.completion_tokens))
        steps = [
            Step("think", prompt[-200:], thoughts[:STEP_LOG_CHARS], think.finish_reason, think.completion_tokens),
            Step("answer", "</think>", ans.text[:STEP_LOG_CHARS], ans.finish_reason, ans.completion_tokens),
        ]
        return Attempt(output=ans.text, cost=cost, steps=steps, signals=signals)


@dataclass
class SelfConsistency(Policy):
    """k independent samples of `inner`, majority vote on the extracted answer."""

    id: str
    inner: Policy = field(default=None)  # type: ignore[assignment]
    k: int = 5

    def supports(self, task: Task) -> bool:
        # Voting needs comparable final answers; code outputs are not.
        return task.kind == "math"

    def run(self, task: Task, ctx: PolicyContext, seed: int) -> Attempt:
        attempts = [self.inner.run(task, ctx, seed * 1000 + j) for j in range(self.k)]
        keys = [_vote_key(a.output) for a in attempts]
        valid = [k for k in keys if k is not None]
        if valid:
            winner, votes = Counter(valid).most_common(1)[0]
            chosen = attempts[keys.index(winner)]
        else:
            votes, chosen = 0, attempts[0]
        cost = Cost()
        for a in attempts:
            cost = cost.add(a.cost)
        signals = {
            **task_signals(task),
            "agreement": votes / self.k,
            "distinct_answers": float(len(set(valid))),
        }
        for name in ("mean_logprob", "tail_mean_logprob"):
            vals = [a.signals[name] for a in attempts if name in a.signals]
            if vals:
                signals[name] = float(np.mean(vals))
        steps = [s for a in attempts for s in a.steps]
        return Attempt(output=chosen.output, cost=cost, steps=steps, signals=signals)


def _vote_key(output: str) -> str | None:
    ans = extract_answer(output)
    return normalize(ans) if ans is not None else None
