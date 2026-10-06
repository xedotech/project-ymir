"""A deterministic fake model for testing the pipeline without a GPU.

It solves the toy arithmetic benchmark ("Compute: 12*7+3") with an accuracy that
rises with thinking budget and falls with the number of operators, and it reports
higher log-probabilities when it is right. That gives routers a real (if synthetic)
signal to learn from, so end-to-end tests exercise the same code paths as real runs.

Tokens are whitespace-separated words. Latency is simulated, not slept.
"""

from __future__ import annotations

import ast
import hashlib
import math
import operator
import random
import re

from ymir.backends.base import Backend
from ymir.types import Generation

_EXPR = re.compile(r"Compute:\s*([0-9+\-*() ]+)")
_PROGRESS = re.compile(r"\[\[progress=([0-9.]+)\]\]")

TOKENS_PER_OPERATOR = 150
THINK_CEILING = 0.97


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def _safe_eval(expr: str) -> int:
    """Evaluate +, -, * over integers, refusing anything else (no powers, no names)."""

    def ev(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError(f"Unsupported expression: {expr}")

    return ev(ast.parse(expr, mode="eval"))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class MockBackend(Backend):
    def __init__(self, seconds_per_token: float = 0.002) -> None:
        self.seconds_per_token = seconds_per_token

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
        stop: list[str] | None = None,
        logprobs: int | None = None,
    ) -> Generation:
        m = _EXPR.search(prompt)
        if not m:
            return self._gen(prompt, "I cannot help with that.", "stop", correct=False, want_lp=bool(logprobs))
        expr = m.group(1).strip()
        truth = _safe_eval(expr)
        difficulty = max(1, sum(expr.count(op) for op in "+-*"))
        rng = random.Random(hashlib.sha256(f"{prompt}|{seed}|{temperature}".encode()).hexdigest())

        base_p = _sigmoid(2.5 - 1.2 * difficulty)
        tail = prompt.rsplit("<|im_start|>assistant\n", 1)[-1]

        if tail.endswith("<think>\n"):
            # Thinking phase: think until solved or until the budget runs out.
            need = int(TOKENS_PER_OPERATOR * difficulty * rng.uniform(0.7, 1.3))
            n = min(need, max_tokens)
            progress = n / need
            words = ["hmm"] * max(n - 1, 0) + [f"[[progress={progress:.2f}]]"]
            finish = "stop" if n >= need else "length"
            return self._gen(prompt, " ".join(words), finish, correct=True, want_lp=bool(logprobs))

        if "</think>" in tail and "<think>\n\n</think>" not in tail:
            # Answer phase after thinking.
            pm = _PROGRESS.search(tail)
            progress = float(pm.group(1)) if pm else 0.0
            p = base_p + (THINK_CEILING - base_p) * progress
            correct = rng.random() < p
            ans = truth if correct else truth + rng.choice([-3, -2, -1, 1, 2, 7])
            return self._gen(prompt, f"The answer is \\boxed{{{ans}}}.", "stop", correct, bool(logprobs), rng)

        # Non-thinking mode.
        step_by_step = "step by step" in prompt
        p = min(base_p + (0.15 if step_by_step else 0.0), THINK_CEILING)
        correct = rng.random() < p
        ans = truth if correct else truth + rng.choice([-3, -2, -1, 1, 2, 7])
        reasoning = " ".join(["so"] * (30 * difficulty)) + " " if step_by_step else ""
        text = f"{reasoning}\\boxed{{{ans}}}"
        words = text.split()
        if len(words) > max_tokens:
            return self._gen(prompt, " ".join(words[:max_tokens]), "length", False, bool(logprobs), rng)
        return self._gen(prompt, text, "stop", correct, bool(logprobs), rng)

    def _gen(
        self,
        prompt: str,
        text: str,
        finish: str,
        correct: bool,
        want_lp: bool,
        rng: random.Random | None = None,
    ) -> Generation:
        n = len(text.split())
        token_lps = None
        if want_lp:
            rng = rng or random.Random(text)
            mu = -0.15 if correct else -0.6
            token_lps = [min(0.0, rng.gauss(mu, 0.15)) for _ in range(n)]
        return Generation(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=n,
            finish_reason=finish,
            latency_s=n * self.seconds_per_token,
            token_logprobs=token_lps,
        )
