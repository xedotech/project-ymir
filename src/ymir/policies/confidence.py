"""Cheap confidence signals computed from one generation, and scorers that turn them
into a "probably correct" score a cascade can threshold.

Signal names are part of the log format: a router trained on logged signals expects
the same names at run time, so add new signals rather than renaming old ones.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ymir.grading import extract_answer, extract_code
from ymir.types import Generation, Task

TAIL_TOKENS = 16


def generation_signals(gen: Generation, task: Task) -> dict[str, float]:
    s: dict[str, float] = {
        "completion_tokens_log": math.log1p(gen.completion_tokens),
        "finished": 1.0 if gen.finish_reason == "stop" else 0.0,
    }
    if task.kind == "math":
        s["has_answer"] = 1.0 if extract_answer(gen.text) is not None and "\\boxed" in gen.text else 0.0
    else:
        s["has_answer"] = 1.0 if "```" in gen.text and extract_code(gen.text) else 0.0

    lps = gen.token_logprobs
    if lps:
        arr = np.asarray(lps, dtype=float)
        s["mean_logprob"] = float(arr.mean())
        s["min_logprob"] = float(arr.min())
        s["tail_mean_logprob"] = float(arr[-TAIL_TOKENS:].mean())
        s["frac_low_conf"] = float((arr < math.log(0.5)).mean())
    if gen.top_logprobs:
        ents = []
        for dist in gen.top_logprobs:
            p = np.exp(np.asarray(list(dist.values()), dtype=float))
            p = p / p.sum()
            ents.append(float(-(p * np.log(p + 1e-12)).sum()))
        s["mean_topk_entropy"] = float(np.mean(ents))
    return s


def task_signals(task: Task) -> dict[str, float]:
    """Features known before any generation (used by a difficulty router)."""
    return {"question_chars_log": math.log1p(len(task.question))}


class Scorer:
    """Maps a signal dict to a score in which higher means more likely correct."""

    def score(self, signals: dict[str, float]) -> float:
        raise NotImplementedError


@dataclass
class SignalScorer(Scorer):
    """Use one signal directly, e.g. mean_logprob."""

    signal: str = "mean_logprob"
    default: float = -math.inf

    def score(self, signals: dict[str, float]) -> float:
        return float(signals.get(self.signal, self.default))


@dataclass
class LogisticScorer(Scorer):
    """P(correct | signals), fit by `ymir train-router` on logged runs."""

    features: list[str]
    weights: list[float]
    bias: float
    mean: list[float]
    std: list[float]
    meta: dict[str, Any] = field(default_factory=dict)

    def score(self, signals: dict[str, float]) -> float:
        x = np.array([signals.get(f, m) for f, m in zip(self.features, self.mean)], dtype=float)
        z = (x - np.asarray(self.mean)) / np.asarray(self.std)
        return float(1.0 / (1.0 + np.exp(-(z @ np.asarray(self.weights) + self.bias))))

    @classmethod
    def fit(
        cls,
        rows: list[dict[str, float]],
        labels: list[int],
        features: list[str],
        l2: float = 1.0,
        iters: int = 50,
    ) -> "LogisticScorer":
        """L2-regularised logistic regression by Newton's method. Missing features get the column mean."""
        raw = np.array([[r.get(f, np.nan) for f in features] for r in rows], dtype=float)
        mean = np.nanmean(raw, axis=0)
        mean = np.where(np.isnan(mean), 0.0, mean)
        raw = np.where(np.isnan(raw), mean, raw)
        std = raw.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        X = (raw - mean) / std
        y = np.asarray(labels, dtype=float)
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])
        w = np.zeros(d + 1)
        reg = np.full(d + 1, l2)
        reg[-1] = 0.0  # do not regularise the bias
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
            grad = Xb.T @ (p - y) + reg * w
            H = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(reg) + 1e-9 * np.eye(d + 1)
            step = np.linalg.solve(H, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        return cls(features=features, weights=w[:-1].tolist(), bias=float(w[-1]), mean=mean.tolist(), std=std.tolist())

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LogisticScorer":
        with open(path) as f:
            return cls(**json.load(f))


def make_scorer(cfg: dict[str, Any] | str) -> Scorer:
    if isinstance(cfg, str):
        return LogisticScorer.load(cfg)
    kind = cfg.get("type", "signal")
    if kind == "signal":
        return SignalScorer(signal=cfg.get("signal", "mean_logprob"))
    if kind == "logistic":
        return LogisticScorer.load(cfg["path"])
    raise ValueError(f"Unknown scorer type: {kind}")
