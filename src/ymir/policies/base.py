from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ymir.backends.base import Backend
from ymir.prompts import QWEN3, PromptFormat
from ymir.types import Attempt, Task


@dataclass
class PolicyContext:
    backend: Backend
    fmt: PromptFormat = QWEN3
    # Top-k log-probabilities requested per generated token (0 disables them).
    logprobs: int = 5


class Policy(ABC):
    """A strategy for spending compute on one task."""

    id: str

    def supports(self, task: Task) -> bool:
        return True

    @abstractmethod
    def run(self, task: Task, ctx: PolicyContext, seed: int) -> Attempt: ...
