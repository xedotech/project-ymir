"""Backend interface.

Policies talk to a raw text-completion endpoint rather than a chat endpoint, because
budget forcing needs to stop the model mid-thought and continue from an exact prefix.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ymir.types import Generation


class Backend(ABC):
    @abstractmethod
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
        """Generate a continuation of `prompt`."""
