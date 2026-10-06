from __future__ import annotations

from typing import Any

from ymir.backends.base import Backend
from ymir.backends.mock import MockBackend
from ymir.backends.openai_compat import OpenAICompatBackend


def make_backend(cfg: dict[str, Any]) -> Backend:
    kind = cfg.get("type", "openai")
    if kind == "mock":
        return MockBackend()
    if kind == "openai":
        return OpenAICompatBackend(
            base_url=cfg.get("base_url", "http://localhost:8000/v1"),
            model=cfg["model"],
            api_key=cfg.get("api_key", "EMPTY"),
            timeout_s=float(cfg.get("timeout_s", 1800)),
            extra_body=cfg.get("extra_body"),
        )
    raise ValueError(f"Unknown backend type: {kind}")


__all__ = ["Backend", "MockBackend", "OpenAICompatBackend", "make_backend"]
