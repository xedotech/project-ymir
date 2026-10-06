"""Client for an OpenAI-compatible /v1/completions server, such as `vllm serve`.

Token counts come from the server's `usage` field, never from local estimates.
Uses only the standard library so it runs in a bare Kaggle notebook.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from ymir.backends.base import Backend
from ymir.types import Generation


class OpenAICompatBackend(Backend):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 1800.0,
        max_retries: int = 4,
        extra_body: dict | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/completions"
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        # Server-specific sampling knobs, e.g. {"top_k": 20, "min_p": 0.0} for vLLM.
        self.extra_body = extra_body or {}

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
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **self.extra_body,
        }
        if seed is not None:
            body["seed"] = seed
        if stop:
            body["stop"] = stop
        if logprobs:
            body["logprobs"] = logprobs

        start = time.perf_counter()
        payload = self._post(body)
        latency = time.perf_counter() - start

        choice = payload["choices"][0]
        usage = payload.get("usage") or {}
        lp = choice.get("logprobs") or {}
        return Generation(
            text=choice.get("text", ""),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason") or "unknown",
            latency_s=latency,
            token_logprobs=[x for x in (lp.get("token_logprobs") or []) if x is not None] or None,
            top_logprobs=[x for x in (lp.get("top_logprobs") or []) if x] or None,
        )

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                # 4xx means the request itself is wrong; retrying will not help.
                if 400 <= e.code < 500:
                    detail = e.read().decode(errors="replace")[:500]
                    raise RuntimeError(f"Server rejected request ({e.code}): {detail}") from e
                if attempt == self.max_retries:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.max_retries:
                    raise
            time.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")
