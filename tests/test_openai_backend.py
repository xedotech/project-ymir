"""The real backend against a fake vLLM-style /v1/completions server, so request
format and response parsing are tested without a GPU."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ymir.backends.openai_compat import OpenAICompatBackend
from ymir.policies import PolicyContext, Think
from ymir.types import Task


class _Handler(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/completions":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append(body)
        if body["prompt"].endswith("<think>\n"):
            text, finish, n = "thinking hard", "stop", 3
        else:
            text, finish, n = "\\boxed{7}", "stop", 4
        resp = {
            "choices": [
                {
                    "text": text,
                    "finish_reason": finish,
                    "logprobs": {
                        "tokens": ["a"] * n,
                        "token_logprobs": [-0.1] * n,
                        "top_logprobs": [{"a": -0.1, "b": -2.5}] * n,
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": n},
        }
        if "</think>" in body["prompt"]:
            resp["usage"]["prompt_tokens"] = 15
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def test_think_policy_over_http(server):
    backend = OpenAICompatBackend(server, model="Qwen/Qwen3-8B", extra_body={"top_k": 20})
    task = Task(id="t", benchmark="b", kind="math", question="What is 3+4?", reference="7")
    a = Think(id="B2", budget=128, answer_tokens=64).run(task, PolicyContext(backend=backend), seed=3)

    first, second = _Handler.requests
    assert first["max_tokens"] == 128 and first["stop"] == ["</think>"] and first["seed"] == 3
    assert first["top_k"] == 20 and first["model"] == "Qwen/Qwen3-8B"
    assert second["prompt"].endswith("thinking hard\n</think>\n\n")
    assert second["logprobs"] == 5

    assert a.output == "\\boxed{7}"
    assert a.cost.thinking_tokens == 3 and a.cost.generated_tokens == 7
    assert a.cost.prompt_tokens == 25 and a.cost.prefill_tokens == 12  # 10 + (15 - 10 - 3)
    assert a.signals["mean_logprob"] == pytest.approx(-0.1)
    assert a.signals["mean_topk_entropy"] > 0


def test_client_error_is_not_retried(server):
    backend = OpenAICompatBackend(server.replace("/v1", "/nope"), model="m", max_retries=3)
    with pytest.raises(RuntimeError, match="rejected"):
        backend.complete("x", max_tokens=1, temperature=0, top_p=1)
