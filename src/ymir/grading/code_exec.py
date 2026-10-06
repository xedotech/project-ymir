"""Execution-based grading for code benchmarks.

Model-written code is run in a separate Python process with a timeout and, on
POSIX, CPU and memory limits. This is NOT a security sandbox: run code benchmarks
only inside a disposable environment (a Kaggle or rented GPU container), never on a
machine with credentials or data you care about.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """The last fenced code block, or the whole text if there is none."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].strip()
    # Unterminated fence (output cut off by the token limit).
    if "```" in text:
        return text.split("```", 1)[1].split("\n", 1)[-1].strip()
    return text.strip()


@dataclass
class ExecResult:
    passed: bool
    timed_out: bool
    stderr_tail: str


def _limits(memory_mb: int, cpu_s: int):
    def apply() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 2**20, memory_mb * 2**20))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            os.setsid()
        except Exception:
            pass

    return apply if os.name == "posix" else None


def run_tests(
    solution: str,
    test_code: str,
    setup_code: str = "",
    timeout_s: float = 20.0,
    memory_mb: int = 2048,
) -> ExecResult:
    """Run `setup_code + solution + test_code` and pass if it exits 0."""
    program = "\n\n".join(x for x in (setup_code, solution, test_code) if x)
    with tempfile.TemporaryDirectory(prefix="ymir_exec_") as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                cwd=d,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=_limits(memory_mb, int(timeout_s) + 1),
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return ExecResult(passed=False, timed_out=True, stderr_tail="timeout")
        return ExecResult(
            passed=proc.returncode == 0,
            timed_out=False,
            stderr_tail=proc.stderr[-500:],
        )
