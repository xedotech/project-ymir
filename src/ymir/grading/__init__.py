from __future__ import annotations

from dataclasses import dataclass

from ymir.grading.code_exec import extract_code, run_tests
from ymir.grading.math_answer import extract_answer, is_equiv
from ymir.types import Task


@dataclass
class Grade:
    correct: bool
    pred: str | None
    detail: str = ""


def grade(task: Task, output: str) -> Grade:
    if task.kind == "math":
        pred = extract_answer(output)
        return Grade(correct=is_equiv(pred, str(task.reference)), pred=pred)
    if task.kind == "code":
        code = extract_code(output)
        ref = task.reference
        res = run_tests(code, test_code=ref["test"], setup_code=ref.get("setup", ""))
        lines = res.stderr_tail.strip().splitlines()
        detail = "timeout" if res.timed_out else (lines[-1] if lines else "")
        return Grade(correct=res.passed, pred=code[:2000], detail=detail)
    raise ValueError(f"Unknown task kind: {task.kind}")


__all__ = ["Grade", "grade", "extract_answer", "is_equiv", "extract_code", "run_tests"]
