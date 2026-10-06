import pytest

from ymir.grading import extract_code, grade, run_tests
from ymir.grading.math_answer import extract_answer, is_equiv, last_boxed, normalize
from ymir.types import Task


@pytest.mark.parametrize(
    "text,expected",
    [
        ("so \\boxed{42}.", "42"),
        ("\\boxed{\\frac{1}{2}} and later \\boxed{3}", "3"),
        ("nested \\boxed{\\sqrt{x^{2}+1}}", "\\sqrt{x^{2}+1}"),
        ("no box here", None),
    ],
)
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


def test_extract_answer_fallbacks():
    assert extract_answer("work...\n#### 1,234") == "1,234"
    assert extract_answer("The answer is 17.") == "17"
    assert extract_answer("first 3 then 8") == "8"
    # Answers written inside the thinking must not be graded.
    assert extract_answer("<think>maybe \\boxed{1}</think> final \\boxed{2}") == "2"


@pytest.mark.parametrize(
    "pred,ref",
    [
        ("1,000", "1000"),
        ("\\dfrac{1}{2}", "\\frac{1}{2}"),
        ("0.5", "\\frac{1}{2}"),
        ("1/2", "\\frac12"),
        ("x = 5", "5"),
        ("10\\text{ cm}", "10"),
        ("90^\\circ", "90"),
        (".5", "0.5"),
        ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})"),
    ],
)
def test_equivalent(pred, ref):
    assert is_equiv(pred, ref, use_math_verify=False)


@pytest.mark.parametrize("pred,ref", [("3", "4"), ("\\frac{1}{3}", "0.33"), (None, "1")])
def test_not_equivalent(pred, ref):
    assert not is_equiv(pred, ref, use_math_verify=False)


def test_normalize_keeps_tuples():
    assert normalize("1,2") == "1,2"


def test_extract_code_takes_last_block():
    text = "plan\n```python\ndef f(): return 1\n```\nfixed:\n```python\ndef f():\n    return 2\n```"
    assert extract_code(text) == "def f():\n    return 2"


def test_extract_code_unterminated_fence():
    assert extract_code("```python\ndef f():\n    return 3\n") == "def f():\n    return 3"


def test_run_tests_pass_fail_timeout():
    assert run_tests("def add(a, b):\n    return a + b", "assert add(2, 3) == 5").passed
    assert not run_tests("def add(a, b):\n    return a - b", "assert add(2, 3) == 5").passed
    res = run_tests("def spin():\n    while True: pass", "spin()", timeout_s=1.0)
    assert not res.passed and res.timed_out


def test_grade_code_task():
    task = Task(
        id="c1",
        benchmark="mbppplus",
        kind="code",
        question="Write add.",
        reference={"test": "assert add(1, 2) == 3", "setup": "import math"},
    )
    good = grade(task, "```python\ndef add(a, b):\n    return a + b\n```")
    bad = grade(task, "```python\ndef add(a, b):\n    return 0\n```")
    assert good.correct and not bad.correct
    assert "AssertionError" in bad.detail
