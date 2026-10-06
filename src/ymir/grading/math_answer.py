"""Final-answer extraction and equivalence for math benchmarks.

The built-in checker handles what GSM8K, MATH-500 and AIME need most of the time:
integers, decimals, simple fractions and LaTeX that differs only in formatting. If
the optional `math-verify` package is installed, it is tried first, because it also
handles symbolic equivalence (for example \\sqrt{8} vs 2\\sqrt{2}).
"""

from __future__ import annotations

import re
from fractions import Fraction


def last_boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...} (or \\fbox{...}), with nested braces handled."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        # "\boxed 5" form
        m = re.match(r"\\boxed\s+([^\s$]+)", text[idx:])
        return m.group(1) if m else None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str) -> str | None:
    """Best guess at the final answer in a model output.

    Order: last \\boxed{}, then a GSM8K-style '#### x', then 'answer is x', then the
    last number in the text.
    """
    text = strip_thinking(text)
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"answer is[:\s]*\$?([^\n$]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    nums = _NUMBER.findall(text)
    return nums[-1] if nums else None


def strip_thinking(text: str) -> str:
    """Drop a <think>...</think> block so answers inside the thoughts are not graded."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


_REMOVE = [
    "\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\displaystyle", "$",
    "^\\circ", "^{\\circ}", "\\%", "%", "\\$",
]


def normalize(ans: str) -> str:
    s = ans.strip()
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|textbf|mbox)\{([^}]*)\}", r"\1", s)
    for tok in _REMOVE:
        s = s.replace(tok, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    # Common unit words after a number ("10 cm", "5 dollars").
    s = re.sub(r"^(-?[\d.,/]+)\s*(?:[a-zA-Z]+\^?\d*)$", r"\1", s)
    s = s.replace(" ", "")
    # "x=5" -> "5" when the left side is a single variable.
    s = re.sub(r"^[a-zA-Z]=", "", s)
    s = s.rstrip(".")
    # Thousands separators: "1,000" -> "1000" (but keep tuples like "1,2").
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    # \frac12 -> \frac{1}{2}
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)
    # Leading zero: ".5" -> "0.5"
    s = re.sub(r"(^|[^\d])\.(\d)", r"\g<1>0.\2", s)
    return s


def _as_number(s: str) -> Fraction | None:
    s = s.replace("\\frac", "")
    m = re.fullmatch(r"(-?)\{(-?\d+)\}\{(\d+)\}", s)
    if m:
        sign = -1 if m.group(1) else 1
        return sign * Fraction(int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2)))
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def is_equiv(pred: str | None, ref: str, use_math_verify: bool = True) -> bool:
    if pred is None:
        return False
    # Either checker accepting counts: math-verify catches symbolic equivalence,
    # the built-in one catches formatting it sometimes fails to parse.
    if use_math_verify and _math_verify(pred, ref):
        return True
    a, b = normalize(pred), normalize(ref)
    if a == b:
        return True
    na, nb = _as_number(a), _as_number(b)
    if na is not None and nb is not None:
        return na == nb
    return False


def _math_verify(pred: str, ref: str) -> bool | None:
    try:
        from math_verify import parse, verify  # type: ignore
    except ImportError:
        return None
    try:
        gold = parse(f"${ref}$")
        answer = parse(f"${pred}$")
        if not gold or not answer:
            return None
        return bool(verify(gold, answer))
    except Exception:
        return None
