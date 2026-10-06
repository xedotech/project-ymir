"""Prompt construction for Qwen3-style chat models with a <think> block.

The strings below reproduce what Qwen3's chat template emits, written out by hand so
the harness controls the exact prefix and can continue it (budget forcing):

- thinking mode:     "<|im_start|>assistant\n<think>\n"   the model then thinks
- non-thinking mode: "<|im_start|>assistant\n<think>\n\n</think>\n\n"   (template's enable_thinking=False)

If you switch to a model family with a different template, add a new PromptFormat
rather than editing this one, so earlier runs stay reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

from ymir.types import Task

MATH_STEP_BY_STEP = "Please reason step by step, and put your final answer within \\boxed{}."
MATH_DIRECT = "Give only the final answer within \\boxed{}, with no explanation."


@dataclass(frozen=True)
class PromptFormat:
    name: str
    user_start: str
    user_end: str
    assistant_start: str
    think_open: str
    think_close: str

    def chat(self, user_message: str) -> str:
        return f"{self.user_start}{user_message}{self.user_end}{self.assistant_start}"

    def thinking(self, user_message: str) -> str:
        """Prefix that makes the model start thinking."""
        return self.chat(user_message) + self.think_open

    def non_thinking(self, user_message: str) -> str:
        """Prefix with an empty think block, so the model answers directly."""
        return self.chat(user_message) + self.think_open + self.think_close

    def close_thinking(self, prefix_with_thoughts: str) -> str:
        """Force the think block shut, wherever the model stopped."""
        return prefix_with_thoughts.rstrip("\n") + self.think_close


QWEN3 = PromptFormat(
    name="qwen3",
    user_start="<|im_start|>user\n",
    user_end="<|im_end|>\n",
    assistant_start="<|im_start|>assistant\n",
    think_open="<think>\n",
    think_close="\n</think>\n\n",
)

FORMATS = {"qwen3": QWEN3}


def user_message(task: Task, style: str) -> str:
    """The user turn for a task.

    style: "direct" (answer only), "cot" (reason visibly, no think block) or
    "think" (instruction used together with thinking mode).
    """
    if task.kind == "math":
        instruction = MATH_DIRECT if style == "direct" else MATH_STEP_BY_STEP
        return f"{task.question}\n\n{instruction}"
    if task.kind == "code":
        example = task.meta.get("example_test", "")
        parts = [task.question.strip()]
        if example:
            parts.append(f"Your code should pass this test:\n{example}")
        if style == "cot":
            parts.append("Briefly plan the solution, then give the complete code in one ```python block.")
        else:
            parts.append("Return only the complete code in one ```python block.")
        return "\n\n".join(parts)
    raise ValueError(f"Unknown task kind: {task.kind}")
