from ymir.backends.mock import MockBackend
from ymir.benchmarks import load_toy
from ymir.policies import (
    Cascade,
    NoThink,
    PolicyContext,
    SelfConsistency,
    SignalScorer,
    Think,
    build_policies,
)
from ymir.prompts import QWEN3, user_message
from ymir.types import Task

CTX = PolicyContext(backend=MockBackend())
TASK = load_toy({"n": 5})[4]  # level 5, the hardest


def test_qwen3_prompt_shapes():
    msg = user_message(TASK, "think")
    assert QWEN3.thinking(msg).endswith("<|im_start|>assistant\n<think>\n")
    assert QWEN3.non_thinking(msg).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert QWEN3.close_thinking(QWEN3.thinking(msg) + "some thoughts\n").endswith("some thoughts\n</think>\n\n")


def test_code_prompt_shows_one_example_test():
    task = Task(id="c", benchmark="b", kind="code", question="Write f.", reference={}, meta={"example_test": "assert f(1) == 2"})
    assert "assert f(1) == 2" in user_message(task, "direct")


def test_nothink_cost_and_signals():
    a = NoThink(id="B0").run(TASK, CTX, seed=0)
    assert a.cost.calls == 1 and a.cost.thinking_tokens == 0
    assert a.cost.generated_tokens > 0
    assert "mean_logprob" in a.signals and "question_chars_log" in a.signals


def test_think_budget_is_enforced_and_charged():
    small = Think(id="t", budget=50).run(TASK, CTX, seed=0)
    big = Think(id="T", budget=5000).run(TASK, CTX, seed=0)
    assert small.cost.thinking_tokens == 50
    assert small.signals["think_hit_budget"] == 1.0
    assert big.signals["think_hit_budget"] == 0.0
    assert big.cost.thinking_tokens > small.cost.thinking_tokens
    assert big.cost.calls == 2
    # Prefix-cached continuation: prefill counts the first prompt plus only the new tokens.
    assert big.cost.prefill_tokens < big.cost.prompt_tokens


def test_self_consistency_charges_all_samples():
    inner = NoThink(id="B0")
    sc = SelfConsistency(id="sc", inner=inner, k=4).run(TASK, CTX, seed=1)
    assert sc.cost.calls == 4
    assert 0 < sc.signals["agreement"] <= 1


def test_cascade_stops_or_escalates_and_charges_every_stage():
    cheap, expensive = NoThink(id="B0"), Think(id="B3", budget=5000)
    always_stop = Cascade(id="c0", stages=[cheap, expensive], scorers=[SignalScorer("mean_logprob")], thresholds=[-1e9])
    never_stop = Cascade(id="c1", stages=[cheap, expensive], scorers=[SignalScorer("mean_logprob")], thresholds=[1e9])
    a = always_stop.run(TASK, CTX, seed=0)
    b = never_stop.run(TASK, CTX, seed=0)
    assert a.stopped_at == "B0" and a.cost.calls == 1
    assert b.stopped_at == "B3" and b.cost.calls == 3
    solo = expensive.run(TASK, CTX, seed=0)
    # Same seed -> the escalated answer is the stage run alone, so offline replay is exact.
    assert b.output == solo.output


def test_build_policies_resolves_references():
    pols = build_policies(
        [
            {"id": "B0", "type": "nothink"},
            {"id": "B3", "type": "think", "budget": 100},
            {"id": "SC", "type": "self_consistency", "inner": "B0", "k": 3},
            {"id": "G", "type": "cascade", "stages": ["B0", "B3"], "thresholds": [0.5], "scorer": {"type": "signal"}},
        ]
    )
    assert set(pols) == {"B0", "B3", "SC", "G"}
    assert not pols["SC"].supports(Task(id="c", benchmark="b", kind="code", question="", reference={}))
