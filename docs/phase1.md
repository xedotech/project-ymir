# Phase I definitions (milestone M0)

These definitions are fixed before any real results exist, so they cannot be tuned to the results. Changing one after M2 starts requires a note in this file saying what changed and why.

## Question

Can a policy that decides per task how much reasoning to spend (Gravitas-lite) achieve a better accuracy-versus-compute frontier than any single fixed policy on the same frozen model?

This is a minimal form of the canonical spec's Adaptive Compute and Value of Computation (§23–24). The spec's roadmap (§46) puts Adaptive Compute at Phase IV; it is pulled forward because it produces the cost measurements every later phase is judged against.

## Substrate

- Model: Qwen3-8B, frozen, served by vLLM. Qwen3-1.7B for debugging only. One other model family on the final configuration only.
- Prompt format: `qwen3` in `src/ymir/prompts.py`. Sampling follows the Qwen3 model card: thinking T=0.6, top-p 0.95; non-thinking T=0.7, top-p 0.8; top-k 20 for both.
- Seeds: 0, 1, 2. Every policy sees the same tasks and seeds.

## Cost

| Quantity | Definition |
| --- | --- |
| Generated tokens | Completion tokens reported by the server, summed over all calls a policy makes on a task. **Primary cost measure.** |
| Thinking tokens | Generated tokens inside the think block |
| Prefill tokens | Prompt tokens the server newly processed, excluding prefixes re-sent for continuation (assumes prefix caching) |
| FLOPs | 2 × parameters × (prefill + generated), attention ignored |
| Latency | Wall-clock seconds summed over calls; reported, not used for the success bar, because it depends on batching |
| Dollars | GPU-hours × the hourly rate actually paid |

A cascade is charged for every stage it ran, including discarded cheap attempts and any router overhead.

## Intelligence per Compute

1. **Accuracy at matched budget**: best accuracy reachable at a given mean generated-token budget, interpolated along the Pareto frontier.
2. **Compute to target**: fewest mean generated tokens at which a policy (or the frontier of fixed policies) reaches a target accuracy.
3. **Marginal efficiency**: (Acc − Acc_B0) / (Tokens − Tokens_B0).

Accuracy is averaged over seeds per task, then over tasks. Confidence intervals are 95% bootstrap over tasks. Policy comparisons are paired on the same tasks and seeds.

## Success bar

Gravitas-lite reaches within 1 accuracy point of B3 ("always think hard") using at most 60% of B3's mean generated tokens, on at least two of MATH-500, GSM8K and MBPP+, with the paired 95% CI for its accuracy gain over the best fixed policy at that budget excluding zero.

## Kill criterion

If no threshold beats the best fixed policy on at least two of the three main benchmarks, record a negative result and move to the Experience Engine. That outcome says gains must come from reuse rather than allocation.

## Leakage rules

- Routers are trained on training splits (GSM8K train, MATH train) or on a hashed fraction of tasks, never on the evaluation tasks they are scored on.
- When a router is trained with `--split-frac F`, every report that uses it passes `--eval-split-frac F`, so all policies are compared on the same held-out tasks.
- Thresholds are chosen on training data; the report shows the whole sweep so readers can see sensitivity.
