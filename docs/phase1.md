# Evaluation definitions

These definitions are fixed before results exist, so they cannot be tuned to the results. Any later change is recorded in this file with the reason.

## Question

Can a strategy that varies how much compute it spends per task achieve a better accuracy-versus-compute frontier than any single fixed strategy on the same frozen model?

## Setup

- A frozen open-weight model served by vLLM; a smaller model of the same family for debugging. One other model family on the final configuration only.
- Sampling follows the model card's recommended settings.
- Seeds: 0, 1, 2. Every policy sees the same tasks and seeds.

## Cost

| Quantity | Definition |
| --- | --- |
| Generated tokens | Completion tokens reported by the server, summed over all calls a policy makes on a task. **Primary cost measure.** |
| Thinking tokens | Generated tokens inside the reasoning block |
| Prefill tokens | Prompt tokens the server newly processed, excluding re-sent prefixes (assumes prefix caching) |
| FLOPs | 2 × parameters × (prefill + generated), attention ignored |
| Latency | Wall-clock seconds summed over calls; reported, not used for the success bar, because it depends on batching |
| Dollars | GPU-hours × the hourly rate actually paid |

A multi-stage policy is charged for every stage it ran, including discarded attempts and any overhead.

## Accuracy per compute

1. **Accuracy at matched budget**: best accuracy reachable at a given mean generated-token budget, interpolated along the Pareto frontier.
2. **Compute to target**: fewest mean generated tokens at which a policy (or the frontier of fixed policies) reaches a target accuracy.
3. **Marginal efficiency**: (Acc − Acc_B0) / (Tokens − Tokens_B0).

Accuracy is averaged over seeds per task, then over tasks. Confidence intervals are 95% bootstrap over tasks. Policy comparisons are paired on the same tasks and seeds.

## Success bar

The adaptive policy reaches within 1 accuracy point of B3 using at most 60% of B3's mean generated tokens, on at least two of MATH-500, GSM8K and MBPP+. The paired 95% CI for its accuracy gain over the best fixed policy at that budget must exclude zero.

## Negative result

If no setting beats the best fixed policy on at least two of the three main benchmarks, the negative result is recorded.

## Leakage rules

- Anything trained is trained on training splits (GSM8K train, MATH train) or on a hashed fraction of tasks, never on the evaluation tasks it is scored on.
- When training uses `--split-frac F`, every report that uses it passes `--eval-split-frac F`, so all policies are compared on the same held-out tasks.
- Thresholds are chosen on training data; the report shows the whole sweep so readers can see sensitivity.
