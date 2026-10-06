# project-ymir

Phase I harness for Ymir Labs' Candid research program (Nutin Technologies).

Phase I asks one narrow question: **can a learned compute-allocation policy (a minimal Gravitas) on an open-weight model match that model's own full-reasoning accuracy with substantially fewer tokens?** This repository runs that experiment. Every policy run on every task records two things: whether the answer was right, and exactly what it cost. "Intelligence per Compute" is computed from those two numbers.

- Plan and rationale: [Candid Phase I Plan](https://claude.ai/code/artifact/4d0bf343-2237-4171-b8c7-895124df331b)
- Pre-registered definitions (milestone M0): [docs/phase1.md](docs/phase1.md)

## What is here

| Path | What it does |
| --- | --- |
| `src/ymir/backends/` | Client for any OpenAI-compatible `/v1/completions` server (vLLM), plus a deterministic mock model for GPU-free testing |
| `src/ymir/prompts.py` | Qwen3 prompt format with explicit thinking and non-thinking prefixes, so thinking can be cut off at a budget and resumed |
| `src/ymir/benchmarks/` | Loaders for MATH-500, GSM8K, MBPP+, AIME 2025, training splits for the router, and a synthetic toy set |
| `src/ymir/grading/` | Final-answer extraction and equivalence for math; execution of hidden tests for code |
| `src/ymir/policies/` | Fixed baselines B0 to B4 and the Gravitas-lite cascade, with confidence signals and a logistic router |
| `src/ymir/runner.py` | Runs benchmark × policy × seed, appends one JSONL record per task, resumes after interruption |
| `src/ymir/metrics.py` | Bootstrap CIs, paired differences, oracle router, offline cascade replay, Pareto frontier, compute-to-target |
| `src/ymir/report.py` | Markdown report and accuracy-versus-tokens plots |
| `configs/` | Experiment configs (TOML) |

## Policies

| Id | Policy | Represents |
| --- | --- | --- |
| B0 | Non-thinking, direct answer | Cheapest possible |
| B1 | Non-thinking, visible step-by-step | Light reasoning |
| B2-1k / 4k / 16k | Thinking, capped by budget forcing | Fixed deep reasoning at several costs |
| B3 | Thinking, full budget (32k) | "Always think hard" |
| B4-sc5 | 5 samples of B1, majority vote (math only) | Parallel brute force |
| B5-oracle | Cheapest correct policy per task, in hindsight | Upper bound for any router (computed from logs) |
| Gravitas-lite | Run a cheap stage; escalate only if a learned router says the answer is probably wrong | The experiment |

Budget forcing: the model thinks with `stop=["</think>"]` and `max_tokens=budget`; the harness then closes the think block and asks for the answer. The second call re-sends the thoughts as prompt, which the server's prefix cache makes nearly free, so `prefill_tokens` counts only new tokens.

## Quick start without a GPU

```bash
pip install -e ".[dev]"
pytest -q

ymir run configs/smoke_mock.toml
ymir train-router runs/smoke-mock.jsonl --policy B0 --split-frac 0.5 --out runs/router_b0.json
ymir report runs/smoke-mock.jsonl --oracle B0,B2-256,B2-1k,B3 --cascade B0,B3 \
    --router runs/router_b0.json --target-ref B3 --eval-split-frac 0.5
```

The mock model solves synthetic arithmetic with accuracy that rises with thinking budget and falls with difficulty. Its numbers mean nothing; it exists so the full pipeline can be tested anywhere.

## Running for real (Kaggle or a rented GPU)

1. **Environment.** In a Kaggle notebook with GPU T4 ×2 and internet on:

   ```bash
   git clone https://github.com/xedotech/project-ymir && cd project-ymir
   pip install -e ".[data,plot,mathverify]" vllm
   ```

2. **Server.** Start vLLM in the background and wait for it to load:

   ```bash
   TP=2 DTYPE=half nohup ./scripts/serve_vllm.sh > vllm.log 2>&1 &
   until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
   ```

   T4s are Turing GPUs (no bf16). If the installed vLLM version refuses them, pin an older vLLM or run on the rented GPU. Use `MODEL=Qwen/Qwen3-1.7B` while debugging.

3. **Data.** `ymir data fetch configs/baselines_qwen3.toml` downloads the benchmarks to `data/` and prints one example of each. Check the examples: hub dataset ids and column names drift, and every loader accepts overrides in the config (`hf_id`, `hf_config(s)`, `split`, `question_field`, `answer_field`).

4. **M1 and M2, baselines.** Smoke test, then the full grid. Runs are resumable, so rerun the same command after a session timeout.

   ```bash
   ymir run configs/baselines_qwen3.toml --limit 20
   ymir run configs/baselines_qwen3.toml
   ymir report runs/baselines-qwen3-8b.jsonl --params-b 8.2 --oracle B0,B1,B2-1k,B2-4k,B2-16k,B3 \
       --target-ref B3 --plot-dir reports/plots --out reports/baselines.md
   ```

5. **M3, Gravitas-lite.** Train the router on training splits only, then replay the cascade over the logged baseline runs to sweep the threshold. No extra model calls are needed for the sweep.

   ```bash
   ymir data fetch configs/router_train_qwen3.toml
   ymir run configs/router_train_qwen3.toml
   ymir train-router runs/router-train-qwen3-8b.jsonl --policy B0 --out routers/b0.json
   ymir report runs/baselines-qwen3-8b.jsonl --params-b 8.2 --cascade B0,B3 --router routers/b0.json \
       --target-ref B3 --plot-dir reports/plots --out reports/cascade.md
   ```

   To confirm the chosen operating point live, add a `cascade` policy to a config (see `tests/test_runner_metrics.py` for the shape) and run it. Offline replay and live runs agree exactly when the stage configs and seeds match; a test checks this.

## Record format

Each line of `runs/<name>.jsonl` is one (benchmark, task, policy, seed):

```json
{"benchmark": "math500", "task_id": "test/algebra/1.json", "policy": "B2-4k", "seed": 0,
 "correct": true, "pred": "\\frac{1}{2}", "reference": "\\frac12",
 "cost": {"calls": 2, "prompt_tokens": 3120, "prefill_tokens": 140, "generated_tokens": 2980,
          "thinking_tokens": 2810, "latency_s": 41.2},
 "signals": {"mean_logprob": -0.21, "tail_mean_logprob": -0.05, "think_hit_budget": 0.0, "...": 0},
 "steps": [{"name": "think", "text": "..."}, {"name": "answer", "text": "..."}]}
```

The `steps` field holds the full trajectory. It is the input Phase II's Experience Engine will consume.

## Caveats

- **Code execution is not a sandbox.** MBPP+ grading runs model-written code in a subprocess with time and memory limits. Only run code benchmarks in a disposable environment such as a Kaggle or rented container.
- **Contamination.** GSM8K and MATH are probably in the base model's training data. Use them for comparisons between policies on the same model, not as absolute capability claims; AIME 2025 is the held-out check.
- **FLOPs** use 2·N·tokens and ignore attention, which understates long-context cost.
