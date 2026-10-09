# project-ymir

Ymir Labs research harness: evaluation and experiment tooling for efficient open models.

The harness runs open-weight models on public benchmarks under a set of answering strategies and records two things for every task: whether the answer was right, and exactly what it cost (calls, tokens, latency). Reports compare strategies on accuracy versus compute.

- Evaluation definitions: [docs/phase1.md](docs/phase1.md)

## What is here

| Path | What it does |
| --- | --- |
| `src/ymir/backends/` | Client for any OpenAI-compatible `/v1/completions` server (e.g. vLLM), plus a deterministic mock model for GPU-free testing |
| `src/ymir/prompts.py` | Prompt formats |
| `src/ymir/benchmarks/` | Loaders for MATH-500, GSM8K, MBPP+, AIME 2025, training splits, and a synthetic toy set |
| `src/ymir/grading/` | Final-answer extraction and equivalence for math; execution of hidden tests for code |
| `src/ymir/policies/` | Answering strategies (fixed baselines and a cascade) |
| `src/ymir/runner.py` | Runs benchmark × policy × seed, appends one JSONL record per task, resumes after interruption |
| `src/ymir/metrics.py` | Bootstrap CIs, paired differences, oracle baseline, offline replay, Pareto frontier, compute-to-target |
| `src/ymir/report.py` | Markdown report and accuracy-versus-tokens plots |
| `configs/` | Experiment configs (TOML) |

## Policies

| Id | Policy |
| --- | --- |
| B0 | Direct answer |
| B1 | Visible step-by-step |
| B2-1k / 4k / 16k | Reasoning capped at a fixed token budget |
| B3 | Reasoning at full budget |
| B4-sc5 | 5 samples of B1, majority vote (math only) |
| B5-oracle | Cheapest correct policy per task, in hindsight (computed from logs) |
| Cascade | Multi-stage strategy evaluated against the baselines |

## Quick start without a GPU

```bash
pip install -e ".[dev]"
pytest -q

ymir run configs/smoke_mock.toml
ymir train-router runs/smoke-mock.jsonl --policy B0 --split-frac 0.5 --out runs/router_b0.json
ymir report runs/smoke-mock.jsonl --oracle B0,B2-256,B2-1k,B3 --cascade B0,B3 \
    --router runs/router_b0.json --target-ref B3 --eval-split-frac 0.5
```

The mock model solves synthetic arithmetic. Its numbers mean nothing; it exists so the full pipeline can be tested anywhere.

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

   T4s are Turing GPUs (no bf16). If the installed vLLM version refuses them, pin an older vLLM or use a rented GPU. Set `MODEL` to a smaller model while debugging.

3. **Data.** `ymir data fetch configs/baselines_qwen3.toml` downloads the benchmarks to `data/` and prints one example of each. Check the examples: hub dataset ids and column names drift, and every loader accepts overrides in the config (`hf_id`, `hf_config(s)`, `split`, `question_field`, `answer_field`).

4. **Baselines.** Smoke test, then the full grid. Runs are resumable, so rerun the same command after a session timeout.

   ```bash
   ymir run configs/baselines_qwen3.toml --limit 20
   ymir run configs/baselines_qwen3.toml
   ymir report runs/baselines-qwen3-8b.jsonl --params-b 8.2 --oracle B0,B1,B2-1k,B2-4k,B2-16k,B3 \
       --target-ref B3 --plot-dir reports/plots --out reports/baselines.md
   ```

5. **Cascade.** Follow the commands below. Replays over logged runs need no extra model calls.

   ```bash
   ymir data fetch configs/router_train_qwen3.toml
   ymir run configs/router_train_qwen3.toml
   ymir train-router runs/router-train-qwen3-8b.jsonl --policy B0 --out routers/b0.json
   ymir report runs/baselines-qwen3-8b.jsonl --params-b 8.2 --cascade B0,B3 --router routers/b0.json \
       --target-ref B3 --plot-dir reports/plots --out reports/cascade.md
   ```

   Offline replay and live runs agree when the stage configs and seeds match; a test checks this.

## Record format

Each line of `runs/<name>.jsonl` is one (benchmark, task, policy, seed):

```json
{"benchmark": "math500", "task_id": "test/algebra/1.json", "policy": "B2-4k", "seed": 0,
 "correct": true, "pred": "\\frac{1}{2}", "reference": "\\frac12",
 "cost": {"calls": 2, "prompt_tokens": 3120, "prefill_tokens": 140, "generated_tokens": 2980,
          "thinking_tokens": 2810, "latency_s": 41.2},
 "signals": {"...": 0},
 "steps": [{"name": "think", "text": "..."}, {"name": "answer", "text": "..."}]}
```

## Caveats

- **Code execution is not a sandbox.** MBPP+ grading runs model-written code in a subprocess with time and memory limits. Only run code benchmarks in a disposable environment such as a Kaggle or rented container.
- **Contamination.** GSM8K and MATH are probably in base models' training data. Use them to compare policies on the same model, not as absolute capability claims; AIME 2025 is the held-out check.
- **FLOPs** use 2·N·tokens and ignore attention, which understates long-context cost.
