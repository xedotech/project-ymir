"""Run a grid of benchmark x policy x seed and append one JSONL record per task.

Runs are resumable: records already in the output file are skipped, so a Kaggle
session that hits its time limit can be restarted with the same command.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Iterable

from ymir.backends import make_backend
from ymir.benchmarks import load_benchmark
from ymir.grading import grade
from ymir.policies import Policy, PolicyContext, build_policies
from ymir.prompts import FORMATS
from ymir.types import Task

RECORD_VERSION = 1


def record_key(benchmark: str, task_id: str, policy: str, seed: int) -> tuple[str, str, str, int]:
    return (benchmark, task_id, policy, int(seed))


def existing_keys(path: str) -> set[tuple[str, str, str, int]]:
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partial line from an interrupted write
            if r.get("error"):
                continue  # failed jobs are retried; metrics keep only the latest record per key
            keys.add(record_key(r["benchmark"], r["task_id"], r["policy"], r["seed"]))
    return keys


def run_one(task: Task, policy: Policy, ctx: PolicyContext, seed: int, run_cfg: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    error = None
    try:
        attempt = policy.run(task, ctx, seed)
        g = grade(task, attempt.output)
    except Exception as e:  # record failures instead of losing the whole run
        error = f"{type(e).__name__}: {e}"
        attempt, g = None, None
    rec: dict[str, Any] = {
        "v": RECORD_VERSION,
        "run": run_cfg["name"],
        "benchmark": task.benchmark,
        "task_id": task.id,
        "policy": policy.id,
        "seed": seed,
        "correct": bool(g.correct) if g else False,
        "pred": g.pred if g else None,
        "reference": task.reference if task.kind == "math" else None,
        "grade_detail": g.detail if g else "",
        "error": error,
        "meta": task.meta if task.kind == "math" else {k: v for k, v in task.meta.items() if k != "example_test"},
        "wall_s": time.perf_counter() - start,
    }
    if attempt is not None:
        rec["cost"] = attempt.cost.to_dict()
        rec["signals"] = attempt.signals
        rec["stopped_at"] = attempt.stopped_at
        if run_cfg.get("log_text", True):
            rec["steps"] = [asdict(s) for s in attempt.steps]
    return rec


def iter_jobs(
    tasks_by_bench: dict[str, list[Task]],
    policies: dict[str, Policy],
    policy_ids: list[str],
    seeds: Iterable[int],
    done: set,
) -> Iterable[tuple[Task, Policy, int]]:
    for bench, tasks in tasks_by_bench.items():
        for pid in policy_ids:
            policy = policies[pid]
            for seed in seeds:
                for task in tasks:
                    if not policy.supports(task):
                        continue
                    if record_key(bench, task.id, pid, seed) in done:
                        continue
                    yield task, policy, seed


def run(cfg: dict[str, Any], only_benchmarks: list[str] | None = None, only_policies: list[str] | None = None, limit: int | None = None) -> str:
    run_cfg = cfg["run"]
    out_dir = run_cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_cfg['name']}.jsonl")

    backend = make_backend(cfg["backend"])
    fmt = FORMATS[cfg["backend"].get("prompt_format", "qwen3")]
    ctx = PolicyContext(backend=backend, fmt=fmt, logprobs=int(run_cfg["logprobs"]))
    policies = build_policies(cfg["policies"])
    # Policies listed under `run.policies` are executed; others may exist only as cascade stages.
    policy_ids = run_cfg.get("policies") or list(policies)
    if only_policies:
        policy_ids = [p for p in policy_ids if p in only_policies]

    tasks_by_bench = {}
    for bcfg in cfg["benchmarks"]:
        if only_benchmarks and bcfg["name"] not in only_benchmarks:
            continue
        bcfg = dict(bcfg)
        if limit:
            bcfg["limit"] = limit
        tasks_by_bench[bcfg["name"]] = load_benchmark(bcfg, run_cfg["data_dir"])

    done = existing_keys(out_path)
    jobs = list(iter_jobs(tasks_by_bench, policies, policy_ids, run_cfg["seeds"], done))
    total = len(jobs)
    print(f"[ymir] {run_cfg['name']}: {total} jobs to run ({len(done)} already done) -> {out_path}", flush=True)
    if total == 0:
        return out_path

    lock = threading.Lock()
    finished = correct = errors = 0
    t0 = time.perf_counter()
    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=int(run_cfg["concurrency"])) as pool:
        futures = [pool.submit(run_one, t, p, ctx, s, run_cfg) for t, p, s in jobs]
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                finished += 1
                correct += rec["correct"]
                errors += rec["error"] is not None
                if finished % 25 == 0 or finished == total:
                    rate = finished / max(time.perf_counter() - t0, 1e-9)
                    print(
                        f"[ymir] {finished}/{total} done, acc so far {correct / finished:.3f}, "
                        f"errors {errors}, {rate:.2f} jobs/s",
                        flush=True,
                    )
    return out_path
