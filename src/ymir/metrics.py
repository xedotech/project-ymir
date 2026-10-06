"""Turning run logs into the Phase I numbers.

All functions work on the JSONL records written by the runner. Accuracy is averaged
over seeds per task first, so confidence intervals resample tasks (the unit that
actually varies), not task-seed pairs.

Intelligence per Compute is reported three ways (see docs/phase1.md):
  1. accuracy at matched budget       -> pareto_frontier + accuracy_at_budget
  2. compute to reach a target        -> compute_to_target
  3. marginal efficiency vs a cheap reference -> marginal_efficiency
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Callable, Iterable

import numpy as np

from ymir.accounting import flops
from ymir.policies.confidence import Scorer

Record = dict[str, Any]


def load_records(paths: Iterable[str]) -> list[Record]:
    """Read run logs. If a (benchmark, task, policy, seed) appears more than once
    (a failed job that was retried), the latest record wins."""
    latest: dict[tuple, Record] = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest[(r["benchmark"], r["task_id"], r["policy"], int(r["seed"]))] = r
    return list(latest.values())


def in_train_split(task_id: str, frac: float) -> bool:
    """Stable hash split, so a router trained on part of a benchmark is only ever
    evaluated on the other part."""
    h = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def gen_tokens(r: Record) -> float:
    return float(r.get("cost", {}).get("generated_tokens", 0))


def index(records: list[Record], benchmark: str) -> dict[str, dict[tuple[str, int], Record]]:
    """policy -> (task_id, seed) -> record, for one benchmark."""
    out: dict[str, dict[tuple[str, int], Record]] = defaultdict(dict)
    for r in records:
        if r["benchmark"] == benchmark:
            out[r["policy"]][(r["task_id"], int(r["seed"]))] = r
    return out


def _per_task(recs: Iterable[Record], value: Callable[[Record], float]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        acc[r["task_id"]].append(value(r))
    return {t: float(np.mean(v)) for t, v in acc.items()}


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(records: list[Record], params_b: float | None = None, n_boot: int = 2000) -> list[dict[str, Any]]:
    """One row per (benchmark, policy)."""
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        groups[(r["benchmark"], r["policy"])].append(r)
    rows = []
    for (bench, pol), recs in sorted(groups.items()):
        ok = [r for r in recs if r.get("error") is None]
        acc_by_task = _per_task(recs, lambda r: float(r["correct"]))
        vals = np.array(list(acc_by_task.values()))
        lo, hi = bootstrap_ci(vals, n_boot)
        cost = lambda key: float(np.mean([r["cost"].get(key, 0) for r in ok])) if ok else float("nan")  # noqa: E731
        row = {
            "benchmark": bench,
            "policy": pol,
            "tasks": len(acc_by_task),
            "records": len(recs),
            "errors": len(recs) - len(ok),
            "accuracy": float(vals.mean()) if len(vals) else float("nan"),
            "acc_lo": lo,
            "acc_hi": hi,
            "gen_tokens": cost("generated_tokens"),
            "think_tokens": cost("thinking_tokens"),
            "prefill_tokens": cost("prefill_tokens"),
            "latency_s": cost("latency_s"),
            "calls": cost("calls"),
        }
        if params_b:
            row["tflops"] = flops(params_b, row["prefill_tokens"], row["gen_tokens"]) / 1e12
        rows.append(row)
    return rows


def paired_diff(records: list[Record], benchmark: str, a: str, b: str, n_boot: int = 2000) -> dict[str, float]:
    """Accuracy and token difference a - b on the tasks both policies ran."""
    idx = index(records, benchmark)
    acc_a = _per_task(idx[a].values(), lambda r: float(r["correct"]))
    acc_b = _per_task(idx[b].values(), lambda r: float(r["correct"]))
    tok_a = _per_task(idx[a].values(), gen_tokens)
    tok_b = _per_task(idx[b].values(), gen_tokens)
    common = sorted(set(acc_a) & set(acc_b))
    d_acc = np.array([acc_a[t] - acc_b[t] for t in common])
    d_tok = np.array([tok_a[t] - tok_b[t] for t in common])
    lo, hi = bootstrap_ci(d_acc, n_boot)
    return {
        "tasks": len(common),
        "acc_diff": float(d_acc.mean()) if len(common) else float("nan"),
        "acc_diff_lo": lo,
        "acc_diff_hi": hi,
        "token_ratio": float(np.mean([tok_a[t] for t in common]) / max(np.mean([tok_b[t] for t in common]), 1e-9)) if common else float("nan"),
        "tok_diff": float(d_tok.mean()) if len(common) else float("nan"),
    }


# --- Synthetic policies built from logs ----------------------------------------------


def oracle(records: list[Record], benchmark: str, policies: list[str], name: str = "B5-oracle") -> list[Record]:
    """Hindsight router: per task, the cheapest policy that was correct (or the
    cheapest overall if none was). An upper bound for any real router."""
    idx = index(records, benchmark)
    keys = set.intersection(*(set(idx[p]) for p in policies)) if policies else set()
    out = []
    for key in sorted(keys):
        cands = sorted((idx[p][key] for p in policies), key=gen_tokens)
        right = [r for r in cands if r["correct"]]
        pick = right[0] if right else cands[0]
        out.append({**pick, "policy": name, "chosen": pick["policy"], "steps": None})
    return out


def simulate_cascade(
    records: list[Record],
    benchmark: str,
    stages: list[str],
    scorers: list[Scorer],
    thresholds: list[float],
    name: str | None = None,
    task_filter: Callable[[str], bool] | None = None,
) -> list[Record]:
    """Replay a cascade over logged stage runs (same task, same seed).

    Exact for the live Cascade policy, because each stage runs with the same seed
    whether it is run alone or inside a cascade."""
    idx = index(records, benchmark)
    keys = set.intersection(*(set(idx[s]) for s in stages))
    name = name or "sim:" + ">".join(stages) + "@" + ",".join(f"{t:g}" for t in thresholds)
    out = []
    for key in sorted(keys):
        if task_filter and not task_filter(key[0]):
            continue
        total: dict[str, float] = defaultdict(float)
        for i, s in enumerate(stages):
            r = idx[s][key]
            for k, v in r.get("cost", {}).items():
                total[k] += v
            if i == len(stages) - 1:
                break
            if scorers[i].score(r.get("signals", {})) >= thresholds[i]:
                break
        out.append(
            {
                "benchmark": benchmark,
                "task_id": key[0],
                "seed": key[1],
                "policy": name,
                "correct": r["correct"],
                "error": r.get("error"),
                "cost": dict(total),
                "stopped_at": s,
                "meta": r.get("meta", {}),
            }
        )
    return out


def sweep_cascade(
    records: list[Record],
    benchmark: str,
    stages: list[str],
    scorers: list[Scorer],
    grid: Iterable[float],
    task_filter: Callable[[str], bool] | None = None,
) -> list[dict[str, Any]]:
    """Accuracy and tokens for one shared threshold per point of `grid`."""
    points = []
    for t in grid:
        recs = simulate_cascade(records, benchmark, stages, scorers, [t] * (len(stages) - 1), task_filter=task_filter)
        if not recs:
            continue
        acc = float(np.mean(list(_per_task(recs, lambda r: float(r["correct"])).values())))
        tok = float(np.mean([gen_tokens(r) for r in recs]))
        escalated = float(np.mean([r["stopped_at"] != stages[0] for r in recs]))
        points.append({"threshold": float(t), "accuracy": acc, "gen_tokens": tok, "escalation_rate": escalated})
    return points


# --- Intelligence-per-compute summaries ------------------------------------------------


def pareto_frontier(points: list[dict[str, Any]], x: str = "gen_tokens", y: str = "accuracy") -> list[dict[str, Any]]:
    """Points not dominated by any cheaper-or-equal point with higher-or-equal accuracy."""
    pts = sorted(points, key=lambda p: (p[x], -p[y]))
    frontier, best = [], -np.inf
    for p in pts:
        if p[y] > best:
            frontier.append(p)
            best = p[y]
    return frontier


def accuracy_at_budget(points: list[dict[str, Any]], budget: float, x: str = "gen_tokens", y: str = "accuracy") -> float:
    """Best accuracy reachable at or under `budget` tokens, interpolating along the frontier."""
    f = pareto_frontier(points, x, y)
    if not f or budget < f[0][x]:
        return float("nan")
    for a, b in zip(f, f[1:]):
        if a[x] <= budget <= b[x]:
            w = (budget - a[x]) / max(b[x] - a[x], 1e-9)
            return a[y] + w * (b[y] - a[y])
    return f[-1][y]


def compute_to_target(points: list[dict[str, Any]], target: float, x: str = "gen_tokens", y: str = "accuracy") -> float:
    """Fewest tokens at which the frontier reaches `target` accuracy (interpolated)."""
    f = pareto_frontier(points, x, y)
    if not f or f[-1][y] < target:
        return float("nan")
    if f[0][y] >= target:
        return f[0][x]
    for a, b in zip(f, f[1:]):
        if a[y] < target <= b[y]:
            w = (target - a[y]) / max(b[y] - a[y], 1e-12)
            return a[x] + w * (b[x] - a[x])
    return float("nan")


def marginal_efficiency(row: dict[str, Any], reference: dict[str, Any]) -> float:
    """(Acc - Acc_ref) / (tokens - tokens_ref): accuracy gained per extra generated token."""
    d_tok = row["gen_tokens"] - reference["gen_tokens"]
    if d_tok <= 0:
        return float("nan")
    return (row["accuracy"] - reference["accuracy"]) / d_tok


def accuracy_by_level(records: list[Record], benchmark: str) -> dict[str, dict[int, float]]:
    """policy -> difficulty level -> accuracy. Shows whether allocation tracks difficulty."""
    out: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["benchmark"] == benchmark and "level" in (r.get("meta") or {}):
            out[r["policy"]][int(r["meta"]["level"])].append(float(r["correct"]))
    return {p: {lvl: float(np.mean(v)) for lvl, v in sorted(d.items())} for p, d in out.items()}
