"""Command line entry point: `ymir <command>` or `python -m ymir <command>`.

  ymir data fetch CONFIG          download benchmarks to data/*.jsonl (needs the data extra)
  ymir run CONFIG                 run the policy grid, append to runs/<name>.jsonl
  ymir train-router RUNS...       fit a P(correct | signals) router on logged runs
  ymir report RUNS...             markdown summary, cascade sweep, optional plots
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from ymir import metrics
from ymir.benchmarks import cache_path, load_benchmark, save_tasks
from ymir.config import load_config
from ymir.policies.confidence import LogisticScorer, make_scorer

DEFAULT_FEATURES = [
    "mean_logprob",
    "min_logprob",
    "tail_mean_logprob",
    "frac_low_conf",
    "mean_topk_entropy",
    "completion_tokens_log",
    "finished",
    "has_answer",
    "question_chars_log",
]


def cmd_data_fetch(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data_dir = cfg["run"]["data_dir"]
    for bcfg in cfg["benchmarks"]:
        if bcfg.get("loader", bcfg["name"]) == "toy":
            continue
        tasks = load_benchmark({**bcfg, "limit": 0}, data_dir="/nonexistent")
        path = cache_path(data_dir, bcfg["name"])
        save_tasks(tasks, path)
        print(f"[ymir] {bcfg['name']}: {len(tasks)} tasks -> {path}")
        if tasks:
            t = tasks[0]
            print(f"        example: id={t.id} reference={str(t.reference)[:60]!r} question={t.question[:80]!r}")


def cmd_run(args: argparse.Namespace) -> None:
    from ymir.runner import run

    cfg = load_config(args.config)
    run(cfg, only_benchmarks=args.benchmarks, only_policies=args.policies, limit=args.limit)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cmd_train_router(args: argparse.Namespace) -> None:
    records = metrics.load_records(args.runs)
    recs = [
        r for r in records
        if r["policy"] == args.policy and r.get("signals") and r.get("error") is None
        and (not args.benchmarks or r["benchmark"] in args.benchmarks)
    ]
    if args.split_frac:
        train = [r for r in recs if metrics.in_train_split(r["task_id"], args.split_frac)]
        test = [r for r in recs if not metrics.in_train_split(r["task_id"], args.split_frac)]
    else:
        train, test = recs, []
    if len(train) < 20:
        sys.exit(f"Only {len(train)} training records for policy {args.policy}; need at least 20.")
    features = args.features or [f for f in DEFAULT_FEATURES if any(f in r["signals"] for r in train)]
    scorer = LogisticScorer.fit([r["signals"] for r in train], [int(r["correct"]) for r in train], features, l2=args.l2)
    scorer.meta = {
        "policy": args.policy,
        "benchmarks": sorted({r["benchmark"] for r in train}),
        "split_frac": args.split_frac,
        "n_train": len(train),
        "base_rate": float(np.mean([r["correct"] for r in train])),
    }
    for name, part in (("train", train), ("test", test)):
        if not part:
            continue
        s = np.array([scorer.score(r["signals"]) for r in part])
        y = np.array([int(r["correct"]) for r in part])
        scorer.meta[f"{name}_auc"] = _auc(s, y)
        print(f"[ymir] {name}: n={len(part)} AUC={scorer.meta[f'{name}_auc']:.3f} base rate={y.mean():.3f}")
    scorer.save(args.out)
    print(f"[ymir] router for {args.policy} on {len(features)} features -> {args.out}")
    print(json.dumps(dict(zip(features, np.round(scorer.weights, 3).tolist())), indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    from ymir.report import build_report

    records = metrics.load_records(args.runs)
    stages = args.cascade.split(",") if args.cascade else None
    scorers = None
    if stages:
        if args.router:
            scorers = [make_scorer(p) for p in args.router]
            if len(scorers) == 1:
                scorers = scorers * (len(stages) - 1)
        else:
            scorers = [make_scorer({"type": "signal", "signal": args.signal})] * (len(stages) - 1)
    grid = [float(x) for x in args.grid.split(",")] if args.grid else None
    if grid is None and stages and not args.router:
        # Raw log-prob signals are not in [0, 1]; sweep over their observed range.
        vals = [r["signals"].get(args.signal) for r in records if r["policy"] == stages[0] and r.get("signals")]
        vals = [v for v in vals if v is not None]
        if vals:
            grid = list(np.quantile(vals, np.linspace(0, 1, 21)))
    text = build_report(
        records,
        params_b=args.params_b,
        oracle_of=args.oracle.split(",") if args.oracle else None,
        cascade_stages=stages,
        cascade_scorers=scorers,
        grid=grid,
        target_ref=args.target_ref,
        eval_split_frac=args.eval_split_frac,
        plot_dir=args.plot_dir,
    )
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"[ymir] report -> {args.out}")
    else:
        print(text)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ymir", description="Ymir Labs Phase I harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="dataset utilities")
    dsub = d.add_subparsers(dest="data_cmd", required=True)
    df = dsub.add_parser("fetch", help="download a config's benchmarks into its data_dir")
    df.add_argument("config")
    df.set_defaults(func=cmd_data_fetch)

    r = sub.add_parser("run", help="run a config's policy grid")
    r.add_argument("config")
    r.add_argument("--benchmarks", nargs="*", help="only these benchmark names")
    r.add_argument("--policies", nargs="*", help="only these policy ids")
    r.add_argument("--limit", type=int, help="first N tasks per benchmark (smoke tests)")
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("train-router", help="fit P(correct | signals) for one policy's outputs")
    t.add_argument("runs", nargs="+")
    t.add_argument("--policy", required=True, help="the cascade stage this router gates, e.g. B0")
    t.add_argument("--benchmarks", nargs="*", help="train only on these benchmarks (e.g. training splits)")
    t.add_argument("--split-frac", type=float, default=None, help="train on this hashed fraction of tasks, report AUC on the rest")
    t.add_argument("--features", nargs="*")
    t.add_argument("--l2", type=float, default=1.0)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_train_router)

    rp = sub.add_parser("report", help="summarize runs")
    rp.add_argument("runs", nargs="+")
    rp.add_argument("--params-b", type=float, help="model size in billions, for FLOP estimates")
    rp.add_argument("--oracle", help="comma-separated policy ids for the B5 oracle router")
    rp.add_argument("--cascade", help="comma-separated stage policy ids to sweep, e.g. B0,B2-4k,B3")
    rp.add_argument("--router", nargs="*", help="router JSON per stage boundary (one is reused for all)")
    rp.add_argument("--signal", default="mean_logprob", help="raw signal to threshold when no router is given")
    rp.add_argument("--grid", help="comma-separated thresholds")
    rp.add_argument("--target-ref", help="policy whose accuracy defines the compute-to-target check, e.g. B3")
    rp.add_argument("--eval-split-frac", type=float, help="evaluate only on tasks outside the router's training split")
    rp.add_argument("--plot-dir")
    rp.add_argument("--out")
    rp.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
