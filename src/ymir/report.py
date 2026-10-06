"""Markdown (and optional PNG) reports from run logs."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from ymir import metrics
from ymir.policies.confidence import Scorer


def _fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def build_report(
    records: list[dict[str, Any]],
    params_b: float | None = None,
    oracle_of: list[str] | None = None,
    cascade_stages: list[str] | None = None,
    cascade_scorers: list[Scorer] | None = None,
    grid: list[float] | None = None,
    target_ref: str | None = None,
    target_slack: float = 0.01,
    eval_split_frac: float | None = None,
    plot_dir: str | None = None,
) -> str:
    if eval_split_frac:
        records = [r for r in records if not metrics.in_train_split(r["task_id"], eval_split_frac)]

    benchmarks = sorted({r["benchmark"] for r in records})
    lines = ["# Ymir Phase I report", ""]
    if eval_split_frac:
        lines += [f"Evaluated on held-out tasks only (router train fraction {eval_split_frac}).", ""]
    lines += [
        "Tokens are mean generated tokens per task (thinking + answer). TFLOPs use 2·N·tokens and ignore attention.",
        "Accuracy CIs are 95% bootstrap over tasks, after averaging seeds per task.",
        "",
    ]

    for bench in benchmarks:
        recs = [r for r in records if r["benchmark"] == bench]
        extra: list[dict[str, Any]] = []
        if oracle_of:
            extra += metrics.oracle(recs, bench, oracle_of)
        rows = metrics.summarize(recs + extra, params_b)
        frontier_ids = {p["policy"] for p in metrics.pareto_frontier(rows)}

        lines += [f"## {bench}", ""]
        header = "| Policy | Accuracy | 95% CI | Gen tokens | Think tokens | Latency s | Calls |"
        sep = "| --- | --- | --- | --- | --- | --- | --- |"
        if params_b:
            header += " TFLOPs |"
            sep += " --- |"
        lines += [header + " Frontier |", sep + " --- |"]
        for row in sorted(rows, key=lambda r: r["gen_tokens"]):
            line = (
                f"| {row['policy']} | {_fmt(row['accuracy'])} | {_fmt(row['acc_lo'])}–{_fmt(row['acc_hi'])} "
                f"| {_fmt(row['gen_tokens'], 0)} | {_fmt(row['think_tokens'], 0)} | {_fmt(row['latency_s'], 2)} | {_fmt(row['calls'], 1)} |"
            )
            if params_b:
                line += f" {_fmt(row.get('tflops', float('nan')), 2)} |"
            line += " yes |" if row["policy"] in frontier_ids else " |"
            if row["errors"]:
                line += f" ({row['errors']} errors)"
            lines.append(line)
        lines.append("")

        sweep = []
        if cascade_stages and cascade_scorers and all(s in {r["policy"] for r in recs} for s in cascade_stages):
            sweep = metrics.sweep_cascade(recs, bench, cascade_stages, cascade_scorers, grid or list(np.linspace(0, 1, 21)))
            lines += [f"### Cascade sweep: {' > '.join(cascade_stages)}", ""]
            lines += ["| Threshold | Accuracy | Gen tokens | Escalated |", "| --- | --- | --- | --- |"]
            for p in sweep:
                lines.append(f"| {p['threshold']:.2f} | {_fmt(p['accuracy'])} | {_fmt(p['gen_tokens'], 0)} | {_fmt(p['escalation_rate'], 2)} |")
            lines.append("")

        fixed_points = [r for r in rows if not r["policy"].startswith("B5")]
        if target_ref:
            ref = next((r for r in rows if r["policy"] == target_ref), None)
            if ref:
                target = ref["accuracy"] - target_slack
                lines += [f"### Compute to reach {target_ref} accuracy minus {target_slack:g} ({_fmt(target)})", ""]
                best_fixed = metrics.compute_to_target(fixed_points, target)
                lines.append(f"- Best fixed policies: {_fmt(best_fixed, 0)} tokens ({_fmt(best_fixed / ref['gen_tokens'], 2)}× of {target_ref})")
                if sweep:
                    casc = metrics.compute_to_target(sweep, target)
                    verdict = "meets" if not math.isnan(casc) and casc <= 0.6 * ref["gen_tokens"] else "does not meet"
                    lines.append(
                        f"- Cascade: {_fmt(casc, 0)} tokens ({_fmt(casc / ref['gen_tokens'], 2)}× of {target_ref}); "
                        f"{verdict} the ≤0.60× success bar"
                    )
                lines.append("")

        levels = metrics.accuracy_by_level(recs, bench)
        if levels:
            all_levels = sorted({lvl for d in levels.values() for lvl in d})
            lines += ["### Accuracy by difficulty level", ""]
            lines += ["| Policy | " + " | ".join(f"L{lvl}" for lvl in all_levels) + " |", "| --- |" + " --- |" * len(all_levels)]
            for pol in sorted(levels):
                lines.append(f"| {pol} | " + " | ".join(_fmt(levels[pol].get(lvl, float("nan")), 2) for lvl in all_levels) + " |")
            lines.append("")

        if plot_dir:
            _plot(bench, rows, sweep, plot_dir)
            lines += [f"![{bench}]({os.path.join(plot_dir, bench + '.png')})", ""]

    return "\n".join(lines)


def _plot(bench: str, rows: list[dict[str, Any]], sweep: list[dict[str, Any]], plot_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ymir] matplotlib not installed; skipping plots (pip install -e .[plot])")
        return
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for r in rows:
        ax.errorbar(
            r["gen_tokens"], r["accuracy"],
            yerr=[[r["accuracy"] - r["acc_lo"]], [r["acc_hi"] - r["accuracy"]]],
            fmt="o", color="#555555", capsize=2, markersize=4,
        )
        ax.annotate(r["policy"], (r["gen_tokens"], r["accuracy"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    if sweep:
        xs = [p["gen_tokens"] for p in sweep]
        ys = [p["accuracy"] for p in sweep]
        # Markers on every sweep point: gaps between thresholds are real, not interpolated.
        ax.plot(xs, ys, "-o", color="#2a6fdb", linewidth=1.2, markersize=2.5, label="Gravitas-lite (threshold sweep)")
        ax.legend(fontsize=7, frameon=False)
    ax.set_xscale("log")
    ax.set_xlabel("Mean generated tokens per task (log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title(bench, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{bench}.png"))
    plt.close(fig)
