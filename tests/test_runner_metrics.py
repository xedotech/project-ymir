import json

import numpy as np

from ymir import metrics
from ymir.cli import main
from ymir.policies import LogisticScorer, SignalScorer
from ymir.runner import run


def _cfg(tmp_path, policies=None, extra=None):
    cfg = {
        "run": {
            "name": "t",
            "out_dir": str(tmp_path),
            "data_dir": str(tmp_path / "data"),
            "seeds": [0, 1],
            "concurrency": 4,
            "logprobs": 5,
            "log_text": True,
            "policies": policies or ["B0", "B3"],
        },
        "backend": {"type": "mock"},
        "benchmarks": [{"name": "toy", "n": 60}],
        "policies": [
            {"id": "B0", "type": "nothink", "max_tokens": 64},
            {"id": "B3", "type": "think", "budget": 4096},
            *(extra or []),
        ],
    }
    return cfg


def test_run_is_resumable(tmp_path):
    path = run(_cfg(tmp_path))
    n1 = sum(1 for _ in open(path))
    assert n1 == 60 * 2 * 2
    run(_cfg(tmp_path))
    assert sum(1 for _ in open(path)) == n1  # nothing re-run
    rec = json.loads(open(path).readline())
    for key in ("benchmark", "task_id", "policy", "seed", "correct", "cost", "signals", "steps"):
        assert key in rec


def test_simulated_cascade_matches_live_cascade(tmp_path):
    thr = -0.3
    live = {"id": "G", "type": "cascade", "stages": ["B0", "B3"], "thresholds": [thr], "scorer": {"type": "signal", "signal": "mean_logprob"}}
    path = run(_cfg(tmp_path, policies=["B0", "B3", "G"], extra=[live]))
    recs = metrics.load_records([path])
    sim = metrics.simulate_cascade(recs, "toy", ["B0", "B3"], [SignalScorer("mean_logprob")], [thr])
    live_recs = {(r["task_id"], r["seed"]): r for r in recs if r["policy"] == "G"}
    assert len(sim) == len(live_recs)
    for s in sim:
        lr = live_recs[(s["task_id"], s["seed"])]
        assert s["correct"] == lr["correct"]
        assert s["cost"]["generated_tokens"] == lr["cost"]["generated_tokens"]
        assert s["stopped_at"] == lr["stopped_at"]


def test_summary_oracle_and_frontier(tmp_path):
    recs = metrics.load_records([run(_cfg(tmp_path))])
    rows = {r["policy"]: r for r in metrics.summarize(recs, params_b=8.0)}
    assert rows["B3"]["accuracy"] > rows["B0"]["accuracy"]
    assert rows["B3"]["gen_tokens"] > rows["B0"]["gen_tokens"]
    assert rows["B0"]["acc_lo"] <= rows["B0"]["accuracy"] <= rows["B0"]["acc_hi"]
    orc = metrics.summarize(metrics.oracle(recs, "toy", ["B0", "B3"]))[0]
    assert orc["accuracy"] >= rows["B3"]["accuracy"]
    assert orc["gen_tokens"] <= rows["B3"]["gen_tokens"]


def test_pareto_and_targets():
    pts = [
        {"policy": "a", "gen_tokens": 10, "accuracy": 0.5},
        {"policy": "b", "gen_tokens": 100, "accuracy": 0.9},
        {"policy": "c", "gen_tokens": 200, "accuracy": 0.8},  # dominated
    ]
    assert [p["policy"] for p in metrics.pareto_frontier(pts)] == ["a", "b"]
    assert np.isclose(metrics.compute_to_target(pts, 0.7), 55.0)
    assert np.isclose(metrics.accuracy_at_budget(pts, 55), 0.7)
    assert np.isnan(metrics.compute_to_target(pts, 0.95))
    assert np.isclose(metrics.marginal_efficiency(pts[1], pts[0]), 0.4 / 90)


def test_logistic_router_learns_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    rows = [{"good": float(v) + rng.normal(0, 0.5), "noise": rng.normal()} for v in y]
    sc = LogisticScorer.fit(rows, y.tolist(), ["good", "noise"])
    assert abs(sc.weights[0]) > 3 * abs(sc.weights[1])
    assert sc.score({"good": 1.0, "noise": 0}) > 0.5 > sc.score({"good": 0.0, "noise": 0})


def test_cli_end_to_end(tmp_path, capsys):
    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text(
        f"""
[run]
name = "cli"
out_dir = "{tmp_path}"
seeds = [0]
policies = ["B0", "B3"]
[backend]
type = "mock"
[[benchmarks]]
name = "toy"
n = 80
[[policies]]
id = "B0"
type = "nothink"
max_tokens = 64
[[policies]]
id = "B3"
type = "think"
budget = 4096
"""
    )
    main(["run", str(cfg_path)])
    runs = str(tmp_path / "cli.jsonl")
    router = str(tmp_path / "r.json")
    main(["train-router", runs, "--policy", "B0", "--split-frac", "0.5", "--out", router])
    out = tmp_path / "report.md"
    main(["report", runs, "--oracle", "B0,B3", "--cascade", "B0,B3", "--router", router,
          "--target-ref", "B3", "--eval-split-frac", "0.5", "--out", str(out)])
    text = out.read_text()
    assert "Cascade sweep" in text and "B5-oracle" in text and "Compute to reach" in text


def test_failed_jobs_are_retried_and_latest_record_wins(tmp_path):
    path = tmp_path / "t.jsonl"
    base = {"benchmark": "toy", "task_id": "toy-0", "policy": "B0", "seed": 0}
    path.write_text(
        json.dumps({**base, "correct": False, "error": "TimeoutError: x"}) + "\n"
    )
    from ymir.runner import existing_keys

    assert existing_keys(str(path)) == set()
    with open(path, "a") as f:
        f.write(json.dumps({**base, "correct": True, "error": None, "cost": {}}) + "\n")
    recs = metrics.load_records([str(path)])
    assert len(recs) == 1 and recs[0]["correct"]
