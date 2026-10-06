"""Benchmark loaders.

Each loader returns a list of `Task`s. Real benchmarks come from the Hugging Face hub
via the optional `datasets` package. `ymir data fetch` saves them as JSONL under
`data/`, and loaders read that cache first, so runs work offline after one fetch.

Hub dataset ids and column names drift. Every hub loader below takes overrides from
config (`hf_id`, `hf_config`, `split`, `question_field`, `answer_field`), and the
generic loader auto-detects common column names.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from typing import Any, Callable

from ymir.backends.mock import _safe_eval
from ymir.grading.math_answer import last_boxed
from ymir.types import Task

Loader = Callable[[dict[str, Any]], list[Task]]


def _hf_rows(hf_id: str, hf_config: str | None, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install the data extra (`pip install -e .[data]`) to fetch hub datasets.") from e
    ds = load_dataset(hf_id, hf_config, split=split) if hf_config else load_dataset(hf_id, split=split)
    return [dict(r) for r in ds]


def _pick(row: dict[str, Any], preferred: str | None, candidates: list[str]) -> Any:
    for key in ([preferred] if preferred else []) + candidates:
        if key and key in row:
            return row[key]
    raise KeyError(f"None of {[preferred] + candidates} found; columns are {sorted(row)}")


# --- toy ---------------------------------------------------------------------------


def load_toy(cfg: dict[str, Any]) -> list[Task]:
    """Synthetic arithmetic with known difficulty (1 to 5 operators). For the mock backend."""
    name = cfg.get("name", "toy")
    n = int(cfg.get("n", 200))
    rng = random.Random(int(cfg.get("seed", 1234)))
    tasks = []
    for i in range(n):
        level = 1 + i % 5
        terms = [str(rng.randint(2, 99)) for _ in range(level + 1)]
        expr = terms[0]
        for t in terms[1:]:
            expr += rng.choice(["+", "-", "*"]) + t
        tasks.append(
            Task(
                id=f"{name}-{i}",
                benchmark=name,
                kind="math",
                question=f"Compute: {expr}",
                reference=str(_safe_eval(expr)),
                meta={"level": level},
            )
        )
    return tasks


# --- math --------------------------------------------------------------------------


def load_gsm8k(cfg: dict[str, Any]) -> list[Task]:
    name = cfg.get("name", "gsm8k")
    rows = _hf_rows(cfg.get("hf_id", "openai/gsm8k"), cfg.get("hf_config", "main"), cfg.get("split", "test"))
    tasks = []
    for i, r in enumerate(rows):
        answer = str(r["answer"]).split("####")[-1].strip().replace(",", "")
        tasks.append(Task(id=f"{name}-{i}", benchmark=name, kind="math", question=r["question"], reference=answer))
    return tasks


def load_math500(cfg: dict[str, Any]) -> list[Task]:
    name = cfg.get("name", "math500")
    rows = _hf_rows(cfg.get("hf_id", "HuggingFaceH4/MATH-500"), cfg.get("hf_config"), cfg.get("split", "test"))
    return [
        Task(
            id=str(r.get("unique_id", f"{name}-{i}")),
            benchmark=name,
            kind="math",
            question=r["problem"],
            reference=str(r["answer"]),
            meta={"level": _level(r.get("level", 0)), "subject": r.get("subject", "")},
        )
        for i, r in enumerate(rows)
    ]


def load_hf_math(cfg: dict[str, Any]) -> list[Task]:
    """Generic question/answer math set, e.g. AIME. Several configs can be concatenated.

    Set `answer_from_solution = true` when the answer column is a worked solution
    (as in the MATH training set); the last \\boxed{} in it is used as the answer.
    """
    name = cfg["name"]
    hf_id = cfg["hf_id"]
    configs = cfg.get("hf_configs") or [cfg.get("hf_config")]
    tasks = []
    for c in configs:
        for i, r in enumerate(_hf_rows(hf_id, c, cfg.get("split", "test"))):
            q = _pick(r, cfg.get("question_field"), ["problem", "question", "Problem", "Question"])
            a = str(_pick(r, cfg.get("answer_field"), ["answer", "Answer", "solution"]))
            if cfg.get("answer_from_solution"):
                boxed = last_boxed(a)
                if boxed is None:
                    continue
                a = boxed
            meta = {"level": _level(r.get("level"))} if r.get("level") is not None else {}
            tasks.append(Task(id=f"{name}-{c or 'default'}-{i}", benchmark=name, kind="math", question=q, reference=a, meta=meta))
    return tasks


def _level(v: Any) -> int:
    """MATH stores levels as "Level 3"; MATH-500 as 3."""
    digits = "".join(ch for ch in str(v) if ch.isdigit())
    return int(digits) if digits else 0


def load_aime2025(cfg: dict[str, Any]) -> list[Task]:
    cfg = {"hf_id": "opencompass/AIME2025", "hf_configs": ["AIME2025-I", "AIME2025-II"], "name": "aime2025", **cfg}
    return load_hf_math(cfg)


# --- code --------------------------------------------------------------------------


def load_mbppplus(cfg: dict[str, Any]) -> list[Task]:
    name = cfg.get("name", "mbppplus")
    rows = _hf_rows(cfg.get("hf_id", "evalplus/mbppplus"), cfg.get("hf_config"), cfg.get("split", "test"))
    tasks = []
    for r in rows:
        imports = r.get("test_imports") or []
        setup = "\n".join(imports) if isinstance(imports, list) else str(imports)
        test_list = r.get("test_list") or []
        tasks.append(
            Task(
                id=f"{name}-{r['task_id']}",
                benchmark=name,
                kind="code",
                question=r["prompt"],
                # The "plus" tests are hidden from the model; one original assert is shown
                # so the model knows the expected function name and signature.
                reference={"test": r["test"], "setup": setup},
                meta={"example_test": test_list[0] if test_list else ""},
            )
        )
    return tasks


LOADERS: dict[str, Loader] = {
    "toy": load_toy,
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "aime2025": load_aime2025,
    "mbppplus": load_mbppplus,
    "hf_math": load_hf_math,
}


def cache_path(data_dir: str, name: str) -> str:
    return os.path.join(data_dir, f"{name}.jsonl")


def save_tasks(tasks: list[Task], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps(asdict(t)) + "\n")


def read_tasks(path: str) -> list[Task]:
    with open(path) as f:
        return [Task(**json.loads(line)) for line in f if line.strip()]


def load_benchmark(cfg: dict[str, Any], data_dir: str = "data") -> list[Task]:
    """Load from the local cache if present, otherwise from the loader. Applies `limit`."""
    name = cfg["name"]
    path = cache_path(data_dir, name)
    if cfg.get("loader", name) != "toy" and os.path.exists(path):
        tasks = read_tasks(path)
    else:
        loader = LOADERS.get(cfg.get("loader", name))
        if loader is None:
            raise ValueError(f"Unknown benchmark '{name}'. Known: {sorted(LOADERS)}")
        tasks = loader(cfg)
    if "shuffle_seed" in cfg:
        # Mixed-source sets (e.g. MATH train, stored by subject) need a shuffle before `limit`.
        tasks = list(tasks)
        random.Random(int(cfg["shuffle_seed"])).shuffle(tasks)
    limit = int(cfg.get("limit", 0))
    return tasks[:limit] if limit > 0 else tasks
