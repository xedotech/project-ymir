"""Experiment configs are TOML files (see configs/). Only the standard library is
needed to read them on Python 3.11+."""

from __future__ import annotations

import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


def load_config(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for section in ("run", "backend", "benchmarks", "policies"):
        if section not in cfg:
            raise ValueError(f"{path}: missing [{section}]")
    cfg["run"].setdefault("seeds", [0])
    cfg["run"].setdefault("concurrency", 8)
    cfg["run"].setdefault("out_dir", "runs")
    cfg["run"].setdefault("data_dir", "data")
    cfg["run"].setdefault("logprobs", 5)
    cfg["run"].setdefault("log_text", True)
    return cfg
