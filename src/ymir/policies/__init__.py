from __future__ import annotations

from typing import Any

from ymir.policies.base import Policy, PolicyContext
from ymir.policies.cascade import Cascade
from ymir.policies.confidence import LogisticScorer, SignalScorer, make_scorer
from ymir.policies.fixed import NoThink, SelfConsistency, Think

_SAMPLING_KEYS = ("temperature", "top_p")


def _sampling(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: float(cfg[k]) for k in _SAMPLING_KEYS if k in cfg}


def build_policies(cfgs: list[dict[str, Any]]) -> dict[str, Policy]:
    """Build policies from config tables. Composite policies (self_consistency,
    cascade) refer to other policies by id, so those must be defined earlier."""
    out: dict[str, Policy] = {}
    for cfg in cfgs:
        pid = cfg["id"]
        kind = cfg["type"]
        if pid in out:
            raise ValueError(f"Duplicate policy id: {pid}")
        if kind == "nothink":
            out[pid] = NoThink(id=pid, style=cfg.get("style", "direct"), max_tokens=int(cfg.get("max_tokens", 1024)), **_sampling(cfg))
        elif kind == "think":
            out[pid] = Think(
                id=pid,
                budget=int(cfg.get("budget", 4096)),
                answer_tokens=int(cfg.get("answer_tokens", 1024)),
                **_sampling(cfg),
            )
        elif kind == "self_consistency":
            out[pid] = SelfConsistency(id=pid, inner=_ref(out, cfg["inner"], pid), k=int(cfg.get("k", 5)))
        elif kind == "cascade":
            stages = [_ref(out, s, pid) for s in cfg["stages"]]
            if "scorers" in cfg:
                scorers = [make_scorer(c) for c in cfg["scorers"]]
            else:
                scorers = [make_scorer(cfg.get("scorer", {"type": "signal"}))] * (len(stages) - 1)
            out[pid] = Cascade(
                id=pid,
                stages=stages,
                scorers=scorers,
                thresholds=[float(t) for t in cfg["thresholds"]],
            )
        else:
            raise ValueError(f"Unknown policy type '{kind}' for {pid}")
    return out


def _ref(policies: dict[str, Policy], ref: str, owner: str) -> Policy:
    if ref not in policies:
        raise ValueError(f"{owner} refers to '{ref}', which must be defined before it")
    return policies[ref]


__all__ = [
    "Policy",
    "PolicyContext",
    "NoThink",
    "Think",
    "SelfConsistency",
    "Cascade",
    "SignalScorer",
    "LogisticScorer",
    "build_policies",
]
