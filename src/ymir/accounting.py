"""Turning token counts into FLOPs and dollars.

FLOPs use the standard dense-transformer estimate of 2 * N * tokens, which ignores
the attention term. That underestimates long-context cost, so reports state it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    params_b: float  # model parameters, in billions
    gpu: str = "unknown"
    usd_per_gpu_hour: float = 0.0
    num_gpus: int = 1


def flops(params_b: float, prefill_tokens: int, generated_tokens: int) -> float:
    """Approximate forward-pass FLOPs: 2 * N * (new prompt tokens + generated tokens)."""
    return 2.0 * params_b * 1e9 * (prefill_tokens + generated_tokens)


def dollars(latency_s: float, hw: Hardware, concurrency: int = 1) -> float:
    """Dollar cost of the wall-clock time a task occupied.

    With `concurrency` requests sharing the GPUs, each task is charged its share.
    This is a rough per-task attribution; total run cost (GPU-hours * rate) is exact.
    """
    gpu_hours = latency_s / 3600.0 * hw.num_gpus / max(concurrency, 1)
    return gpu_hours * hw.usd_per_gpu_hour
