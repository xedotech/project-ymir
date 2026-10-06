#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server for the harness.
#
# Kaggle (2x T4, 16 GB each):   TP=2 DTYPE=half ./scripts/serve_vllm.sh
# Single 24-48 GB GPU:           ./scripts/serve_vllm.sh
# Fast iteration:                MODEL=Qwen/Qwen3-1.7B ./scripts/serve_vllm.sh
#
# T4s have no bf16, so use DTYPE=half there. Qwen3-8B in fp16 is about 16 GB of
# weights, which leaves limited KV cache on 2x T4; long thinking budgets (B3) are
# better run on the rented GPU, or use MODEL=Qwen/Qwen3-8B-AWQ with QUANT=awq.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
TP="${TP:-1}"
DTYPE="${DTYPE:-auto}"
MAX_LEN="${MAX_LEN:-36864}"   # prompt + 32k thinking + answer for B3
GPU_UTIL="${GPU_UTIL:-0.92}"
PORT="${PORT:-8000}"
QUANT_ARGS=()
if [[ -n "${QUANT:-}" ]]; then QUANT_ARGS=(--quantization "$QUANT"); fi

exec vllm serve "$MODEL" \
  --dtype "$DTYPE" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching \
  --max-logprobs 20 \
  --port "$PORT" \
  "${QUANT_ARGS[@]}"
