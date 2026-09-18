#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi
export QWEN_ATTENTION_BACKEND="${QWEN_ATTENTION_BACKEND:-sdpa}"
export QWEN_CPU_OFFLOAD="${QWEN_CPU_OFFLOAD:-1}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec .venv/bin/python app.py