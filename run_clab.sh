#!/usr/bin/env bash
# Submit an npc-text-gen experiment or benchmark to clab using uv (works on cn19 / CS-541866 / GPU nodes)
set -euo pipefail

CLUSTER="${CLUSTER:-siue}"
CPUS="${CPUS:-16}"
RAM="${RAM:-32}"
GPUS="${GPUS:-0}"
EST="${EST:-15m}"
NAME="${NAME:-npc-exp}"
BUDGET="${BUDGET:-30}"
MODE="${MODE:-looped-scratch}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

GPU_FLAG=()
DEPS="--with torch --with numpy"
if [[ "$GPUS" != "0" ]]; then
  GPU_FLAG=(--gpus "$GPUS")
fi
if [[ "$MODE" == "adapt" ]]; then
  DEPS="$DEPS --with transformers --with accelerate --with peft --with sentencepiece"
fi

clab submit \
  --project npc-text-gen \
  --name "$NAME" \
  --cluster "$CLUSTER" \
  "${GPU_FLAG[@]}" \
  --cpus "$CPUS" \
  --ram "$RAM" \
  --est "$EST" \
  --wait -- \
  bash -c "set -e; export UV_CACHE_DIR=\"\${UV_CACHE_DIR:-\$HOME/.cache/uv}\"; UV=\"\$HOME/.local/bin/uv\"; \"\$UV\" run -q --no-project --python 3.12 python prepare.py; \"\$UV\" run -q --no-project --python 3.12 $DEPS python train.py --mode $MODE --model $MODEL --budget-sec $BUDGET"
