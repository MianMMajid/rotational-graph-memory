#!/bin/bash
set -euo pipefail

# Configuration
BASE_MODEL="${BASE_MODEL:-mlx-community/Llama-3-8B-Instruct-4bit}"
DATA_DIR="./data"
ADAPTER_PATH="./adapters"
FINAL_MODEL_PATH="./model/topological-agent-v1"
ITERATIONS=500

echo "===================================================="
echo "🧠 INITIATING AGENT SLEEP CYCLE (MLX LoRA TRAINING)"
echo "===================================================="
echo "[1/3] Verifying dataset in ${DATA_DIR}..."

if [ ! -f "${DATA_DIR}/topological_train.jsonl" ]; then
  echo "❌ Error: topological_train.jsonl not found in ${DATA_DIR}!"
  exit 1
fi

# mlx_lm expects train.jsonl and valid.jsonl
cp "${DATA_DIR}/topological_train.jsonl" "${DATA_DIR}/train.jsonl"
cp "${DATA_DIR}/topological_train.jsonl" "${DATA_DIR}/valid.jsonl"

echo "✅ Dataset verified."
echo

echo "[2/3] pip install mlx-lm"
python3 -m pip install -U mlx-lm

echo "[2/3] Spinning up Apple Silicon GPU for LoRA Fine-Tuning..."
echo "      Training ${ITERATIONS} iterations. You will hear your Mac's fans spin up."
echo "----------------------------------------------------"

python3 -m mlx_lm.lora \
  --model "${BASE_MODEL}" \
  --train \
  --data "${DATA_DIR}" \
  --iters "${ITERATIONS}" \
  --batch-size 2 \
  --adapter-path "${ADAPTER_PATH}"

echo "----------------------------------------------------"
echo "✅ Fine-tuning complete. Adapters saved to ${ADAPTER_PATH}."
echo

echo "[3/3] Fusing learned weights into the baseline model..."
python3 -m mlx_lm.fuse \
  --model "${BASE_MODEL}" \
  --adapter-path "${ADAPTER_PATH}" \
  --save-path "${FINAL_MODEL_PATH}"

echo "===================================================="
echo "🚀 FUSION COMPLETE: New Autonomous Agent deployed to ${FINAL_MODEL_PATH}"
echo "===================================================="
