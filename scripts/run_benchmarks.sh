#!/usr/bin/env bash
# Run the full benchmark grid on your Mac. Each run is its own process so peak
# memory is measured per model. Results land in results/summary.csv.
#
# Usage: bash scripts/run_benchmarks.sh [limit]
#   limit = number of test examples (default 0 = all). Try 20 first as a smoke test.
set -euo pipefail

LIMIT="${1:-0}"
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

rm -f results/summary.csv

# 1. Base model, no fine-tuning: the zero-shot baseline
python -m trialx.evaluate --run-name base_fp16 --backend hf --model "$BASE_MODEL" \
  --dtype float16 --device mps --limit "$LIMIT"

# 2. Fine-tuned, full Hugging Face checkpoint
python -m trialx.evaluate --run-name ft_fp16 --backend hf --model models/ft-merged \
  --dtype float16 --device mps --limit "$LIMIT"

# 3-5. Fine-tuned GGUF on CPU with llama.cpp: 16-bit, 8-bit, 4-bit
python -m trialx.evaluate --run-name ft_gguf_f16 --backend gguf --model models/ft-f16.gguf --limit "$LIMIT"
python -m trialx.evaluate --run-name ft_gguf_q8 --backend gguf --model models/ft-q8_0.gguf --limit "$LIMIT"
python -m trialx.evaluate --run-name ft_gguf_q4 --backend gguf --model models/ft-q4_k_m.gguf --limit "$LIMIT"

echo
column -s, -t < results/summary.csv
