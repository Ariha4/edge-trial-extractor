#!/usr/bin/env bash
# Convert the merged fine-tuned model to GGUF and make 8-bit and 4-bit versions.
#
# Needs (on your Mac):
#   brew install llama.cpp                       # gives the llama-quantize binary
#   git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
#   pip install -r ~/llama.cpp/requirements.txt  # for the converter script
#
# Usage:
#   bash scripts/quantize.sh models/ft-merged ~/llama.cpp
set -euo pipefail

MERGED_MODEL_DIR="${1:-models/ft-merged}"
LLAMA_CPP_DIR="${2:-$HOME/llama.cpp}"
OUT_DIR="models"

mkdir -p "$OUT_DIR"

echo "1/3 converting to GGUF (16-bit)"
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_MODEL_DIR" \
  --outfile "$OUT_DIR/ft-f16.gguf" --outtype f16

echo "2/3 quantizing to 8-bit (Q8_0)"
llama-quantize "$OUT_DIR/ft-f16.gguf" "$OUT_DIR/ft-q8_0.gguf" Q8_0

echo "3/3 quantizing to 4-bit (Q4_K_M)"
llama-quantize "$OUT_DIR/ft-f16.gguf" "$OUT_DIR/ft-q4_k_m.gguf" Q4_K_M

ls -lh "$OUT_DIR"/*.gguf
